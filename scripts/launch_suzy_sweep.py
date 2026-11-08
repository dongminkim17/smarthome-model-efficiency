#!/usr/bin/env python3
"""Seven independent single-GPU LoRA runs on Suzy GPUs 1-7; validation only."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import yaml

from launch_experiments import PROJECT_ENV, ROOT, full_checkpoint, stamp, stop_owned_processes
from smarthome_efficiency.training import load_config, validate_resume

# gpu, name, task, seed, target, learning rate
EXPERIMENTS = (
    (1, 'mer_qv_seed42', 'mer6', 42, 'qv', 1e-4),
    (2, 'mer_qv_seed43', 'mer6', 43, 'qv', 1e-4),
    (3, 'mer_linear_lr1e4_seed42', 'mer6', 42, 'decoder_linear', 1e-4),
    (4, 'mer_linear_lr5e5_seed42', 'mer6', 42, 'decoder_linear', 5e-5),
    (5, 'mintrec2_qv_seed43', 'mintrec2', 43, 'qv', 1e-4),
    (6, 'mintrec2_qv_seed44', 'mintrec2', 44, 'qv', 1e-4),
    (7, 'mintrec20_linear_seed42', 'mintrec20', 42, 'decoder_linear', 1e-4),
)


def write_json(path, value):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fresh_plans(output, smoke=False, epochs=None):
    plans = []
    for gpu, name, task, seed, target, rate in EXPERIMENTS:
        config = load_config(ROOT / 'configs' / f'{task}.yaml')
        config.update(seed=seed, split_seed=42, epochs=epochs or 5,
                      micro_batch_size=1, gradient_accumulation_steps=4,
                      lora_rank=16, lora_alpha=32, lora_dropout=0.0,
                      lora_target=target, learning_rate=rate,
                      eval_modes=['logit', 'greedy'], selection_mode='logit',
                      eval_every_fraction=0.5, train_limit=None, dev_limit=None)
        if smoke:
            config.update(epochs=1, train_limit=24, dev_limit=20)
        plans.append({'name': name, 'gpu': gpu, 'config': config,
                      'output': str(output / name), 'status': 'pending'})
    return plans


def check_gpus(indices):
    query = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    devices = {}
    for row in csv.reader(io.StringIO(query.stdout)):
        if row:
            index, uuid, name, total, used, utilization = [v.strip() for v in row]
            devices[int(index)] = dict(index=int(index), uuid=uuid, name=name,
                                       total_mib=int(total), used_mib=int(used),
                                       utilization_percent=int(utilization))
    if any(i not in devices for i in indices):
        raise RuntimeError(f'Unavailable GPU indices: {indices}')
    selected = [devices[i] for i in indices]
    query = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=20)
    uuids = {d['uuid'] for d in selected}
    occupied = [r for r in csv.reader(io.StringIO(query.stdout)) if r and r[0].strip() in uuids]
    if occupied or any(d['used_mib'] > 512 or d['utilization_percent'] > 5 for d in selected):
        raise RuntimeError(f'Selected GPUs are occupied; nothing launched: {selected}, {occupied}')
    return {d['index']: d for d in selected}


def child_environment(config, gpu_uuid):
    environment = os.environ.copy()
    # A physical UUID avoids CUDA enumeration differences on Suzy's mixed GPU types.
    environment.update(CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       RANK='0', LOCAL_RANK='0', WORLD_SIZE='1', LOCAL_WORLD_SIZE='1',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS=str(config['num_threads']),
                       MKL_NUM_THREADS=str(config['num_threads']),
                       PATH=str(PROJECT_ENV / 'bin') + os.pathsep + environment.get('PATH', ''))
    return environment


def completion(plan, returncode, smoke):
    if returncode != 0:
        return {'status': 'failed', 'returncode': returncode}
    try:
        report = json.loads((Path(plan['output']) / 'report.json').read_text())
        expected = 'paused' if smoke else 'completed'
        if report['status'] != expected or report['global_step'] < (2 if smoke else 1):
            raise ValueError(f'Unexpected training status/step: {report.get("status")}/{report.get("global_step")}')
        if report.get('test_set_used') is not False:
            raise ValueError('Sweep training must not use test data')
        return {'status': 'completed', 'returncode': returncode,
                'training_status': report['status'], 'global_step': report['global_step'],
                'best_score': report['best_score'], 'best_step': report['best_step'],
                'best_checkpoint': str((Path(plan['output']) / 'checkpoints/best').resolve()),
                'training_seconds': report.get('total_elapsed_seconds')}
    except (OSError, ValueError, KeyError) as error:
        return {'status': 'failed', 'returncode': returncode, 'error': str(error)}


def prepare_resume(plans, epochs, micro_batch_size=None, gradient_checkpointing=None):
    """Use saved scientific configs; resume unfinished runs and skip completed ones."""
    for plan in plans:
        output = Path(plan['output'])
        saved = None
        if (output / 'checkpoints/latest').exists():
            checkpoint = full_checkpoint(output)
            saved = json.loads((checkpoint / 'metadata.json').read_text())
            config = dict(saved['config'])
            if epochs is not None:
                config['epochs'] = epochs
            if micro_batch_size is not None:
                config.update(micro_batch_size=micro_batch_size, gradient_accumulation_steps=4 // micro_batch_size)
            if gradient_checkpointing is not None:
                config['gradient_checkpointing'] = gradient_checkpointing
            validate_resume(saved['config'], config, saved['world_size'], 1)
            plan['config'] = config
            plan['resume_checkpoint'] = str(checkpoint)
        elif output.exists() and any(output.iterdir()):
            raise ValueError(f'{output}: partial run has no checkpoint; use a new sweep directory')
        else:
            if epochs is not None:
                plan['config']['epochs'] = epochs
            if micro_batch_size is not None:
                plan['config'].update(micro_batch_size=micro_batch_size, gradient_accumulation_steps=4 // micro_batch_size)
        if gradient_checkpointing is not None:
            plan['config']['gradient_checkpointing'] = gradient_checkpointing
        report_path = output / 'report.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        finished = (saved is not None and report.get('status') == 'completed'
                    and report.get('config', {}).get('epochs') == plan['config']['epochs'])
        plan['status'] = 'completed' if finished else 'pending'


def apply_execution_overrides(plans, overrides):
    """Allow only compute settings, preserving the four-example update budget."""
    names = {p['name'] for p in plans}
    if not isinstance(overrides, dict) or set(overrides) - names:
        raise ValueError('Execution overrides must map known job names to settings')
    for plan in plans:
        values = overrides.get(plan['name'], {})
        if not isinstance(values, dict) or set(values) - {'micro_batch_size', 'gradient_checkpointing'}:
            raise ValueError('Only micro_batch_size and gradient_checkpointing may be overridden')
        config = dict(plan['config'])
        if 'micro_batch_size' in values:
            size = values['micro_batch_size']
            if type(size) is not int or size not in (1, 2, 4):
                raise ValueError('micro_batch_size must be 1, 2, or 4')
            config.update(micro_batch_size=size, gradient_accumulation_steps=4 // size)
        if 'gradient_checkpointing' in values:
            if type(values['gradient_checkpointing']) is not bool:
                raise ValueError('gradient_checkpointing must be boolean')
            config['gradient_checkpointing'] = values['gradient_checkpointing']
        validate_resume(plan['config'], config, 1, 1)
        plan['config'] = config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--execution-overrides', type=Path, help='JSON per-job microbatch/checkpointing settings')
    parser.add_argument('--epochs', type=int, help='Total epoch target (default 5); may extend on resume')
    parser.add_argument('--gradient-checkpointing', choices=['on', 'off'], help='Trade activation memory for recomputation')
    parser.add_argument('--micro-batch-size', type=int, choices=[1, 2, 4], help='Clips per forward; accumulation preserves effective batch four')
    parser.add_argument('--resume', action='store_true', help='Resume an existing sweep using saved configs')
    parser.add_argument('--smoke', action='store_true', help='24 train/20 dev examples and two updates per GPU')
    parser.add_argument('--dry-run', action='store_true', help='Preview without filesystem writes or GPU use')
    parser.add_argument('--expected-host', default='suzy', help='Refuse a real launch on another host')
    args = parser.parse_args()
    checkpointing = None if args.gradient_checkpointing is None else args.gradient_checkpointing == 'on'
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.resume and (args.output_root is None or args.smoke):
        parser.error('--resume requires --output-root and cannot be combined with --smoke')
    output = (args.output_root or ROOT / 'artifacts/full' / f'suzy_sweep_{stamp()}').expanduser().resolve()
    if args.resume:
        manifest = json.loads((output / 'sweep.json').read_text())
        if manifest.get('smoke'):
            parser.error('Smoke sweeps cannot be continued as full experiments')
        plans = manifest['jobs']
        if [(p['gpu'], p['name']) for p in plans] != [(e[0], e[1]) for e in EXPERIMENTS]:
            raise ValueError('Saved sweep does not match the seven-job experiment layout')
        prepare_resume(plans, args.epochs, args.micro_batch_size, checkpointing)
    else:
        if output.exists():
            raise FileExistsError(f'Output already exists: {output}; choose a new path or --resume')
        plans = fresh_plans(output, args.smoke, args.epochs)
        if args.micro_batch_size is not None:
            for plan in plans:
                plan['config'].update(micro_batch_size=args.micro_batch_size, gradient_accumulation_steps=4 // args.micro_batch_size)
        if checkpointing is not None:
            for plan in plans:
                plan['config']['gradient_checkpointing'] = checkpointing
        manifest = {'smoke': args.smoke, 'jobs': plans, 'created_at_utc': stamp(),
                    'official_test_evaluation': False, 'world_size_per_job': 1,
                    'effective_batch_size': 4}
    if args.execution_overrides:
        apply_execution_overrides(plans, json.loads(args.execution_overrides.read_text()))
    for plan in plans:
        c = plan['config']
        print(f"GPU {plan['gpu']}: {plan['name']}, {c['epochs']} epochs, "
              f"{c['lora_target']}, lr={c['learning_rate']}, seed={c['seed']}, "
              f"batch=4 (micro={c['micro_batch_size']}, accumulation={c['gradient_accumulation_steps']}); {plan['status']}", flush=True)
    print(f'Output: {output}; automatic validation only; official tests remain separate.', flush=True)
    if args.dry_run:
        return 0
    host = socket.gethostname().split('.')[0]
    if host != args.expected_host:
        raise RuntimeError(f'Expected host {args.expected_host}, found {host}; no jobs launched')
    pending = [p for p in plans if p['status'] != 'completed']
    if not pending:
        print('All runs already completed at the requested epoch target.', flush=True)
        return 0
    if not (PROJECT_ENV / 'bin/python').is_file():
        raise FileNotFoundError(PROJECT_ENV / 'bin/python')
    for plan in pending:
        if not (Path(plan['config']['model']) / 'config.json').is_file():
            raise FileNotFoundError(plan['config']['model'])
        if not Path(plan['config']['data_root']).is_dir():
            raise FileNotFoundError(plan['config']['data_root'])
    gpu_state = check_gpus([p['gpu'] for p in pending])
    output.mkdir(parents=True, exist_ok=args.resume)
    lock = (output / '.launcher.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another launcher owns this sweep')
    launch_id = stamp()
    launch_dir = output / 'launches' / launch_id
    launch_dir.mkdir(parents=True, exist_ok=False)
    manifest.pop('error', None)
    manifest.pop('finished_at_utc', None)
    manifest.update(status='running', host=host, launcher_pid=os.getpid(),
                    latest_launch=str(launch_dir), gpu_state_before_launch=gpu_state,
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__).resolve(), ROOT / 'scripts/train.py']})
    active = []

    def save():
        write_json(output / 'sweep.json', manifest)
        write_json(launch_dir / 'sweep.json', manifest)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        save()
        for plan in pending:
            for key in ('returncode', 'error', 'finished_at_utc'):
                plan.pop(key, None)
            path = launch_dir / f"{plan['name']}.yaml"
            path.write_text(yaml.safe_dump(plan['config'], sort_keys=False))
            command = [str(PROJECT_ENV / 'bin/python'), '-u', str(ROOT / 'scripts/train.py'),
                       '--config', str(path), '--output', plan['output']]
            if plan.get('resume_checkpoint'):
                command += ['--resume', plan['resume_checkpoint']]
            if args.smoke:
                command += ['--max-steps', '2']
            log_path = launch_dir / f"{plan['name']}.log"
            environment = child_environment(plan['config'], gpu_state[plan['gpu']]['uuid'])
            with log_path.open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            active.append({'process': process, 'plan': plan})
            plan.update(status='running', pid=process.pid, command=command, log=str(log_path),
                        started_at_utc=stamp(), gpu_uuid=gpu_state[plan['gpu']]['uuid'])
            save()
            print(f"Started {plan['name']} on GPU {plan['gpu']}; PID {process.pid}; log {log_path}", flush=True)
        while active:
            for job in list(active):
                code = job['process'].poll()
                if code is None:
                    continue
                plan = job['plan']
                plan.update(completion(plan, code, args.smoke), finished_at_utc=stamp())
                active.remove(job)
                save()
                print(f"{plan['name']}: {plan['status']} (exit {code})", flush=True)
            if active:
                time.sleep(2)
        manifest['status'] = 'completed' if all(p['status'] == 'completed' for p in plans) else 'failed'
        manifest['finished_at_utc'] = stamp()
        save()
        return 0 if manifest['status'] == 'completed' else 1
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=f'{type(error).__name__}: {error}')
        raise
    finally:
        stop_owned_processes(active)
        for job in active:
            job['plan'].update(status='interrupted', returncode=job['process'].returncode)
        save()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        lock.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
