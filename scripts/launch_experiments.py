#!/usr/bin/env python3
"""Launch the two independent four-GPU fine-tuning jobs on rose."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV = Path(sys.prefix)
TASKS = ('mintrec2', 'mer6')
GPU_SETS = {'mintrec2': '0,1,2,3', 'mer6': '4,5,6,7'}
sys.path.insert(0, str(ROOT / 'src'))
from smarthome_efficiency.training import load_config, validate_resume


def stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def read_config(path: Path, task: str) -> dict:
    config = load_config(path)
    if config['task'] != task:
        raise ValueError(f'{path}: expected task: {task}')
    return config


def full_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / 'checkpoints/latest').exists():
        path = (path / 'checkpoints/latest').resolve()
    elif (path / 'latest').exists():
        path = (path / 'latest').resolve()
    for name in ('metadata.json', 'training_state.pt', 'adapter/adapter_model.safetensors'):
        if not (path / name).is_file():
            raise ValueError(f'Incomplete full checkpoint: missing {path / name}; use adapter initialization for weights only')
    return path


def latest_checkpoint(task_root: Path) -> Path:
    pointer = task_root / 'checkpoints' / 'latest'
    if pointer.is_dir():
        return pointer.resolve()
    if pointer.is_file():
        target = pointer.read_text(encoding='utf-8').strip()
        if not target:
            raise ValueError(f'Empty latest checkpoint pointer: {pointer}')
        path = Path(target)
        if not path.is_absolute():
            path = pointer.parent / path
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f'Latest checkpoint target does not exist: {path}')
    # A .json pointer is accepted for compatibility with trainer metadata.
    json_pointer = pointer.with_suffix('.json')
    if json_pointer.is_file():
        value = json.loads(json_pointer.read_text(encoding='utf-8'))
        target = value.get('path') if isinstance(value, dict) else value
        if not isinstance(target, str) or not target:
            raise ValueError(f'Expected a checkpoint path in {json_pointer}')
        path = Path(target)
        if not path.is_absolute():
            path = json_pointer.parent / path
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f'Latest checkpoint target does not exist: {path}')
    raise FileNotFoundError(f'No latest checkpoint pointer found in {pointer.parent}')


def check_gpus() -> list[dict]:
    """Refuse occupied devices; this never terminates existing compute processes."""
    query = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=20,
    )
    devices = []
    for row in csv.reader(io.StringIO(query.stdout)):
        if not row:
            continue
        index, used, total, utilization = (int(part.strip()) for part in row)
        devices.append({'index': index, 'used_mib': used, 'total_mib': total, 'utilization_percent': utilization})
    by_index = {device['index']: device for device in devices}
    if not all(index in by_index for index in range(8)):
        raise RuntimeError('This paired launch needs rose GPU indices 0 through 7')
    processes = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=20,
    )
    active = [line for line in processes.stdout.splitlines() if line.strip() and not line.strip().lower().startswith('no running')]
    if active:
        raise RuntimeError('Existing GPU compute processes detected; no experiments were launched. Check nvidia-smi and retry when all eight GPUs are free.')
    occupied = [device for device in devices if device['index'] < 8 and (device['used_mib'] > 512 or device['utilization_percent'] > 5)]
    if occupied:
        raise RuntimeError(f'GPU memory/utilization indicates occupied devices; no experiments were launched: {occupied}')
    return [by_index[index] for index in range(8)]


def stop_owned_processes(jobs: list[dict]) -> None:
    """Signal only process groups created by this invocation."""
    for job in jobs:
        try:
            os.killpg(job['process'].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(job['process'].poll() is None for job in jobs):
        time.sleep(0.2)
    for job in jobs:
        try:
            os.killpg(job['process'].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        job['process'].wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mintrec-config', type=Path)
    parser.add_argument('--mer-config', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--resume-root', type=Path, help='Resume both tasks from their saved configs and latest checkpoints')
    parser.add_argument('--resume-mintrec', type=Path, help='Explicit full checkpoint for MIntRec')
    parser.add_argument('--resume-mer', type=Path, help='Explicit full checkpoint for MER')
    parser.add_argument('--init-adapter-mintrec', '--init-mintrec-adapter', dest='init_mintrec_adapter', type=Path, help='Warm-start adapter weights without optimizer state')
    parser.add_argument('--init-adapter-mer', '--init-mer-adapter', dest='init_mer_adapter', type=Path, help='Warm-start adapter weights without optimizer state')
    parser.add_argument('--epochs', type=int, help='Override target total epochs for both jobs')
    parser.add_argument('--max-steps', type=int, help='Absolute optimizer-step stopping cap for both jobs, including resumed steps')
    parser.add_argument('--smoke', action='store_true', help='Use two-epoch small-subset configs and stop after two optimizer steps')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved jobs; do not write files, check GPUs, or launch processes')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.smoke and (args.resume_root or args.resume_mintrec or args.resume_mer):
        parser.error('--smoke cannot be combined with resume; resume must retain its original dataset and schedule')

    launch_id = stamp()
    resume_root = args.resume_root.resolve() if args.resume_root else None
    output_root = (args.output_root or resume_root or ROOT / 'artifacts' / 'full' / launch_id).resolve()
    if output_root.exists() and not (resume_root or args.resume_mintrec or args.resume_mer):
        raise FileExistsError(f'Output root already exists; choose a new path or explicitly resume: {output_root}')
    launch_dir = output_root / 'launches' / launch_id
    plans = []
    for task, supplied, explicit_resume, init_adapter in (
        ('mintrec2', args.mintrec_config, args.resume_mintrec, args.init_mintrec_adapter),
        ('mer6', args.mer_config, args.resume_mer, args.init_mer_adapter),
    ):
        checkpoint = explicit_resume.resolve() if explicit_resume else latest_checkpoint(resume_root / task) if resume_root else None
        metadata = None
        if checkpoint is not None:
            checkpoint = full_checkpoint(checkpoint)
            metadata = json.loads((checkpoint / 'metadata.json').read_text(encoding='utf-8'))
        if checkpoint is not None and supplied is None:
            # Checkpoint metadata, including any previous epoch extension, is authoritative.
            source = checkpoint / 'metadata.json'
            config = dict(metadata['config'])
            if config.get('task') != task:
                raise ValueError(f'{checkpoint}: expected checkpoint for {task}')
        else:
            source = supplied or ROOT / 'configs' / f'{task}.yaml'
            config = read_config(source.resolve(), task)
        if args.epochs is not None:
            config['epochs'] = args.epochs
        if args.smoke:
            config['epochs'] = 2
            config['train_limit'] = 10 if task == 'mintrec2' else 14
            config['dev_limit'] = 5 if task == 'mintrec2' else 7
            config['max_new_tokens'] = 8
        if checkpoint is not None:
            validate_resume(metadata['config'], config, metadata['world_size'], 4)
        if (output_root / task).exists() and checkpoint is None:
            raise FileExistsError(f'{task}: output already exists and no full checkpoint was provided: {output_root / task}')
        if checkpoint is not None and init_adapter:
            parser.error(f'{task}: choose either full resume or adapter initialization')
        if init_adapter and not init_adapter.is_dir():
            raise FileNotFoundError(init_adapter)
        resolved_path = launch_dir / f'{task}.yaml'
        command = [str(PROJECT_ENV / 'bin' / 'python'), '-u', '-m', 'torch.distributed.run',
                   '--standalone', '--nnodes=1', '--nproc_per_node=4', '--rdzv-endpoint=localhost:0',
                   str(ROOT / 'scripts' / 'train.py'), '--config', str(resolved_path),
                   '--output', str(output_root / task)]
        if checkpoint is not None:
            command.extend(['--resume', str(checkpoint)])
        if init_adapter:
            command.extend(['--init-adapter', str(init_adapter.resolve())])
        step_cap = args.max_steps if args.max_steps is not None else 2 if args.smoke else None
        if step_cap is not None:
            command.extend(['--max-steps', str(step_cap)])
        plans.append({'task': task, 'source_config': str(source.resolve()), 'config': config,
                      'config_path': str(resolved_path), 'command': command,
                      'gpus': GPU_SETS[task], 'log_path': str(launch_dir / f'{task}.log'),
                      'resume': str(checkpoint) if checkpoint else None})
    print(f'Output root: {output_root}', flush=True)
    for plan in plans:
        global_batch = 4 * int(plan['config']['micro_batch_size']) * int(plan['config']['gradient_accumulation_steps'])
        print(f"{plan['task']}: GPUs {plan['gpus']}, global batch {global_batch}, eval {plan['config']['eval_modes']}, total epochs {plan['config']['epochs']}", flush=True)
        print(f"CUDA_VISIBLE_DEVICES={plan['gpus']} {shlex.join(plan['command'])}", flush=True)
    if args.dry_run:
        print('Dry run only: GPU availability has not been checked and no files or processes were created.', flush=True)
        return 0
    if not (PROJECT_ENV / 'bin' / 'python').is_file():
        raise FileNotFoundError(f'Project Python is missing: {PROJECT_ENV}')
    if not (ROOT / 'scripts' / 'train.py').is_file():
        raise FileNotFoundError(ROOT / 'scripts' / 'train.py')
    gpu_state = check_gpus()
    launch_dir.mkdir(parents=True, exist_ok=False)
    (output_root / 'configs').mkdir(exist_ok=True)
    for plan in plans:
        text = yaml.safe_dump(plan['config'], sort_keys=False)
        Path(plan['config_path']).write_text(text, encoding='utf-8')
        original_config = output_root / 'configs' / f"{plan['task']}.yaml"
        if not original_config.exists():
            original_config.write_text(text, encoding='utf-8')
    manifest_path = launch_dir / 'launch.json'
    manifest = {'launch_id': launch_id, 'output_root': str(output_root), 'smoke': args.smoke,
                'gpu_state_before_launch': gpu_state, 'jobs': plans, 'status': 'starting'}

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        temporary.replace(manifest_path)

    received_signal = None

    def on_signal(signum, frame):
        nonlocal received_signal
        received_signal = signum

    old_handlers = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    jobs = []
    return_code = 1
    try:
        save_manifest()
        for plan in plans:
            environment = os.environ.copy()
            environment.update({'CUDA_VISIBLE_DEVICES': plan['gpus'], 'HF_HUB_OFFLINE': '1',
                                'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1',
                                'OMP_NUM_THREADS': str(plan['config'].get('num_threads', 2)),
                                'MKL_NUM_THREADS': str(plan['config'].get('num_threads', 2)),
                                'PATH': str(PROJECT_ENV / 'bin') + os.pathsep + environment.get('PATH', '')})
            log = Path(plan['log_path']).open('w', encoding='utf-8')
            try:
                process = subprocess.Popen(plan['command'], cwd=ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            jobs.append({'task': plan['task'], 'process': process, 'log': log})
            plan['pid'] = process.pid
            print(f"Started {plan['task']} (launcher PID {process.pid}); log: {plan['log_path']}", flush=True)
        manifest['status'] = 'running'
        save_manifest()
        while True:
            if received_signal is not None:
                manifest['status'] = 'interrupted'
                manifest['signal'] = received_signal
                return_code = 128 + received_signal
                break
            exit_codes = {job['task']: job['process'].poll() for job in jobs}
            failed = {task: code for task, code in exit_codes.items() if code is not None and code != 0}
            if failed:
                manifest['status'] = 'failed'
                manifest['failed_exit_codes'] = failed
                print(f'A job failed: {failed}. Stopping only this invocation\'s process groups; inspect the per-task logs.', flush=True)
                return_code = 1
                break
            if all(code == 0 for code in exit_codes.values()):
                manifest['status'] = 'completed'
                return_code = 0
                break
            time.sleep(1)
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if return_code != 0:
            stop_owned_processes(jobs)
        for job in jobs:
            job['log'].close()
        manifest['exit_codes'] = {job['task']: job['process'].poll() for job in jobs}
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save_manifest()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        print(f"Launch {manifest['status']}. Details: {manifest_path}", flush=True)
    return return_code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
