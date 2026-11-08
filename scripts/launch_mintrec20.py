#!/usr/bin/env python3
"""Train MIntRec-20 on four GPUs, then evaluate the validation-selected checkpoint."""
from __future__ import annotations

import argparse
import csv
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

from launch_experiments import PROJECT_ENV, ROOT, full_checkpoint, stamp, stop_owned_processes
from evaluate import resolve_evaluation_checkpoint
from smarthome_efficiency.training import load_config, validate_resume


def check_gpus(indices):
    query = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=20)
    devices = {int(row[0]): {'index': int(row[0]), 'uuid': row[1].strip(),
                            'used_mib': int(row[2]), 'utilization_percent': int(row[3])}
               for row in csv.reader(io.StringIO(query.stdout)) if row}
    if any(index not in devices for index in indices):
        raise RuntimeError(f'Requested GPU indices are unavailable: {indices}')
    selected = [devices[index] for index in indices]
    query = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=20)
    uuids = {device['uuid'] for device in selected}
    occupied = [row for row in csv.reader(io.StringIO(query.stdout)) if row and row[0].strip() in uuids]
    if occupied or any(d['used_mib'] > 512 or d['utilization_percent'] > 5 for d in selected):
        raise RuntimeError(f'Selected GPUs are occupied; nothing was launched: {selected}')
    return selected


def run_phase(command, log_path, environment):
    print(f'Running: {shlex.join(command)}\nLog: {log_path}', flush=True)
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                time.sleep(1)
            if process.returncode != 0:
                raise RuntimeError(f'Process exited {process.returncode}; see {log_path}')
        finally:
            # Stop only this launcher's process group on interruption or failure.
            stop_owned_processes([{'process': process}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Default: configs/mintrec20.yaml; resume defaults to saved config')
    parser.add_argument('--output-root', type=Path, default=ROOT/'artifacts/full/mintrec20_5epochs')
    parser.add_argument('--gpus', default='0,1,2,3', help='Four physical GPU indices')
    parser.add_argument('--resume', action='store_true', help='Continue the latest checkpoint under output-root/train')
    parser.add_argument('--epochs', type=int, help='Target total epochs; increasing on resume restarts the remaining schedule')
    parser.add_argument('--smoke', action='store_true', help='One epoch on 24 train/20 dev clips; final evaluation uses dev only')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    indices = [int(x) for x in args.gpus.split(',')]
    if len(indices) != 4 or len(set(indices)) != 4 or min(indices) < 0:
        parser.error('--gpus must contain four distinct nonnegative indices')
    if args.smoke and args.resume:
        parser.error('--smoke and --resume cannot be combined')
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    output = args.output_root.expanduser().resolve()
    train_output = output/'train'
    checkpoint = full_checkpoint(train_output) if args.resume else None
    saved = json.loads((checkpoint/'metadata.json').read_text()) if checkpoint else None
    config = (dict(saved['config']) if saved and args.config is None else
              load_config(args.config or ROOT/'configs/mintrec20.yaml'))
    if config['task'] != 'mintrec20':
        parser.error('This launcher requires task: mintrec20')
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.smoke:
        config.update(epochs=1, train_limit=24, dev_limit=20)
    if saved:
        validate_resume(saved['config'], config, saved['world_size'], 4)
    elif output.exists():
        raise FileExistsError(f'Output already exists: {output}; use a new directory or --resume')
    launch_id = stamp()
    launch_dir = output/'launches'/launch_id
    config_path = launch_dir/'config.yaml'
    # A subset run must never trigger automatic official-test evaluation.
    subset = config['train_limit'] is not None or config['dev_limit'] is not None
    split = 'dev' if subset else 'test'
    evaluation_output = output/'evaluations'/launch_id
    prefix = [str(PROJECT_ENV/'bin/python'), '-u', '-m', 'torch.distributed.run',
              '--standalone', '--nnodes=1', '--nproc_per_node=4']
    train_command = prefix + [str(ROOT/'scripts/train.py'), '--config', str(config_path),
                              '--output', str(train_output)]
    if checkpoint:
        train_command.extend(['--resume', str(checkpoint)])
    evaluation_command = prefix + [str(ROOT/'scripts/evaluate.py'), '--checkpoint', str(train_output/'checkpoints/best'),
                                   '--split', split, '--output', str(evaluation_output)]
    if subset and config['dev_limit'] is not None:
        evaluation_command.extend(['--limit', str(config['dev_limit'])])
    print(f'MIntRec-20: GPUs {indices}, epochs {config["epochs"]}, eval {config["eval_modes"]}, final split {split}', flush=True)
    if args.dry_run:
        for command in (train_command, evaluation_command):
            print(shlex.join(command))
        return 0
    gpu_state = check_gpus(indices)
    launch_dir.mkdir(parents=True, exist_ok=False)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=','.join(map(str, indices)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS=str(config['num_threads']), MKL_NUM_THREADS=str(config['num_threads']),
                       PATH=str(PROJECT_ENV/'bin')+os.pathsep+environment.get('PATH',''))
    manifest = {'status': 'training', 'config': config, 'gpus': indices, 'gpu_state_before_launch': gpu_state,
                'train_command': train_command, 'evaluation_command': evaluation_command,
                'evaluation_split': split, 'evaluation_output': str(evaluation_output)}
    manifest_path = launch_dir/'launch.json'

    def save():
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2)+'\n')
        temporary.replace(manifest_path)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        save()
        run_phase(train_command, launch_dir/'train.log', environment)
        report = json.loads((train_output/'report.json').read_text())
        if report['status'] != 'completed':
            raise RuntimeError('Training did not complete; final evaluation was not started')
        best = resolve_evaluation_checkpoint(train_output)
        metadata = json.loads((best/'metadata.json').read_text())
        if metadata['global_step'] != report['best_step']:
            raise RuntimeError('Best-checkpoint pointer disagrees with the training report')
        evaluation_command[evaluation_command.index('--checkpoint')+1] = str(best)
        manifest.update(status='evaluating', selected_checkpoint=str(best))
        save()
        run_phase(evaluation_command, launch_dir/'evaluation.log', environment)
        result = json.loads((evaluation_output/'report.json').read_text())
        if result['status'] != 'completed':
            raise RuntimeError('Evaluation did not complete')
        pointer = output/f'{split}-latest'
        temporary = output/f'.{split}-latest-{launch_id}'
        temporary.symlink_to(os.path.relpath(evaluation_output, output))
        temporary.replace(pointer)
        manifest['status'] = 'completed'
        print(json.dumps({'status': 'completed', 'split': split, 'report': str(evaluation_output/'report.json'),
                          **{mode: result[mode] for mode in config['eval_modes']}}, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
