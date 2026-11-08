#!/usr/bin/env python3
"""Queue A1–A4 after existing experiments, smoke all arms, then train and test."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import yaml

from launch_experiments import PROJECT_ENV, ROOT, stamp, stop_owned_processes
from launch_pruning_sweep import available_gpus
from launch_pruning_followup import checkpoint_identity, pin_best
from launch_suzy_sweep import child_environment, write_json
from gpu_queue_coordination import allocation_lock
from recovery_experiments import TASKS, ARMS, CONTROL_ARMS, DESCRIPTIONS, queued_jobs, select_sources, materialize_jobs
from smarthome_efficiency.training import config_fingerprint, validate_resume

DEFAULT_NAIVE = ROOT / 'artifacts/full/naive_pruning_20261019'
DEFAULT_CALIBRATED = ROOT / 'artifacts/full/calibrated_pruning_20261019'


def prerequisite_state(paths):
    waiting = []
    for path in paths:
        if not path.exists():
            waiting.append({'path': str(path), 'status': 'not_started'})
            continue
        payload = json.loads(path.read_text())
        if payload['status'] in ('failed', 'interrupted'):
            raise ValueError(f'Prior experiment needs attention: {path} ({payload["status"]})')
        if payload['status'] != 'completed':
            waiting.append({'path': str(path), 'status': payload['status']})
    return waiting


def validate_pinned_sources(selected):
    if set(selected) != set(TASKS):
        raise ValueError('Pinned source selection must cover all three tasks')
    for task in TASKS:
        entry = selected[task]
        if entry.get('test_results_used') is not False or entry.get('distillation_results_used') is not False:
            raise ValueError('Sources must have been selected from SFT validation only')
        winner = entry['winner']
        if (checkpoint_identity(winner['checkpoint']) != winner['checkpoint_identity']
                or pin_best(winner['sft_run']) != winner['checkpoint']):
            raise ValueError('Pinned source adapter/checkpoint changed')


def refresh_tests(jobs, smoke):
    by_id = {job['id']: job for job in jobs}
    for job in jobs:
        if job['kind'] != 'test' or job['status'] not in ('waiting', 'ready'):
            continue
        training = by_id[job['deps'][0]]
        if training['status'] in ('failed', 'blocked'):
            job.update(status='blocked', error='Training prerequisite failed')
        elif training['status'] == 'completed':
            checkpoint = pin_best(training['run_dir'], allow_paused=smoke)
            if checkpoint is None:
                raise ValueError('Completed training has no validation-selected checkpoint')
            job.update(checkpoint=checkpoint, checkpoint_identity=checkpoint_identity(checkpoint), status='ready')


def validate_result(job, smoke):
    run = Path(job['run_dir'])
    report = json.loads((run / 'report.json').read_text())
    if job['kind'] == 'train':
        cfg = job['config']
        if (report['status'] != ('paused' if smoke else 'completed')
                or report['test_set_used'] is not False
                or config_fingerprint(report['config']) != job['config_sha256']
                or report['objective'] != cfg['objective']):
            raise ValueError('Training report does not match the requested experiment')
        if report['global_step'] != (2 if smoke else report['requested_steps']):
            raise ValueError('Training did not finish the requested updates')
        provenance = json.loads((run / 'provenance.json').read_text())
        if provenance.get('init_adapter') != job['init_adapter']:
            raise ValueError('Training adapter initialization does not match the arm')
        if smoke:
            events = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
            updates = [e for e in events if e['event'] == 'train']
            if len(updates) != 2 or any(not math.isfinite(e['loss']) for e in updates):
                raise ValueError('Smoke optimizer updates were not finite')
            # Verify that the adapter was actually updated and serialized.
            from safetensors import safe_open
            import torch
            checkpoint = Path(report['latest_checkpoint'])
            with safe_open(checkpoint / 'adapter/adapter_model.safetensors', framework='pt', device='cpu') as weights:
                if not any(torch.count_nonzero(weights.get_tensor(key)).item()
                           for key in weights.keys() if 'lora_B' in key):
                    raise ValueError('Smoke adapter has no nonzero update')
        return {'report': str(run / 'report.json'), 'best_score': report['best_score'],
                'best_step': report['best_step'], 'global_step': report['global_step']}
    if checkpoint_identity(job['checkpoint']) != job['checkpoint_identity']:
        raise ValueError('Pinned evaluation checkpoint changed')
    if (report['status'] != 'completed' or report['checkpoint'] != job['checkpoint']
            or report['split'] != job['split'] or report['eval_modes'] != ['logit', 'greedy']
            or report['test_set_used'] != (not smoke)):
        raise ValueError('Evaluation report does not match the pinned checkpoint/protocol')
    if not smoke and (report['partial_split'] or report['evaluated_count'] != report['full_split_count']):
        raise ValueError('Official test evaluation must cover the full split')
    return {'report': str(run / 'report.json'), 'evaluated_count': report['evaluated_count'],
            'logit': report['logit'], 'greedy': report['greedy'], 'checkpoint': job['checkpoint']}


def build_command(job, run, config_path, smoke, resume=None):
    if job['kind'] == 'train':
        command = [str(PROJECT_ENV / 'bin/python'), '-B', '-u', str(ROOT / 'scripts/train.py'),
                   '--config', str(config_path), '--output', str(run)]
        if resume is not None:
            command += ['--resume', str(resume)]
        elif job['init_adapter'] is not None:
            command += ['--init-adapter', job['init_adapter']]
        if smoke:
            command += ['--max-steps', '2']
    else:
        command = [str(PROJECT_ENV / 'bin/python'), '-B', '-u', str(ROOT / 'scripts/evaluate.py'),
                   '--checkpoint', job['checkpoint'], '--split', job['split'], '--modes', 'both', '--output', str(run)]
        if smoke:
            command += ['--limit', '6']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['rank-recovery', 'recovery-controls'], default='rank-recovery')
    parser.add_argument('--source-selection', type=Path, help="Reuse an earlier queue's pinned validation-only sources")
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke-root', type=Path, required=True)
    parser.add_argument('--source-sweeps', type=Path, nargs='+', default=[DEFAULT_NAIVE / 'sweep.json', DEFAULT_CALIBRATED / 'sweep.json'])
    parser.add_argument('--wait-for', type=Path, nargs='+', default=[DEFAULT_NAIVE / 'followup/workflow.json', DEFAULT_CALIBRATED / 'followup/workflow.json'])
    parser.add_argument('--teacher-cache-root', type=Path, default=DEFAULT_NAIVE / 'followup/teachers')
    parser.add_argument('--gpus', default='1,2,3,4,5,6,7')
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    arms = ARMS if args.suite == 'rank-recovery' else CONTROL_ARMS
    if args.suite == 'recovery-controls' and args.source_selection is None:
        parser.error('Recovery controls must reuse --source-selection from the A1–A4 queue')
    indices = [int(value) for value in args.gpus.split(',')]
    if not indices or len(set(indices)) != len(indices) or any(i not in range(1, 8) for i in indices):
        parser.error('Use unique physical Suzy GPU indices 1–7')
    if args.poll_seconds < 1:
        parser.error('Polling interval must be positive')
    output = args.output_root.expanduser().resolve()
    smoke_root = args.smoke_root.expanduser().resolve()
    definition = {'source_sweeps': [str(p.expanduser().resolve()) for p in args.source_sweeps],
                  'wait_for': [str(p.expanduser().resolve()) for p in args.wait_for],
                  'teacher_cache_root': str(args.teacher_cache_root.expanduser().resolve()),
                  'smoke_root': str(smoke_root), 'arms': list(arms), 'tasks': list(TASKS)}
    if args.source_selection is not None:
        definition['source_selection'] = str(args.source_selection.expanduser().resolve())
    definition_sha = config_fingerprint(definition)
    if args.resume:
        manifest = json.loads((output / 'workflow.json').read_text())
        if manifest.get('controller') == 'project_queue':
            raise RuntimeError('This batch is managed by scripts/project_queue.py; resume the project queue instead')
        if manifest['definition_sha256'] != definition_sha:
            raise ValueError('Cannot change recovery experiment inputs on resume')
        for job in manifest['jobs'] + manifest.get('smoke_jobs', []):
            if job['status'] != 'completed':
                job['status'] = 'ready' if job['kind'] == 'train' and job.get('config') else 'waiting'
    else:
        manifest = {'format_version': 1, 'created_at_utc': stamp(), 'definition': definition,
                    'definition_sha256': definition_sha, 'descriptions': {a: DESCRIPTIONS[a] for a in arms},
                    'status': 'waiting_for_existing_experiments', 'jobs': queued_jobs(output, arms),
                    'smoke_jobs': [], 'phase': 'waiting', 'A5_enabled': False,
                    'test_results_used_for_training_or_source_selection': False,
                    'shared_gpu_allocation': True, 'allowed_physical_gpus': indices}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if socket.gethostname().split('.')[0] != 'suzy':
        raise RuntimeError('Run this queue on Suzy')
    output.mkdir(parents=True, exist_ok=args.resume)
    lock = (output / '.workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch_dir = output / 'launches' / stamp()
    launch_dir.mkdir(parents=True)
    manifest.update(launcher_pid=os.getpid(), host=socket.gethostname(), latest_launch=str(launch_dir))
    manifest.pop('error', None)
    active = []
    pools = []

    def save():
        write_json(output / 'workflow.json', manifest)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}

    def start(job, gpu, smoke):
        for key in ('error', 'returncode'):
            job.pop(key, None)
        if job.get('init_adapter') is not None:
            source = Path(job['init_adapter']).parent
            if checkpoint_identity(source) != job['init_checkpoint_identity']:
                raise ValueError('Pinned SFT initialization changed')
        resume = None
        if job['kind'] == 'train':
            cfg = job['config']
            if job.get('run_dir') and (Path(job['run_dir']) / 'checkpoints/latest').exists():
                run = Path(job['run_dir'])
                resume = (run / 'checkpoints/latest').resolve()
                metadata = json.loads((resume / 'metadata.json').read_text())
                validate_resume(metadata['config'], cfg, metadata['world_size'], 1)
            else:
                job['attempt'] = job.get('attempt', 0) + 1
                run = Path(job['output']) / f'attempt-{job["attempt"]:03d}'
            config_path = launch_dir / f'{"smoke" if smoke else "full"}_{job["name"]}.yaml'
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        else:
            if checkpoint_identity(job['checkpoint']) != job['checkpoint_identity']:
                raise ValueError('Pinned evaluation checkpoint changed')
            cfg = json.loads((Path(job['checkpoint']) / 'metadata.json').read_text())['config']
            job['attempt'] = job.get('attempt', 0) + 1
            run = Path(job['output']) / f'attempt-{job["attempt"]:03d}'
            config_path = None
        command = build_command(job, run, config_path, smoke, resume)
        log = launch_dir / f'{"smoke" if smoke else "full"}_{job["id"].replace(":", "_")}.log'
        env = child_environment(cfg, gpu['uuid'])
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        job.update(status='running', gpu=gpu['index'], gpu_uuid=gpu['uuid'], pid=process.pid,
                   run_dir=str(run), command=command, log=str(log), started_at_utc=stamp())
        active.append({'process': process, 'plan': job})
        save()
        print(f"Started {'smoke' if smoke else 'full'} {job['id']} on GPU {gpu['index']}", flush=True)

    try:
        save()
        paths = [Path(p) for p in definition['source_sweeps'] + definition['wait_for']]
        while True:
            waiting = prerequisite_state(paths)
            manifest['waiting_for'] = waiting
            if not waiting:
                break
            manifest['status'] = 'waiting_for_existing_experiments'
            save()
            time.sleep(args.poll_seconds)
        # Existing queue coordinators own these pools until their final cleanup.
        for name in ('.pruning_gpu_pool.lock', '.pruning_followup_gpu_pool.lock'):
            pool = (ROOT / 'artifacts' / name).open('a')
            pools.append(pool)
            while True:
                try:
                    fcntl.flock(pool, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    manifest['status'] = 'waiting_for_prior_queue_cleanup'
                    save()
                    time.sleep(args.poll_seconds)
        if not manifest.get('selected_sources'):
            sweeps = [json.loads(Path(p).read_text()) for p in definition['source_sweeps']]
            if definition.get('source_selection'):
                selected = json.loads(Path(definition['source_selection']).read_text())
                validate_pinned_sources(selected)
            else:
                selected = select_sources(sweeps)
            manifest.update(selected_sources=selected,
                            jobs=materialize_jobs(output, selected, definition['teacher_cache_root'], arms=arms),
                            smoke_jobs=materialize_jobs(smoke_root, selected, definition['teacher_cache_root'], smoke=True, arms=arms),
                            phase='smoke')
            write_json(output / 'source_selection.json', selected)
            save()
        for phase in ('smoke', 'full'):
            smoke = phase == 'smoke'
            jobs = manifest['smoke_jobs'] if smoke else manifest['jobs']
            manifest.update(phase=phase, status='smoke_testing' if smoke else 'running')
            save()
            while True:
                for owned in list(active):
                    code = owned['process'].poll()
                    if code is None:
                        continue
                    active.remove(owned)
                    job = owned['plan']
                    try:
                        if code != 0:
                            raise RuntimeError(f'Process exited {code}; inspect {job["log"]}')
                        job.update(result=validate_result(job, smoke), status='completed',
                                   returncode=code, finished_at_utc=stamp())
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        job.update(status='failed', error=str(error), returncode=code)
                    save()
                refresh_tests(jobs, smoke)
                if any(j['status'] in ('failed', 'blocked') for j in jobs):
                    raise RuntimeError(f'{phase} prerequisite failed; inspect workflow/logs before resuming')
                if all(j['status'] == 'completed' for j in jobs):
                    break
                with allocation_lock(ROOT):
                    reserved = {owned['plan']['gpu'] for owned in active}
                    free = [g for g in available_gpus(indices) if g['index'] not in reserved]
                    ready = sorted((j for j in jobs if j['status'] == 'ready'),
                                   key=lambda j: (j['kind'] != 'test', j['task'] != 'mer6', j['name']))
                    for job, gpu in zip(ready, free):
                        start(job, gpu, smoke)
                save()
                time.sleep(args.poll_seconds)
            if smoke:
                manifest['gpu_smoke_passed'] = True
                write_json(output / 'gpu_smoke_verification.json',
                           {'passed': True, 'training_configs': len(arms) * len(TASKS),
                            'dev_reload_evaluations': len(arms) * len(TASKS),
                            'optimizer_updates_per_config': 2, 'test_set_used': False})
                save()
        manifest.update(status='completed', phase='completed', finished_at_utc=stamp())
        write_json(output / 'results.json', {j['id']: j['result'] for j in manifest['jobs']})
        with (output / 'results.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['arm', 'task', 'source', 'logit_accuracy', 'logit_weighted_f1',
                             'greedy_accuracy', 'greedy_weighted_f1', 'report'])
            for j in manifest['jobs']:
                if j['kind'] != 'test':
                    continue
                result = j['result']
                winner = manifest['selected_sources'][j['task']]['winner']
                writer.writerow([j['arm'], j['task'], winner['name'], result['logit']['accuracy'],
                                 result['logit']['weighted_f1'], result['greedy']['accuracy'],
                                 result['greedy']['weighted_f1'], result['report']])
        save()
        return 0
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=f'{type(error).__name__}: {error}')
        raise
    finally:
        stop_owned_processes(active)
        for owned in active:
            owned['plan'].update(status='interrupted')
        save()
        for pool in pools:
            pool.close()
        lock.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
