#!/usr/bin/env python3
"""Persistent per-job scheduler for approved Suzy experiments; stop detaches workers."""
from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from smarthome_efficiency.queueing import (
    ACTIVE, TERMINAL, StateStore, atomic_json, classify_failure, exclusive_lock, fingerprint,
    memory_fallback, now, phase_from_log, process_identity, refresh_readiness, reservations,
    same_process, scientific_fingerprint, estimate_schedule, validate_graph, is_transient_io,
)
from project_queue_jobs import (
    CALIBRATED, NAIVE, RANK, CONTROLS, TASKS, configure_recovery, import_project,
    native_command, read, select_sources, validate_native,
)
from launch_experiments import PROJECT_ENV
from launch_pruning_sweep import available_gpus
from launch_suzy_sweep import child_environment
from smarthome_efficiency.training import config_fingerprint

DEFAULT_ROOT = ROOT/'artifacts/queue/suzy'


def execution_host(state):
    expected = state.get('host', 'suzy')
    actual = socket.gethostname().split('.')[0]
    if actual != expected:
        raise RuntimeError(f'This queue belongs to {expected}; current host is {actual}')
    indices = state['allowed_gpus']
    if (not indices or len(indices) != len(set(indices))
            or any(type(i) is not int or i < 0 or i > 7 for i in indices)
            or (expected == 'suzy' and 0 in indices)):
        raise ValueError('Invalid approved GPU pool; Suzy GPU0 remains excluded')
    return expected



def tail(path, limit=12000):
    if not path or not Path(path).exists():
        return ''
    with Path(path).open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-limit))
        return stream.read().decode(errors='replace')


def validate_submission(state, spec):
    """Validate an additive submission; False means an already committed replay."""
    if not isinstance(spec, dict) or spec.get('schema_version') != 1:
        raise ValueError('Unsupported submission schema')
    if fingerprint(spec) in {s['sha256'] for s in state.get('submissions', [])}:
        return False
    additions = spec.get('jobs')
    documents = spec.get('documents', {})
    if not isinstance(additions, list) or not additions or not isinstance(documents, dict):
        raise ValueError('A submission needs a nonempty jobs list and optional documents object')
    resources = {'select': 'inline', 'prepare': 'cpu', 'prune_export': 'gpu', 'followup': 'gpu',
                 'recovery_train': 'gpu', 'recovery_test': 'gpu', 'encoder_cache': 'gpu',
                 'smoke_train': 'gpu', 'smoke_test': 'gpu', 'shortlist': 'inline', 'checkpoint_average':'gpu', 'merge_export':'gpu', 'benchmark_comparison':'cpu'}
    for job in additions:
        if not isinstance(job, dict) or job.get('kind') not in resources:
            raise ValueError('New submissions must use registered job types')
        if (job.get('status') not in ('waiting', 'ready') or job.get('attempts') != []
                or job.get('retry_count') != 0 or job.get('resource') != resources[job['kind']]
                or not isinstance(job.get('id'), str) or not job['id']
                or job.get('task') not in TASKS or not isinstance(job.get('payload'), dict)
                or not isinstance(job.get('deps'), list)
                or not all(isinstance(d, str) for d in job['deps'])
                or not isinstance(job.get('priority'), int)
                or any(job.get(k) is not None for k in ('pid', 'wrapper_pid', 'gpu', 'attempt_dir'))
                or job.get('imported')):
            raise ValueError('Submitted jobs must be unstarted, with valid resource and entry fields')
        if job['payload'].get('arm') == 'A5':
            raise ValueError('A5 is deliberately deferred; it is not enabled by the queue')
    validate_graph(state['jobs'] + additions)
    if set(documents).intersection(state['documents']):
        raise ValueError('Submission cannot replace an existing batch manifest')
    return True


class OwnershipPersistenceError(Exception):
    """Detach safely rather than mark a spawned live worker failed."""


class Scheduler:
    def __init__(self, root):
        self.store = StateStore(root)
        self.state = self.store.load()
        self.state['queue_root'] = str(Path(root).resolve())
        self.by_id = {j['id']: j for j in self.state['jobs']}
        self.processes = {}
        self.last_publish = 0
        self.dirty = True

    def save(self):
        self.store.save(self.state)

    def event(self, name, **values):
        self.store.event(name, **values)
        self.dirty = True

    def fail(self, job, message):
        category = classify_failure(message)
        retry = (not job.get('imported') and job['retry_count'] < self.state['policy']['max_retries']
                 and category in ('transient', 'memory'))
        if category == 'memory':
            cfg = job['payload'].get('config')
            retry = retry and cfg is not None and job['kind'] in ('smoke_train', 'recovery_train', 'followup')
            if retry:
                changed = memory_fallback(cfg)
                retry = changed != cfg
                if retry:
                    if scientific_fingerprint(changed) != scientific_fingerprint(cfg):
                        raise ValueError('Execution fallback changed scientific settings')
                    job['payload'].update(config=changed, config_sha256=config_fingerprint(changed))
                    job['execution_override'] = {k: changed[k] for k in
                                                 ('micro_batch_size', 'gradient_accumulation_steps', 'gradient_checkpointing')}
        job.setdefault('errors', []).append({'at': now(), 'category': category, 'message': message})
        for key in ('gpu', 'gpu_uuid', 'pid', 'identity', 'wrapper_pid', 'wrapper_identity', 'attempt_dir'):
            job.pop(key, None)
        if retry:
            job.update(status='waiting', retry_count=job['retry_count']+1, reason='Retry after '+category)
            self.event('retry_scheduled', job=job['id'], category=category, execution_override=job.get('execution_override'))
        else:
            job.update(status='failed', reason=message, finished_at=now())
            self.event('job_failed', job=job['id'], category=category, message=message)
        self.save()

    def complete(self, job, exit_code=None):
        result = validate_native(job)
        if job['kind'] == 'smoke_train':
            events = [json.loads(line) for line in (Path(job['payload']['run_dir'])/'metrics.jsonl').read_text().splitlines()]
            peak = max(e.get('peak_allocated_gib', 0) for e in events)
            limit = job.get('memory_total_gib', 24.) - self.state['policy']['memory_headroom_gib']
            if peak > limit:
                raise RuntimeError(f'Memory headroom insufficient: {peak:.3f} GiB > {limit:.3f} GiB')
            result.update(peak_allocated_gib=peak, memory_limit_gib=limit,
                          memory_probe_manifest=job['payload']['memory_cases'])
        job.update(status='completed', result=result, finished_at=now(),
                   validated_completion=True, exit_code=exit_code)
        job['payload'].update(status='completed', result=result, finished_at_utc=now())
        if job.get('gpu') is not None:
            job['last_gpu'] = job['gpu']
        for key in ('gpu', 'gpu_uuid'):
            job.pop(key, None)
        self.event('job_completed', job=job['id'], exit_code=exit_code,
                   exit_status_known=exit_code is not None)
        self.save()

    def ingest(self):
        inbox = self.store.root/'inbox'
        inbox.mkdir(exist_ok=True)
        for path in sorted(inbox.glob('*.json')):
            try:
                spec = read(path)
                if not validate_submission(self.state, spec):
                    path.rename(path.with_suffix('.accepted'))
                    continue
                additions = spec['jobs']
                candidate = copy.deepcopy(self.state)
                candidate['jobs'].extend(additions)
                candidate['documents'].update(spec.get('documents', {}))
                candidate.setdefault('submissions', []).append({'path': str(path), 'sha256': fingerprint(spec), 'at': now()})
                self.store.save(candidate)
                self.state = candidate
                self.by_id = {j['id']: j for j in self.state['jobs']}
                self.event('submission_accepted', path=str(path), jobs=len(additions))
                path.rename(path.with_suffix('.accepted'))
            except (ValueError, KeyError) as error:
                self.event('submission_rejected', path=str(path), error=str(error))
                path.rename(path.with_suffix('.rejected'))

    def ingest_controls(self):
        from smarthome_efficiency.queueing import cancel_unstarted_runs, reprioritize_unstarted_jobs
        folder = self.store.root/'control_requests'
        folder.mkdir(exist_ok=True)
        for path in sorted(folder.glob('*.json')):
            try:
                request = read(path)
                digest = fingerprint(request)
                if digest in {r['sha256'] for r in self.state.get('control_requests', [])}:
                    path.rename(path.with_suffix('.accepted'));continue
                if request.get('schema_version') != 1 or request.get('action') not in ('cancel_unstarted_runs','reprioritize_unstarted_jobs'):
                    raise ValueError('Unsupported queue control request')
                candidate = copy.deepcopy(self.state)
                if request['action'] == 'cancel_unstarted_runs':
                    result = cancel_unstarted_runs(candidate['jobs'], request.get('training_jobs'), request.get('reason'))
                else:
                    result = reprioritize_unstarted_jobs(candidate['jobs'], request.get('updates'), request.get('reason'))
                candidate.setdefault('control_requests', []).append(
                    {'sha256':digest,'path':str(path),'at':now(),'action':request['action'],'result':result})
                self.store.save(candidate)
                self.state = candidate
                self.by_id = {j['id']:j for j in self.state['jobs']}
                self.event('control_request_accepted', path=str(path), result=result)
                path.rename(path.with_suffix('.accepted'))
            except (ValueError, KeyError, TypeError) as error:
                self.event('control_request_rejected', path=str(path), error=str(error))
                path.rename(path.with_suffix('.rejected'))

    def observe(self):
        for pid, process in list(self.processes.items()):
            if process.poll() is not None:
                process.wait()
                self.processes.pop(pid)
        for job in list(self.state['jobs']):
            if job['status'] not in ACTIVE:
                continue
            try:
                job.pop('observation_error', None)
                if job.get('imported'):
                    if same_process(job['pid'], job.get('identity')):
                        job['progress'] = phase_from_log(job.get('log'), job.get('progress'))
                        continue
                    # Unknown exit code: complete only from a strictly valid native report.
                    self.complete(job, None)
                    continue
                attempt = Path(job['attempt_dir'])
                result_path, worker_path = attempt/'result.json', attempt/'worker.json'
                if result_path.exists():
                    result = read(result_path)
                    if result.get('job_id') != job['id']:
                        raise ValueError('Worker result belongs to another job')
                    if result['returncode'] != 0:
                        raise RuntimeError(result.get('supervisor_error', '')+'\n'+tail(attempt/'stdout.log'))
                    self.complete(job, result['returncode'])
                    continue
                if worker_path.exists():
                    worker = read(worker_path)
                    if worker['job_id'] != job['id'] or worker['command'] != job['payload']['command']:
                        raise ValueError('Worker identity/command mismatch')
                    job.update(pid=worker['child_pid'], identity=worker['child_identity'],
                               wrapper_pid=worker['wrapper_pid'], wrapper_identity=worker['wrapper_identity'])
                    if same_process(worker['wrapper_pid'], worker['wrapper_identity']) or same_process(worker['child_pid'], worker['child_identity']):
                        job.update(status='running', progress=phase_from_log(attempt/'stdout.log', job.get('progress')))
                        continue
                    self.complete(job, None)
                    continue
                if same_process(job.get('wrapper_pid'), job.get('wrapper_identity')):
                    continue
                # A supervisor can be lost after spawning its child. Keep that
                # child's reservation, including a crash before worker.json.
                if job.get('pid') and job.get('identity'):
                    if same_process(job['pid'], job['identity']):
                        job.update(status='running', progress=phase_from_log(attempt/'stdout.log', job.get('progress')))
                    else:
                        self.complete(job, None)
                    continue
                descriptor = str(attempt/'descriptor.json')
                discovered = None
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit():
                        continue
                    identity = process_identity(int(proc.name))
                    if identity and identity['command'] == job['payload']['command']:
                        job.update(pid=int(proc.name), identity=identity, status='running')
                        discovered = True
                        break
                    if identity and descriptor in identity['command'] and any(str(ROOT/'scripts/queue_worker.py') == arg for arg in identity['command']):
                        job.update(wrapper_pid=int(proc.name), wrapper_identity=identity)
                        discovered = True
                        break
                if discovered:
                    continue
                if time.time()-job['spawn_intent_unix'] < 45:
                    continue
                raise RuntimeError('Worker failed to start; no matching supervisor, child or result')
            except OSError as error:
                if is_transient_io(error):
                    job['observation_error'] = {'at': now(), 'error': str(error)}
                    self.event('observation_deferred', job=job['id'], error=str(error))
                    continue  # Retain ownership; unavailable evidence is not a failed worker.
                self.fail(job, f'{type(error).__name__}: {error}')
            except (ValueError, KeyError, RuntimeError) as error:
                self.fail(job, f'{type(error).__name__}: {error}')

    def plan(self):
        refresh_readiness(self.state['jobs'])
        for job in self.state['jobs']:
            if job['status'] != 'ready' or job.get('planning_retry_after', 0) > time.time():
                continue
            try:
                job.pop('planning_error', None)
                job.pop('planning_retry_after', None)
                if job['kind'] == 'select':
                    candidates = [dict(self.by_id[key]['payload'], status='completed') for key in job['deps']]
                    selected = select_sources([{'status': 'completed', 'jobs': candidates}], tasks=(job['task'],))
                    self.state['selected_sources'][job['task']] = selected[job['task']]
                    job.update(status='completed', result=selected[job['task']], finished_at=now())
                    self.event('source_selected', task=job['task'], source=selected[job['task']]['winner']['name'])
                    self.save()
                elif job['kind'] == 'shortlist':
                    from validation_shortlist import select_validation_shortlist
                    result = select_validation_shortlist(self.state, job)
                    job.update(status='completed', result=result, finished_at=now())
                    self.event('validation_shortlist_selected', job=job['id'], selected=result['selected'])
                    self.save()
                elif job['kind'] in ('smoke_train', 'recovery_train'):
                    configure_recovery(self.state, job)
                    if job['kind'] == 'recovery_train':
                        parent = self.by_id[job['deps'][0]]
                        smoke = self.by_id[parent['deps'][0]]
                        if smoke.get('execution_override'):
                            job['payload']['config'].update(smoke['execution_override'])
                            job['payload']['config_sha256'] = config_fingerprint(job['payload']['config'])
                            job['execution_override'] = smoke['execution_override']
            except OSError as error:
                if is_transient_io(error):
                    job.update(planning_error={'at': now(), 'error': str(error)}, planning_retry_after=time.time()+30)
                    self.event('planning_deferred', job=job['id'], error=str(error))
                    continue
                self.fail(job, f'Planning failed: {error}')
            except (ValueError, KeyError, RuntimeError) as error:
                self.fail(job, f'Planning failed: {error}')
        refresh_readiness(self.state['jobs'])

    def start(self, job, gpu=None):
        if job['status'] != 'ready':
            raise ValueError('Job is not ready')
        if gpu and gpu['index'] not in self.state['allowed_gpus']:
            raise ValueError('GPU is outside the approved pool')
        if gpu and gpu['index'] in reservations(self.state['jobs']):
            raise ValueError('GPU already reserved')
        attempt = self.store.root/'attempts'/job['id'].replace('/', '__').replace(':', '_')/f'{len(job["attempts"])+1:03d}'
        attempt.mkdir(parents=True, exist_ok=False)
        job['attempts'].append({'path': str(attempt), 'started_at': now()})
        command, cfg = native_command(self.state, job, attempt)
        environment = child_environment(cfg, gpu['uuid']) if gpu else {
            'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
            'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        keys = {'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE',
                'LOCAL_WORLD_SIZE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TOKENIZERS_PARALLELISM',
                'PYTHONUNBUFFERED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PATH', 'PYTHONDONTWRITEBYTECODE'}
        environment = {k: v for k, v in environment.items() if k in keys}
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        lease = str(self.store.root/'leases'/f'gpu-{gpu["index"]}.lock') if gpu else None
        descriptor = {'job_id': job['id'], 'command': command, 'cwd': str(ROOT), 'environment': environment, 'lease': lease}
        atomic_json(attempt/'descriptor.json', descriptor)
        job.update(status='starting', attempt_dir=str(attempt), spawn_intent_unix=time.time(), started_at=now(), log=str(attempt/'stdout.log'))
        job['payload'].update(command=command, log=job['log'])
        if gpu:
            job.update(gpu=gpu['index'], gpu_uuid=gpu['uuid'], memory_total_gib=gpu.get('total_gib', 24.))
        self.save()  # Persist the reservation/spawn intent before starting CUDA work.
        worker_command = [str(PROJECT_ENV/'bin/python'), '-B', '-u', str(ROOT/'scripts/queue_worker.py'),
                          '--descriptor', str(attempt/'descriptor.json')]
        with (attempt/'supervisor.log').open('a') as stream:
            process = subprocess.Popen(worker_command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes[process.pid] = process
        job.update(wrapper_pid=process.pid, wrapper_identity=process_identity(process.pid))
        try:
            self.event('job_started', job=job['id'], gpu=job.get('gpu'), attempt=str(attempt))
            self.save()
        except OSError as error:
            raise OwnershipPersistenceError('Spawned worker retains ownership; restart controller after storage recovers') from error

    def dispatch(self):
        active_cpu = sum(j['resource'] == 'cpu' and j['status'] in ACTIVE for j in self.state['jobs'])
        ready = sorted((j for j in self.state['jobs'] if j['status'] == 'ready' and j.get('planning_retry_after', 0) <= time.time()),
                       key=lambda j: (-j['priority'], j['task'] != 'mer6', j['id']))
        for job in ready:
            if job['resource'] == 'cpu' and active_cpu < self.state['policy']['cpu_workers']:
                try:
                    self.start(job)
                    active_cpu += 1
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    self.fail(job, f'Launch failed: {error}')
        host = self.state.get('host', 'suzy')
        allocation = ROOT/'artifacts'/('.shared_gpu_allocation.lock' if host == 'suzy'
                                      else f'.shared_gpu_allocation.{host}.lock')
        with exclusive_lock(allocation, blocking=True):
            occupied = reservations(self.state['jobs'])
            free = available_gpus(self.state['allowed_gpus'], allowed_indices=self.state['allowed_gpus'])
            usable = []
            for gpu in free:
                if gpu['index'] in occupied:
                    continue
                try:
                    with exclusive_lock(self.store.root/'leases'/f'gpu-{gpu["index"]}.lock'):
                        usable.append(gpu)
                except BlockingIOError:
                    pass
            ready_gpu = [j for j in ready if j['resource'] == 'gpu' and j['status'] == 'ready']
            for job, gpu in zip(ready_gpu, usable):
                try:
                    self.start(job, gpu)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    self.fail(job, f'Launch failed: {error}')

    def estimate(self):
        import statistics
        baseline = read(ROOT/'artifacts/full/suzy_sweep_20261019/sweep.json')
        unpruned = {task: [j['training_seconds'] for j in baseline['jobs']
                           if j['config']['task'] == task and j.get('training_seconds')] for task in TASKS}
        durations = {}
        for job in self.state['jobs']:
            p = job['payload']
            if job['kind'] in ('select', 'shortlist'):
                seconds = .1
            elif job['kind'] == 'prepare':
                seconds = 60.
            elif job['kind'] == 'prune_export':
                seconds = float(p.get('estimated_seconds', 180.))
            elif job['kind'] == 'smoke_train':
                seconds = 90.
            elif job['kind'] == 'smoke_test':
                seconds = 25.
            elif job['kind'] == 'recovery_test' or (job['kind'] == 'followup' and p['kind'].endswith('test')):
                seconds = 720. if job['task'] == 'mer6' else 400.
            else:
                seconds = 10800. if job['task'] == 'mer6' else 5400.
                selected = self.state['selected_sources'].get(job['task'])
                if selected:
                    report = read(Path(selected['winner']['sft_run'])/'report.json')
                    seconds = report.get('total_elapsed_seconds', seconds)
                if p.get('arm') == 'A4' and unpruned[job['task']]:
                    seconds = statistics.median(unpruned[job['task']]) * 1.2
                elif p.get('config', {}).get('lora_rank') == 64:
                    seconds *= 1.15
            if p.get('estimated_seconds') is not None:
                seconds = float(p['estimated_seconds'])
            if job['status'] in ACTIVE:
                epoch = job.get('progress', {}).get('epoch')
                if epoch is not None:
                    requested_epochs = float(p.get('config', {}).get('epochs', p.get('training_epochs', 5)))
                    seconds *= max(0., 1 - epoch / requested_epochs)
            durations[job['id']] = seconds
        estimate = estimate_schedule(self.state['jobs'], durations, gpu_count=len(self.state['allowed_gpus']))
        estimate['rough_range_hours'] = [round(estimate['seconds']*.8/3600, 1), round(estimate['seconds']*1.4/3600, 1)]
        return estimate

    def publish(self):
        for path, document in self.state['documents'].items():
            related = [j for j in self.state['jobs'] if (j.get('location') or {}).get('manifest') == path]
            for job in related:
                location = job['location']
                values = document.setdefault(location['field'], [])
                target = next((v for v in values if v.get(location['key_field']) == location['key']), None)
                if target is None:
                    target = dict(job['payload'])
                    values.append(target)
                target.update(job['payload'])
                target.update(status='running' if job['status'] == 'starting' else job['status'])
                if job.get('pid'):
                    target['pid'] = job['pid']
                if job['status'] in ACTIVE and job.get('gpu') is not None:
                    target.update(gpu=job['gpu'], gpu_uuid=job.get('gpu_uuid'))
                if job.get('result'):
                    target['result'] = job['result']
                if job.get('reason') and job['status'] in ('failed', 'blocked'):
                    target['error'] = job['reason']
                if job['status'] == 'cancelled':
                    target['cancellation_reason'] = job.get('reason')
                target['queue_job_id'] = job['id']
                target['progress'] = job.get('progress')
            statuses = [j['status'] for j in related]
            terminal = statuses and all(s in TERMINAL for s in statuses)
            status = ('failed' if any(s in ('failed','blocked') for s in statuses) else
                      'completed_with_cancellations' if 'cancelled' in statuses else 'completed') if terminal else 'running'
            document.update(controller='project_queue', status=status, launcher_pid=os.getpid(),
                            queue_state=str(self.store.path), heartbeat_at=now(), shared_gpu_allocation=True)
            if str(RANK) in path or str(CONTROLS) in path:
                document.update(selected_sources=self.state['selected_sources'], phase='per_job_dependencies')
                atomic_json(Path(path).parent/'source_selection.json', self.state['selected_sources'])
            atomic_json(path, document)
        estimate = self.estimate()
        self.state['eta'] = estimate
        running = [j for j in self.state['jobs'] if j['status'] in ACTIVE]
        lines = ['# Live project queue', '', 'Snapshot: '+now(), '',
                 'Authoritative state: '+str(self.store.path), '',
                 'Rough remaining hours: '+str(estimate['rough_range_hours'])+'; heuristic, not a completion guarantee.', '',
                 '| Job | Resource | Phase | Step |', '| --- | --- | --- | ---: |']
        for job in running:
            progress = job.get('progress', {})
            lines.append(f"| {job['id']} | GPU {job['gpu']} | {progress.get('phase', job['status'])} | {progress.get('step', '—')} |" if job.get('gpu') is not None else f"| {job['id']} | CPU | preparing inputs | — |")
        lines += ['', '## Waiting, failed and cancelled jobs', '']
        for job in self.state['jobs']:
            if job['status'] in ('waiting', 'ready', 'failed', 'blocked', 'cancelled'):
                lines.append(f"- **{job['id']}**: {job['status']}; {job.get('reason') or 'eligible for an available resource'}; prerequisites: {', '.join(job.get('blocked_by', [])) or 'none'}")
        queue_view = 'QUEUE.md' if self.state.get('host', 'suzy') == 'suzy' else 'QUEUE_'+self.state['host']+'.md'
        (ROOT/'experiments'/queue_view).write_text('\n'.join(lines)+'\n')
        result = subprocess.run([str(PROJECT_ENV/'bin/python'), '-B', str(ROOT/'scripts/update_experiment_index.py')],
                                cwd=ROOT, capture_output=True, text=True, timeout=45)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])

    def tick(self, dispatch=True):
        self.ingest()
        self.ingest_controls()
        self.observe()
        self.plan()
        if dispatch:
            self.dispatch()
        terminal = all(j['status'] in TERMINAL for j in self.state['jobs'])
        statuses = [j['status'] for j in self.state['jobs']]
        self.state['status'] = ('completed_with_failures' if any(s in ('failed','blocked') for s in statuses) else
                                'completed_with_cancellations' if 'cancelled' in statuses else 'completed') if terminal else 'running'
        self.save()
        if self.dirty or time.time()-self.last_publish > 30:
            try:
                self.publish()
                self.dirty = False
                self.last_publish = time.time()
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
                self.state['reporting_errors'].append({'at': now(), 'error': str(error)})
                self.event('reporting_error', error=str(error))
                self.save()
        return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['shadow', 'run', 'status', 'pause', 'resume', 'stop', 'inspect', 'submit'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--job')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--poll-seconds', type=int, default=10)
    args = parser.parse_args()
    if args.action == 'shadow':
        state = import_project(args.root)
        refresh_readiness(state['jobs'])
        text = json.dumps(state, indent=2)
        if args.output:
            args.output.write_text(text+'\n')
        else:
            print(text)
        return 0
    if args.action in ('pause', 'resume'):
        marker = args.root/'dispatch.enabled'
        if args.action == 'pause':
            marker.unlink(missing_ok=True)
        else:
            marker.write_text('Dispatch enabled by operator '+now()+'\n')
        print('Running workers continue; dispatch '+('paused' if args.action == 'pause' else 'enabled'))
        return 0
    if args.action == 'inspect':
        state = StateStore(args.root).load()
        job = next((j for j in state['jobs'] if j['id'] == args.job), None)
        if job is None:
            raise ValueError('Specify a known --job ID')
        print(json.dumps(job, indent=2))
        return 0
    if args.action == 'submit':
        if args.spec is None:
            parser.error('--spec is required')
        spec = read(args.spec)
        validate_submission(StateStore(args.root).load(), spec)
        inbox = args.root/'inbox'
        inbox.mkdir(exist_ok=True)
        path = inbox/(fingerprint(spec)+'.json')
        atomic_json(path, spec)
        print('Submitted for scheduler validation: '+str(path))
        return 0
    if args.action == 'stop':
        state = StateStore(args.root).load()
        execution_host(state)
        pid = state.get('scheduler_pid')
        if pid and same_process(pid, state.get('scheduler_identity')):
            os.kill(pid, signal.SIGTERM)
        print('Scheduler stopping; native workers continue')
        return 0
    if args.action == 'status':
        state = StateStore(args.root).load()
        from collections import Counter
        print(json.dumps({'status': state['status'], 'heartbeat': state.get('heartbeat_at'),
                          'counts': dict(Counter(j['status'] for j in state['jobs'])),
                          'running': [{'id':j['id'],'gpu':j.get('gpu'),'progress':j.get('progress')} for j in state['jobs'] if j['status'] in ACTIVE]}, indent=2))
        return 0
    state = StateStore(args.root).load()
    host = execution_host(state)
    with ExitStack() as locks:
        locks.enter_context(exclusive_lock(args.root/'scheduler.lock'))
        # Prevent old coordinators from starting duplicate project work.
        if host == 'suzy':
            for name in ('.pruning_gpu_pool.lock', '.pruning_followup_gpu_pool.lock'):
                locks.enter_context(exclusive_lock(ROOT/'artifacts'/name))
        scheduler = Scheduler(args.root)
        for path in scheduler.state['documents']:
            name = '.launcher.lock' if Path(path).name == 'sweep.json' else '.workflow.lock'
            locks.enter_context(exclusive_lock(Path(path).parent/name))
        stopped = False
        def stop(signum, frame):
            nonlocal stopped
            stopped = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        scheduler.state.update(scheduler_pid=os.getpid(), scheduler_identity=process_identity(os.getpid()))
        scheduler.event('scheduler_started', pid=os.getpid())
        try:
            while not stopped:
                if scheduler.tick(dispatch=(args.root/'dispatch.enabled').exists()):
                    break
                time.sleep(args.poll_seconds)
        finally:
            scheduler.state['scheduler_pid'] = None
            if stopped:
                scheduler.state['status'] = 'detached_workers_continue'
            scheduler.event('scheduler_stopped', workers_preserved=True)
            scheduler.save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
