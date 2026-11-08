"""Persistent job graph primitives. Standard library only; no model/GPU imports."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import errno
import hashlib
import json
import os
from pathlib import Path
import time

SCHEMA_VERSION = 1
TERMINAL = {'completed', 'failed', 'blocked', 'cancelled'}
ACTIVE = {'starting', 'running'}


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def is_transient_io(error):
    return isinstance(error, OSError) and error.errno in {
        errno.ESTALE, errno.EIO, errno.ETIMEDOUT, errno.EAGAIN,
    }


def retry_io(operation, attempts=4):
    """Reopen after transient NFS errors; never reuse a stale file descriptor."""
    for index in range(attempts):
        try:
            return operation()
        except OSError as error:
            if not is_transient_io(error) or index == attempts-1:
                raise
            time.sleep(.05 * 2**index)


def read_json(path):
    return retry_io(lambda: json.loads(Path(path).read_text()))


def atomic_json(path, value):
    path = Path(path)
    def commit():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        with tmp.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
    retry_io(commit)


@contextmanager
def exclusive_lock(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield stream


def process_identity(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'start_ticks': fields[19],
                'command': (root / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(pid, identity):
    return identity is not None and process_identity(pid) == identity


def validate_graph(jobs):
    ids = [j['id'] for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate job IDs')
    by_id = {j['id']: j for j in jobs}
    visiting, visited = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError('Dependency cycle')
        if key in visited:
            return
        if key not in by_id:
            raise ValueError(f'Missing dependency: {key}')
        visiting.add(key)
        for dep in by_id[key].get('deps', []):
            visit(dep)
        visiting.remove(key)
        visited.add(key)
    for key in ids:
        visit(key)
    return by_id


def refresh_readiness(jobs):
    by_id = validate_graph(jobs)
    changed = True
    while changed:
        changed = False
        for job in jobs:
            if job['status'] not in ('waiting', 'ready'):
                continue
            deps = [by_id[d] for d in job.get('deps', [])]
            if job.get('kind') == 'shortlist':
                unmet = [d['id'] for d in deps if d['status'] not in TERMINAL]
                job.update(status='waiting' if unmet else 'ready', blocked_by=unmet,
                           reason='Waiting for all candidates to finish' if unmet else None)
                continue
            failed = [d['id'] for d in deps if d['status'] in ('failed', 'blocked')]
            cancelled = [d['id'] for d in deps if d['status'] == 'cancelled']
            unmet = [d['id'] for d in deps if d['status'] != 'completed']
            if failed:
                job.update(status='blocked', blocked_by=failed, reason='A prerequisite failed')
                changed = True
            elif cancelled:
                job.update(status='cancelled', blocked_by=cancelled, reason='A prerequisite was cancelled', finished_at=now())
                changed = True
            elif unmet:
                job.update(status='waiting', blocked_by=unmet, reason='Waiting for prerequisites')
            else:
                job.update(status='ready', blocked_by=[], reason=None)
    return by_id


def reservations(jobs):
    devices = {}
    for job in jobs:
        if job['status'] in ACTIVE and job.get('gpu') is not None:
            if job['gpu'] in devices:
                raise ValueError(f'Duplicate GPU reservation: {job["gpu"]}')
            devices[job['gpu']] = job['id']
    return devices


def classify_failure(text):
    value = text.lower()
    if 'out of memory' in value or 'memory headroom' in value:
        return 'memory'
    if any(token in value for token in ('stale file handle', 'input/output error', 'connection timed out',
                                         'resource temporarily unavailable', 'worker failed to start')):
        return 'transient'
    return 'deterministic'


def memory_fallback(config):
    result = dict(config)
    batch = config['micro_batch_size'] * config['gradient_accumulation_steps']
    result.update(micro_batch_size=1, gradient_accumulation_steps=batch, gradient_checkpointing=True)
    return result


def scientific_fingerprint(config):
    return fingerprint({k: v for k, v in config.items()
                        if k not in {'micro_batch_size', 'gradient_accumulation_steps', 'gradient_checkpointing'}})


def phase_from_log(path, previous=None):
    if not path or not Path(path).exists():
        return previous or {'phase': 'loading', 'updated_at': now()}
    with Path(path).open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 24000))
        lines = stream.read().splitlines()
    event = None
    for line in reversed(lines):
        try:
            item = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if item.get('event'):
            event = item
            break
    if event is None:
        return previous or {'phase': 'loading', 'updated_at': now()}
    kind = event['event']
    phase = {'evaluation_progress': 'evaluating', 'preparing_inputs': 'preparing_inputs', 'train': 'training', 'validation': 'validating', 'checkpoint': 'saving',
             'run_start': 'preparing_or_validating', 'run_resume': 'preparing_or_training',
             'run_complete': 'finalizing'}.get(kind, kind)
    if kind == 'phase':
        phase = event.get('phase', phase)
    return {'phase': phase, 'event': kind, 'step': event.get('global_step'),
            'epoch': event.get('epoch'), 'elapsed_seconds': event.get('elapsed_seconds'),
            'last_event_unix': event.get('recorded_at') or Path(path).stat().st_mtime,
            'completed': event.get('local_done', event.get('completed')),
            'total': event.get('local_total', event.get('total'))}


class StateStore:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'state.json'

    def load(self):
        state = read_json(self.path)
        if state.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported queue schema')
        validate_graph(state['jobs'])
        reservations(state['jobs'])
        return state

    def save(self, state):
        validate_graph(state['jobs'])
        reservations(state['jobs'])
        state['heartbeat_at'] = now()
        atomic_json(self.path, state)

    def event(self, event, **values):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'at': now(), 'event': event, **values}, allow_nan=False) + '\n')


def estimate_schedule(jobs, durations, gpu_count=7, cpu_count=1):
    """Greedy dependency/resource estimate; failures are reported, not timed away."""
    validate_graph(jobs)
    gpu = [0.] * gpu_count
    cpu = [0.] * cpu_count
    finished = {j['id']: 0. for j in jobs if j['status'] == 'completed'}
    excluded = [j['id'] for j in jobs if j['status'] in ('failed', 'blocked')]
    for job in jobs:
        if job['status'] in ACTIVE:
            seconds = max(0., durations[job['id']])
            finished[job['id']] = seconds
            slots = gpu if job['resource'] == 'gpu' else cpu
            index = job.get('gpu', 1)-1 if job['resource'] == 'gpu' else slots.index(min(slots))
            slots[index] = max(slots[index], seconds)
    cancelled = [j['id'] for j in jobs if j['status'] == 'cancelled']
    pending = [j for j in jobs if j['id'] not in finished and j['id'] not in excluded and j['id'] not in cancelled]
    while pending:
        ready = [j for j in pending if all(d in finished for d in j.get('deps', []))]
        if not ready:
            break
        def earliest(job):
            deps = max((finished[d] for d in job.get('deps', [])), default=0.)
            slots = gpu if job['resource'] == 'gpu' else cpu if job['resource'] == 'cpu' else [0.]
            return max(deps, min(slots))
        job = min(ready, key=lambda j: (earliest(j), -j.get('priority', 0), j['id']))
        start = earliest(job)
        end = start + durations[job['id']]
        if job['resource'] in ('gpu', 'cpu'):
            slots = gpu if job['resource'] == 'gpu' else cpu
            slots[slots.index(min(slots))] = end
        finished[job['id']] = end
        pending.remove(job)
    return {'seconds': max(finished.values(), default=0.), 'failed_or_blocked': excluded, 'cancelled': cancelled,
            'unresolved': [j['id'] for j in pending],
            'note': 'Approximate dependency/resource simulation; excludes failed branches and future external GPU contention.'}


def cancel_unstarted_runs(jobs, training_ids, reason):
    """Cancel requested scientific runs and their private pending gates, never signal processes.

    A root with any full-training attempt is preserved together with its test.
    Shared exports/caches and completed or active smoke gates are untouched.
    Validate the entire request before changing any job.
    """
    by_id = validate_graph(jobs)
    if (not isinstance(training_ids, list) or not training_ids or len(set(training_ids)) != len(training_ids)
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError('Cancellation requires unique training IDs and an explicit reason')
    for key in training_ids:
        if key not in by_id or by_id[key]['kind'] != 'recovery_train':
            raise ValueError('Cancellation root must be a known recovery_train job: '+str(key))
        if not by_id[key]['payload'].get('arm') or not by_id[key].get('task'):
            raise ValueError('Cancellation root needs a task and scientific arm')
    result = {'cancelled': [], 'skipped': []}
    for key in training_ids:
        root = by_id[key]
        if root['status'] == 'cancelled':
            result['skipped'].append({'id':key,'reason':'already cancelled'});continue
        if root['status'] not in ('waiting','ready') or root.get('attempts'):
            result['skipped'].append({'id':key,'reason':'full training started or is terminal; preserve entire run'});continue
        family = key.split('/')[0]
        for job in jobs:
            if (job['id'].split('/')[0] != family or job.get('task') != root['task']
                    or job['payload'].get('arm') != root['payload']['arm']
                    or job['kind'] not in ('smoke_train','smoke_test','recovery_train','recovery_test')
                    or job['status'] not in ('waiting','ready')):
                continue
            job.update(status='cancelled',reason=reason,cancelled_at=now(),finished_at=now(),blocked_by=[])
            result['cancelled'].append(job['id'])
    refresh_readiness(jobs)
    return result


def reprioritize_unstarted_jobs(jobs, updates, reason):
    """Durably adjust scheduling only; never alter active attempts or configs."""
    by_id = validate_graph(jobs)
    if not isinstance(updates, list) or not updates or not isinstance(reason, str) or not reason.strip():
        raise ValueError('Priority changes require job updates and an explicit reason')
    seen = set()
    for value in updates:
        if (not isinstance(value, dict) or set(value) != {'id','priority'}
                or not isinstance(value['id'], str) or value['id'] not in by_id
                or value['id'] in seen or type(value['priority']) is not int):
            raise ValueError('Priority updates must contain unique known job IDs and integer priorities')
        seen.add(value['id'])
    result = {'changed': [], 'skipped': []}
    for value in updates:
        job = by_id[value['id']]
        if job['status'] not in ('waiting','ready') or job.get('attempts'):
            result['skipped'].append({'id':job['id'],'reason':'already started or terminal'})
            continue
        result['changed'].append({'id':job['id'],'old_priority':job['priority'],'new_priority':value['priority']})
        job['priority'] = value['priority']
    return result
