#!/usr/bin/env python3
"""Independent worker supervisor: durable identity/result and lifetime GPU lease."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from smarthome_efficiency.queueing import atomic_json, exclusive_lock, now, process_identity, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.descriptor)
    attempt = args.descriptor.parent
    result = {'job_id': spec['job_id'], 'started_at': now(), 'wrapper_pid': os.getpid()}
    started = time.monotonic()
    code = 1
    try:
        lease = exclusive_lock(spec['lease']) if spec.get('lease') else nullcontext()
        with lease as lease_stream:
            env = os.environ.copy()
            env.update(spec.get('environment', {}))
            child = None
            try:
                with (attempt / 'stdout.log').open('a') as stream:
                    # The child keeps the lease even if its supervisor is killed.
                    # This closes the spawn-before-worker-metadata crash window.
                    descriptors = (lease_stream.fileno(),) if lease_stream else ()
                    child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=env,
                                             stdout=stream, stderr=subprocess.STDOUT,
                                             start_new_session=True, pass_fds=descriptors)
                    identity = {'job_id': spec['job_id'], 'wrapper_pid': os.getpid(),
                                'wrapper_identity': process_identity(os.getpid()), 'child_pid': child.pid,
                                'child_identity': process_identity(child.pid), 'command': spec['command'],
                                'started_at': result['started_at']}
                    atomic_json(attempt / 'worker.json', identity)
            finally:
                # A metadata/reporting failure cannot release ownership while
                # the native training process is still writing its artifacts.
                if child is not None:
                    code = child.wait()
            result['returncode'] = code
    except BaseException as error:
        code = 1
        result.update(returncode=1, supervisor_error=f'{type(error).__name__}: {error}')
    finally:
        result.update(finished_at=now(), elapsed_seconds=time.monotonic() - started)
        atomic_json(attempt / 'result.json', result)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
