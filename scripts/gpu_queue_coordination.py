"""Cross-queue allocation lock and coordinator-only process handoff."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time


@contextmanager
def allocation_lock(root):
    with (Path(root)/'artifacts/.shared_gpu_allocation.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def peer_reservations(path):
    if path is None:
        return set()
    payload=json.loads(Path(path).read_text())
    return {job['gpu'] for job in payload['jobs']
            if job['status']=='running' and job.get('gpu') is not None}


def process_identity(pid):
    path=Path('/proc')/str(pid)
    try:
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':
            return None
        return {'start_ticks':fields[19],
                'command':(path/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')}
    except FileNotFoundError:
        return None


class AdoptedProcess:
    """Observe an existing worker; completion reports validate its outcome.

    A non-child's exit code cannot be recovered. Zero here means it exited;
    the caller must still validate its complete output/report as usual.
    """
    def __init__(self, job):
        self.pid=job['pid']
        self.identity=job.get('process_identity')
        if self.identity is None:
            raise ValueError('Handoff requires a recorded worker identity')
        current=process_identity(self.pid)
        if current is not None and (current!=self.identity or current['command']!=job['command']):
            raise ValueError('Worker identity changed during handoff')
        self.returncode=None

    def poll(self):
        if process_identity(self.pid)!=self.identity:
            self.returncode=0
        return self.returncode

    def wait(self):
        while self.poll() is None:
            time.sleep(.2)
        return self.returncode
