import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gpu_queue_coordination import AdoptedProcess, allocation_lock, peer_reservations, process_identity
from launch_pruning_sweep import coexist_allocation

class QueueCoordinationTests(unittest.TestCase):
    def test_adoption_observes_worker_without_restarting(self):
        command=[sys.executable,'-c','import time;time.sleep(30)']
        worker=subprocess.Popen(command,start_new_session=True)
        try:
            identity=process_identity(worker.pid)
            adopted=AdoptedProcess({'pid':worker.pid,'command':command,'process_identity':identity})
            self.assertIsNone(adopted.poll())
            with self.assertRaises(ValueError):
                AdoptedProcess({'pid':worker.pid,'command':command,'process_identity':identity|{'start_ticks':'wrong'}})
            worker.terminate();worker.wait(timeout=5)
            self.assertEqual(adopted.poll(),0)
        finally:
            if worker.poll() is None:worker.kill();worker.wait()

    def test_shared_peers_reserve_loading_workers_and_allow_all_devices(self):
        peer={'status':'running','shared_gpu_allocation':True,'jobs':[
            {'status':'running','gpu':4}, {'status':'completed','gpu':6}]}
        self.assertEqual(coexist_allocation(list(range(1,8)),peer),(list(range(1,8)),{4}))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'peer.json';path.write_text(json.dumps(peer))
            self.assertEqual(peer_reservations(path),{4})
        self.assertEqual(peer_reservations(None),set())

    def test_allocation_lock_excludes_other_processes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'artifacts').mkdir()
            command=[sys.executable,'-c',
                'import fcntl,sys;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',
                str(root/'artifacts/.shared_gpu_allocation.lock')]
            with allocation_lock(root):
                result=subprocess.run(command,capture_output=True)
                self.assertNotEqual(result.returncode,0)
            self.assertEqual(subprocess.run(command,capture_output=True).returncode,0)
