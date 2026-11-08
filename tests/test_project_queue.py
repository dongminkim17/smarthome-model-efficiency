"""Persistent scheduling, real supervisor processes and isolation guarantees."""
import copy
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import (StateStore, atomic_json, validate_graph, refresh_readiness,
    reservations, exclusive_lock, memory_fallback, scientific_fingerprint, process_identity)
from project_queue import Scheduler, OwnershipPersistenceError
from project_queue_jobs import entry


def state_for(jobs, folder):
    value={'schema_version':1,'jobs':jobs,'documents':{},'selected_sources':{},'reporting_errors':[],
           'allowed_gpus':list(range(1,8)),'policy':{'max_retries':1,'memory_headroom_gib':1.,'cpu_workers':1}}
    StateStore(folder).save(value)
    return value


def job(key='a', deps=()):
    return entry(key,'fake',{'config':{'micro_batch_size':2,'gradient_accumulation_steps':2,
                                     'gradient_checkpointing':False,'learning_rate':1e-4}},deps=deps,resource='cpu')


class QueueGraphTests(unittest.TestCase):
    def test_missing_dependencies_cycles_and_duplicate_ids_are_rejected(self):
        for jobs in [[job('a',['missing'])],[job('a',['b']),job('b',['a'])],[job('a'),job('a')]]:
            with self.assertRaises(ValueError):validate_graph(jobs)

    def test_failure_blocks_only_descendants_and_independent_work_stays_ready(self):
        jobs=[job('a'),job('b',['a']),job('c',['b']),job('d')]
        jobs[0]['status']='failed';refresh_readiness(jobs)
        self.assertEqual([j['status'] for j in jobs],['failed','blocked','blocked','ready'])
        self.assertEqual(jobs[1]['blocked_by'],['a'])

    def test_reservations_include_loading_and_reject_double_allocations(self):
        a,b=job('a'),job('b');a.update(status='starting',gpu=3)
        self.assertEqual(reservations([a,b]),{3:'a'})
        b.update(status='running',gpu=3)
        with self.assertRaises(ValueError):reservations([a,b])

    def test_memory_fallback_preserves_scientific_config_and_batch(self):
        cfg=job()['payload']['config'];changed=memory_fallback(cfg)
        self.assertEqual(changed['micro_batch_size'],1)
        self.assertEqual(changed['gradient_accumulation_steps'],4)
        self.assertTrue(changed['gradient_checkpointing'])
        self.assertEqual(scientific_fingerprint(cfg),scientific_fingerprint(changed))

    def test_atomic_store_survives_failed_write_without_replacing_valid_state(self):
        with tempfile.TemporaryDirectory() as folder:
            store=StateStore(folder);state_for([job()],folder)
            original=store.path.read_bytes()
            with patch('smarthome_efficiency.queueing.os.fsync',side_effect=OSError('simulated I/O')):
                with self.assertRaises(OSError):store.save({'schema_version':1,'jobs':[job('new')]})
            self.assertEqual(store.path.read_bytes(),original)
            self.assertEqual(store.load()['jobs'][0]['id'],'a')

    def test_state_read_reopens_after_transient_nfs_error(self):
        with tempfile.TemporaryDirectory() as folder:
            state_for([job()],folder);store=StateStore(folder);valid=store.path.read_text()
            with patch('pathlib.Path.read_text',side_effect=[OSError(errno.ESTALE,'stale file handle'),valid]):
                self.assertEqual(store.load()['jobs'][0]['id'],'a')


class SchedulerTests(unittest.TestCase):
    def test_retry_is_bounded_and_does_not_stop_independent_running_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=job('a'),job('b');a.update(status='running',gpu=3);b.update(status='running',gpu=4)
            a['kind']='smoke_train';state_for([a,b],folder);scheduler=Scheduler(folder)
            a,b=scheduler.state['jobs'];scheduler.fail(a,'CUDA out of memory')
            self.assertEqual(a['status'],'waiting');self.assertEqual(a['retry_count'],1)
            self.assertEqual(a['payload']['config']['micro_batch_size'],1)
            self.assertEqual(b['status'],'running');self.assertEqual(b['gpu'],4)
            scheduler.fail(a,'CUDA out of memory')
            self.assertEqual(a['status'],'failed');self.assertEqual(b['status'],'running')

    def test_reporting_error_does_not_fail_experiments(self):
        with tempfile.TemporaryDirectory() as folder:
            state_for([job()],folder);scheduler=Scheduler(folder)
            with patch.object(scheduler,'publish',side_effect=OSError('reporting unavailable')):
                scheduler.tick(dispatch=False)
            self.assertEqual(scheduler.state['jobs'][0]['status'],'ready')
            self.assertEqual(len(scheduler.state['reporting_errors']),1)

    def test_gpu_zero_is_rejected_before_any_worker_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            a=job();a['status']='ready';state_for([a],folder);scheduler=Scheduler(folder)
            with self.assertRaises(ValueError):scheduler.start(scheduler.state['jobs'][0],{'index':0,'uuid':'excluded'})
            self.assertEqual(scheduler.state['jobs'][0]['attempts'],[])

    def test_imported_vanished_worker_requires_a_valid_completion_report(self):
        with tempfile.TemporaryDirectory() as folder:
            a=job();a.update(status='running',imported=True,pid=999999999,identity=None)
            state_for([a],folder);scheduler=Scheduler(folder)
            with patch('project_queue.validate_native',side_effect=ValueError('incomplete report')):
                scheduler.observe()
            self.assertEqual(scheduler.state['jobs'][0]['status'],'failed')
            self.assertEqual(scheduler.state['jobs'][0]['retry_count'],0)

    def test_native_worker_survives_scheduler_recreation_and_executes_only_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=job();a['status']='ready';state_for([a],folder)
            scheduler=Scheduler(folder);marker=root/'marker'
            command=[sys.executable,'-c',f'import time;from pathlib import Path;time.sleep(.3);Path({str(marker)!r}).write_text("once")']
            with patch('project_queue.native_command',return_value=(command,{})):
                scheduler.start(scheduler.state['jobs'][0])
            # A new controller observes the same durable attempt; it launches nothing.
            resumed=Scheduler(folder)
            deadline=time.monotonic()+10
            with patch('project_queue.validate_native',return_value={'verified':True}):
                while resumed.state['jobs'][0]['status']!='completed' and time.monotonic()<deadline:
                    resumed.observe();time.sleep(.03)
            self.assertEqual(resumed.state['jobs'][0]['status'],'completed')
            self.assertEqual(marker.read_text(),'once')
            self.assertEqual(len(resumed.state['jobs'][0]['attempts']),1)
            self.assertEqual(resumed.state['jobs'][0]['exit_code'],0)
            for process in scheduler.processes.values():process.wait(timeout=5)

    def test_start_intent_can_discover_an_existing_supervisor(self):
        with tempfile.TemporaryDirectory() as folder:
            a=job();a['status']='ready';state_for([a],folder);scheduler=Scheduler(folder)
            command=[sys.executable,'-c','import time;time.sleep(.4)']
            with patch('project_queue.native_command',return_value=(command,{})):
                scheduler.start(scheduler.state['jobs'][0])
            j=scheduler.state['jobs'][0];j.pop('wrapper_pid');j.pop('wrapper_identity');scheduler.save()
            resumed=Scheduler(folder)
            with patch('project_queue.validate_native',return_value={'verified':True}):
                for _ in range(200):
                    resumed.observe()
                    if resumed.state['jobs'][0]['status']=='completed':break
                    time.sleep(.02)
            self.assertEqual(resumed.state['jobs'][0]['status'],'completed')
            self.assertEqual(len(resumed.state['jobs'][0]['attempts']),1)
            for process in scheduler.processes.values():process.wait(timeout=5)

    def test_orphan_child_is_discovered_without_supervisor_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);release=root/'release';a=job();a['status']='starting'
            command=[sys.executable,'-c',
                     f'from pathlib import Path;import time;p=Path({str(release)!r});\nwhile not p.exists(): time.sleep(.02)']
            child=subprocess.Popen(command)
            try:
                attempt=root/'attempt';attempt.mkdir()
                a.update(attempt_dir=str(attempt),spawn_intent_unix=time.time()-60)
                a['payload']['command']=command
                state_for([a],folder);scheduler=Scheduler(folder)
                scheduler.observe();j=scheduler.state['jobs'][0]
                self.assertEqual(j['status'],'running');self.assertEqual(j['pid'],child.pid)
                release.touch();child.wait(timeout=5)
                with patch('project_queue.validate_native',return_value={'verified':True}):scheduler.observe()
                self.assertEqual(j['status'],'completed');self.assertIsNone(j['exit_code'])
            finally:
                release.touch();child.wait(timeout=5)

    def test_transient_log_read_error_preserves_live_worker_ownership(self):
        with tempfile.TemporaryDirectory() as folder:
            a=job();a.update(status='running',imported=True,pid=123,identity={'test':True},gpu=3)
            state_for([a],folder);scheduler=Scheduler(folder)
            with patch('project_queue.same_process',return_value=True), \
                 patch('project_queue.phase_from_log',side_effect=OSError(errno.ESTALE,'stale file handle')):
                scheduler.observe()
            j=scheduler.state['jobs'][0]
            self.assertEqual(j['status'],'running');self.assertEqual(j['gpu'],3)
            self.assertIn('observation_error',j);self.assertEqual(j['retry_count'],0)

    def test_postspawn_persistence_error_keeps_recoverable_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            a=job();a['status']='ready';state_for([a],folder);scheduler=Scheduler(folder)
            real_save=scheduler.save;calls=[]
            def save():
                calls.append(True)
                if len(calls)>1:raise OSError(errno.EIO,'simulated shared-storage outage')
                real_save()
            command=[sys.executable,'-c','import time;time.sleep(.2)']
            with patch('project_queue.native_command',return_value=(command,{})),patch.object(scheduler,'save',side_effect=save):
                with self.assertRaises(OwnershipPersistenceError):scheduler.start(scheduler.state['jobs'][0])
            resumed=Scheduler(folder)
            self.assertEqual(resumed.state['jobs'][0]['status'],'starting')
            deadline=time.monotonic()+10
            with patch('project_queue.validate_native',return_value={'verified':True}):
                while resumed.state['jobs'][0]['status']!='completed' and time.monotonic()<deadline:
                    resumed.observe();time.sleep(.02)
            self.assertEqual(resumed.state['jobs'][0]['status'],'completed')
            self.assertEqual(len(resumed.state['jobs'][0]['attempts']),1)
            for process in scheduler.processes.values():process.wait(timeout=5)


class WorkerTests(unittest.TestCase):
    def test_durable_exit_code_and_gpu_lease_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);lease=root/'gpu.lock'
            def launch(name, code):
                path=root/name;path.mkdir()
                atomic_json(path/'descriptor.json',{'job_id':name,'cwd':str(root),'lease':str(lease),
                    'command':[sys.executable,'-c',code]})
                return subprocess.Popen([sys.executable,str(ROOT/'scripts/queue_worker.py'),'--descriptor',str(path/'descriptor.json')])
            first=launch('first','import time;time.sleep(.5)')
            for _ in range(100):
                if (root/'first/worker.json').exists():break
                time.sleep(.01)
            second=launch('second','raise SystemExit(0)')
            self.assertNotEqual(second.wait(timeout=10),0)
            self.assertEqual(first.wait(timeout=10),0)
            third=launch('third','raise SystemExit(7)')
            self.assertEqual(third.wait(timeout=10),7)
            self.assertEqual(json.loads((root/'third/result.json').read_text())['returncode'],7)

    def test_native_child_keeps_lease_after_supervisor_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);release=root/'release';lease=root/'gpu.lock'
            command=[sys.executable,'-c',
                     f'from pathlib import Path;import time;p=Path({str(release)!r});\nwhile not p.exists(): time.sleep(.02)']
            atomic_json(root/'descriptor.json',{'job_id':'a','cwd':str(root),'lease':str(lease),'command':command})
            wrapper=subprocess.Popen([sys.executable,str(ROOT/'scripts/queue_worker.py'),'--descriptor',str(root/'descriptor.json')])
            try:
                deadline=time.monotonic()+5
                while not (root/'worker.json').exists() and time.monotonic()<deadline:time.sleep(.01)
                worker=json.loads((root/'worker.json').read_text())
                wrapper.kill();wrapper.wait(timeout=5)
                self.assertIsNotNone(process_identity(worker['child_pid']))
                with self.assertRaises(BlockingIOError):
                    with exclusive_lock(lease):pass
                release.touch()
                deadline=time.monotonic()+5
                while process_identity(worker['child_pid']) and time.monotonic()<deadline:time.sleep(.01)
                with exclusive_lock(lease):pass
            finally:
                release.touch()
                if wrapper.poll() is None:wrapper.kill()
                wrapper.wait(timeout=5)


class MemoryProbeTests(unittest.TestCase):
    def test_large_cases_cover_prompt_audio_and_vision_extremes(self):
        from prepare_memory_cases import choose_cases
        rows=[{'id':str(i),'tokens':i,'audio_frames':100-i,'vision_elements':1000 if i==5 else i} for i in range(20)]
        chosen=choose_cases(rows)
        self.assertEqual(len(chosen),8);self.assertEqual(len(set(chosen)),8)
        self.assertIn('19',chosen);self.assertIn('0',chosen);self.assertIn('5',chosen)

    def test_memory_probe_rejects_nontraining_or_duplicated_ids(self):
        from train import memory_probe_records
        records=[{'id':str(i)} for i in range(10)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cases.json'
            valid={'split':'train','test_set_used':False,'ids':[str(i) for i in range(8)]}
            atomic_json(path,valid);self.assertEqual(len(memory_probe_records(records,path)),8)
            for change in [{'split':'test'},{'test_set_used':True},{'ids':['0']*8},{'ids':[str(i+5) for i in range(8)]}]:
                atomic_json(path,valid|change)
                with self.assertRaises(ValueError):memory_probe_records(records,path)

class ScheduleEstimateTests(unittest.TestCase):
    def test_parallel_branches_and_per_job_dependencies(self):
        from smarthome_efficiency.queueing import estimate_schedule
        a,b,c=job('a'),job('b'),job('c',['a'])
        for j in (a,b,c):j.update(resource='gpu',status='ready')
        result=estimate_schedule([a,b,c],{'a':10.,'b':20.,'c':5.},gpu_count=2)
        self.assertEqual(result['seconds'],20.)
        self.assertFalse(result['unresolved'])

    def test_current_reservations_and_failed_branches_remain_visible(self):
        from smarthome_efficiency.queueing import estimate_schedule
        a,b,c=job('a'),job('b'),job('c',['b'])
        a.update(resource='gpu',status='running',gpu=1)
        b.update(resource='gpu',status='failed');c['status']='blocked'
        result=estimate_schedule([a,b,c],{'a':10.,'b':20.,'c':5.},gpu_count=2)
        self.assertEqual(result['seconds'],10.)
        self.assertEqual(result['failed_or_blocked'],['b','c'])

class SubmissionTests(unittest.TestCase):
    def test_additional_jobs_are_atomic_and_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            state_for([job('a')],folder);scheduler=Scheduler(folder)
            b=entry('b','prepare',{'task':'mer6','output':'/future'},deps=['a'],resource='cpu')
            inbox=Path(folder)/'inbox';inbox.mkdir()
            atomic_json(inbox/'one.json',{'schema_version':1,'jobs':[b]})
            scheduler.ingest()
            self.assertIn('b',scheduler.by_id);self.assertTrue((inbox/'one.accepted').exists())
            atomic_json(inbox/'replay.json',{'schema_version':1,'jobs':[b]})
            scheduler.ingest()
            self.assertTrue((inbox/'replay.accepted').exists())
            changed=copy.deepcopy(b);changed['payload']['output']='/different'
            atomic_json(inbox/'duplicate.json',{'schema_version':1,'jobs':[changed]})
            scheduler.ingest()
            self.assertEqual(len(scheduler.state['jobs']),2)
            self.assertTrue((inbox/'duplicate.rejected').exists())

    def test_a5_is_not_implicitly_enabled(self):
        with tempfile.TemporaryDirectory() as folder:
            state_for([job('a')],folder);scheduler=Scheduler(folder)
            b=entry('b','recovery_train',{'arm':'A5','task':'mer6'})
            inbox=Path(folder)/'inbox';inbox.mkdir()
            atomic_json(inbox/'a5.json',{'schema_version':1,'jobs':[b]})
            scheduler.ingest()
            self.assertNotIn('b',scheduler.by_id)
            self.assertTrue((inbox/'a5.rejected').exists())

    def test_malformed_submission_is_rejected_without_stopping_scheduler(self):
        with tempfile.TemporaryDirectory() as folder:
            state_for([job('a')],folder);scheduler=Scheduler(folder)
            inbox=Path(folder)/'inbox';inbox.mkdir()
            b=entry('b','prepare',{'task':'mer6'},resource='cpu');b['resource']='unknown'
            for index, spec in enumerate([[],{'schema_version':1,'jobs':None},
                                          {'schema_version':1,'jobs':[b]}]):
                path=inbox/f'bad{index}.json';atomic_json(path,spec)
                scheduler.ingest()
                self.assertTrue(path.with_suffix('.rejected').exists())
                self.assertEqual(len(scheduler.state['jobs']),1)


class SourcePlanningTests(unittest.TestCase):
    def test_source_selection_is_task_local(self):
        from test_recovery_experiments import RecoveryExperimentTests
        from recovery_experiments import select_sources
        with tempfile.TemporaryDirectory() as folder:
            sweep,caches=RecoveryExperimentTests().fixture(Path(folder))
            task_jobs=[j for j in sweep['jobs'] if j['task']=='mintrec2']
            selected=select_sources([{'status':'completed','jobs':task_jobs}],tasks=('mintrec2',))
            self.assertEqual(set(selected),{'mintrec2'})
            self.assertEqual(selected['mintrec2']['winner']['method'],'ffn_magnitude')

    def test_a4_can_materialize_without_compact_source_selection(self):
        from test_recovery_experiments import RecoveryExperimentTests
        from project_queue_jobs import configure_recovery
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=RecoveryExperimentTests().fixture(root)
            teacher=root/'teacher/mer6'
            cfg=json.loads((teacher/'metadata.json').read_text())['config']
            j=entry('recovery_ablation_20261019/smoke_train:A4_mer6','smoke_train',
                    {'name':'A4_mer6','task':'mer6','arm':'A4','kind':'train'})
            with patch('project_queue_jobs.original_teacher',return_value=(str(teacher),cfg)):
                configure_recovery({'queue_root':str(root),'selected_sources':{}},j)
            self.assertEqual(j['payload']['config']['lora_rank'],64)
            self.assertEqual(j['payload']['config']['model'],str(root/'pretrained'))
            self.assertIsNone(j['payload']['init_adapter'])
