"""Cancellation cannot signal active jobs or remove results/shared preparation."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from project_queue import Scheduler
from project_queue_jobs import entry
from test_project_queue import state_for
from smarthome_efficiency.queueing import cancel_unstarted_runs,refresh_readiness,estimate_schedule,atomic_json,reprioritize_unstarted_jobs


def run_jobs(arm='B_SFT'):
    p={'task':'mintrec2','arm':arm,'name':'mintrec2_'+arm.lower()};name='family/'+arm
    probe=entry(name+'/probe','smoke_train',p,deps=['shared'])
    reload=entry(name+'/reload','smoke_test',p,deps=[probe['id']])
    train=entry(name+'/train','recovery_train',p,deps=[reload['id']])
    test=entry(name+'/test','recovery_test',p,deps=[train['id']])
    return [probe,reload,train,test]


def fixture():
    shared=entry('shared','prune_export',{'task':'mintrec2','status':'completed'})
    jobs=[shared,*run_jobs(),*run_jobs('B_LOWLR')];refresh_readiness(jobs);return jobs


class CancellationTests(unittest.TestCase):
    def test_only_pending_private_chain_is_cancelled(self):
        jobs=fixture();before=copy.deepcopy(jobs);result=cancel_unstarted_runs(jobs,['family/B_SFT/train'],'Approved by user')
        self.assertEqual(len(result['cancelled']),4);self.assertEqual(jobs[0],before[0])
        self.assertEqual([j['status'] for j in jobs[1:5]],['cancelled']*4)
        self.assertEqual(jobs[5:],before[5:])
        estimate=estimate_schedule(jobs,{j['id']:60 for j in jobs})
        self.assertFalse(estimate['failed_or_blocked']);self.assertFalse(estimate['unresolved']);self.assertEqual(len(estimate['cancelled']),4)

    def test_active_or_previously_started_training_keeps_its_test(self):
        for status,attempts in [('running',[{'path':'running'}]),('starting',[{'path':'starting'}]),('waiting',[{'path':'retry'}]),('completed',[{'path':'done'}])]:
            jobs=fixture();train=jobs[3];train.update(status=status,attempts=attempts,pid=123,identity={'fixture':True})
            original=copy.deepcopy(train);result=cancel_unstarted_runs(jobs,[train['id']],'Approved')
            self.assertFalse(result['cancelled']);self.assertEqual(train,original)
            self.assertNotEqual(jobs[4]['status'],'cancelled')

    def test_active_smoke_is_preserved_when_future_full_training_is_cancelled(self):
        jobs=fixture();jobs[1].update(status='running',pid=123,gpu=7,attempts=[{'path':'smoke'}]);active=copy.deepcopy(jobs[1])
        result=cancel_unstarted_runs(jobs,[jobs[3]['id']],'Approved')
        self.assertEqual(jobs[1],active);self.assertEqual(len(result['cancelled']),3)

    def test_request_is_validated_before_mutation_and_cancelled_dependency_propagates(self):
        jobs=fixture();before=copy.deepcopy(jobs)
        with self.assertRaises(ValueError):cancel_unstarted_runs(jobs,[jobs[3]['id'],'missing'],'Approved')
        self.assertEqual(jobs,before)
        jobs.append(entry('future','recovery_test',{'task':'mintrec2'},deps=[jobs[3]['id']]))
        cancel_unstarted_runs(jobs,[jobs[3]['id']],'Approved')
        self.assertEqual(jobs[-1]['status'],'cancelled')

    def test_control_inbox_is_durable_idempotent_and_isolates_bad_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state_for(fixture(),root);scheduler=Scheduler(root)
            request={'schema_version':1,'action':'cancel_unstarted_runs','training_jobs':['family/B_SFT/train'],'reason':'Explicit user choice'}
            path=root/'control_requests/valid.json';atomic_json(path,request)
            scheduler.ingest_controls();self.assertTrue(path.with_suffix('.accepted').exists())
            self.assertEqual(len(scheduler.state['control_requests']),1)
            atomic_json(path,request);scheduler.ingest_controls();self.assertEqual(len(scheduler.state['control_requests']),1)
            bad=root/'control_requests/bad.json';atomic_json(bad,request|{'training_jobs':['missing']})
            scheduler.ingest_controls();self.assertTrue(bad.with_suffix('.rejected').exists())
            self.assertTrue(all(j['status']!='cancelled' for j in scheduler.state['jobs'] if j['payload'].get('arm')=='B_LOWLR'))



class PriorityControlTests(unittest.TestCase):
    def test_atomic_validation_and_active_attempt_preservation(self):
        jobs=fixture();jobs[1].update(status='running',pid=123,attempts=[{'path':'active'}]);before=copy.deepcopy(jobs)
        with self.assertRaises(ValueError):reprioritize_unstarted_jobs(jobs,[{'id':jobs[3]['id'],'priority':30},{'id':'missing','priority':1}],'Test')
        self.assertEqual(jobs,before)
        result=reprioritize_unstarted_jobs(jobs,[{'id':jobs[1]['id'],'priority':40},{'id':jobs[3]['id'],'priority':30}],'Approved scheduling')
        self.assertEqual(jobs[1],before[1]);self.assertEqual(jobs[3]['priority'],30)
        self.assertEqual(len(result['changed']),1);self.assertEqual(len(result['skipped']),1)
        for bad in [[{'id':jobs[3]['id'],'priority':True}],[{'id':jobs[3]['id'],'priority':1}]*2]:
            with self.assertRaises(ValueError):reprioritize_unstarted_jobs(jobs,bad,'Test')

    def test_priority_inbox_is_audited_and_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state_for(fixture(),root);scheduler=Scheduler(root)
            request={'schema_version':1,'action':'reprioritize_unstarted_jobs','updates':[{'id':'family/B_SFT/train','priority':30}],'reason':'Faster first GPU preflight'}
            path=root/'control_requests/priority.json';atomic_json(path,request);scheduler.ingest_controls()
            self.assertTrue(path.with_suffix('.accepted').exists());self.assertEqual(scheduler.by_id['family/B_SFT/train']['priority'],30)
            atomic_json(path,request);scheduler.ingest_controls();self.assertEqual(len(scheduler.state['control_requests']),1)

if __name__=='__main__':unittest.main()

