"""Joy resource isolation and validation-only official-test selection contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from project_queue import execution_host
from project_queue_jobs import entry
from smarthome_efficiency.queueing import refresh_readiness
from validation_shortlist import select_validation_shortlist
from queue_mer_focus import definitions


class MerFocusContracts(unittest.TestCase):
    def test_joy_allows_zero_but_suzy_exclusion_is_preserved(self):
        with patch('project_queue.socket.gethostname',return_value='joy'):
            self.assertEqual(execution_host({'host':'joy','allowed_gpus':list(range(8))}),'joy')
            with self.assertRaises(RuntimeError): execution_host({'host':'suzy','allowed_gpus':[1]})
        with patch('project_queue.socket.gethostname',return_value='suzy'):
            with self.assertRaises(ValueError): execution_host({'host':'suzy','allowed_gpus':[0,1]})

    def test_sixteen_runs_keep_only_training_hyperparameters_variable(self):
        arms=definitions()
        self.assertEqual(len(arms),16)
        self.assertEqual(sum(item[2] for item in arms),4)
        self.assertEqual(sum(item[0].startswith('FRESH') for item in arms),8)
        self.assertEqual(sum(item[0].startswith('FEATURE') for item in arms),4)
        allowed={'epochs','learning_rate','encoder_learning_rate','distillation_alpha',
                 'feature_audio_weight','feature_vision_weight'}
        self.assertTrue(all(set(item[3])<=allowed for item in arms))
        self.assertEqual(len({item[0] for item in arms}),16)

    def test_failed_candidate_does_not_block_shortlist_or_independent_work(self):
        failed=entry('failed','recovery_train',{'task':'mer6'});failed['status']='failed'
        complete=entry('good','recovery_train',{'task':'mer6'});complete['status']='completed'
        select=entry('select','shortlist',{'task':'mer6'},deps=['failed','good'],resource='inline')
        ordinary=entry('dependent','recovery_test',{'task':'mer6'},deps=['failed'])
        refresh_readiness([failed,complete,select,ordinary])
        self.assertEqual(select['status'],'ready')
        self.assertEqual(ordinary['status'],'blocked')

    def make_state(self,folder,scores):
        jobs=[]
        for i,score in enumerate(scores):
            path=Path(folder)/f'train{i}';path.mkdir()
            cfg={'task':'mer6','train_limit':None,'dev_limit':None,'selection_mode':'logit'}
            (path/'report.json').write_text(json.dumps({'status':'completed','test_set_used':False,
                'dev_count':675,'config':cfg,'best_score':score}))
            train=entry(f'train{i}','recovery_train',{'task':'mer6','run_dir':str(path),'name':f'run{i}'})
            train['status']='completed'
            test=entry(f'test{i}','recovery_test',{'task':'mer6','shortlist_job':'select'},deps=[train['id'],'select'])
            jobs.extend([train,test])
        shortlist=entry('select','shortlist',{'task':'mer6','top_k':2,'dev_count':675,
            'output':str(Path(folder)/'decision.json')},
            deps=[j['id'] for j in jobs if j['kind']=='recovery_train'],resource='inline')
        jobs.append(shortlist)
        return {'jobs':jobs},shortlist

    def test_selects_validation_winners_cancels_only_nonselected_tests_and_pins_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            state,job=self.make_state(folder,[.70,.74,.72])
            with patch('launch_pruning_followup.pin_best',side_effect=lambda x:str(Path(x)/'checkpoints/best')), \
                 patch('launch_pruning_followup.checkpoint_identity',return_value={'metadata.json':'pinned'}):
                decision=select_validation_shortlist(state,job)
            self.assertEqual(decision['selected'],['train1','train2'])
            self.assertFalse(decision['test_results_used'])
            self.assertEqual(next(j for j in state['jobs'] if j['id']=='test0')['status'],'cancelled')
            self.assertTrue(all(j['status']=='completed' for j in state['jobs'] if j['kind']=='recovery_train'))
            self.assertEqual(json.loads(Path(job['payload']['output']).read_text())['selected'],['train1','train2'])

    def test_partial_dev_or_test_using_training_report_cannot_select(self):
        with tempfile.TemporaryDirectory() as folder:
            state,job=self.make_state(folder,[.70])
            report=Path(folder)/'train0/report.json';d=json.loads(report.read_text())
            d['test_set_used']=True;report.write_text(json.dumps(d))
            with self.assertRaises(ValueError): select_validation_shortlist(state,job)
            self.assertFalse(Path(job['payload']['output']).exists())
            self.assertEqual(state['jobs'][1]['status'],'waiting')


if __name__=='__main__': unittest.main()
