"""Priority, isolated changes and pinned-cache reuse for all 24 A feature controls."""
import copy,sys,tempfile,unittest
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import test_full_model_followups as fixtures
from queue_full_model_a_feature import build_submission,FAMILY,TASKS,VARIANTS
from project_queue_jobs import entry,native_command
from smarthome_efficiency.queueing import atomic_json,validate_graph
from smarthome_efficiency.training import load_config


class AFeatureQueueTests(unittest.TestCase):
    def fixture(self,root):
        state=fixtures.FollowupQueueTests().fixture(root)
        by={j['id']:j for j in state['jobs']}
        for task in TASKS:
            base=by['full_model_2b_20261020/export:'+task+'_a_lm1536_a1024_v1024']['payload']['output']
            cfg=copy.deepcopy(by['full_model_2b_20261020/train:'+task+'_a_lm1536_a1024_v1024']['payload']['config'])
            path=root/task/'features';cfg.update(model=base,feature_teacher_cache=str(path),feature_audio_weight=.1,feature_vision_weight=.1)
            atomic_json(path/'cache.json',{'status':'completed','spec':{'teacher':{'split':'train','test_set_used':False,'teacher_checkpoint':cfg['teacher_checkpoint']}}})
            state['jobs'].append(entry('full_model_followups_20261020/train:'+task+'_a_feature_r16','recovery_train',{'task':task,'status':'completed','config':cfg}))
            state['jobs'].append(entry('full_model_followups_20261020/encoder_cache:'+task,'encoder_cache',{'task':task,'status':'completed'}))
        return state

    def test_24_runs_are_single_variable_controls_with_independent_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);original=copy.deepcopy(state);spec=build_submission(state,root)
            self.assertEqual(state,original);validate_graph(state['jobs']+spec['jobs']);self.assertEqual(len(spec['jobs']),96)
            self.assertFalse(any(j['kind'] in ('prune_export','encoder_cache') for j in spec['jobs']))
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train'];self.assertEqual(len(trains),24)
            old={j['id']:j for j in state['jobs']};by={j['id']:j for j in spec['jobs']}
            for task in TASKS:
                ref=old['full_model_followups_20261020/train:'+task+'_a_feature_r16']['payload']['config']
                selected=[j for j in trains if j['task']==task]
                self.assertEqual({j['payload']['arm'] for j in selected},{a for a,_,_ in VARIANTS})
                for j in selected:
                    p=j['payload'];changes=next(c for a,_,c in VARIANTS if a==p['arm'])
                    self.assertEqual(p['config'],ref|changes);self.assertIsNone(p['init_adapter'])
                    self.assertEqual(p['expected_parameters']['total'],1978160384)
                    reload=by[j['deps'][0]];probe=by[reload['deps'][0]]
                    self.assertEqual(probe['kind'],'smoke_train');self.assertEqual(reload['kind'],'smoke_test')
                    self.assertEqual(by[FAMILY+'/test:'+p['name']]['deps'],[j['id']])
                    self.assertIn('full_model_followups_20261020/encoder_cache:'+task,probe['deps'])
                    path=root/'cfg.yaml';path.write_text(yaml.safe_dump(p['config']));self.assertEqual(load_config(path),p['config'])
            for task in TASKS:
                group=[j for j in trains if j['task']==task]
                top=[j['priority'] for j in group if j['payload']['arm'] in ['FEATURE_WEAK','FEATURE_STRONG','FEATURE_LOWLR','FEATURE_SPLITLR']]
                other=[j['priority'] for j in group if j['payload']['arm'] not in ['FEATURE_WEAK','FEATURE_STRONG','FEATURE_LOWLR','FEATURE_SPLITLR']]
                self.assertGreater(min(top),max(other))
            self.assertGreater(min(j['priority'] for j in trains if j['task']=='mintrec2'),max(j['priority'] for j in trains if j['task']!='mintrec2'))
            self.assertGreater(min(j['priority'] for j in trains),2)
            j=trains[0];Path(j['payload']['config']['feature_teacher_cache'],'cache.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Pinned feature cache'):native_command(state,j,root)


if __name__=='__main__':unittest.main()
