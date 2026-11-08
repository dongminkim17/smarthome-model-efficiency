import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import mer_avt_pipeline as subject
from smarthome_efficiency.queueing import atomic_json,validate_graph
from smarthome_efficiency.training import DEFAULTS,config_fingerprint
from project_queue_jobs import entry,native_command

class MerAvtPipelineTests(unittest.TestCase):
 def fixture(self,root):
  cfg=dict(DEFAULTS,task='mer6',model=str(root/'original'),data_root=str(root/'avt'),use_transcript=True,micro_batch_size=2,gradient_accumulation_steps=2)
  return {'jobs':[],'documents':{},'queue_root':str(root/'queue')},cfg
 def source(self,root,cfg,name):
  p=root/name;(p/'adapter').mkdir(parents=True);(p/'adapter/adapter_model.safetensors').write_bytes(b'adapter')
  atomic_json(p/'metadata.json',{'config':cfg});return p
 def test_teacher_is_fresh_avt_with_private_preparation_and_updated_adapter_reload(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);state,cfg=self.fixture(root)
   with patch.object(subject,'source_config',return_value=cfg):spec=subject.build_teacher(state,root,root/'avt')
   self.assertEqual(len(spec['jobs']),5);validate_graph(spec['jobs'])
   prep=next(j for j in spec['jobs'] if j['kind']=='prepare');self.assertEqual(prep['payload']['config'],cfg)
   (root/'attempt').mkdir();command,actual=native_command(state,prep,root/'attempt');self.assertEqual(actual,cfg)
   train=next(j for j in spec['jobs'] if j['kind']=='recovery_train');self.assertIsNone(train['payload']['init_adapter'])
   self.assertEqual(train['payload']['config']['objective'],'sft')
   self.assertTrue(next(j for j in spec['jobs'] if j['kind']=='smoke_test')['payload']['reload_latest'])
 def test_lm_and_encoder_phases_use_new_teacher_caches_and_exact_three_recipes(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);state,cfg=self.fixture(root);teacher=self.source(root,cfg,'teacher')
   teacher_job=entry(subject.train_id(subject.TEACHER,'SFT_QV16'),'recovery_train',{'task':'mer6','status':'completed'})
   state['jobs']=[teacher_job,entry(subject.prepare_id(),'prepare',{'task':'mer6','status':'completed'},resource='cpu')]
   with patch.object(subject,'completed_source',return_value=(teacher_job,teacher)):spec=subject.build_lm(state,root)
   self.assertEqual(next(j for j in spec['jobs'] if j['kind']=='encoder_cache')['payload']['expected_examples'],2698)
   self.assertEqual(len(spec['jobs']),7);validate_graph(state['jobs']+spec['jobs'])
   lm_train=next(j for j in spec['jobs'] if j['kind']=='recovery_train');c=lm_train['payload']['config']
   self.assertEqual(c['teacher_checkpoint'],str(teacher));self.assertIn(subject.FAMILY,c['teacher_cache'])
   self.assertTrue(c['use_transcript']);self.assertEqual(c['distillation_alpha'],.75)
   self.assertNotIn('feature_teacher_cache',c)
   export=next(j for j in spec['jobs'] if j['kind']=='prune_export');self.assertTrue(export['payload']['generate_calibration'])
   state['jobs']+=spec['jobs']
   for j in state['jobs']:j['status']='completed'
   compact=root/'compact';atomic_json(compact/'compression.json',{'fixture':True})
   recovered=self.source(root,dict(c,model=str(compact)),'recovered')
   feature=root/'artifacts/cache'/subject.FAMILY/'encoder_features/cache.json';atomic_json(feature,{'fixture':True})
   with patch.object(subject,'completed_source',side_effect=[(teacher_job,teacher),(lm_train,recovered)]):enc=subject.build_encoder(state,root)
   self.assertEqual(len(enc['jobs']),14);validate_graph(state['jobs']+enc['jobs'])
   trains=[j for j in enc['jobs'] if j['kind']=='recovery_train'];self.assertEqual(len(trains),3)
   self.assertEqual({j['payload']['arm'] for j in trains},{x[0] for x in subject.ARMS})
   for j in trains:
    p=j['payload'];c=p['config'];self.assertIsNone(p['init_adapter']);self.assertEqual(c['teacher_checkpoint'],str(teacher))
    self.assertEqual(c['data_root'],cfg['data_root']);self.assertEqual(c['epochs'],5)
    self.assertLess(p['expected_parameters']['audio_encoder']+p['expected_parameters']['vision_encoder'],1e9)
    self.assertLess(p['expected_parameters']['language'],2e9);self.assertIn('expected_feature_cache_sha256',p)
    self.assertEqual(p['config_sha256'],config_fingerprint(c))
 def test_failed_prerequisite_is_not_silently_waited_on(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);state,cfg=self.fixture(root)
   j=entry(subject.train_id(subject.TEACHER,'SFT_QV16'),'recovery_train',{'task':'mer6'});j['status']='failed';state['jobs']=[j]
   self.assertEqual(subject.tick(state,root)[0],'failed')

if __name__=='__main__':unittest.main()
