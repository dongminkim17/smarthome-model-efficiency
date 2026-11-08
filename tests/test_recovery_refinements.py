"""Approved refinement matrix, fresh initialization and provenance safeguards."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import test_joint_pruning as fixtures
import queue_recovery_refinements as subject
from smarthome_efficiency.queueing import atomic_json,read_json,validate_graph
from smarthome_efficiency.training import load_config,config_fingerprint
from project_queue_jobs import entry

class RecoveryRefinementTests(unittest.TestCase):
 def fixture(self,root):
  state=fixtures.JointPruningTests().fixture(root)
  by={j['id']:j for j in state['jobs']}
  for task in subject.TASKS:
   cfg=by['encoder_budget_20261021/train:'+task+'_balanced_feature_splitlr']['payload']['config']
   for family,model in [('encoder_budget_20261021',cfg['model']),('joint_pruning_20261022',str(root/task/'joint'))]:
    suffix=task+'_balanced' if family.startswith('encoder') else task
    state['jobs'].append(entry(family+'/export:'+suffix,'prune_export',{'task':task,'status':'completed','output':model}))
    if family.startswith('joint'):
     atomic_json(Path(model)/'compression.json',read_json(Path(cfg['model'])/'compression.json'))
  return state

 def test_exact_matrix_fresh_sources_dependencies_and_intended_changes_only(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);state=self.fixture(root);before=copy.deepcopy(state)
   with patch.object(subject,'validate_export') as validator:spec=subject.build_submission(state,root)
   self.assertEqual(validator.call_count,6);self.assertEqual(state,before)
   self.assertEqual(len(spec['jobs']),72);validate_graph(state['jobs']+spec['jobs'])
   by={j['id']:j for j in spec['jobs']};old={j['id']:j for j in state['jobs']}
   trains=[j for j in spec['jobs'] if j['kind']=='recovery_train'];self.assertEqual(len(trains),18)
   self.assertFalse(any(j['kind'] in ('prune_export','teacher_cache','encoder_cache') for j in spec['jobs']))
   for task in subject.TASKS:
    ref=old['encoder_budget_20261021/train:'+task+'_balanced_feature_splitlr']['payload']['config']
    for arm,joint,changes,priority in subject.VARIANTS:
     job=by[subject.FAMILY+'/train:'+task+'_'+arm.lower()];p=job['payload'];cfg=p['config']
     expected=copy.deepcopy(ref);expected.update(changes)
     expected['model']=str(root/task/'joint') if joint else ref['model']
     self.assertEqual(cfg,expected);self.assertEqual(job['priority'],priority)
     self.assertIsNone(p['init_adapter']);self.assertIsNone(p['parent_checkpoint'])
     self.assertEqual(p['training_epochs'],10 if joint else 5)
     self.assertEqual(cfg['split_seed'],42);self.assertEqual(cfg['teacher_checkpoint'],ref['teacher_checkpoint'])
     self.assertEqual(cfg['teacher_cache'],ref['teacher_cache']);self.assertEqual(cfg['feature_teacher_cache'],ref['feature_teacher_cache'])
     self.assertEqual(p['expected_parameters']['language'],1924902912)
     self.assertEqual(p['config_sha256'],config_fingerprint(cfg))
     path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(path),cfg)
     reload=by[job['deps'][0]];probe=by[reload['deps'][0]]
     self.assertEqual(probe['payload']['config']['epochs'],2);self.assertEqual(probe['payload']['training_epochs'],2)
     self.assertGreater(probe['priority'],job['priority'])
     self.assertIn(p['base_export'],probe['deps'])
     self.assertFalse(any('train:' in dep for dep in probe['deps']))
     self.assertIn('expected_feature_cache_sha256',probe['payload'])
     test=by[subject.FAMILY+'/test:'+p['name']]
     self.assertEqual(test['deps'],[job['id']]);self.assertEqual(test['payload']['split'],'test')
     self.assertEqual(test['payload']['modes'],['logit','greedy']);self.assertIsNone(test['payload']['limit'])

 def test_rejects_changed_cache_recipe_architecture_or_unfinished_export(self):
  for change in ('cache','recipe','architecture','export'):
   with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
    root=Path(folder);state=self.fixture(root)
    ref=state['jobs'][0];cfg=ref['payload']['config']
    if change=='cache':
     p=Path(cfg['teacher_cache']);v=read_json(p);v['spec']['split']='test';atomic_json(p,v)
    elif change=='recipe':cfg['split_seed']=99
    elif change=='architecture':
     p=Path(cfg['model'])/'compression.json';v=read_json(p);v['parameters_after']['language']+=1;atomic_json(p,v)
    else:
     next(j for j in state['jobs'] if j['id']=='joint_pruning_20261022/export:mintrec2')['status']='running'
    with patch.object(subject,'validate_export'),self.assertRaises(ValueError):subject.build_submission(state,root)

if __name__=='__main__':unittest.main()
