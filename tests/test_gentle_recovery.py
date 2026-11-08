"""Nine-run selection and dense-validation configuration/ETA checks."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import test_recovery_refinements as fixtures
import queue_gentle_recovery as subject
from smarthome_efficiency.queueing import validate_graph
from smarthome_efficiency.training import load_config,config_fingerprint

class GentleRecoveryTests(unittest.TestCase):
 def fixture(self,root):
  state=fixtures.RecoveryRefinementTests().fixture(root)
  state['jobs']=[j for j in state['jobs'] if not j['id'].startswith('joint_pruning_20261022/')]
  for j in state['jobs']:
   if j['kind']=='recovery_train':
    (Path(j['payload']['run_dir'])/'metrics.jsonl').write_text('{"event":"validation","seconds":400}\n')
  return state

 def test_exact_approved_matrix_fixed_sources_dense_binary_validation_and_eta(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);state=self.fixture(root);before=copy.deepcopy(state)
   with patch('queue_recovery_refinements.validate_export') as validate:spec=subject.build_submission(state,root)
   self.assertEqual(state,before);self.assertEqual(validate.call_count,3)
   validate_graph(state['jobs']+spec['jobs']);self.assertEqual(len(spec['jobs']),36)
   self.assertTrue(all(j['id'].startswith(subject.FAMILY+'/') for j in spec['jobs']))
   trains=[j for j in spec['jobs'] if j['kind']=='recovery_train'];self.assertEqual(len(trains),9)
   old={j['id']:j for j in state['jobs']};by={j['id']:j for j in spec['jobs']}
   for j in trains:
    p=j['payload'];c=p['config'];ref=old['encoder_budget_20261021/train:'+j['task']+'_balanced_feature_splitlr']['payload']['config']
    changes=next(v[2] for v in subject.VARIANTS if v[0]==p['arm'])
    expected=copy.deepcopy(ref);expected.update(changes)
    if j['task']=='mintrec2':expected['eval_every_fraction']=.1
    self.assertEqual(c,expected);self.assertEqual(c['epochs'],5);self.assertEqual(c['lora_rank'],16)
    self.assertIsNone(p['init_adapter']);self.assertIsNone(p['parent_checkpoint'])
    self.assertEqual(p['config_sha256'],config_fingerprint(c))
    self.assertEqual(p['estimated_seconds'],17000 if j['task']=='mintrec2' else 1000)
    path=root/'cfg.yaml';path.write_text(yaml.safe_dump(c));self.assertEqual(load_config(path),c)
    reload=by[j['deps'][0]];probe=by[reload['deps'][0]]
    self.assertFalse(any('joint_pruning' in dep for dep in probe['deps']))
    self.assertEqual(probe['payload']['config']['eval_every_fraction'],.1 if j['task']=='mintrec2' else .5)
    test=by[subject.FAMILY+'/test:'+p['name']]
    self.assertEqual(test['deps'],[j['id']]);self.assertEqual(test['payload']['split'],'test')
    self.assertEqual(test['payload']['modes'],['logit','greedy'])
   doc=next(iter(spec['documents'].values()));self.assertIn('Nine',doc['description'])

 def test_dense_validation_requires_measured_timing(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);state=self.fixture(root)
   j=next(j for j in state['jobs'] if j['kind']=='recovery_train' and j['task']=='mintrec2')
   (Path(j['payload']['run_dir'])/'metrics.jsonl').write_text('')
   with patch('queue_recovery_refinements.validate_export'),self.assertRaises(ValueError):subject.build_submission(state,root)

if __name__=='__main__':unittest.main()
