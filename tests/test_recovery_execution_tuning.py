import copy,sys,unittest
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
from apply_recovery_batch_tuning import choose_execution

class RecoveryExecutionTuningTests(unittest.TestCase):
 def fixture(self):
  def case(micro,seconds=2.,peak=15.,status='passed'):
   return {'micro_batch_size':micro,'gradient_checkpointing':True,'status':status,'groups':{k:{'mean_update_seconds':seconds,'peak_allocated_gib':peak} for k in ('typical_a','typical_b','largest_a','largest_b')}}
  a={'status':'completed','task':'mer6','test_set_used':False,'cases':[case(2,3),case(4)]}
  b=copy.deepcopy(a)
  for c in b['cases']:c['groups']={k:v for k,v in c['groups'].items() if k.startswith('largest')}
  return a,b
 def test_speedup_with_verified_headroom_preserves_batch(self):
  a,b=self.fixture();r=choose_execution(a,b);self.assertEqual(r['execution'],{'micro_batch_size':4,'gradient_accumulation_steps':1,'gradient_checkpointing':True});self.assertEqual(r['typical_update_speedup'],1.5)
 def test_rejects_oom_numerical_failure_missing_stress_memory_and_slower_execution(self):
  for change in ('oom','numerical','missing','memory','slow'):
   with self.subTest(change=change):
    a,b=self.fixture()
    if change=='oom':b['cases'][1]['status']='oom'
    if change=='numerical':b['cases'][1]['status']='numerical_mismatch'
    if change=='missing':b['cases'][1]['groups'].pop('largest_b')
    if change=='memory':b['cases'][1]['groups']['largest_a']['peak_allocated_gib']=22.
    if change=='slow':a['cases'][1]['groups']['typical_a']['mean_update_seconds']=5.
    self.assertEqual(choose_execution(a,b)['execution']['micro_batch_size'],2)
 def test_rejects_unfinished_or_test_set_or_different_task(self):
  for change in ('status','test_set_used','task'):
   a,b=self.fixture();b[change]={'status':'running','test_set_used':True,'task':'mintrec2'}[change]
   with self.assertRaises(ValueError):choose_execution(a,b)
