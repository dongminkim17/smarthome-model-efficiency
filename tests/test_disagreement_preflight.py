import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from prepare_memory_cases import choose_cases,ensure_teacher_cases
from smarthome_efficiency.disagreement_preflight import validate_disagreement_smoke

class DisagreementPreflightTests(unittest.TestCase):
    def test_preserves_memory_selection_when_coverage_exists_and_retains_largest_when_missing(self):
        rows=[{'id':str(i),'tokens':i,'audio_frames':i,'vision_elements':i} for i in range(20)]
        ids=choose_cases(rows)
        self.assertEqual(ensure_teacher_cases(ids,rows,{ids[0]}),ids)
        added=ensure_teacher_cases(ids,rows,{'0','1'})
        self.assertEqual(added[:4],ids[:4]);self.assertIn('1',added);self.assertEqual(len(set(added)),8)
        for wrong in [set(),{r['id'] for r in rows}]:
            with self.assertRaises(ValueError):ensure_teacher_cases(ids,rows,wrong)

    def fixture(self,gated):
        ids=[str(i) for i in range(8)]
        probe={'split':'train','test_set_used':False,'ids':ids,'teacher_coverage':{
            'teacher_cache_sha256':'pinned','disagreement_ids':['0'],'agreement_ids':ids[1:]}}
        events=[{'event':'train','global_step':i+1,'examples':4,'teacher_disagreement_count':int(i==0),
            'teacher_disagreement_rate':.25 if i==0 else 0.,
            'distillation_alpha_effective_mean':(.75*3+gated)/4 if i==0 else .75} for i in range(2)]
        return {'distillation_alpha':.75,'distillation_disagreement_alpha':gated},events,probe,[{'id':i} for i in ids]

    def test_gpu_gate_verifies_counts_weights_and_train_provenance_for_both_gates(self):
        for gated in [0.,.25]:
            cfg,events,probe,train=self.fixture(gated)
            r=validate_disagreement_smoke(cfg,events,probe,train,'pinned')
            self.assertEqual(r['teacher_disagreements'],1);self.assertTrue(r['both_branches_exercised'])
            for change in ['no_gate','wrong_count','wrong_cache','test_split','duplicate_train']:
                ev=copy.deepcopy(events);pr=copy.deepcopy(probe);tr=copy.deepcopy(train);sha='pinned'
                if change=='no_gate':ev[0]['distillation_alpha_effective_mean']=.75
                if change=='wrong_count':ev[0]['teacher_disagreement_count']=0
                if change=='wrong_cache':sha='changed'
                if change=='test_split':pr['split']='test'
                if change=='duplicate_train':tr[0]=tr[1]
                with self.assertRaises(ValueError):validate_disagreement_smoke(cfg,ev,pr,tr,sha)

if __name__=='__main__':unittest.main()
