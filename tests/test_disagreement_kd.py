"""Per-example teacher-disagreement recovery: gradients, packing, provenance guards."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.distillation import distillation_kl, disagreement_distillation_loss
from smarthome_efficiency.training import DEFAULTS, config_fingerprint, load_config, rank_update, validate_resume
import test_evaluation


class DisagreementKDTests(unittest.TestCase):
    def test_mixed_agreement_uses_per_example_weights_and_frozen_teacher(self):
        student=torch.tensor([[1.,-1.,.5],[2.,0.,-1.],[-1.,1.,0.]],requires_grad=True)
        teacher=torch.tensor([[3.,0.,-1.],[1.,3.,0.],[-1.,0.,2.]],requires_grad=True)
        sft=torch.tensor([.2,2.,.6],requires_grad=True)
        gold=[0,0,2];temperature=2.
        for wrong_alpha in [0.,.25]:
            student.grad=None;teacher.grad=None;sft.grad=None
            result=disagreement_distillation_loss(sft,student,teacher,gold,temperature,.75,wrong_alpha)
            weights=torch.tensor([.75,wrong_alpha,.75])
            self.assertEqual(result['disagreement'].tolist(),[False,True,False])
            torch.testing.assert_close(result['effective_alpha'],weights)
            self.assertFalse(result['effective_alpha'].requires_grad)
            per_kl=torch.stack([distillation_kl(student[i:i+1],teacher[i:i+1],temperature) for i in range(3)])
            expected=((1-weights)*sft+weights*per_kl).mean()
            torch.testing.assert_close(result['loss'],expected)
            # Averaging alpha before mixing would misweight these nonuniform SFT losses.
            wrong=(1-weights.mean())*sft.mean()+weights.mean()*per_kl.mean()
            self.assertGreater(abs(float((result['loss']-wrong).detach())),1e-3)
            result['loss'].backward()
            expected_gradient=(weights[:,None]*temperature/3)*(
                torch.softmax(student.detach()/temperature,-1)-torch.softmax(teacher.detach()/temperature,-1))
            torch.testing.assert_close(student.grad,expected_gradient,atol=1e-7,rtol=1e-5)
            torch.testing.assert_close(sft.grad,(1-weights)/3)
            self.assertIsNone(teacher.grad)
            if wrong_alpha==0:self.assertTrue(torch.equal(student.grad[1],torch.zeros(3)))

    def test_gate_uses_teacher_argmax_not_student_and_ties_are_deterministic(self):
        teacher=torch.tensor([[1.,1.],[0.,2.]])
        # Argmax ties select the first class, exactly as ordinary teacher scoring.
        for student in [torch.tensor([[5.,0.],[4.,0.]]),torch.tensor([[0.,5.],[0.,4.]])]:
            out=disagreement_distillation_loss(torch.ones(2),student,teacher,[0,0],2.,.75,.25)
            self.assertEqual(out['disagreement'].tolist(),[False,True])

    def test_single_example_and_partial_ddp_packing_preserve_gradient(self):
        # Simulate DDP's gradient mean after local world*real/valid weighting.
        torch.manual_seed(7)
        for world in [1,2,4]:
            for count in [1,3,7]:
                original=torch.randn(count,3);teacher=torch.randn(count,3);gold=torch.arange(count)%3
                for micro in [1,2,4]:
                    accumulation=4//micro
                    for update in range((count+world*4-1)//(world*4)):
                        start=update*world*4;stop=min(start+world*4,count)
                        student=original.clone().requires_grad_();sft=F.cross_entropy(student,gold,reduction='none')
                        out=disagreement_distillation_loss(sft[start:stop],student[start:stop],teacher[start:stop],gold[start:stop],2.,.75,.25)
                        out['loss'].backward();expected=student.grad.clone()
                        student=original.clone().requires_grad_();observed_count=0;observed_weight=0.
                        for rank in range(world):
                            indices,valid=rank_update(list(range(count)),update,world,rank,4)
                            for begin in range(0,4,micro):
                                real=[i for i in indices[begin:begin+micro] if i is not None]
                                ids=real or [0]
                                loss=F.cross_entropy(student[ids],gold[ids],reduction='none')
                                out=disagreement_distillation_loss(loss,student[ids],teacher[ids],gold[ids],2.,.75,.25)
                                # /world models DDP gradient averaging, including zero-weight padding.
                                (out['loss']*(world*len(real)/valid)/world).backward()
                                if real:
                                    observed_count+=int(out['disagreement'].sum())
                                    observed_weight+=float(out['effective_alpha'].sum())
                        torch.testing.assert_close(student.grad,expected,atol=1e-7,rtol=1e-5)
                        wrong=(teacher[start:stop].argmax(-1)!=gold[start:stop])
                        self.assertEqual(observed_count,int(wrong.sum()))
                        self.assertAlmostEqual(observed_weight,float(torch.where(wrong,.25,.75).sum()))

    def test_live_model_per_example_sft_matches_masked_scalar_and_packing(self):
        fixture=test_evaluation.ModelEvaluationTests();fixture.setUp();runner=fixture.runner
        batches=[{'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long)}
                 for ids in ([1,2],[1,2,3,4],[2],[1,3,2])]
        labels=[0,1,0,1];teacher=torch.tensor([[2.,0.],[2.,0.],[0.,2.],[0.,2.]])
        per_sft=[]
        for i,(batch,label) in enumerate(zip(batches,labels)):
            r=runner.loss_batch(batch,label,return_class_gradients=True,return_per_example_sft=True)
            self.assertEqual(r['sft_per_example_for_kd'].shape,(1,))
            self.assertTrue(r['sft_per_example_for_kd'].requires_grad)
            torch.testing.assert_close(r['sft_per_example_for_kd'].mean(),r['sft_loss'])
            per_sft.append(r['sft_per_example_for_kd'].detach())
            out=disagreement_distillation_loss(r['sft_per_example_for_kd'],r['class_logits_for_kd'],teacher[i:i+1],[label],2.,.75,.25)
            (out['loss']/4).backward()
        expected=runner.model.weight.grad.clone()
        runner.model.zero_grad(set_to_none=True)
        r=runner.loss_microbatch(batches,labels,return_class_gradients=True,return_per_example_sft=True)
        torch.testing.assert_close(r['sft_per_example_for_kd'],torch.cat(per_sft))
        torch.testing.assert_close(r['sft_per_example_for_kd'].mean(),r['sft_loss'])
        out=disagreement_distillation_loss(r['sft_per_example_for_kd'],r['class_logits_for_kd'],teacher,labels,2.,.75,.25)
        out['loss'].backward();torch.testing.assert_close(runner.model.weight.grad,expected)
        ordinary=runner.loss_batch(batches[0],0)
        self.assertNotIn('sft_per_example_for_kd',ordinary)
        with self.assertRaisesRegex(ValueError,'training-only'):
            runner.loss_batch(batches[0],0,train=False,return_per_example_sft=True)

    def test_ordinary_kd_path_preserves_exact_existing_formula(self):
        # The existing helper's batchmean reduction must remain exactly unchanged.
        student=torch.tensor([[.1,.2,.3],[4.,2.,1.]])
        teacher=torch.tensor([[.4,.2,.8],[1.,2.,4.]])
        expected=F.kl_div(F.log_softmax(student.float()/2.,dim=-1),F.softmax(teacher.float()/2.,dim=-1),reduction='batchmean')*4.
        self.assertTrue(torch.equal(distillation_kl(student,teacher,2.),expected))
        out=disagreement_distillation_loss(torch.tensor([.2,.7]),student,teacher,[2,2],2.,.75,.75)
        torch.testing.assert_close(out['loss'],.25*torch.tensor([.2,.7]).mean()+.75*expected)

    def test_invalid_inputs_fail_closed(self):
        args=[torch.ones(2),torch.ones(2,3),torch.ones(2,3),[0,1],2.,.75,.25]
        for index,value in [(0,torch.ones(1)),(1,torch.ones(1,3)),(2,torch.full((2,3),float('nan'))),
                            (3,[0,3]),(3,[0.,1.]),(3,[True,False]),(4,0.),(4,float('inf')),
                            (5,True),(5,float('nan')),(6,-.1),(6,1.1)]:
            bad=args.copy();bad[index]=value
            with self.subTest(index=index,value=str(value)),self.assertRaises(ValueError):
                disagreement_distillation_loss(*bad)

    def test_optional_config_preserves_fingerprint_and_resume_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'c.yaml'
            base={'model':'fixture','objective':'sft_kd','distillation_alpha':.75,
                  'teacher_checkpoint':'teacher','teacher_cache':'cache'}
            path.write_text(yaml.safe_dump(base));ordinary=load_config(path)
            self.assertNotIn('distillation_disagreement_alpha',ordinary)
            self.assertEqual(config_fingerprint(ordinary),config_fingerprint(json.loads(json.dumps(ordinary))))
            for val in [0.,.25]:
                path.write_text(yaml.safe_dump(base|{'distillation_disagreement_alpha':val}));gated=load_config(path)
                self.assertEqual(gated['distillation_disagreement_alpha'],val)
                with self.assertRaisesRegex(ValueError,'distillation_disagreement_alpha'):
                    validate_resume(ordinary,gated,1,1)
                with self.assertRaisesRegex(ValueError,'distillation_disagreement_alpha'):
                    validate_resume(gated,ordinary,1,1)
                validate_resume(gated,json.loads(json.dumps(gated)),1,1)
                packed=gated|{'micro_batch_size':2,'gradient_accumulation_steps':2}
                validate_resume(packed,packed|{'micro_batch_size':1,'gradient_accumulation_steps':4},1,1)
            for change in [{'distillation_disagreement_alpha':v} for v in [None,True,-.1,1.1,float('inf'),float('nan')]]+[
                {'distillation_disagreement_alpha':.25,'objective':'sft'},
                {'distillation_disagreement_alpha':.25,'distillation_alpha_final':0.},
            ]:
                path.write_text(yaml.safe_dump(base|change))
                with self.subTest(change=change),self.assertRaises(ValueError):load_config(path)
            a=DEFAULTS|base|{'distillation_disagreement_alpha':.25}
            with self.assertRaisesRegex(ValueError,'distillation_disagreement_alpha'):
                validate_resume(a,a|{'distillation_disagreement_alpha':0.},1,1)


if __name__=='__main__':unittest.main()
