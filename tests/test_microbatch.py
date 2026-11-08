"""Microbatch masking, feature ordering, gradient equivalence, and resume guards."""
from pathlib import Path
import tempfile
import unittest
import torch
import yaml

import test_evaluation
from smarthome_efficiency.model import collate_prepared_batches, trim_audio_padding
from smarthome_efficiency.training import DEFAULTS, load_config, rank_update, validate_resume


class MicrobatchTests(unittest.TestCase):
    def test_masked_audio_selection_is_exactly_preserved(self):
        features = torch.arange(2*3*9).reshape(2,3,9).float()
        mask = torch.tensor([[1,0,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0]])
        compact = trim_audio_padding({'input_features':features,'feature_attention_mask':mask})
        self.assertEqual(compact['input_features'].shape, (2,3,3))
        old = features.permute(0,2,1)[mask.bool()]
        new = compact['input_features'].permute(0,2,1)[compact['feature_attention_mask'].bool()]
        self.assertTrue(torch.equal(old,new))
        self.assertTrue(torch.equal(mask.sum(-1),compact['feature_attention_mask'].sum(-1)))
        self.assertEqual(compact['input_features'].untyped_storage().nbytes(), compact['input_features'].numel()*4)

    def test_collation_left_pads_tokens_and_preserves_patch_audio_order(self):
        batches=[]
        for length,patches in [(2,3),(4,5)]:
            batches.append({'input_ids':torch.full((1,length),length),
                            'attention_mask':torch.ones(1,length,dtype=torch.long),
                            'pixel_values_videos':torch.full((patches,2),length),
                            'video_grid_thw':torch.tensor([[1,patches,1]]),
                            'video_second_per_grid':torch.tensor([2.0]),
                            'input_features':torch.full((1,3,length),float(length)),
                            'feature_attention_mask':torch.ones(1,length,dtype=torch.long)})
        b=collate_prepared_batches(batches,0)
        self.assertEqual(b['input_ids'].tolist(),[[0,0,2,2],[4,4,4,4]])
        self.assertEqual(b['attention_mask'].tolist(),[[0,0,1,1],[1,1,1,1]])
        self.assertEqual(b['feature_attention_mask'].tolist(),[[1,1,0,0],[1,1,1,1]])
        self.assertEqual(b['pixel_values_videos'][:,0].tolist(),[2]*3+[4]*5)
        self.assertEqual(batches[0]['input_ids'].shape,(1,2))

    def test_batched_loss_and_gradient_equal_accumulated_examples(self):
        fixture=test_evaluation.ModelEvaluationTests(); fixture.setUp(); runner=fixture.runner
        batches=[{'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long)}
                 for ids in ([1,2],[1,2,3,4],[2],[1,3,2])]
        targets=[0,1,0,1]
        losses=[]; logits=[]
        for b,t in zip(batches,targets):
            r=runner.loss_batch(b,t); (r['loss']/4).backward()
            losses.append(r['loss'].detach()); logits.append(r['class_logits'])
        expected_grad=runner.model.weight.grad.clone()
        for size in [2,4]:
            runner.model.zero_grad(set_to_none=True)
            values=[]
            for start in range(0,4,size):
                r=runner.loss_microbatch(batches[start:start+size],targets[start:start+size])
                (r['loss']*size/4).backward()
                values.extend(r['class_logits'].unbind(0))
            torch.testing.assert_close(runner.model.weight.grad,expected_grad)
            torch.testing.assert_close(torch.stack(values),torch.stack(logits))
        torch.testing.assert_close(r['loss'],torch.stack(losses).mean())
        self.assertEqual(runner.model.last_labels.tolist(),[[-100]*4+[5,9],[-100]*4+[7,9],[-100]*4+[5,9],[-100]*4+[7,9]])

    def test_resume_allows_repacking_but_not_a_different_effective_batch(self):
        old=dict(DEFAULTS,micro_batch_size=1,gradient_accumulation_steps=4)
        for micro in [1,2,4]:
            new=old|{'micro_batch_size':micro,'gradient_accumulation_steps':4//micro}
            validate_resume(old,new,1,1)
        with self.assertRaisesRegex(ValueError,'effective batch'):
            validate_resume(old,old|{'micro_batch_size':2},1,1)
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'config.yaml';p.write_text(yaml.safe_dump({'model':'local','micro_batch_size':2}))
            self.assertEqual(load_config(p)['micro_batch_size'],2)

    def test_partial_batch_weighting_for_multiple_micro_sizes(self):
        for world in [1,4]:
            for count in [1,3,7,17]:
                for micro in [1,2,4]:
                    accumulation=4//micro
                    visited=[]
                    for update in range((count+world*4-1)//(world*4)):
                        rank_grad=[]; expected=[]
                        for rank in range(world):
                            indices,valid=rank_update(list(range(count)),update,world,rank,4)
                            gradient=0.0
                            for start in range(0,4,micro):
                                real=[x for x in indices[start:start+micro] if x is not None]
                                visited.extend(real); expected.extend(x+1.0 for x in real)
                                if real:
                                    mean=sum(x+1.0 for x in real)/len(real)
                                    gradient+=mean*world*len(real)/valid
                            rank_grad.append(gradient)
                        self.assertAlmostEqual(sum(rank_grad)/world,sum(expected)/len(expected))
                    self.assertEqual(sorted(visited),list(range(count)))
