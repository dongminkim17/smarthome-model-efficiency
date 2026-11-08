"""Encoder BI, protected global attention, coupled surgery and native reload."""
import copy,sys,tempfile,unittest
from pathlib import Path
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniThinkerConfig
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_pruning import tiny_model
from test_feature_distillation import media_batch,encode
from smarthome_efficiency.hybrid_pruning import HybridCollector,select_encoder_layers,D_COUNTS
from smarthome_efficiency.pruning import apply_pruning,parameter_counts
from smarthome_efficiency.multimodal_pruning import component_ffns,output_projection


def hybrid_model():
    cfg=copy.deepcopy(tiny_model().config);cfg.audio_config.encoder_layers=4;cfg.vision_config.depth=4;cfg.vision_config.fullatt_block_indexes=[1,3]
    return Qwen2_5OmniThinkerForConditionalGeneration(cfg).eval()


class HybridPruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_protected_layers_override_ranking_with_stable_ties(self):
        self.assertEqual(select_encoder_layers([0.,0.,1.,0.,1.,0.],4,[0,5]),[0,2,4,5])
        self.assertEqual(select_encoder_layers([1.]*6,4,[0,5]),[0,1,2,5])
        with self.assertRaises(ValueError):select_encoder_layers([1.]*4,2,[0,1,3])

    def test_hybrid_collector_and_surgery_preserve_protected_blocks_reload_and_av_outputs(self):
        model=hybrid_model();batch=media_batch();ids=torch.tensor([[1,4,7,8]])
        with torch.no_grad(),HybridCollector(model) as collector:
            model(input_ids=ids,use_cache=False);encode(model,batch)
        scores=collector.scores(1)
        self.assertEqual(len(scores['audio_layer_bi']),4);self.assertEqual(len(scores['vision_layer_bi']),4)
        audio_original=list(model.audio_tower.layers);vision_original=list(model.visual.blocks)
        result=apply_pruning(model,'multimodal_depth_ffn_activation',ffn_width=12,audio_ffn_width=8,
                            vision_ffn_width=12,audio_layers=3,vision_layers=3,importance=scores)
        self.assertEqual(result['kept_original_vision_layer_indices'],[0,1,3])
        self.assertEqual(model.visual.fullatt_block_indexes,[1,2]);self.assertEqual(model.config.vision_config.fullatt_block_indexes,[1,2])
        for new,old in enumerate(result['kept_original_audio_layer_indices']):self.assertIs(model.audio_tower.layers[new],audio_original[old])
        for new,old in enumerate(result['kept_original_vision_layer_indices']):self.assertIs(model.visual.blocks[new],vision_original[old])
        self.assertEqual(len(model.model.layers),4);self.assertTrue(result['retained_non_ffn_tensors_unchanged'])
        with torch.no_grad():expected=encode(model,batch);logits=model(input_ids=ids,use_cache=False).logits;tokens=model.generate(ids,max_new_tokens=2,do_sample=False)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder,safe_serialization=True)
            loaded=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(folder,attn_implementation='sdpa').eval()
            self.assertEqual(parameter_counts(loaded),parameter_counts(model))
            with torch.no_grad():
                actual=encode(loaded,batch)
                for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
                torch.testing.assert_close(loaded(input_ids=ids,use_cache=False).logits,logits,rtol=0,atol=0)
                self.assertTrue(torch.equal(loaded.generate(ids,max_new_tokens=2,do_sample=False),tokens))

    def test_bad_layer_budget_rejected_before_ffn_mutation(self):
        model=hybrid_model();before={n:p.data_ptr() for n,p in model.named_parameters()}
        scores={'audio_layer_bi':[.1]*4,'vision_layer_bi':[.1]*4,'components':{}}
        with self.assertRaises(ValueError):apply_pruning(model,'multimodal_depth_ffn_activation',ffn_width=12,
                    audio_ffn_width=8,vision_ffn_width=12,audio_layers=3,vision_layers=2,importance=scores)
        self.assertEqual(before,{n:p.data_ptr() for n,p in model.named_parameters()})

    def test_production_d_exact_counts_match_native_meta_model(self):
        source=ROOT/'artifacts/compact/calibrated_pruning_20261019/mintrec2_ffn_activation'
        if not source.exists():self.skipTest('Production config not available')
        cfg=Qwen2_5OmniThinkerConfig.from_pretrained(source,local_files_only=True)
        cfg.text_config.intermediate_size=2048;cfg.audio_config.encoder_layers=24;cfg.audio_config.encoder_ffn_dim=1536
        cfg.vision_config.depth=24;cfg.vision_config.intermediate_size=1280;cfg.vision_config.fullatt_block_indexes=[5,11,17,23];cfg._attn_implementation='sdpa'
        with torch.device('meta'):model=Qwen2_5OmniThinkerForConditionalGeneration(cfg)
        self.assertEqual(parameter_counts(model),D_COUNTS);self.assertLess(D_COUNTS['total'],2000000000)

if __name__=='__main__':unittest.main()
