"""Structural pruning correctness, native reload/cache, and fresh-LoRA recovery."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from smarthome_efficiency.pruning import (
    apply_pruning, evenly_spaced_layers, magnitude_scores, parameter_counts,
    processor_cache_identity, processor_hashes, prune_ffn_channels, prune_layers, select_top_channels,
)


def tiny_model():
    torch.manual_seed(13)
    config = Qwen2_5OmniThinkerConfig(
        text_config={'vocab_size':64, 'hidden_size':32, 'intermediate_size':48,
                     'num_hidden_layers':4, 'num_attention_heads':4, 'num_key_value_heads':2,
                     'max_position_embeddings':128, 'rope_scaling':{'rope_type':'default', 'mrope_section':[1,1,2]},
                     'pad_token_id':0, 'bos_token_id':1, 'eos_token_id':2, 'use_sliding_window':False},
        audio_config={'encoder_layers':1, 'd_model':16, 'encoder_attention_heads':2,
                      'encoder_ffn_dim':32, 'output_dim':32, 'max_source_positions':16},
        vision_config={'depth':1, 'hidden_size':32, 'intermediate_size':48, 'num_heads':4,
                       'out_hidden_size':32, 'patch_size':2, 'temporal_patch_size':2,
                       'fullatt_block_indexes':[0]},
        pad_token_id=0, bos_token_id=1, eos_token_id=2, vision_start_token_id=60,
    )
    config._attn_implementation = 'sdpa'
    return Qwen2_5OmniThinkerForConditionalGeneration(config).eval()


class PruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_even_spacing_retains_endpoints_and_original_order(self):
        self.assertEqual(evenly_spaced_layers(6, 4), [0,2,3,5])
        self.assertEqual(evenly_spaced_layers(36,17), [0,2,4,7,9,11,13,15,18,20,22,24,26,28,31,33,35])
        self.assertEqual(evenly_spaced_layers(4,4), list(range(4)))
        for bad in [(4,1),(4,5),(4,True)]:
            with self.assertRaises(ValueError): evenly_spaced_layers(*bad)

    def test_scores_cover_all_three_coupled_matrices_and_stable_ties(self):
        model=tiny_model(); mlp=model.model.layers[0].mlp
        expected=torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight.T],dim=1).square().sum(1)
        torch.testing.assert_close(magnitude_scores(mlp),expected)
        self.assertEqual(select_top_channels(torch.tensor([1.,3.,3.,0.]),2),[1,2])
        self.assertEqual(select_top_channels(torch.ones(4),2),[0,1])
        with self.assertRaises(ValueError): select_top_channels(torch.tensor([float('nan')]),1)

    def test_channel_surgery_matches_explicit_mask_and_preserves_encoders(self):
        model=tiny_model(); original=copy.deepcopy(model)
        keep={i:[0,3,7,20] for i in range(4)}
        rng=torch.get_rng_state().clone()
        apply_result=prune_ffn_channels(model,keep)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(model.config.text_config.intermediate_size,4)
        for i,layer in enumerate(model.model.layers):
            x=torch.randn(2,3,32); old=original.model.layers[i].mlp
            hidden=old.act_fn(old.gate_proj(x))*old.up_proj(x)
            expected=torch.nn.functional.linear(hidden[...,keep[i]],old.down_proj.weight[:,keep[i]])
            torch.testing.assert_close(layer.mlp(x),expected)
        for name,value in original.state_dict().items():
            if name.startswith(('visual.','audio_tower.')):
                self.assertTrue(torch.equal(value,model.state_dict()[name]))
        self.assertEqual(apply_result['kept_original_ffn_channel_indices']['0'],keep[0])
        self.assertLess(parameter_counts(model)['language'],parameter_counts(original)['language'])

    def test_invalid_selection_is_rejected_before_mutation(self):
        model=tiny_model(); before=model.model.layers[0].mlp.gate_proj
        for bad in [{0:[1]}, {i:([1,2] if i==0 else [1]) for i in range(4)},
                    {i:[2,1] for i in range(4)}, {i:[1,1] for i in range(4)}]:
            with self.assertRaises(ValueError): prune_ffn_channels(model,bad)
            self.assertIs(model.model.layers[0].mlp.gate_proj,before)
        with self.assertRaises(ValueError): prune_layers(model,[2,1])
        with self.assertRaises(ValueError): apply_pruning(model,'unimplemented')

    def test_native_save_reload_and_cached_generation_for_both_methods(self):
        ids=torch.tensor([[1,5,7,9]])
        for method,kwargs in [('depth_uniform',{'keep_layers':2}),('ffn_magnitude',{'ffn_width':16})]:
            model=tiny_model(); apply_pruning(model,method,**kwargs)
            with torch.no_grad():
                before=model(input_ids=ids,use_cache=False).logits
                generated=model.generate(ids,max_new_tokens=3,do_sample=False,use_cache=True)
            self.assertEqual([layer.self_attn.layer_idx for layer in model.model.layers],list(range(len(model.model.layers))))
            with tempfile.TemporaryDirectory() as path:
                model.save_pretrained(path,safe_serialization=True)
                reloaded=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(path,attn_implementation='sdpa').eval()
                with torch.no_grad():
                    after=reloaded(input_ids=ids,use_cache=False).logits
                    actual=reloaded.generate(ids,max_new_tokens=3,do_sample=False,use_cache=True)
                torch.testing.assert_close(before,after,rtol=0,atol=0)
                self.assertTrue(torch.equal(generated,actual))
                self.assertEqual(parameter_counts(model),parameter_counts(reloaded))

    def test_merge_then_prune_and_fresh_lora_can_recover(self):
        model=tiny_model(); ids=torch.tensor([[1,5,7,9]])
        target=r'model\.layers\.\d+\.self_attn\.(q_proj|v_proj)'
        adapter=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=target,task_type=TaskType.CAUSAL_LM))
        with torch.no_grad():
            for name,p in adapter.named_parameters():
                if 'lora_B' in name: p.normal_(std=.01)
        adapter.eval()
        with torch.no_grad(): expected=adapter(input_ids=ids,use_cache=False).logits
        with self.assertRaisesRegex(ValueError,'Merge'): apply_pruning(adapter,'depth_uniform',keep_layers=2)
        merged=adapter.merge_and_unload(safe_merge=True).eval()
        with torch.no_grad(): torch.testing.assert_close(merged(input_ids=ids,use_cache=False).logits,expected,atol=1e-6,rtol=1e-5)
        apply_pruning(merged,'ffn_magnitude',ffn_width=16)
        with torch.no_grad(): pruned_logits=merged(input_ids=ids,use_cache=False).logits
        fresh=get_peft_model(merged,LoraConfig(r=2,lora_alpha=4,target_modules=target,task_type=TaskType.CAUSAL_LM)).eval()
        with torch.no_grad(): torch.testing.assert_close(fresh(input_ids=ids,use_cache=False).logits,pruned_logits,atol=0,rtol=0)
        optimizer=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad],lr=.001)
        fresh.train(); loss=fresh(input_ids=ids,labels=ids,use_cache=False).loss
        loss.backward(); optimizer.step()
        self.assertTrue(any(torch.count_nonzero(p).item() for n,p in fresh.named_parameters() if 'lora_B' in n))
        self.assertFalse(any(p.requires_grad for n,p in fresh.named_parameters() if 'lora_' not in n))

    def test_processor_cache_reuse_requires_identical_assets(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path); source=root/'source'; source.mkdir(); compact=root/'compact'; compact.mkdir()
            for name in ['config.json','tokenizer.json','preprocessor_config.json']:
                (source/name).write_text('{}'); (compact/name).write_text('{}')
            identity=processor_cache_identity(source)
            (compact/'config.json').write_text('{"pruned":true}')
            manifest={'format_version':1,'processor_cache_identity':identity,'processor_files_sha256':processor_hashes(source)}
            (compact/'compression.json').write_text(json.dumps(manifest))
            self.assertEqual(processor_cache_identity(compact),identity)
            (compact/'tokenizer.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'processor files'): processor_cache_identity(compact)


if __name__=='__main__': unittest.main()
