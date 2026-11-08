"""All-component surgery, packed-token importance, native reload and recovery gradients."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from test_pruning import tiny_model
from smarthome_efficiency.pruning import apply_pruning, parameter_counts
from smarthome_efficiency.training import lora_targets
from smarthome_efficiency.multimodal_pruning import (
    COMPONENTS, BUDGETS, expected_counts, component_ffns, output_projection,
    is_prunable_ffn_tensor, MultimodalImportanceCollector, validate_multimodal_calibration,
    adapter_gradient_coverage,
)


def scores_for(model):
    return {'components': {key: {'ffn_activation': {str(i): list(range(output_projection(key, f).in_features))
                          for i, f in enumerate(ffns)}} for key, ffns in component_ffns(model).items()}}


def apply_tiny(model):
    return apply_pruning(model, 'multimodal_ffn_activation', ffn_width=12, audio_ffn_width=8,
                         vision_ffn_width=12, importance=scores_for(model))


def audio_mlp(layer, x):
    return layer.fc2(layer.activation_fn(layer.fc1(x)))


class MultimodalPruningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_production_budget_counts_match_native_meta_architectures(self):
        source = ROOT/'artifacts/compact/calibrated_pruning_20261019/mintrec2_ffn_activation/config.json'
        if not source.exists():
            self.skipTest('Production config not available')
        expected = [1_978_160_384, 1_963_496_704, 1_969_820_928]
        for (name, widths), total in zip(BUDGETS.items(), expected):
            config = Qwen2_5OmniThinkerConfig.from_pretrained(source.parent, local_files_only=True)
            config.text_config.intermediate_size = widths[0]
            config.audio_config.encoder_ffn_dim = widths[1]
            config.vision_config.intermediate_size = widths[2]
            config._attn_implementation = 'sdpa'
            with torch.device('meta'):
                model = Qwen2_5OmniThinkerForConditionalGeneration(config)
            self.assertEqual(parameter_counts(model), expected_counts(*widths), name)
            self.assertEqual(parameter_counts(model)['total'], total)
            self.assertLess(total, 2_000_000_000)
            # Even the training/evaluation representation with unmerged adapters
            # fits; deploying a merged adapter restores the smaller base count.
            wrapped = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
                target_modules=lora_targets('decoder_encoder_ffn'), task_type=TaskType.CAUSAL_LM, bias='none'))
            self.assertLess(sum(p.numel() for p in wrapped.parameters()), 2_000_000_000)
            counts = {key: 0 for key in COMPONENTS}
            for name, parameter in wrapped.named_parameters():
                if parameter.requires_grad:
                    key = 'audio_encoder' if 'audio_tower.' in name else 'vision_encoder' if 'visual.' in name else 'language'
                    counts[key] += parameter.numel()
            self.assertTrue(all(counts.values()))

    def test_coupled_surgery_equals_masked_ffns_and_preserves_every_other_tensor(self):
        model = tiny_model(); original = copy.deepcopy(model)
        protected = {n:(p.data_ptr(),p._version) for n,p in model.named_parameters() if not is_prunable_ffn_tensor(n)}
        rng = torch.get_rng_state().clone(); selection = apply_tiny(model)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        after = {n:(p.data_ptr(),p._version) for n,p in model.named_parameters() if not is_prunable_ffn_tensor(n)}
        self.assertEqual(protected,after)
        for key, ffns in component_ffns(model).items():
            for i, ffn in enumerate(ffns):
                old = component_ffns(original)[key][i]
                selected = selection['kept_ffn_channel_indices_by_component'][key][str(i)]
                x = torch.randn(2,3,output_projection(key,old).out_features)
                hidden = old.activation_fn(old.fc1(x)) if key=='audio_encoder' else old.act_fn(old.gate_proj(x))*old.up_proj(x)
                out = output_projection(key,old)
                expected = torch.nn.functional.linear(hidden[...,selected],out.weight[:,selected],out.bias)
                actual = audio_mlp(ffn,x) if key=='audio_encoder' else ffn(x)
                torch.testing.assert_close(actual,expected)
        self.assertEqual(model.config.audio_config.encoder_ffn_dim,8)
        self.assertEqual(model.config.vision_config.intermediate_size,12)
        self.assertEqual(len(model.model.layers),4)
        self.assertEqual(len(model.audio_tower.layers),1)
        self.assertEqual(len(model.visual.blocks),1)
        self.assertLess(parameter_counts(model)['total'],parameter_counts(original)['total'])

    def test_bad_scores_or_widths_are_rejected_before_any_mutation(self):
        for change in ('missing','wrong_width','nan','negative','bad_budget'):
            model=tiny_model();before={n:p.data_ptr() for n,p in model.named_parameters()}
            scores=scores_for(model);kwargs=dict(ffn_width=12,audio_ffn_width=8,vision_ffn_width=12)
            if change=='missing':del scores['components']['vision_encoder']
            if change=='wrong_width':scores['components']['vision_encoder']['ffn_activation']['0'].pop()
            if change=='nan':scores['components']['audio_encoder']['ffn_activation']['0'][0]=float('nan')
            if change=='negative':scores['components']['vision_encoder']['ffn_activation']['0'][0]=-1
            if change=='bad_budget':kwargs['vision_ffn_width']=100
            with self.assertRaises(ValueError):apply_pruning(model,'multimodal_ffn_activation',importance=scores,**kwargs)
            self.assertEqual(before,{n:p.data_ptr() for n,p in model.named_parameters()})

    def test_packed_statistics_use_real_tokens_and_gated_or_gelu_inputs(self):
        model=tiny_model();ids=torch.tensor([[1,4,7,8]])
        with torch.no_grad(),MultimodalImportanceCollector(model) as collector:
            model(input_ids=ids,use_cache=False)
            audio=model.audio_tower.layers[0];vision=model.visual.blocks[0].mlp
            x=torch.randn(7,16);y=torch.randn(11,32)
            hidden_a=audio.activation_fn(audio.fc1(x));hidden_v=vision.act_fn(vision.gate_proj(y))*vision.up_proj(y)
            audio.fc2(hidden_a);vision.down_proj(hidden_v)
        scores=collector.scores(1)
        torch.testing.assert_close(torch.tensor(scores['components']['audio_encoder']['ffn_activation']['0']),hidden_a.abs().mean(0))
        torch.testing.assert_close(torch.tensor(scores['components']['vision_encoder']['ffn_activation']['0']),hidden_v.abs().mean(0))
        self.assertEqual(scores['components']['audio_encoder']['tokens_per_layer'],[7])
        self.assertEqual(scores['components']['vision_encoder']['tokens_per_layer'],[11])
        self.assertEqual(validate_multimodal_calibration({'status':'completed','spec':{'examples':1},'scores':scores},{'examples':1},model),scores)
        scores['components']['audio_encoder']['examples_per_layer']=[2]
        with self.assertRaises(ValueError):validate_multimodal_calibration({'status':'completed','spec':{'examples':1},'scores':scores},{'examples':1},model)
        self.assertFalse(model.audio_tower.layers[0].fc2._forward_pre_hooks)

    def test_native_reload_preserves_all_ffns_counts_and_cached_generation(self):
        model=tiny_model();apply_tiny(model);ids=torch.tensor([[1,5,7,9]])
        x=torch.randn(7,16);y=torch.randn(11,32)
        with torch.no_grad():
            logits=model(input_ids=ids,use_cache=False).logits
            tokens=model.generate(ids,max_new_tokens=2,do_sample=False,use_cache=True)
            audio=audio_mlp(model.audio_tower.layers[0],x);vision=model.visual.blocks[0].mlp(y)
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder,safe_serialization=True)
            loaded=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(folder,attn_implementation='sdpa').eval()
            self.assertEqual(parameter_counts(model),parameter_counts(loaded))
            with torch.no_grad():
                torch.testing.assert_close(loaded(input_ids=ids,use_cache=False).logits,logits,rtol=0,atol=0)
                self.assertTrue(torch.equal(loaded.generate(ids,max_new_tokens=2,do_sample=False,use_cache=True),tokens))
                torch.testing.assert_close(audio_mlp(loaded.audio_tower.layers[0],x),audio,rtol=0,atol=0)
                torch.testing.assert_close(loaded.visual.blocks[0].mlp(y),vision,rtol=0,atol=0)

    def test_encoder_and_decoder_adapters_receive_gradients_and_merge_without_size_growth(self):
        model=tiny_model();apply_tiny(model);counts=parameter_counts(model)
        wrapped=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=lora_targets('decoder_encoder_ffn'),task_type=TaskType.CAUSAL_LM,bias='none'))
        ids=torch.tensor([[1,5,7,9]])
        base=wrapped.get_base_model()
        loss=wrapped(input_ids=ids,labels=ids,use_cache=False).loss
        loss=loss+audio_mlp(base.audio_tower.layers[0],torch.randn(7,16)).square().mean()
        loss=loss+base.visual.blocks[0].mlp(torch.randn(11,32)).square().mean()
        loss.backward();coverage=adapter_gradient_coverage(wrapped)
        self.assertEqual(set(coverage),set(COMPONENTS))
        self.assertTrue(all(v['gradient_l2']>0 for v in coverage.values()))
        with torch.no_grad():
            for n,p in wrapped.named_parameters():
                if 'audio_tower' in n and p.requires_grad:p.grad=None
        with self.assertRaisesRegex(ValueError,'Missing'):adapter_gradient_coverage(wrapped)
        torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=.001).step()
        merged=wrapped.merge_and_unload(safe_merge=True)
        self.assertEqual(parameter_counts(merged),counts)


if __name__=='__main__':unittest.main()
