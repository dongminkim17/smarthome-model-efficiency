"""Encoder-only surgery, exact parameter limits and overnight queue provenance."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
import torch
import yaml
from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import test_a_feature_queue as fixtures
from test_pruning import tiny_model
from test_multimodal_pruning import scores_for, audio_mlp
from smarthome_efficiency.pruning import apply_pruning, parameter_counts, sha256_file
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.encoder_pruning import encoder_parameters, validate_separate_budget, is_prunable_encoder_tensor
from smarthome_efficiency.queueing import atomic_json, read_json, validate_graph
from smarthome_efficiency.training import load_config
from project_queue_jobs import entry, native_command, validate_export
from queue_encoder_budget import build_submission, FAMILY, TASKS, ARCHITECTURES, VARIANTS


class EncoderBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_surgery_preserves_every_language_parameter_and_matches_mask(self):
        model=tiny_model(); original=copy.deepcopy(model); scores=scores_for(model)
        protected={n:(id(p),p.data_ptr(),p._version,p.detach().clone()) for n,p in model.named_parameters() if not is_prunable_encoder_tensor(n)}
        rng=torch.get_rng_state().clone()
        result=apply_pruning(model,'encoder_ffn_activation',ffn_width=48,audio_ffn_width=16,vision_ffn_width=24,importance=scores)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(model.config.text_config.intermediate_size,48)
        for n,p in model.named_parameters():
            if n in protected:
                ident,pointer,version,value=protected[n]
                self.assertEqual((id(p),p.data_ptr(),p._version),(ident,pointer,version))
                self.assertTrue(torch.equal(p,value),n)
        self.assertEqual(parameter_counts(model)['language'],parameter_counts(original)['language'])
        self.assertTrue(result['language_tensors_unchanged'])
        for key,old,new in [('audio_encoder',original.audio_tower.layers[0],model.audio_tower.layers[0]),('vision_encoder',original.visual.blocks[0].mlp,model.visual.blocks[0].mlp)]:
            keep=result['kept_ffn_channel_indices_by_component'][key]['0']
            if key=='audio_encoder':
                x=torch.randn(7,16); hidden=old.activation_fn(old.fc1(x))
                expected=torch.nn.functional.linear(hidden[:,keep],old.fc2.weight[:,keep],old.fc2.bias)
                torch.testing.assert_close(audio_mlp(new,x),expected)
            else:
                x=torch.randn(9,32);hidden=old.act_fn(old.gate_proj(x))*old.up_proj(x)
                expected=torch.nn.functional.linear(hidden[:,keep],old.down_proj.weight[:,keep],old.down_proj.bias)
                torch.testing.assert_close(new(x),expected)

    def test_invalid_second_encoder_cannot_partially_prune_audio_or_lm(self):
        for bad in ('nan','negative','width','missing','language_width'):
            model=tiny_model();scores=scores_for(model);kwargs=dict(ffn_width=48,audio_ffn_width=16,vision_ffn_width=24)
            before={n:(id(p),p._version) for n,p in model.named_parameters()}
            if bad=='nan':scores['components']['vision_encoder']['ffn_activation']['0'][0]=float('nan')
            if bad=='negative':scores['components']['vision_encoder']['ffn_activation']['0'][0]=-1
            if bad=='width':kwargs['vision_ffn_width']=100
            if bad=='missing':del scores['components']['vision_encoder']
            if bad=='language_width':kwargs['ffn_width']=24
            with self.assertRaises(ValueError):apply_pruning(model,'encoder_ffn_activation',importance=scores,**kwargs)
            self.assertEqual(before,{n:(id(p),p._version) for n,p in model.named_parameters()})

    def test_native_reload_and_av_outputs_generation_are_unchanged(self):
        model=tiny_model();apply_pruning(model,'encoder_ffn_activation',ffn_width=48,audio_ffn_width=16,vision_ffn_width=24,importance=scores_for(model))
        ids=torch.tensor([[1,4,6,8]]);x=torch.randn(7,16);y=torch.randn(9,32)
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
                self.assertTrue(torch.equal(tokens,loaded.generate(ids,max_new_tokens=2,do_sample=False,use_cache=True)))
                torch.testing.assert_close(audio_mlp(loaded.audio_tower.layers[0],x),audio,rtol=0,atol=0)
                torch.testing.assert_close(loaded.visual.blocks[0].mlp(y),vision,rtol=0,atol=0)

    def test_native_meta_architectures_meet_both_caps_but_not_old_total_cap(self):
        source=ROOT/'artifacts/compact/calibrated_pruning_20261019/mintrec2_ffn_activation'
        if not source.exists():self.skipTest('Production config unavailable')
        for name,widths in ARCHITECTURES.items():
            cfg=Qwen2_5OmniThinkerConfig.from_pretrained(source,local_files_only=True)
            cfg.text_config.intermediate_size=widths[0];cfg.audio_config.encoder_ffn_dim=widths[1];cfg.vision_config.intermediate_size=widths[2];cfg._attn_implementation='sdpa'
            with torch.device('meta'):model=Qwen2_5OmniThinkerForConditionalGeneration(cfg)
            counts=parameter_counts(model);self.assertEqual(counts,expected_counts(*widths),name)
            validate_separate_budget(expected_counts(4352,5120,3420),counts)
            self.assertEqual(counts['language'],1_924_902_912);self.assertLess(encoder_parameters(counts),1_000_000_000)
            self.assertGreater(counts['total'],2_000_000_000)
        before=dict(language=2_000_000_000,audio_encoder=700_000_000,vision_encoder=700_000_000,other=0)
        after=before|dict(audio_encoder=500_000_000,vision_encoder=500_000_000)
        validate_separate_budget(before,after)
        for change in [dict(language=1_999_999_999),dict(audio_encoder=500_000_001),dict(other=1)]:
            with self.assertRaises(ValueError):validate_separate_budget(before,after|change)

    def fixture(self,root):
        state=fixtures.AFeatureQueueTests().fixture(root)
        by={j['id']:j for j in state['jobs']}
        for task in TASKS:
            a=by['full_model_2b_20261020/export:'+task+'_a_lm1536_a1024_v1024']
            side=read_json(Path(a['payload']['output'])/'compression.json');source=Path(side['source_checkpoint'])
            meta=read_json(source/'metadata.json');base=Path(meta['config']['model'])
            atomic_json(base/'compression.json',{'parameters_after':expected_counts(4352,5120,3420),'verification':{'passed':True}})
            meta['config'].update(objective='sft_kd',distillation_alpha=.75);atomic_json(source/'metadata.json',meta)
            cache=Path(meta['config']['teacher_cache']);atomic_json(cache,read_json(cache)|{'status':'completed'})
            cal=root/task/'calibration.json';atomic_json(cal,{'fixture':'training-only pinned scores'})
            side.update(calibration_cache=str(cal),calibration_samples=128,calibration_seed=42,calibration={'sha256':sha256_file(cal)})
            side['source_metadata_sha256']=sha256_file(source/'metadata.json');atomic_json(Path(a['payload']['output'])/'compression.json',side)
            state['jobs'].append(entry('recovery_controls_20261019/train:KD075_'+task,'recovery_train',{'task':task,'status':'completed','run_dir':str(source.parent.parent)}))
        return state

    def test_24_runs_pin_unchanged_lm_and_have_independent_smoke_and_official_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);original=copy.deepcopy(state);spec=build_submission(state,root)
            self.assertEqual(state,original);validate_graph(state['jobs']+spec['jobs'])
            self.assertEqual(len(spec['jobs']),105)
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train'];exports=[j for j in spec['jobs'] if j['kind']=='prune_export']
            self.assertEqual(len(trains),24);self.assertEqual(len(exports),9)
            by={j['id']:j for j in spec['jobs']}
            for task in TASKS:
                self.assertEqual({j['payload']['arm'] for j in trains if j['task']==task},{v[0] for v in VARIANTS})
            for job in trains:
                p=job['payload'];cfg=p['config'];self.assertEqual(p['budget_scope'],'separate');self.assertIsNone(p['init_adapter'])
                self.assertEqual(cfg['epochs'],5);self.assertEqual(cfg['micro_batch_size']*cfg['gradient_accumulation_steps'],4)
                self.assertEqual(cfg['eval_modes'],['logit','greedy']);self.assertEqual(cfg['selection_mode'],'logit')
                self.assertEqual(p['expected_parameters']['language'],1_924_902_912)
                reload=by[job['deps'][0]];smoke=by[reload['deps'][0]]
                self.assertEqual(smoke['kind'],'smoke_train');self.assertEqual(reload['kind'],'smoke_test')
                self.assertEqual(by[FAMILY+'/test:'+p['name']]['deps'],[job['id']])
                self.assertEqual('feature_teacher_cache' in cfg,'FEATURE' in p['arm'])
                path=root/'cfg.yaml';path.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(path),cfg)
            for job in exports:
                command,_=native_command(state,job,root)
                self.assertIn('separate',command);self.assertIn('encoder_ffn_activation',command)
                self.assertFalse(job['payload']['generate_calibration'])
            # Changed lineage/cache metadata cannot silently enter the batch.
            source=Path(exports[0]['payload']['request']['source_checkpoint']);meta=read_json(source/'metadata.json');meta['config']['distillation_alpha']=.5;atomic_json(source/'metadata.json',meta)
            with self.assertRaises(ValueError):build_submission(state,root)


    def test_export_validator_enforces_both_caps_and_original_total_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);spec=build_submission(state,root)
            p=next(j['payload'] for j in spec['jobs'] if j['kind']=='prune_export')
            request=p['request'];out=Path(p['output']);counts=p['expected_parameters']
            report=dict(request,parameters_before=expected_counts(4352,5120,3420),parameters_after=counts,
                verification={'passed':True,'save_reload_class_logits_match':True,'save_reload_generation_match':True},
                test_set_used=False,verification_split='train',recovery_updates=0,encoder_tensors_unchanged=False,
                language_tensors_unchanged=True,non_ffn_tensors_unchanged=True,av_interfaces_unchanged=True,all_layer_counts_unchanged=True,
                parameter_limits={'language':2_000_000_000,'combined_encoders':1_000_000_000},
                combined_encoder_parameters=encoder_parameters(counts),
                calibration={'sha256':sha256_file(Path(request['calibration_cache'])),'spec':{'scope':'all_thinker_components'}})
            atomic_json(out/'compression.json',report);atomic_json(out/'model.safetensors.index.json',{})
            atomic_json(out/'config.json',{'text_config':{'intermediate_size':4352,'num_hidden_layers':36},
                'audio_config':{'encoder_ffn_dim':request['audio_ffn_width'],'encoder_layers':32},
                'vision_config':{'intermediate_size':request['vision_ffn_width'],'depth':32}})
            self.assertEqual(validate_export(p)['combined_encoder_parameters'],990823680)
            for change in [{'language_tensors_unchanged':False},
                           {'parameters_before':report['parameters_before']|{'language':counts['language']+1}},
                           {'parameters_after':counts|{'audio_encoder':1_000_000_001}}]:
                atomic_json(out/'compression.json',report|change)
                with self.assertRaises(ValueError):validate_export(p)
            # Granting a separate budget must never loosen the historic total<2B validation.
            old=copy.deepcopy(p);old['request'].update(budget_scope='total',method='multimodal_ffn_activation')
            atomic_json(out/'compression.json',report|old['request'])
            with self.assertRaises(ValueError):validate_export(old)

if __name__=='__main__':unittest.main()
