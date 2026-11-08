"""Reference-equation parity, multimodal-independent surgery, and cache contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from test_pruning import tiny_model
from smarthome_efficiency.pruning import apply_pruning,parameter_counts
from smarthome_efficiency.pruning_calibration import ImportanceCollector,block_influence,ffn_example_squares,obtain_calibration,validate_calibration
from launch_pruning_sweep import CALIBRATED_METHODS,coexist_allocation,fresh_plans
from launch_pruning_followup import build_jobs


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_block_influence_matches_shortgpt_diagonal_without_square_matrix(self):
        torch.manual_seed(2);x=torch.randn(2,3,5);y=torch.randn_like(x)
        x[0,0]=0
        a=x.reshape(-1,5);b=y.reshape(-1,5)
        # Literal released ShortGPT formula: retain the diagonal of pairwise sim.
        reference=1-((a@b.T)/(a.norm(dim=-1,keepdim=True)*b.norm(dim=-1,keepdim=True))).diagonal().nan_to_num(nan=.5)
        torch.testing.assert_close(block_influence(x,y).reshape(-1),reference,atol=2e-7,rtol=1e-6)
        torch.testing.assert_close(block_influence(y,y),torch.zeros(2,3),atol=2e-7,rtol=0)
        with self.assertRaises(ValueError):block_influence(x,y[:,:2])

    def test_ffn_reduction_matches_minitron_seq_mean_batch_l2(self):
        clips=[torch.tensor([[[1.,-2.],[-3.,4.]]]),torch.tensor([[[2.,1.],[2.,-1.],[2.,1.]]])]
        squared=sum(ffn_example_squares(x) for x in clips)
        # Official Model Optimizer uses [sequence,batch,channel].
        reference=sum(x.permute(1,0,2).abs().mean(0).pow(2).sum(0) for x in clips).sqrt()
        torch.testing.assert_close(squared.sqrt(),reference)
        torch.testing.assert_close(squared,torch.tensor([8.,10.]))
        with self.assertRaises(ValueError):ffn_example_squares(torch.zeros(2,3,4))
        with self.assertRaises(ValueError):ffn_example_squares(torch.full((1,2,3),float('nan')))

    def test_streaming_hooks_measure_gated_inputs_and_raw_block_outputs(self):
        model=tiny_model();ids=torch.tensor([[1,4,7,8]])
        expected={};handles=[]
        for i,layer in enumerate(model.model.layers):
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m,a,i=i:expected.__setitem__(('ffn',i),ffn_example_squares(a[0]).double())))
            def save(module,args,kwargs,output,i=i):
                x=kwargs.get('hidden_states',args[0] if args else None);y=output[0] if isinstance(output,tuple) else output
                expected[('bi',i)]=block_influence(x,y).double().mean().item()
            handles.append(layer.register_forward_hook(save,with_kwargs=True))
        try:
            with torch.no_grad(),ImportanceCollector(model) as collector:model(input_ids=ids,use_cache=False)
            scores=collector.scores()
        finally:
            for h in handles:h.remove()
        self.assertEqual(scores['tokens_per_layer'],[4]*4)
        self.assertEqual(scores['examples_per_layer'],[1]*4)
        for i in range(4):
            self.assertAlmostEqual(scores['layer_bi'][i],expected['bi',i],places=8)
            torch.testing.assert_close(torch.tensor(scores['ffn_activation'][str(i)],dtype=torch.float64),expected['ffn',i].sqrt())
        self.assertFalse(any(l._forward_hooks or l.mlp.down_proj._forward_pre_hooks for l in model.model.layers))

    def test_hooks_removed_on_error_and_incomplete_statistics_rejected(self):
        model=tiny_model();collector=ImportanceCollector(model)
        with self.assertRaises(RuntimeError):
            with collector:raise RuntimeError('interrupted calibration')
        self.assertFalse(any(l._forward_hooks or l.mlp.down_proj._forward_pre_hooks for l in model.model.layers))
        with self.assertRaises(ValueError):collector.scores()

    def test_native_reload_for_calibrated_depth_ffn_and_hybrid(self):
        ids=torch.tensor([[1,5,7,9]])
        scores={'layer_bi':[.1,.8,.2,.9], 'ffn_activation':{str(i):list(range(48)) for i in range(4)}}
        for method,kwargs in [('depth_bi',{'keep_layers':2}),('ffn_activation',{'ffn_width':16}),('depth_ffn_activation',{'keep_layers':2,'ffn_width':16})]:
            model=tiny_model();original=copy.deepcopy(model)
            selected=apply_pruning(model,method,importance=scores,**kwargs)
            if 'keep_layers' in kwargs:
                self.assertEqual(selected['kept_original_layer_indices'],[1,3])
                self.assertEqual([l.self_attn.layer_idx for l in model.model.layers],[0,1])
            if 'ffn_width' in kwargs:
                for original_id,channels in selected['kept_original_ffn_channel_indices'].items():
                    self.assertEqual(channels,list(range(32,48)))
                    compact_id=selected.get('kept_original_layer_indices',list(range(4))).index(int(original_id))
                    self.assertTrue(torch.equal(model.model.layers[compact_id].mlp.down_proj.weight,original.model.layers[int(original_id)].mlp.down_proj.weight[:,channels]))
            for name,p in model.named_parameters():
                if name.startswith(('audio_tower.','visual.')):self.assertTrue(torch.equal(p,original.state_dict()[name]))
            with torch.no_grad():before=model(input_ids=ids,use_cache=False).logits;tokens=model.generate(ids,max_new_tokens=2,do_sample=False,use_cache=True)
            with tempfile.TemporaryDirectory() as path:
                model.save_pretrained(path,safe_serialization=True)
                loaded=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(path,attn_implementation='sdpa').eval()
                with torch.no_grad():after=loaded(input_ids=ids,use_cache=False).logits;actual=loaded.generate(ids,max_new_tokens=2,do_sample=False,use_cache=True)
                torch.testing.assert_close(before,after,rtol=0,atol=0);self.assertTrue(torch.equal(tokens,actual))
                self.assertEqual(parameter_counts(model),parameter_counts(loaded))

    def test_bad_hybrid_scores_rejected_before_mutation(self):
        model=tiny_model();layer=model.model.layers[0]
        bad={'layer_bi':[.1,.8,.2,.9],'ffn_activation':{str(i):[1.]*48 for i in range(4)}}
        bad['ffn_activation']['3']=[1.]*47
        with self.assertRaises(ValueError):apply_pruning(model,'depth_ffn_activation',keep_layers=2,ffn_width=16,importance=bad)
        self.assertIs(model.model.layers[0],layer);self.assertEqual(len(model.model.layers),4)

    def test_calibration_cache_reuse_and_stale_provenance(self):
        model=tiny_model()
        class Runner:
            def __init__(self):self.model=model;self.calls=0
            def prepare(self,r):return {'input_ids':torch.tensor([[1,4,7,8]]),'attention_mask':torch.ones(1,4,dtype=torch.long)}
            def score(self,b):
                self.calls+=1
                with torch.no_grad():self.model(**b,use_cache=False)
        runner=Runner();spec={'examples':2,'split':'train','test_set_used':False};records=[{'id':'a'},{'id':'b'}]
        with tempfile.TemporaryDirectory() as path,patch('smarthome_efficiency.pruning_calibration.calibration_spec',return_value=spec):
            cache=Path(path)/'scores.json'
            first,p1=obtain_calibration(runner,{},Path(path),records,42,cache)
            second,p2=obtain_calibration(runner,{},Path(path),records,42,cache)
            self.assertEqual(first,second);self.assertEqual(runner.calls,2);self.assertFalse(p1['reused']);self.assertTrue(p2['reused']);self.assertEqual(p1['sha256'],p2['sha256'])
            payload=json.loads(cache.read_text());changed=spec|{'examples':3}
            with self.assertRaises(ValueError):validate_calibration(payload,changed,model)
            payload['scores']['ffn_activation']['0'][0]=float('nan')
            with self.assertRaises(ValueError):validate_calibration(payload,spec,model)

    def test_legacy_queue_allocations_are_disjoint_until_it_finishes(self):
        external={'status':'running','jobs':[{'status':'running','gpu':6},{'status':'completed','gpu':7}]}
        self.assertEqual(coexist_allocation(list(range(1,8)),external),([7],{6}))
        self.assertEqual(coexist_allocation([1,2,3],external),([],{6}))
        external['status']='completed'
        self.assertEqual(coexist_allocation(list(range(1,8)),external),(list(range(1,8)),{6}))

    def test_nine_job_configs_and_reused_teacher_cache_dependencies(self):
        from smarthome_efficiency.training import load_config
        with tempfile.TemporaryDirectory() as folder:
            sources={}
            for task in ['mer6','mintrec2','mintrec20']:
                source=Path(folder)/task;(source/'adapter').mkdir(parents=True)
                (source/'metadata.json').write_text(json.dumps({'config':load_config(ROOT/'configs'/f'{task}.yaml')}))
                (source/'adapter/adapter_config.json').write_text('{}');(source/'adapter/adapter_model.safetensors').touch()
                sources[task]=source
            with patch('launch_pruning_sweep.SOURCES',sources):
                plans=fresh_plans(Path('/unused/runs'),Path('/unused/compact'),['mer6','mintrec2','mintrec20'],methods=CALIBRATED_METHODS)
        self.assertEqual(len(plans),9)
        for task in ['mer6','mintrec2','mintrec20']:
            group=[p for p in plans if p['task']==task]
            self.assertEqual(len({p['pruning_settings']['calibration_cache'] for p in group}),1)
            self.assertTrue(all(p['pruning_settings']['calibration_samples']==128 for p in group))
            self.assertTrue(all(p['config']['epochs']==5 and p['config']['lora_target']=='decoder_linear' for p in group))
        jobs=build_jobs({'jobs':plans},Path('/unused/followup'),teacher_cache_root=Path('/existing/teachers'))
        self.assertEqual(len(jobs),30)
        self.assertEqual(len([j for j in jobs if j['kind']=='teacher']),3)
        self.assertTrue(all(j['config']['teacher_cache'].startswith('/existing/teachers/') for j in jobs if j['kind']=='kd_train'))
        hybrid=next(p for p in plans if p['method']=='depth_ffn_activation')
        self.assertEqual(hybrid['pruning_settings']['keep_layers'],28)
        self.assertEqual(hybrid['pruning_settings']['ffn_width'],6144)


if __name__=='__main__':unittest.main()
