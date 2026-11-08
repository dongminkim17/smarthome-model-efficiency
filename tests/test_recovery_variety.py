"""Scheduled-KD resume, exact LoRA averaging and the shared-method batch contracts."""
import copy,json,tempfile,unittest
from pathlib import Path
import sys
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.adapter_averaging import average_factors,averaging_windows
from smarthome_efficiency.training import DEFAULTS,distillation_weight,load_config,validate_resume
from queue_mer_recovery_variety import definitions


class RecoveryVarietyTests(unittest.TestCase):
    def test_kd_schedule_holds_decays_and_is_identical_after_resume(self):
        old=DEFAULTS|{'distillation_alpha':.75}
        self.assertEqual([distillation_weight(old,s,101) for s in (0,20,50,100)],[.75]*4)
        cfg=old|{'distillation_alpha_final':.25,'distillation_decay_start':.2}
        self.assertAlmostEqual(distillation_weight(cfg,20,101),.75)
        self.assertAlmostEqual(distillation_weight(cfg,60,101),.5)
        self.assertAlmostEqual(distillation_weight(cfg,100,101),.25)
        self.assertAlmostEqual(distillation_weight(cfg,150,101),.25)
        reloaded=json.loads(json.dumps(cfg))
        self.assertEqual([distillation_weight(cfg,s,101) for s in range(41,101)],
                         [distillation_weight(reloaded,s,101) for s in range(41,101)])
        with self.assertRaises(ValueError):validate_resume(old,cfg,1,1)
        validate_resume(cfg,reloaded,1,1)

    def test_config_schedule_validation_preserves_legacy(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'c.yaml';base={'model':'fixture','task':'mer6','objective':'sft_kd',
                                           'teacher_checkpoint':'fixture','teacher_cache':'fixture'}
            path.write_text(yaml.safe_dump(base));self.assertNotIn('distillation_alpha_final',load_config(path))
            path.write_text(yaml.safe_dump(base|{'distillation_alpha_final':0.}));cfg=load_config(path)
            self.assertEqual(cfg['distillation_decay_start'],.2)
            for bad in ({'distillation_alpha_final':-1},{'distillation_alpha_final':True},
                        {'distillation_alpha_final':.2,'distillation_decay_start':1.},
                        {'distillation_alpha_final':.2,'objective':'sft'}):
                path.write_text(yaml.safe_dump(base|bad))
                with self.assertRaises(ValueError):load_config(path)

    def test_effective_update_mean_is_exact_and_avoids_factor_cross_terms(self):
        torch.manual_seed(4)
        cfg={'r':2,'lora_alpha':4,'bias':'none'}
        a=torch.randn(2,5);b=torch.randn(7,2)
        states=[{'x.lora_A.weight':a,'x.lora_B.weight':b},
                {'x.lora_A.weight':-a,'x.lora_B.weight':-b}]
        combined,c=average_factors(states,[cfg,cfg])
        expected=2*b@a
        actual=c['lora_alpha']/c['r']*combined['x.lora_B.weight']@combined['x.lora_A.weight']
        torch.testing.assert_close(actual,expected)
        self.assertEqual((c['r'],c['lora_alpha']),(4,8))
        self.assertGreater(expected.norm().item(),0)
        self.assertEqual(((states[0]['x.lora_B.weight']+states[1]['x.lora_B.weight'])/2).norm().item(),0)
        three=states+[{'x.lora_A.weight':torch.randn(2,5),'x.lora_B.weight':torch.randn(7,2)}]
        combined,c=average_factors(three,[cfg]*3,[.2,.3,.5])
        expected=sum(w*2*s['x.lora_B.weight']@s['x.lora_A.weight'] for w,s in zip([.2,.3,.5],three))
        torch.testing.assert_close(c['lora_alpha']/c['r']*combined['x.lora_B.weight']@combined['x.lora_A.weight'],expected)

    def test_incompatible_adapters_fail_closed(self):
        state={'x.lora_A.weight':torch.ones(2,3),'x.lora_B.weight':torch.ones(4,2)}
        cfg={'r':2,'lora_alpha':4}
        for other in (cfg|{'r':3},cfg|{'use_dora':True},cfg|{'rank_pattern':{'x':1}}):
            with self.assertRaises(ValueError):average_factors([state,state],[cfg,other])
        with self.assertRaises(ValueError):average_factors([state,state],[cfg,cfg],[.5,.8])
        bad=copy.deepcopy(state);bad['x.lora_B.weight'][0,0]=float('nan')
        with self.assertRaises(ValueError):average_factors([state,bad],[cfg,cfg])

    def test_windows_exclude_untrained_baseline_and_are_predeclared(self):
        self.assertEqual(averaging_windows([0,5,10,15,20,25],15),{'near_best':[10,15,20],'last_three':[15,20,25]})
        self.assertEqual(averaging_windows([0,5,10,15],0),{'near_best':[5,10,15]})
        with self.assertRaises(ValueError):averaging_windows([0,5,10],5)

    def test_real_peft_average_merges_into_same_dense_parameter_count(self):
        from peft import LoraConfig,get_peft_model,PeftModel
        from safetensors.torch import load_file,save_file
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.proj=torch.nn.Linear(5,7,bias=False)
            def forward(self,x):return self.proj(x)
        torch.manual_seed(5);base=Tiny();weights=copy.deepcopy(base.state_dict());count=sum(p.numel() for p in base.parameters())
        states=[];configs=[];input=torch.randn(4,5);predictions=[]
        with tempfile.TemporaryDirectory() as folder:
            for i in range(3):
                model=Tiny();model.load_state_dict(weights)
                model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=['proj']))
                for n,p in model.named_parameters():
                    if 'lora_' in n:p.data.normal_()
                model.eval();predictions.append(model(input).detach())
                target=Path(folder)/str(i);model.save_pretrained(target)
                states.append(load_file(str(target/'adapter_model.safetensors')));configs.append(json.loads((target/'adapter_config.json').read_text()))
                checkpoint=target/'checkpoint';(checkpoint/'adapter').mkdir(parents=True)
                import shutil
                shutil.copy2(target/'adapter_model.safetensors',checkpoint/'adapter/adapter_model.safetensors')
                shutil.copy2(target/'adapter_config.json',checkpoint/'adapter/adapter_config.json')
                (checkpoint/'metadata.json').write_text(json.dumps({'config':{'model':'same','lora_rank':2,'lora_alpha':4},
                    'global_step':i+1,'data_fingerprint':'same','base_model_manifest_sha256':None}))
            from select_averaged_checkpoint import make_average
            checkpoint=make_average([Path(folder)/str(i)/'checkpoint' for i in range(3)],Path(folder)/'checkpoint_mean')
            self.assertEqual(json.loads((checkpoint/'metadata.json').read_text())['config']['lora_rank'],6)
            self.assertFalse((checkpoint/'training_state.pt').exists())
            self.assertEqual(checkpoint,make_average([Path(folder)/str(i)/'checkpoint' for i in range(3)],checkpoint))
            factors,config=average_factors(states,configs)
            avg=Path(folder)/'average';avg.mkdir();save_file(factors,str(avg/'adapter_model.safetensors'))
            (avg/'adapter_config.json').write_text(json.dumps(config))
            restored=Tiny();restored.load_state_dict(weights)
            model=PeftModel.from_pretrained(restored,avg).eval()
            torch.testing.assert_close(model(input),sum(predictions)/3,rtol=1e-5,atol=1e-5)
            merged=model.merge_and_unload()
            self.assertEqual(sum(p.numel() for p in merged.parameters()),count)
            torch.testing.assert_close(merged(input),sum(predictions)/3,rtol=1e-5,atol=1e-5)


    def test_shortlist_accepts_full_dev_postprocessing_candidates(self):
        from unittest.mock import patch
        import test_mer_focus_queue as fixture
        from validation_shortlist import select_validation_shortlist
        with tempfile.TemporaryDirectory() as folder:
            state,job=fixture.MerFocusContracts().make_state(folder,[.70,.74,.72])
            for entry in state['jobs']:
                if entry['kind']=='recovery_train':entry['kind']='checkpoint_average'
            with patch('launch_pruning_followup.pin_best',side_effect=lambda x:str(Path(x)/'checkpoints/best')), \
                 patch('launch_pruning_followup.checkpoint_identity',return_value={'metadata.json':'pinned'}):
                result=select_validation_shortlist(state,job)
            self.assertEqual(result['selected'],['train1','train2'])
            self.assertFalse(result['test_results_used'])
            self.assertEqual(state['jobs'][1]['status'],'cancelled')

    def test_32_runs_change_recovery_details_only(self):
        arms=definitions();self.assertEqual(len(arms),32)
        self.assertEqual(sum(a[2] for a in arms),8)
        self.assertEqual(sum(a[3]['epochs'] for a in arms),144)
        self.assertEqual(len({a[0] for a in arms}),32)
        self.assertEqual({a[1] for a in arms},{16,64})
        forbidden={'model','mer_partition','split_seed','max_frames','max_pixels','use_transcript'}
        self.assertTrue(all(not(forbidden&set(a[3])) for a in arms))
        self.assertTrue(all(a[3]['epochs']==(3 if a[2] else 5) for a in arms))
        self.assertTrue(all(not a[2] for a in arms if a[3].get('lora_dropout') or a[3].get('lora_target')))


if __name__=='__main__':unittest.main()
