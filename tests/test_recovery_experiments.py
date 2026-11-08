"""Recovery controls, validation-only selection, initialization and queue gates."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from recovery_experiments import TASKS, ARMS, queued_jobs, select_sources, materialize_jobs
from launch_recovery_sweep import prerequisite_state, refresh_tests, build_command, validate_result
from smarthome_efficiency.training import DEFAULTS, config_fingerprint, load_config


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class RecoveryExperimentTests(unittest.TestCase):
    def fixture(self, root):
        jobs=[]
        caches=root/'teacher_cache'
        for task in TASKS:
            teacher=root/'teacher'/task
            cfg=dict(DEFAULTS,task=task,model=str(root/'pretrained'),lora_target='qv')
            write_json(teacher/'metadata.json',{'config':cfg})
            write_json(caches/task/'cache.json',{'status':'completed','spec':{'split':'train','test_set_used':False,'teacher_checkpoint':str(teacher)}})
            for method,score in [('depth_uniform',.71),('ffn_magnitude',.81)]:
                name=f'{task}_{method}';run=root/'sft'/name;compact=root/'compact'/name
                config=dict(cfg,model=str(compact),lora_target='decoder_linear',micro_batch_size=2,gradient_accumulation_steps=2)
                checkpoint=run/'checkpoints/step-00000002'
                write_json(checkpoint/'metadata.json',{'config':config,'global_step':2})
                (checkpoint/'adapter').mkdir();(checkpoint/'adapter/adapter_model.safetensors').write_bytes(b'fixture')
                (checkpoint/'adapter/adapter_config.json').write_text('{}')
                (run/'checkpoints/best').symlink_to(checkpoint,target_is_directory=True)
                write_json(run/'report.json',{'status':'completed','config':config,'best_score':score,'best_step':2,'test_set_used':False})
                write_json(compact/'compression.json',{'verification':{'passed':True},'parameters_after':{'language':1_925_000_000}})
                # Contradictory test scores must never influence selection.
                write_json(run/'test/report.json',{'accuracy':1-score})
                jobs.append({'name':name,'task':task,'method':method,'status':'completed','config':config,
                             'output':str(run),'compact_model':str(compact),'source_checkpoint':str(teacher)})
        return {'status':'completed','jobs':jobs},caches

    def test_selection_uses_sft_validation_and_checks_completion_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=self.fixture(root)
            selected=select_sources([sweep])
            for task in TASKS:
                self.assertEqual(selected[task]['winner']['method'],'ffn_magnitude')
                self.assertFalse(selected[task]['test_results_used'])
            for report_change in [{'test_set_used':True},{'status':'running'},{'best_score':float('nan')}]:
                path=Path(sweep['jobs'][0]['output'])/'report.json';original=json.loads(path.read_text())
                write_json(path,original|report_change)
                with self.assertRaises(ValueError):select_sources([sweep])
                write_json(path,original)
            with self.assertRaises(ValueError):select_sources([sweep|{'status':'running'}])
            sidecar=Path(sweep['jobs'][0]['compact_model'])/'compression.json'
            write_json(sidecar,{'verification':{'passed':True},'parameters_after':{'language':2_000_000_000}})
            with self.assertRaises(ValueError):select_sources([sweep])

    def test_exact_twelve_runs_no_a5_and_controlled_arm_differences(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=self.fixture(root)
            selected=select_sources([sweep]);jobs=materialize_jobs(root/'output',selected,caches)
            trains={j['name']:j for j in jobs if j['kind']=='train'}
            self.assertEqual(len(trains),12);self.assertEqual(len(jobs),24)
            self.assertEqual({j['arm'] for j in jobs},set(ARMS))
            for task in TASKS:
                arms={a:trains[f'{a}_{task}'] for a in ARMS}
                a,b,c,d=[arms[a] for a in ARMS]
                for job in arms.values():
                    cfg=job['config'];self.assertEqual(cfg['epochs'],5)
                    self.assertEqual(cfg['micro_batch_size']*cfg['gradient_accumulation_steps'],4)
                    path=root/'config.yaml'
                    import yaml
                    path.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(path),cfg)
                self.assertEqual(a['config']['lora_rank'],64);self.assertEqual(a['config']['lora_alpha'],128)
                self.assertIsNone(a['init_adapter']);self.assertIsNone(b['init_adapter'])
                self.assertEqual(a['config']['model'],b['config']['model'])
                differences={k for k in a['config'] if a['config'][k]!=b['config'][k]}
                self.assertEqual(differences,{'objective','teacher_checkpoint','teacher_cache'})
                self.assertEqual(c['config']['lora_rank'],16);self.assertEqual(c['config']['learning_rate'],3e-5)
                self.assertEqual(c['init_adapter'],str(Path(selected[task]['winner']['checkpoint'])/'adapter'))
                self.assertEqual(d['config']['lora_rank'],64);self.assertEqual(d['config']['lora_target'],'qv')
                self.assertEqual(d['config']['model'],str(root/'pretrained'))
                self.assertIsNone(d['init_adapter']);self.assertEqual(d['config']['objective'],'sft')

    def test_smoke_has_separate_outputs_full_resolution_and_no_test_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=self.fixture(root);selected=select_sources([sweep])
            full=materialize_jobs(root/'full',selected,caches)
            smoke=materialize_jobs(root/'smoke',selected,caches,smoke=True)
            for a,b in zip(full,smoke):
                self.assertNotEqual(a['output'],b['output'])
                if a['kind']=='train':
                    self.assertEqual(b['config']['train_limit'],24);self.assertEqual(b['config']['dev_limit'],20)
                    for key in ['max_frames','max_pixels','lora_rank','lora_alpha','objective','model']:
                        self.assertEqual(a['config'][key],b['config'][key])
                else:
                    self.assertEqual(a['split'],'test');self.assertEqual(b['split'],'dev');self.assertEqual(b['limit'],6)

    def test_teacher_cache_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=self.fixture(root);selected=select_sources([sweep])
            write_json(caches/'mer6/cache.json',{'status':'completed','spec':{'split':'test','test_set_used':True,'teacher_checkpoint':'wrong'}})
            with self.assertRaises(ValueError):materialize_jobs(root/'output',selected,caches)

    def test_prerequisites_wait_for_absent_followup_and_fail_on_prior_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'workflow.json'
            self.assertEqual(prerequisite_state([path])[0]['status'],'not_started')
            write_json(path,{'status':'running'});self.assertEqual(len(prerequisite_state([path])),1)
            write_json(path,{'status':'completed'});self.assertEqual(prerequisite_state([path]),[])
            write_json(path,{'status':'failed'})
            with self.assertRaises(ValueError):prerequisite_state([path])

    def test_test_job_waits_for_finished_training_and_pins_best(self):
        jobs=queued_jobs(Path('/output'))
        refresh_tests(jobs,False)
        self.assertTrue(all(j['status']=='waiting' for j in jobs))
        jobs[0]['status']='failed';refresh_tests(jobs,False)
        self.assertEqual(jobs[1]['status'],'blocked')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=self.fixture(root);selected=select_sources([sweep])
            jobs=materialize_jobs(root/'out',selected,caches);jobs[0].update(status='completed',run_dir=selected['mer6']['winner']['sft_run'])
            refresh_tests(jobs,False)
            self.assertEqual(jobs[1]['status'],'ready')
            self.assertEqual(jobs[1]['checkpoint'],selected['mer6']['winner']['checkpoint'])

    def test_warm_start_and_resume_commands_do_not_mix_initialization_modes(self):
        job={'kind':'train','init_adapter':'/source/adapter'}
        command=build_command(job,Path('/out'),Path('/cfg'),False)
        self.assertIn('--init-adapter',command);self.assertNotIn('--resume',command)
        command=build_command(job,Path('/out'),Path('/cfg'),False,resume=Path('/latest'))
        self.assertIn('--resume',command);self.assertNotIn('--init-adapter',command)
        self.assertIn('--max-steps',build_command(job,Path('/out'),Path('/cfg'),True))
        test={'kind':'test','checkpoint':'/best','split':'dev'}
        command=build_command(test,Path('/out'),None,True)
        self.assertIn('--limit',command);self.assertEqual(command[command.index('--split')+1],'dev')

    def test_training_completion_requires_correct_initialization_and_epoch_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=dict(DEFAULTS);job={'kind':'train','run_dir':str(root),'config':cfg,
                'config_sha256':config_fingerprint(cfg),'init_adapter':'/parent/adapter'}
            report={'status':'completed','test_set_used':False,'config':cfg,'objective':'sft',
                    'global_step':10,'requested_steps':10,'best_score':.8,'best_step':8}
            write_json(root/'report.json',report);write_json(root/'provenance.json',{'init_adapter':'/parent/adapter'})
            self.assertEqual(validate_result(job,False)['global_step'],10)
            write_json(root/'provenance.json',{'init_adapter':None})
            with self.assertRaises(ValueError):validate_result(job,False)
            write_json(root/'provenance.json',{'init_adapter':'/parent/adapter'})
            write_json(root/'report.json',report|{'global_step':8})
            with self.assertRaises(ValueError):validate_result(job,False)

class RecoveryAdapterExecutionTests(unittest.TestCase):
    def test_rank64_and_warm_rank16_execute_native_forward_backward_and_reload(self):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, PeftModel
        from smarthome_efficiency.pruning import prune_layers
        from smarthome_efficiency.distillation import distillation_kl
        from test_pruning import tiny_model
        torch.set_num_threads(2)
        ids=torch.tensor([[1,5,7,9]])
        target=r'model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
        for arm in ARMS:
            base=tiny_model()
            if arm!='A4':prune_layers(base,[0,3])
            rank=16 if arm=='A3' else 64
            scope=r'model\.layers\.\d+\.self_attn\.(q_proj|v_proj)' if arm=='A4' else target
            model=get_peft_model(base,LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=0.,target_modules=scope,task_type=TaskType.CAUSAL_LM))
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
            for step in range(2):
                model.train();optimizer.zero_grad(set_to_none=True)
                output=model(input_ids=ids,labels=ids,use_cache=False)
                loss=output.loss
                if arm in ('A2','A3'):
                    loss=.5*loss+.5*distillation_kl(output.logits[:,-2,:2],torch.tensor([[2.,-1.]]),2.)
                self.assertTrue(torch.isfinite(loss).item());loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None))
                optimizer.step()
            self.assertTrue(any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if 'lora_B' in n))
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder)
                fresh=tiny_model()
                if arm!='A4':prune_layers(fresh,[0,3])
                reloaded=PeftModel.from_pretrained(fresh,folder,is_trainable=True)
                self.assertEqual(reloaded.peft_config['default'].r,rank)
                # A3 continues saved adapter weights with an entirely new optimizer.
                new_optimizer=torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad],lr=3e-5)
                self.assertEqual(len(new_optimizer.state),0)
                reloaded(input_ids=ids,labels=ids,use_cache=False).loss.backward()
                new_optimizer.step()

class RecoveryControlTests(unittest.TestCase):
    fixture = RecoveryExperimentTests.fixture

    def test_twelve_controls_match_the_existing_comparators(self):
        from recovery_experiments import CONTROL_ARMS
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=self.fixture(root);selected=select_sources([sweep])
            controls=materialize_jobs(root/'controls',selected,caches,arms=CONTROL_ARMS)
            reference=materialize_jobs(root/'reference',selected,caches)
            actual={j['name']:j for j in controls if j['kind']=='train'}
            baseline={j['name']:j for j in reference if j['kind']=='train'}
            self.assertEqual(len(actual),12);self.assertEqual(len(controls),24)
            self.assertEqual({j['arm'] for j in controls},set(CONTROL_ARMS))
            for task in TASKS:
                continued=actual[f'SFT_CONT_{task}'];a3=baseline[f'A3_{task}']
                self.assertEqual(continued['init_adapter'],a3['init_adapter'])
                self.assertEqual(continued['init_checkpoint_identity'],a3['init_checkpoint_identity'])
                self.assertEqual(continued['config']['objective'],'sft')
                self.assertEqual(continued['config']['learning_rate'],3e-5)
                differences={k for k in continued['config'] if continued['config'][k]!=a3['config'][k]}
                self.assertEqual(differences,{'objective','teacher_checkpoint','teacher_cache'})
                low,high=actual[f'KD025_{task}'],actual[f'KD075_{task}']
                self.assertIsNone(low['init_adapter']);self.assertIsNone(high['init_adapter'])
                self.assertEqual(low['config']['lora_rank'],16)
                self.assertEqual(low['config']['lora_alpha'],32)
                self.assertEqual(low['config']['distillation_alpha'],.25)
                self.assertEqual(high['config']['distillation_alpha'],.75)
                self.assertEqual(low['config']['learning_rate'],1e-4)
                self.assertEqual(low['config']['distillation_temperature'],2.)
                differences={k for k in low['config'] if low['config'][k]!=high['config'][k]}
                self.assertEqual(differences,{'distillation_alpha'})
                for j in (continued,low,high):
                    self.assertEqual(j['config']['epochs'],5)
                    self.assertEqual(j['source_name'],selected[task]['winner']['name'])

    def test_pinned_sources_are_reused_and_mutations_rejected(self):
        from launch_recovery_sweep import validate_pinned_sources
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=self.fixture(root);selected=select_sources([sweep])
            validate_pinned_sources(selected)
            changed=copy.deepcopy(selected);changed['mer6']['test_results_used']=True
            with self.assertRaises(ValueError):validate_pinned_sources(changed)
            path=Path(selected['mer6']['winner']['checkpoint'])/'adapter/adapter_model.safetensors'
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_pinned_sources(selected)

    def test_control_smokes_preserve_full_objectives_and_initializations(self):
        from recovery_experiments import CONTROL_ARMS
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=self.fixture(root);selected=select_sources([sweep])
            full=materialize_jobs(root/'full',selected,caches,arms=CONTROL_ARMS)
            smoke=materialize_jobs(root/'smoke',selected,caches,smoke=True,arms=CONTROL_ARMS)
            for a,b in zip(full,smoke):
                self.assertNotEqual(a['output'],b['output'])
                if a['kind']=='train':
                    self.assertEqual(a['init_adapter'],b['init_adapter'])
                    for key in ['objective','distillation_alpha','lora_rank','model','learning_rate']:
                        self.assertEqual(a['config'][key],b['config'][key])
                    self.assertEqual(b['config']['train_limit'],24)
                else:
                    self.assertEqual(b['split'],'dev');self.assertEqual(a['split'],'test')

    def test_lower_lr_rank64_changes_only_learning_rate_from_a1(self):
        from recovery_experiments import CONTROL_ARMS
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sweep,caches=self.fixture(root);selected=select_sources([sweep])
            controls=materialize_jobs(root/'controls',selected,caches,arms=CONTROL_ARMS)
            reference=materialize_jobs(root/'reference',selected,caches)
            for task in TASKS:
                low=next(j for j in controls if j['name']==f'R64_LOWLR_{task}' and j['kind']=='train')
                a1=next(j for j in reference if j['name']==f'A1_{task}' and j['kind']=='train')
                self.assertEqual(low['config']['learning_rate'],3e-5)
                self.assertEqual(low['config']['lora_rank'],64);self.assertEqual(low['config']['lora_alpha'],128)
                self.assertEqual(low['config']['epochs'],5);self.assertEqual(low['config']['objective'],'sft')
                self.assertIsNone(low['init_adapter']);self.assertIsNone(a1['init_adapter'])
                differences={k for k in low['config'] if low['config'][k]!=a1['config'][k]}
                self.assertEqual(differences,{'learning_rate'})
