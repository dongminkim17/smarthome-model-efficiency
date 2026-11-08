"""Approved six-configuration expansion: exact coverage and isolated input provenance."""
import copy
from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import test_ffn_width_queue
from queue_ffn_width_experiments import build_submission as build_width
from queue_ffn_expansion import build_submission,FAMILY,TASKS,VARIANTS
from project_queue_jobs import native_command,validate_export
from smarthome_efficiency.queueing import read_json,atomic_json,validate_graph
from smarthome_efficiency.pruning import sha256_file


class ExpansionQueueTests(unittest.TestCase):
    def fixture(self,root):
        combined={'jobs':[],'documents':{},'queue_root':str(root/'queue')}
        for task in TASKS:
            state=test_ffn_width_queue.WidthQueueTests().fixture(root/task)
            for job in state['jobs']:
                job['id']=job['id'].replace('mintrec2',task)
                job['task']=task;job['payload']['task']=task
                if job['kind']=='sft':
                    job['payload']['config']['task']=task
                    path=Path(job['payload']['compact_model'])/'compression.json';report=read_json(path)
                    source=Path(report['source_checkpoint']);metadata=read_json(source/'metadata.json')
                    metadata['config']['task']=task;atomic_json(source/'metadata.json',metadata)
                    report.update(task=task,source_metadata_sha256=sha256_file(source/'metadata.json'))
                    atomic_json(path,report)
            if task=='mintrec2':
                wide=build_width(state,root)['jobs'][0];wide['status']='completed'
                combined['jobs'].append(wide)
            combined['jobs'].extend(state['jobs'])
        return combined

    def test_exact_six_variants_per_task_and_five_required_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);original=copy.deepcopy(state)
            spec=build_submission(state,root);self.assertEqual(state,original)
            validate_graph(state['jobs']+spec['jobs'])
            self.assertEqual(len(spec['jobs']),77)
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train']
            self.assertEqual(len(trains),18)
            for task in TASKS:
                self.assertEqual({j['payload']['arm'] for j in trains if j['task']==task},{v[0] for v in VARIANTS})
            exports=[j for j in spec['jobs'] if j['kind']=='prune_export']
            self.assertEqual(len(exports),5)
            self.assertEqual(sum(j['payload']['generate_calibration'] for j in exports),3)
            self.assertTrue(all(j['payload']['expected_language_parameters']==1981526016 for j in exports))

    def test_every_arm_has_matching_stress_reload_train_and_official_test_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec=build_submission(self.fixture(root),root);by_id={j['id']:j for j in spec['jobs']}
            for job in spec['jobs']:
                if job['kind']!='recovery_train':continue
                p=job['payload'];cfg=p['config'];self.assertEqual(cfg['epochs'],5)
                self.assertEqual(cfg['lora_rank'],16);self.assertEqual(cfg['lora_alpha'],32)
                self.assertEqual(cfg['micro_batch_size']*cfg['gradient_accumulation_steps'],4)
                self.assertIsNone(p['init_adapter'])
                reload=by_id[job['deps'][0]];probe=by_id[reload['deps'][0]]
                a=probe['payload']['config'];self.assertEqual(a['epochs'],2)
                self.assertEqual({k for k in cfg if cfg[k]!=a[k]},{'epochs','train_limit','dev_limit'})
                self.assertTrue(probe['payload']['memory_cases'].endswith(job['task']+'.json'))
                test=by_id[FAMILY+'/test:'+p['name']]
                self.assertEqual(test['deps'],[job['id']]);self.assertEqual(test['payload']['split'],'test')
                self.assertIsNone(test['payload']['limit']);self.assertEqual(test['payload']['modes'],['logit','greedy'])

    def test_variants_change_only_the_approved_scientific_variables(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);spec=build_submission(state,root)
            for job in spec['jobs']:
                if job['kind']!='recovery_train':continue
                p=job['payload'];cfg=p['config'];arm=p['arm']
                self.assertEqual(cfg['learning_rate'],3e-5 if 'LR3E5' in arm else 1e-4)
                self.assertEqual(cfg['lora_dropout'],.05 if 'DROP005' in arm else 0.)
                self.assertEqual(cfg['objective'],'sft_kd' if 'KD_T4' in arm else 'sft')
                self.assertEqual(cfg['distillation_temperature'],4. if 'KD_T4' in arm else 2.)
                self.assertEqual(cfg['distillation_alpha'],.5)
                self.assertEqual(cfg['seed'],42);self.assertEqual(cfg['split_seed'],42)
                if 'KD_T4' in arm:
                    self.assertIn('teacher:'+job['task'],','.join(next(j for j in spec['jobs'] if j['id']==FAMILY+'/smoke_train:'+p['name'])['deps']))
                if 'CAL512' in arm:self.assertTrue(cfg['model'].endswith('_n512'))

    def test_fresh_512_calibration_command_and_post_export_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);spec=build_submission(state,root)
            job=next(j for j in spec['jobs'] if j['kind']=='prune_export' and j['payload']['generate_calibration'])
            p=job['payload'];request=p['request']
            self.assertFalse(Path(request['calibration_cache']).exists())
            command,cfg=native_command(state,job,root)
            self.assertEqual(command[command.index('--calibration-samples')+1],'512')
            model=Path(p['output']);atomic_json(request['calibration_cache'],{'fixture':'new training-only cache'})
            calibration_spec={k:request[k] for k in ('source_checkpoint','source_metadata_sha256','source_adapter_sha256','calibration_seed')}
            calibration_spec.update(examples=512,split='train',test_set_used=False)
            report=dict(request,test_set_used=False,verification_split='train',encoder_tensors_unchanged=True,
                        recovery_updates=0,parameters_after={'language':1981526016},
                        calibration={'sha256':sha256_file(Path(request['calibration_cache'])),'spec':calibration_spec},
                        verification={'passed':True,'save_reload_class_logits_match':True,'save_reload_generation_match':True})
            atomic_json(model/'compression.json',report)
            atomic_json(model/'config.json',{'text_config':{'intermediate_size':4608,'num_hidden_layers':36}})
            atomic_json(model/'model.safetensors.index.json',{})
            self.assertEqual(validate_export(p)['language_parameters'],1981526016)
            for key,value in [('examples',128),('split','test'),('test_set_used',True)]:
                bad=copy.deepcopy(report);bad['calibration']['spec'][key]=value
                atomic_json(model/'compression.json',bad)
                with self.assertRaises(ValueError):validate_export(p)
