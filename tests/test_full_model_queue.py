"""Full-model queue covers all budgets/tasks and never changes previous experiments."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from queue_full_model_2b import build_submission,FAMILY,TASKS
from project_queue_jobs import entry,native_command,validate_export,pin_evaluation
from smarthome_efficiency.multimodal_pruning import BUDGETS,expected_counts
from smarthome_efficiency.queueing import atomic_json,read_json,validate_graph
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.training import DEFAULTS


class FullModelQueueTests(unittest.TestCase):
    def fixture(self,root):
        jobs=[]
        for task in TASKS:
            base=root/task/'base';source=root/task/'source';cache=root/task/'teacher_cache.json'
            atomic_json(base/'compression.json',{'method':'ffn_activation','ffn_width':4352,
                'verification':{'passed':True},'parameters_after':{'language':1924902912,'total':3231263744}})
            (source/'adapter').mkdir(parents=True)
            (source/'adapter/adapter_model.safetensors').write_bytes(b'fixture')
            cfg=dict(DEFAULTS,task=task,model=str(base),lora_target='decoder_linear',
                     objective='sft_kd',distillation_alpha=.75,teacher_checkpoint=str(root/task/'teacher'),teacher_cache=str(cache))
            atomic_json(source/'metadata.json',{'config':cfg,'base_model_manifest_sha256':sha256_file(base/'compression.json')})
            atomic_json(source/'report.json',{'status':'completed','test_set_used':False,'config':cfg})
            atomic_json(cache,{'status':'completed','spec':{'split':'train','test_set_used':False,'teacher_checkpoint':cfg['teacher_checkpoint']}})
            jobs.extend([
                entry('recovery_controls_20261019/train:KD075_'+task,'recovery_train',{'task':task,'status':'completed','run_dir':str(source)}),
                entry('calibrated_pruning_20261019/teacher:'+task,'followup',{'task':task,'status':'completed'}),
                entry('prepare:'+task,'prepare',{'task':task,'status':'completed'},resource='cpu')])
        return {'jobs':jobs,'queue_root':str(root/'queue'),'documents':{}}

    def build(self,state,root):
        with patch('queue_full_model_2b.pin_best',side_effect=lambda run:str(run)):
            return build_submission(state,root)

    def test_exact_nine_runs_independent_gates_and_immutable_old_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);before=copy.deepcopy(state)
            spec=self.build(state,root);self.assertEqual(state,before)
            validate_graph(state['jobs']+spec['jobs'])
            self.assertEqual(len(spec['jobs']),45)
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train']
            self.assertEqual(len(trains),9)
            for task in TASKS:self.assertEqual({j['payload']['arm'] for j in trains if j['task']==task},set(BUDGETS))
            by_id={j['id']:j for j in spec['jobs']}
            for job in trains:
                p=job['payload'];cfg=p['config']
                self.assertEqual(cfg['epochs'],5);self.assertEqual(cfg['lora_target'],'decoder_encoder_ffn')
                self.assertEqual(cfg['distillation_alpha'],.75);self.assertEqual(cfg['distillation_temperature'],2.)
                self.assertEqual(cfg['objective'],'sft_kd');self.assertTrue(cfg['gradient_checkpointing'])
                self.assertEqual(cfg['micro_batch_size']*cfg['gradient_accumulation_steps'],4)
                self.assertIsNone(p['init_adapter'])
                reload=by_id[job['deps'][0]];smoke=by_id[reload['deps'][0]]
                self.assertEqual(smoke['kind'],'smoke_train');self.assertIn('prepare:'+job['task'],smoke['deps'])
                test=by_id[FAMILY+'/test:'+p['name']]
                self.assertEqual(test['deps'],[job['id']]);self.assertEqual(test['payload']['split'],'test')
                self.assertEqual(test['payload']['modes'],['logit','greedy']);self.assertIsNone(test['payload']['limit'])

    def test_shared_fresh_calibration_serializes_only_task_exports_and_pins_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);spec=self.build(state,root)
            for task in TASKS:
                exports=[j for j in spec['jobs'] if j['task']==task and j['kind']=='prune_export']
                self.assertEqual(len({j['payload']['request']['calibration_cache'] for j in exports}),1)
                self.assertEqual(len({j['payload']['request']['source_checkpoint'] for j in exports}),1)
                for i,job in enumerate(exports):
                    self.assertEqual(exports[0]['id'] in job['deps'],i>0)
                    command,_=native_command(state,job,root)
                    self.assertEqual(command[command.index('--budget-scope')+1],'total')
                    p=job['payload'];request=p['request']
                    self.assertEqual(p['expected_parameters'],expected_counts(request['ffn_width'],request['audio_ffn_width'],request['vision_ffn_width']))
                    self.assertLess(p['expected_parameters']['total'],2000000000)
            changed=copy.deepcopy(state)
            changed['jobs'][0]['status']='running'
            with self.assertRaises(ValueError):self.build(changed,root)

    def test_full_export_checks_counts_encoder_shapes_and_protected_interfaces(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);spec=self.build(state,root)
            for job in [j for j in spec['jobs'] if j['kind']=='prune_export']:
                p=job['payload'];r=p['request'];model=Path(p['output'])
                atomic_json(r['calibration_cache'],{'fixture':'fresh'})
                calibration_spec={k:r[k] for k in ('source_checkpoint','source_metadata_sha256','source_adapter_sha256','calibration_seed')}
                calibration_spec.update(examples=128,split='train',test_set_used=False,scope='all_thinker_components')
                report=dict(r,parameters_after=p['expected_parameters'],test_set_used=False,verification_split='train',recovery_updates=0,
                    encoder_tensors_unchanged=False,non_ffn_tensors_unchanged=True,av_interfaces_unchanged=True,all_layer_counts_unchanged=True,
                    calibration={'sha256':sha256_file(Path(r['calibration_cache'])),'spec':calibration_spec},
                    verification={'passed':True,'save_reload_class_logits_match':True,'save_reload_generation_match':True})
                config={'text_config':{'intermediate_size':r['ffn_width'],'num_hidden_layers':36},
                        'audio_config':{'encoder_ffn_dim':r['audio_ffn_width'],'encoder_layers':32},
                        'vision_config':{'intermediate_size':r['vision_ffn_width'],'depth':32}}
                atomic_json(model/'config.json',config);atomic_json(model/'model.safetensors.index.json',{})
                atomic_json(model/'compression.json',report)
                self.assertEqual(validate_export(p)['total_parameters'],p['expected_parameters']['total'])
                for changes in [dict(non_ffn_tensors_unchanged=False),dict(av_interfaces_unchanged=False),
                                dict(parameters_after=dict(p['expected_parameters'],total=2000000000))]:
                    atomic_json(model/'compression.json',report|changes)
                    with self.assertRaises(ValueError):validate_export(p)
                atomic_json(model/'compression.json',report)
                config['audio_config']['encoder_layers']=31;atomic_json(model/'config.json',config)
                with self.assertRaises(ValueError):validate_export(p)


    def test_smoke_reload_uses_updated_adapter_even_if_validation_best_is_step_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);cfg={'lora_target':'decoder_encoder_ffn'}
            latest=run/'checkpoints/latest';(latest/'adapter').mkdir(parents=True)
            atomic_json(latest/'metadata.json',{'global_step':2,'config':cfg})
            (latest/'adapter/adapter_model.safetensors').write_bytes(b'trained fixture')
            atomic_json(run/'report.json',{'status':'paused','test_set_used':False,'global_step':2,'best_step':0,'config':cfg})
            parent={'id':'smoke_parent','payload':{'run_dir':str(run),'config':cfg}}
            job={'kind':'smoke_test','deps':['smoke_parent'],'payload':{}}
            with patch('project_queue_jobs.pin_best',side_effect=AssertionError('Must not reload zero-update validation winner')):
                pin_evaluation({'jobs':[parent]},job)
            self.assertEqual(job['payload']['checkpoint'],str(latest.resolve()))
            self.assertEqual(job['payload']['checkpoint_selection'],'latest_two_step_smoke_adapter')
            atomic_json(latest/'metadata.json',{'global_step':0,'config':cfg})
            with self.assertRaises(ValueError):pin_evaluation({'jobs':[parent]},job)


if __name__=='__main__':unittest.main()
