"""Matched joint compression: original-source provenance, budget checks and 10-epoch execution."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.pruning import sha256_file,processor_hashes
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.joint_pruning import validate_joint_budget
from smarthome_efficiency.queueing import atomic_json,read_json,validate_graph
from smarthome_efficiency.training import DEFAULTS,load_config
from project_queue_jobs import entry,native_command,validate_export
from project_queue import Scheduler
from queue_joint_pruning import build_submission,FAMILY,TASKS,WIDTHS
from test_project_queue import state_for


class JointPruningTests(unittest.TestCase):
    def fixture(self,root):
        jobs=[]
        for task in TASKS:
            base=root/task/'original';teacher=root/task/'teacher';compact=root/task/'two_stage_balanced';run=root/task/'reference_run'
            config={'text_config':{'intermediate_size':11008,'num_hidden_layers':36},
                'audio_config':{'encoder_ffn_dim':5120,'encoder_layers':32},'vision_config':{'intermediate_size':3420,'depth':32}}
            # The official release nests the Thinker inside the full Omni config.
            atomic_json(base/'config.json',{'thinker_config':config});atomic_json(base/'tokenizer.json',{'fixture':True})
            (teacher/'adapter').mkdir(parents=True);(teacher/'adapter/adapter_model.safetensors').write_bytes(b'fixture')
            tc=dict(DEFAULTS,task=task,model=str(base),use_transcript=task!='mer6',lora_target='qv')
            atomic_json(teacher/'metadata.json',{'config':tc})
            cache=root/task/'class_cache.json';features=root/task/'features'
            spec={'teacher_checkpoint':str(teacher),'split':'train','test_set_used':False,
                'teacher_metadata_sha256':sha256_file(teacher/'metadata.json'),
                'teacher_adapter_sha256':sha256_file(teacher/'adapter/adapter_model.safetensors'),
                'teacher_base_config_sha256':sha256_file(base/'config.json')}
            atomic_json(cache,{'status':'completed','spec':spec});atomic_json(features/'cache.json',{'status':'completed','spec':{'teacher':spec}})
            cfg=dict(tc,model=str(compact),objective='sft_kd',teacher_checkpoint=str(teacher),teacher_cache=str(cache),
                feature_teacher_cache=str(features),feature_audio_weight=.1,feature_vision_weight=.1,
                distillation_alpha=.75,distillation_temperature=2.,lora_target='decoder_encoder_ffn',
                encoder_learning_rate=1e-5,micro_batch_size=2,gradient_accumulation_steps=2)
            atomic_json(compact/'compression.json',{'parameters_after':expected_counts(*WIDTHS),'verification':{'passed':True},'processor_files_sha256':processor_hashes(base)})
            atomic_json(run/'report.json',{'status':'completed','config':cfg,'test_set_used':False,'total_elapsed_seconds':1000})
            jobs.extend([
                entry('encoder_budget_20261021/train:'+task+'_balanced_feature_splitlr','recovery_train',{'task':task,'status':'completed','config':cfg,'run_dir':str(run)}),
                entry('encoder_budget_20261021/test:'+task+'_balanced_feature_splitlr','recovery_test',{'task':task,'status':'completed'}),
                entry('calibrated_pruning_20261019/teacher:'+task,'followup',{'task':task,'status':'completed'}),
                entry('full_model_followups_20261020/encoder_cache:'+task,'encoder_cache',{'task':task,'status':'completed'}),
                entry('prepare:'+task,'prepare',{'task':task,'status':'completed'},resource='cpu')])
            jobs[-4]['result']={'report':str(root/task/'reference_test/report.json')}
        return {'jobs':jobs,'documents':{},'queue_root':str(root/'queue')}

    def test_exact_six_run_matrix_is_independent_and_configuration_matched(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);before=copy.deepcopy(state);spec=build_submission(state,root)
            self.assertEqual(state,before);validate_graph(state['jobs']+spec['jobs']);self.assertEqual(len(spec['jobs']),27)
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train'];exports=[j for j in spec['jobs'] if j['kind']=='prune_export']
            self.assertEqual(len(trains),6);self.assertEqual(len(exports),3);by={j['id']:j for j in spec['jobs']}
            for task in TASKS:
                pair=[j for j in trains if j['task']==task];self.assertEqual({j['payload']['config']['epochs'] for j in pair},{5,10})
                a,b=[j['payload']['config'] for j in pair]
                self.assertEqual({k for k in a if a[k]!=b[k]},{'epochs'})
            for j in trains:
                p=j['payload'];cfg=p['config'];self.assertEqual(p['training_epochs'],cfg['epochs'])
                self.assertIsNone(p['init_adapter']);self.assertIsNone(p['parent_checkpoint'])
                self.assertEqual(cfg['teacher_checkpoint'],p['source_checkpoint'])
                self.assertEqual(cfg['micro_batch_size']*cfg['gradient_accumulation_steps'],4)
                self.assertEqual(p['expected_parameters'],expected_counts(*WIDTHS))
                reload=by[j['deps'][0]];smoke=by[reload['deps'][0]]
                self.assertEqual(smoke['payload']['config']['epochs'],2);self.assertEqual(smoke['payload']['training_epochs'],2)
                self.assertEqual(by[FAMILY+'/test:'+p['name']]['deps'],[j['id']])
                path=root/'cfg.yaml';path.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(path),cfg)
                # Full ten-epoch execution must not inherit the smoke's two-step cap.
                (root/'attempt').mkdir(exist_ok=True);j['attempts']=[{'path':str(root/'attempt')}]
                command,actual=native_command(state,j,root/'attempt');self.assertNotIn('--max-steps',command)
                self.assertNotIn('--init-adapter',command);self.assertEqual(actual['epochs'],cfg['epochs'])
            for j in exports:
                p=j['payload'];self.assertTrue(p['generate_calibration']);self.assertIsNone(p['calibration_sha256'])
                command,_=native_command(state,j,root);self.assertIn('separate',command);self.assertIn('multimodal_ffn_activation',command)
                self.assertEqual(p['request']['calibration_samples'],128)

    def test_joint_caps_require_original_unpruned_source_and_both_reductions(self):
        before=expected_counts(11008,5120,3420);after=expected_counts(*WIDTHS);validate_joint_budget(before,after)
        self.assertGreater(after['total'],2e9)
        for original,pruned in [(expected_counts(4352,5120,3420),after),
                (before,after|{'language':2_000_000_001}),
                (before,after|{'audio_encoder':1_000_000_001}),
                (before,after|{'other':1})]:
            with self.assertRaises(ValueError):validate_joint_budget(original,pruned)

    def test_source_snapshot_and_processor_are_pinned(self):
        for field in ('config','processor','compact'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);state=self.fixture(root);spec=build_submission(state,root)
                job=next(j for j in spec['jobs'] if j['kind']=='prune_export');base=Path(job['payload']['source_base_snapshot'])
                if field=='config':atomic_json(base/'config.json',{'changed':True})
                elif field=='processor':atomic_json(base/'tokenizer.json',{'changed':True})
                else:atomic_json(base/'compression.json',{'not_original':True})
                with self.assertRaises(ValueError):native_command(state,job,root)
                with self.assertRaises((ValueError,KeyError)):build_submission(state,root)

    def test_export_validation_enforces_joint_provenance_and_native_architecture(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);spec=build_submission(state,root);p=next(j['payload'] for j in spec['jobs'] if j['kind']=='prune_export');r=p['request'];out=Path(p['output'])
            atomic_json(r['calibration_cache'],{'fixture':True})
            cal={k:r[k] for k in ('source_checkpoint','source_metadata_sha256','source_adapter_sha256','calibration_seed')}
            cal.update(examples=128,split='train',test_set_used=False,scope='all_thinker_components')
            report=dict(r,parameters_before=expected_counts(11008,5120,3420),parameters_after=expected_counts(*WIDTHS),
                parameter_limits={'language':2_000_000_000,'combined_encoders':1_000_000_000},combined_encoder_parameters=990823680,
                joint_language_encoder_pruning=True,language_tensors_unchanged=False,non_ffn_tensors_unchanged=True,
                av_interfaces_unchanged=True,all_layer_counts_unchanged=True,encoder_tensors_unchanged=False,
                verification={'passed':True,'save_reload_class_logits_match':True,'save_reload_generation_match':True},
                recovery_updates=0,verification_split='train',test_set_used=False,
                calibration={'sha256':sha256_file(Path(r['calibration_cache'])),'spec':cal})
            atomic_json(out/'compression.json',report);atomic_json(out/'model.safetensors.index.json',{})
            atomic_json(out/'config.json',{'text_config':{'intermediate_size':4352,'num_hidden_layers':36},
                'audio_config':{'encoder_ffn_dim':3328,'encoder_layers':32},'vision_config':{'intermediate_size':2048,'depth':32}})
            self.assertEqual(validate_export(p)['total_parameters'],2915726592)
            for change in ({'joint_language_encoder_pruning':False},{'parameters_before':expected_counts(4352,5120,3420)},
                    {'non_ffn_tensors_unchanged':False},{'test_set_used':True}):
                atomic_json(out/'compression.json',report|change)
                with self.assertRaises(ValueError):validate_export(p)

    def test_ten_epoch_remaining_time_is_not_zero_after_five_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            p={'task':'mer6','config':{'epochs':10},'estimated_seconds':1000}
            j=entry('joint','recovery_train',p);j.update(status='running',gpu=1,progress={'epoch':5})
            state_for([j],folder);scheduler=Scheduler(folder)
            with patch('project_queue.read',return_value={'jobs':[]}):result=scheduler.estimate()
            self.assertEqual(result['seconds'],500)

if __name__=='__main__':unittest.main()
