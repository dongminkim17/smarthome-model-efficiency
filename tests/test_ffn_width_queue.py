"""Wider FFN experiments: pinned export, independent recovery and exact budget."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from queue_ffn_width_experiments import build_submission,FAMILY
from project_queue_jobs import entry,native_command,validate_export
from smarthome_efficiency.queueing import atomic_json,validate_graph
from smarthome_efficiency.training import DEFAULTS
from smarthome_efficiency.pruning import sha256_file


class WidthQueueTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'teacher';(source/'adapter').mkdir(parents=True)
        (source/'adapter/adapter_model.safetensors').write_bytes(b'fixture')
        cfg=dict(DEFAULTS,task='mintrec2',model=str(root/'reference'),lora_target='decoder_linear',
                 micro_batch_size=2,gradient_accumulation_steps=2,gradient_checkpointing=False)
        atomic_json(source/'metadata.json',{'config':cfg})
        calibration=root/'calibration.json';atomic_json(calibration,{'fixture':True})
        reference=root/'reference'
        atomic_json(reference/'compression.json',{'method':'ffn_activation','ffn_width':4352,'task':'mintrec2',
            'verification':{'passed':True},'parameters_after':{'language':1924902912},
            'source_checkpoint':str(source),'source_adapter_sha256':sha256_file(source/'adapter/adapter_model.safetensors'),
            'source_metadata_sha256':sha256_file(source/'metadata.json'),'calibration_samples':128,
            'calibration_seed':42,'calibration_cache':str(calibration)})
        cache=root/'cache';atomic_json(cache/'cache.json',{'status':'completed','spec':{'split':'train',
            'test_set_used':False,'teacher_checkpoint':str(source)}})
        jobs=[entry('calibrated_pruning_20261019/sft:mintrec2_ffn_activation','sft',
                    {'task':'mintrec2','compact_model':str(reference),'config':cfg}),
              entry('calibrated_pruning_20261019/teacher:mintrec2','followup',
                    {'task':'mintrec2','status':'completed','output':str(cache)}),
              entry('prepare:mintrec2','prepare',{'task':'mintrec2','status':'completed'},resource='cpu')]
        return {'jobs':jobs,'documents':{},'queue_root':str(root/'queue')}

    def test_two_fresh_runs_same_base_and_only_objective_differs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);original=copy.deepcopy(state)
            spec=build_submission(state,root)
            self.assertEqual(state,original);self.assertEqual(len(spec['jobs']),9)
            validate_graph(state['jobs']+spec['jobs'])
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train']
            self.assertEqual(len(trains),2)
            a,b=[j['payload'] for j in trains]
            self.assertEqual({k for k in a['config'] if a['config'][k]!=b['config'][k]},
                             {'objective','teacher_checkpoint','teacher_cache'})
            for j in trains:
                p=j['payload'];self.assertIsNone(p['init_adapter'])
                self.assertEqual(p['config']['epochs'],5);self.assertEqual(p['config']['lora_rank'],16)
                self.assertEqual(p['config']['micro_batch_size']*p['config']['gradient_accumulation_steps'],4)
                self.assertTrue(j['deps'][0].startswith(FAMILY+'/smoke_test:'))
            for j in spec['jobs']:
                if j['kind']=='smoke_train':
                    self.assertIn(FAMILY+'/export:mintrec2_ffn_activation_4608',j['deps'])
                    self.assertEqual(j['payload']['config']['epochs'],2)
                if j['kind']=='recovery_test':self.assertEqual(j['payload']['split'],'test')

    def test_export_command_reuses_exact_source_and_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);spec=build_submission(state,root)
            export=spec['jobs'][0];command,cfg=native_command(state,export,root)
            self.assertIn('--reuse',command)
            self.assertEqual(command[command.index('--ffn-width')+1],'4608')
            self.assertEqual(export['payload']['expected_language_parameters'],1981526016)
            (root/'calibration.json').write_text('{}')
            with self.assertRaises(ValueError):native_command(state,export,root)

    def test_export_validation_requires_budget_architecture_and_verified_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec=build_submission(self.fixture(root),root);p=spec['jobs'][0]['payload']
            model=Path(p['output']);report=dict(p['request'],test_set_used=False,verification_split='train',
                 encoder_tensors_unchanged=True,recovery_updates=0,
                 parameters_after={'language':1981526016},calibration={'sha256':p['calibration_sha256']},
                 verification={'passed':True,'save_reload_class_logits_match':True,'save_reload_generation_match':True})
            atomic_json(model/'compression.json',report)
            atomic_json(model/'config.json',{'text_config':{'intermediate_size':4608,'num_hidden_layers':36}})
            atomic_json(model/'model.safetensors.index.json',{})
            self.assertEqual(validate_export(p)['language_parameters'],1981526016)
            for changes in [{'parameters_after':{'language':2000000000}}, {'verification':{'passed':True}},
                            {'test_set_used':True}, {'ffn_width':4352}]:
                atomic_json(model/'compression.json',report|changes)
                with self.assertRaises(ValueError):validate_export(p)
            atomic_json(model/'compression.json',report)
            atomic_json(model/'config.json',{'text_config':{'intermediate_size':4352,'num_hidden_layers':36}})
            with self.assertRaises(ValueError):validate_export(p)
