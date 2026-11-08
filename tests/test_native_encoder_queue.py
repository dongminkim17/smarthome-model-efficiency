"""Original-encoder recovery provenance, dependency graph and zero-distance checks."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.training import DEFAULTS,load_config,config_fingerprint
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.pruning import sha256_file
from project_queue import validate_submission
from project_queue_jobs import native_command
from queue_mer_native_encoders import definitions,build_submission,REFERENCES
from export_recovered_model import validate_merge_export
from native_encoder_preflight import validate_feature_smoke
import yaml

class NativeEncoderQueueTests(unittest.TestCase):
    def fixture(self,root):
        counts=expected_counts(4352,5120,3420);source=root/'source';source.mkdir();model=root/'old4352';model.mkdir();teacher=root/'teacher';teacher.mkdir()
        cache=root/'class.json';cache.write_text('{}');features=root/'features';features.mkdir();(features/'cache.json').write_text('{}')
        cfg=dict(DEFAULTS,task='mer6',model=str(root/'balanced'),teacher_checkpoint=str(teacher),teacher_cache=str(cache),feature_teacher_cache=str(features),feature_audio_weight=.1,feature_vision_weight=.1,encoder_learning_rate=1e-5,objective='sft_kd',distillation_alpha=.75,use_transcript=False,lora_target='decoder_encoder_ffn')
        sourcecfg=dict(cfg,model=str(model),lora_target='decoder_linear');sourcecfg.pop('encoder_learning_rate')
        (model/'compression.json').write_text(json.dumps({'parameters_after':counts}))
        (source/'report.json').write_text(json.dumps({'config':sourcecfg,'train_count':2698,'dev_count':675,'test_set_used':False,'teacher_cache_sha256':sha256_file(cache)}))
        wide=root/'wide';wide.mkdir();(wide/'compression.json').write_text(json.dumps({'parameters_after':expected_counts(4608,5120,3420),'source_checkpoint':str(teacher),'calibration_samples':128}))
        state={'jobs':[{'id':REFERENCES[16],'payload':{'config':cfg}}, {'id':'recovery_controls_20261019/train:KD075_mer6','status':'completed','payload':{'run_dir':str(source)}}, {'id':'ffn_expansion_20261020/export:mer6_ffn4608_n128','status':'completed','payload':{'output':str(wide)}}]}
        return state,source

    def test_matrix_has_six_per_wave_and_two_fixed_replicates(self):
        ds=definitions();self.assertEqual(len(ds),14);self.assertEqual(sum(d['wave']==1 for d in ds),6)
        self.assertEqual(sum(d['seed']==43 for d in ds),2)
        self.assertEqual({(d['width'],d['recipe'],d['rank']) for d in ds if d['seed']==42}, {(w,p,r) for w in (4352,4608) for p in ('STANDARD','BROAD','GENTLE') for r in (16,64)})

    def test_pipeline_preserves_inputs_and_releases_each_test_independently(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);legacy,source=self.fixture(root)
            with patch('queue_mer_native_encoders.validate_reference_contract'),patch('queue_mer_native_encoders.validate_export'),patch('queue_mer_native_encoders.pin_best',return_value=str(source/'checkpoints/step-1')),patch('queue_mer_native_encoders.checkpoint_identity',return_value={'metadata.json':'pin'}):spec=build_submission(legacy,root/'queue',root)
            self.assertEqual(len(spec['jobs']),62);validate_submission({'jobs':[],'documents':{}},spec)
            jobs=spec['jobs'];by={j['id']:j for j in jobs};train=[j for j in jobs if j['kind']=='recovery_train'];self.assertEqual(len(train),15)
            tests=[j for j in jobs if j['kind']=='recovery_test'];self.assertEqual(len(tests),14)
            self.assertFalse(any(j['kind']=='shortlist' for j in jobs))
            for test in tests:
                self.assertEqual(len(test['deps']),1);self.assertEqual(by[test['deps'][0]]['kind'],'recovery_train');self.assertEqual(test['payload']['modes'],['logit','greedy']);self.assertGreater(test['priority'],max(j['priority'] for j in train))
            for j in jobs:
                if j['kind'] not in ('recovery_train','smoke_train'):continue
                cfg=j['payload']['config'];p=root/'c.yaml';p.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(p),cfg)
                self.assertEqual(config_fingerprint(load_config(p)),j['payload']['config_sha256'])
                self.assertEqual(cfg['split_seed'],42);self.assertFalse(cfg['use_transcript']);self.assertEqual(cfg['max_frames'],8)
                self.assertNotIn('distillation_disagreement_alpha',cfg)
            for j in train:
                cfg=j['payload']['config'];self.assertEqual(cfg['epochs'],5)
                counts=j['payload']['expected_parameters'];self.assertLessEqual(counts['language'],2_000_000_000);self.assertEqual(counts['audio_encoder'],637676544);self.assertEqual(counts['vision_encoder'],668684288)
            wide_merge=next(j for j in jobs if j['kind']=='merge_export' and j['payload'].get('training_job'))
            self.assertEqual(wide_merge['deps'],[wide_merge['payload']['training_job']]);self.assertIn('W4608_LM_PREPARATION',by[wide_merge['deps'][0]]['payload']['arm'])
            for j in jobs:
                if j['kind']=='smoke_train' and j['payload'].get('wave')==1:self.assertNotIn(wide_merge['id'],j['deps'])
            self.assertFalse(any(j['kind']=='recovery_test' and 'PREPARATION' in j['payload']['arm'] for j in jobs))

    def test_matching_encoders_may_have_zero_distance_but_pruned_encoders_may_not(self):
        values=[{'audio_feature_loss':0.,'vision_feature_loss':-1e-8},{'audio_feature_loss':.001,'vision_feature_loss':.002}]
        with self.assertRaises(ValueError):validate_feature_smoke({},values)
        with tempfile.TemporaryDirectory() as f:
            model=Path(f);manifest={'method':'merged_lm_recovery','encoder_tensors_unchanged':True,'parameters_after':expected_counts(4352,5120,3420)}
            (model/'compression.json').write_text(json.dumps(manifest));payload={'original_encoders':True,'config':{'model':str(model)}}
            validate_feature_smoke(payload,values)
            for bad in [float('nan'),-.01,float('inf')]:
                with self.assertRaises(ValueError):validate_feature_smoke(payload,[{'audio_feature_loss':bad,'vision_feature_loss':0.}])
            manifest['encoder_tensors_unchanged']=False;(model/'compression.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):validate_feature_smoke(payload,values)

    def test_dynamic_merge_pins_validation_source_and_rejects_change(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);base=root/'base';base.mkdir();(base/'compression.json').write_text('{}');cp=root/'cp';cp.mkdir();(cp/'metadata.json').write_text(json.dumps({'config':{'model':str(base)}}))
            parent={'id':'wide_train','status':'completed','payload':{'run_dir':'source_run'}};job={'kind':'merge_export','payload':{'training_job':'wide_train','output':'merged'}}
            with patch('project_queue_jobs.pin_best',return_value=str(cp)),patch('project_queue_jobs.checkpoint_identity',return_value={'metadata.json':'fixed'}):
                command,cfg=native_command({'jobs':[parent]},job,root);self.assertIn(str(cp),command);self.assertIn(str(ROOT/'scripts/export_recovered_model.py'),command)
                job['payload']['source_identity']={'metadata.json':'changed'}
                with self.assertRaises(ValueError):native_command({'jobs':[parent]},job,root)

    def test_merge_validation_rejects_wrong_source_config_or_counts(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);source=root/'source';source.mkdir();base=root/'base';base.mkdir();output=root/'output';output.mkdir()
            cfg={'text_config':{'intermediate_size':4352},'audio_config':{'encoder_ffn_dim':5120},'vision_config':{'intermediate_size':3420}}
            counts=expected_counts(4352,5120,3420);(base/'config.json').write_text(json.dumps(cfg));(output/'config.json').write_text(json.dumps(cfg));(base/'compression.json').write_text(json.dumps({'parameters_after':counts}));(source/'metadata.json').write_text(json.dumps({'config':{'model':str(base)}}));(output/'model.safetensors').write_text('fixture')
            report={'method':'merged_lm_recovery','source_checkpoint':str(source),'source_identity':{'metadata.json':'fixed'},'upstream_compression_sha256':sha256_file(base/'compression.json'),'parameters_after':counts,'parameters_before':counts,'encoder_tensors_unchanged':True,'verification_split':'train','test_set_used':False,'parameter_limits':{'language':2_000_000_000,'audio_encoder':1_000_000_000,'vision_encoder':1_000_000_000},'verification':{'passed':True,'save_reload_class_logits_match':True,'save_reload_generation_match':True}}
            path=output/'compression.json';path.write_text(json.dumps(report))
            with patch('export_recovered_model.checkpoint_identity',return_value={'metadata.json':'fixed'}),patch('export_recovered_model.processor_cache_identity',return_value={'same':True}):
                self.assertEqual(validate_merge_export(output,source,counts)['parameters'],counts)
                for key,value in [('test_set_used',True),('encoder_tensors_unchanged',False),('upstream_compression_sha256','changed'),('source_identity',{'metadata.json':'wrong'})]:
                    bad=copy.deepcopy(report);bad[key]=value;path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):validate_merge_export(output,source,counts)
                path.write_text(json.dumps(report));(output/'config.json').write_text('{}')
                with self.assertRaises(ValueError):validate_merge_export(output,source,counts)

if __name__=='__main__':unittest.main()
