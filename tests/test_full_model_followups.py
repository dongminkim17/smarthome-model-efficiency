"""Approved 27-run batch: factorial coverage, strict lineage and full-budget exports."""
import copy,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from queue_full_model_followups import build_submission,FAMILY,TASKS,ARMS
from smarthome_efficiency.queueing import atomic_json,validate_graph
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.training import DEFAULTS,load_config
from smarthome_efficiency.multimodal_pruning import expected_counts
from project_queue_jobs import entry,native_command,validate_export


class FollowupQueueTests(unittest.TestCase):
    def fixture(self,root):
        jobs=[]
        for task in TASKS:
            base=root/task/'source_base';source=root/task/'source';teacher=root/task/'teacher';a=root/task/'A';cache=root/task/'class_cache.json'
            atomic_json(base/'compression.json',{'fixture':True})
            for path in [source,teacher]:
                (path/'adapter').mkdir(parents=True);(path/'adapter/adapter_model.safetensors').write_bytes(b'fixture')
            cfg=dict(DEFAULTS,task=task,model=str(base),data_root='/data',cache_dir='/cache',
                objective='sft_kd',teacher_checkpoint=str(teacher),teacher_cache=str(cache),distillation_alpha=.75,
                lora_target='decoder_encoder_ffn',gradient_checkpointing=True,micro_batch_size=2,gradient_accumulation_steps=2)
            atomic_json(source/'metadata.json',{'config':cfg});atomic_json(teacher/'metadata.json',{'config':cfg})
            atomic_json(cache,{'spec':{'teacher_checkpoint':str(teacher),'split':'train','test_set_used':False}})
            side={'method':'multimodal_ffn_activation','budget_scope':'total','parameters_after':expected_counts(1536,1024,1024),
                  'verification':{'passed':True},'source_checkpoint':str(source),
                  'source_adapter_sha256':sha256_file(source/'adapter/adapter_model.safetensors'),
                  'source_metadata_sha256':sha256_file(source/'metadata.json')}
            atomic_json(a/'compression.json',side)
            jobs.extend([
                entry('full_model_2b_20261020/export:'+task+'_a_lm1536_a1024_v1024','prune_export',{'task':task,'status':'completed','output':str(a)}),
                entry('full_model_2b_20261020/train:'+task+'_a_lm1536_a1024_v1024','recovery_train',{'task':task,'config':cfg}),
                entry('calibrated_pruning_20261019/teacher:'+task,'followup',{'task':task,'status':'completed'}),
                entry('prepare:'+task,'prepare',{'task':task,'status':'completed'},resource='cpu')])
        return {'jobs':jobs,'queue_root':str(root/'queue'),'documents':{}}

    def test_exact_27_runs_independent_gates_and_only_three_shared_caches(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);before=copy.deepcopy(state);spec=build_submission(state,root)
            self.assertEqual(state,before);validate_graph(state['jobs']+spec['jobs']);self.assertEqual(len(spec['jobs']),114)
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train'];self.assertEqual(len(trains),27)
            self.assertEqual(sum(j['kind']=='encoder_cache' for j in spec['jobs']),3)
            self.assertEqual(sum(j['kind']=='prune_export' for j in spec['jobs']),3)
            self.assertEqual(sum(bool(j['payload']['config'].get('feature_teacher_cache')) for j in trains),12)
            by={j['id']:j for j in spec['jobs']}
            for task in TASKS:self.assertEqual({j['payload']['arm'] for j in trains if j['task']==task},{a[0] for a in ARMS})
            for j in trains:
                p=j['payload'];cfg=p['config'];self.assertIsNone(p['init_adapter']);self.assertEqual(cfg['epochs'],5)
                self.assertEqual(cfg['micro_batch_size']*cfg['gradient_accumulation_steps'],4);self.assertEqual(cfg['distillation_alpha'],.75)
                self.assertLess(p['expected_parameters']['total'],2000000000)
                reload=by[j['deps'][0]];smoke=by[reload['deps'][0]];self.assertEqual(smoke['kind'],'smoke_train')
                test=by[FAMILY+'/test:'+p['name']];self.assertEqual(test['deps'],[j['id']]);self.assertEqual(test['payload']['split'],'test')
                if cfg.get('feature_teacher_cache'):self.assertIn(FAMILY+'/encoder_cache:'+j['task'],smoke['deps'])
                import yaml
                path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(path),cfg)

    def test_matched_rank_lr_and_feature_controls_change_only_the_intended_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);spec=build_submission(self.fixture(root),root)
            cfgs={j['payload']['arm']:j['payload']['config'] for j in spec['jobs'] if j['kind']=='recovery_train' and j['task']=='mintrec2'}
            def changed(a,b):return {k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
            self.assertEqual(changed(cfgs['A_WIDE_R16'],cfgs['A_WIDE_R64']),{'lora_rank','lora_alpha'})
            self.assertEqual(changed(cfgs['A_WIDE_R16'],cfgs['A_WIDE_LR3E5']),{'learning_rate'})
            self.assertEqual(changed(cfgs['A_WIDE_R16'],cfgs['A_WIDE_FEATURE_R16']),{'feature_teacher_cache','feature_audio_weight','feature_vision_weight'})
            self.assertEqual(changed(cfgs['D_BASE_R16'],cfgs['D_WIDE_R16']),{'lora_target'})

    def test_native_preparation_commands_and_hybrid_export_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=self.fixture(root);spec=build_submission(state,root)
            for j in spec['jobs']:
                if j['kind']=='encoder_cache':
                    command,cfg=native_command(state,j,root);self.assertIn(str(ROOT/'scripts/cache_encoder_features.py'),command);self.assertIn('--resume',command)
                if j['kind']!='prune_export':continue
                p=j['payload'];r=p['request'];command,_=native_command(state,j,root)
                self.assertEqual(command[command.index('--audio-layers')+1],'24');self.assertEqual(command[command.index('--vision-layers')+1],'24')
                atomic_json(r['calibration_cache'],{'fixture':True})
                cal={k:r[k] for k in ['source_checkpoint','source_adapter_sha256','source_metadata_sha256','calibration_seed']}
                cal.update(examples=128,split='train',test_set_used=False,scope='all_thinker_components')
                a=list(range(23))+[31];v=list(range(22))+[23,31];full=[7,15,23,31];mapped=[i for i,x in enumerate(v) if x in full]
                report=dict(r,parameters_after=p['expected_parameters'],retained_non_ffn_tensors_unchanged=True,av_interfaces_unchanged=True,
                    all_layer_counts_unchanged=False,encoder_tensors_unchanged=False,non_ffn_tensors_unchanged=False,
                    test_set_used=False,verification_split='train',recovery_updates=0,
                    calibration={'sha256':sha256_file(Path(r['calibration_cache'])),'spec':cal},
                    verification={'passed':True,'save_reload_class_logits_match':True,'save_reload_generation_match':True},
                    kept_original_audio_layer_indices=a,kept_original_vision_layer_indices=v,original_vision_full_attention_indices=full)
                out=Path(p['output']);atomic_json(out/'compression.json',report);atomic_json(out/'model.safetensors.index.json',{})
                config={'text_config':{'intermediate_size':2048,'num_hidden_layers':36},'audio_config':{'encoder_layers':24,'encoder_ffn_dim':1536},
                        'vision_config':{'depth':24,'intermediate_size':1280,'fullatt_block_indexes':mapped}}
                atomic_json(out/'config.json',config);self.assertEqual(validate_export(p)['total_parameters'],1989016832)
                config['vision_config']['fullatt_block_indexes']=[];atomic_json(out/'config.json',config)
                with self.assertRaises(ValueError):validate_export(p)

if __name__=='__main__':unittest.main()
