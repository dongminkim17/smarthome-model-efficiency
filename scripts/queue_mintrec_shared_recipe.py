#!/usr/bin/env python3
"""Verify the successful MER native-encoder recipe on both official MIntRec tasks."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from project_queue import validate_submission
from project_queue_jobs import entry
from queue_ffn_expansion import training_chain
from launch_pruning_followup import checkpoint_identity,pin_best
from compare_project_benchmarks import RECIPE_KEYS,EXPECTED_COUNTS
from smarthome_efficiency.queueing import atomic_json,read_json,fingerprint
from smarthome_efficiency.pruning import sha256_file
FAMILY='mintrec_shared_recipe_20261106'
MER_RUN=ROOT/'artifacts/full/mer_native_encoders_20261106/train/mer6_w4352_broad_r64_seed42/attempt-001'
MER_TEST=ROOT/'artifacts/full/mer_native_encoders_20261106/test/mer6_w4352_broad_r64_seed42/attempt-001/report.json'


def reference_recipe(run=MER_RUN,test_path=MER_TEST):
    run,test_path=Path(run),Path(test_path);r=read_json(run/'report.json');t=read_json(test_path);cfg=r['config']
    expected={'epochs':5,'seed':42,'split_seed':42,'lora_rank':64,'lora_alpha':128,'lora_dropout':0.,'lora_target':'decoder_encoder_linear',
              'learning_rate':1e-4,'encoder_learning_rate':1e-5,'objective':'sft_kd','distillation_alpha':.75,'distillation_temperature':2.,
              'feature_audio_weight':.1,'feature_vision_weight':.1,'max_frames':8,'max_pixels':224**2,'use_transcript':False,'gradient_checkpointing':True}
    if (r['status']!='completed' or r.get('test_set_used') is not False or r['train_count']!=2698 or r['dev_count']!=675
            or any(cfg.get(k)!=v for k,v in expected.items())
            or t['status']!='completed' or t['partial_split'] or t['evaluated_count']!=834
            or t['checkpoint']!=pin_best(run) or t['config']!=cfg
            or read_json(Path(cfg['model'])/'compression.json')['parameters_after']!=EXPECTED_COUNTS):
        raise ValueError('Reference must be the completed, verified broad-rank64 LM4352 original-encoder MER setting')
    return copy.deepcopy(cfg)


def build_submission(legacy,queue_root,root=ROOT,reference_run=MER_RUN,reference_test=MER_TEST):
    reference_run,reference_test=Path(reference_run),Path(reference_test)
    root,queue_root=Path(root),Path(queue_root);cfg=reference_recipe(reference_run,reference_test);by={j['id']:j for j in legacy['jobs']}
    base=root/'artifacts/full'/FAMILY;manifest=base/'workflow.json';jobs=[];test_ids={};plans=[]
    def located(key,kind,payload,deps=(),priority=100,resource='gpu'):
        return entry(FAMILY+'/'+key,kind,dict(payload,id=key),{'manifest':str(manifest),'field':'preparation_jobs','key_field':'id','key':key},deps,priority=priority,resource=resource)
    for task in ('mintrec2','mintrec20'):
        src=by['recovery_controls_20261019/train:KD075_'+task];run=Path(src['payload']['run_dir']);report=read_json(run/'report.json');sc=report['config'];source=pin_best(run)
        if (src['status']!='completed' or report['status']!='completed' or report.get('test_set_used') is not False or not source
                or report['train_count']!=1334 or report['dev_count']!=445 or report['world_size']!=1
                or sc['task']!=task or sc['lora_target']!='decoder_linear' or sc['lora_rank']!=16
                or sc['distillation_alpha']!=.75 or sc['distillation_temperature']!=2. or sc['epochs']!=5
                or sc['learning_rate']!=1e-4 or sc['seed']!=42 or sc['split_seed']!=42 or sc['use_transcript'] is not True
                or sc['max_frames']!=cfg['max_frames'] or sc['max_pixels']!=cfg['max_pixels']):
            raise ValueError('Expected the matching completed task LM4352 KD075 recovery')
        side=read_json(Path(sc['model'])/'compression.json')
        if side['parameters_after']!=EXPECTED_COUNTS or side['method']!='ffn_activation' or side['ffn_width']!=4352 or side['calibration_samples']!=128 or not side['verification']['passed'] or side.get('source_checkpoint')!=sc['teacher_checkpoint'] or side.get('encoder_tensors_unchanged') is not True:
            raise ValueError('Source architecture or activation calibration differs')
        feature=root/'artifacts/cache/full_model_followups_20261020/encoder_features'/task
        c=read_json(sc['teacher_cache']);f=read_json(feature/'cache.json');identity=checkpoint_identity(sc['teacher_checkpoint'])
        if (c['status']!='completed' or f['status']!='completed' or len(c['rows'])!=1334 or len(f['rows'])!=1334
                or c['spec']['split']!='train' or c['spec']['test_set_used'] is not False
                or f['spec']['teacher']!=c['spec'] or c['spec']['teacher_checkpoint']!=sc['teacher_checkpoint']
                or c['spec']['teacher_metadata_sha256']!=identity['metadata.json']
                or c['spec']['teacher_adapter_sha256']!=identity['adapter/adapter_model.safetensors']):raise ValueError('Task-specific teacher targets or provenance differ')
        class_hash=sha256_file(Path(sc['teacher_cache']));feature_hash=sha256_file(feature/'cache.json')
        if report.get('teacher_cache_sha256')!=class_hash:raise ValueError('Source recovery used different class targets')
        model=root/'artifacts/compact'/FAMILY/(task+'_lm4352_recovered_original_encoders')
        new=copy.deepcopy(cfg);new.update(task=task,model=str(model),teacher_checkpoint=sc['teacher_checkpoint'],teacher_cache=sc['teacher_cache'],feature_teacher_cache=str(feature),use_transcript=True)
        if any(new.get(k)!=cfg.get(k) for k in RECIPE_KEYS):raise ValueError('Final recovery recipe changed')
        prep_cfg=dict(new,model=sc['model']);stress=queue_root/'stress_cases'/(task+'_shared_recipe.json')
        prep=located('prepare:'+task,'prepare',{'task':task,'config':prep_cfg,'output':str(stress),'estimated_seconds':120},priority=170,resource='cpu')
        merge=located('merge:'+task,'merge_export',{'task':task,'source_run':str(run),'source_identity':checkpoint_identity(source),
            'source_base_manifest_sha256':sha256_file(Path(sc['model'])/'compression.json'),'output':str(model),'expected_parameters':EXPECTED_COUNTS,'estimated_seconds':240},priority=160)
        jobs.extend([prep,merge])
        chain=training_chain(FAMILY,manifest,queue_root,root,task,'NATIVE_BROAD_R64_SEED42',new,sc['teacher_checkpoint'],[prep['id'],merge['id']])
        for j in chain:
            p=j['payload'];p.update(architecture='LM4352_ORIGINAL_ENCODERS',budget_scope='per_modality',expected_parameters=EXPECTED_COUNTS,original_encoders=True,
                reference_method='MER W4352_BROAD_R64_SEED42',initialization='Fresh final-recovery adapters on matching task-specific merged LM4352 KD075 source; original encoders retained')
            if j['kind'] in ('smoke_train','recovery_train'):
                j['priority']=130 if j['kind']=='smoke_train' else 120
                p.update(expected_teacher_cache_sha256=class_hash,expected_feature_cache_sha256=feature_hash,estimated_seconds=240 if j['kind']=='smoke_train' else 3*3600)
            elif j['kind']=='smoke_test':j['priority']=140
            elif j['kind']=='recovery_test':j['priority']=200;p['estimated_seconds']=700;test_ids[task]=j['id']
            if j['kind']=='smoke_train':p['memory_cases']=str(stress)
        jobs.extend(chain);plans.append({'task':task,'counts':{'train':1334,'dev':445,'official_test':445},'source_checkpoint':source,'source_identity':checkpoint_identity(source),'config':new})
    catalog=root/'artifacts/analysis/baselines_20261103/report_comparison_tables.json'
    comparison=located('compare:shared_recipe','benchmark_comparison',{'task':'mintrec2','test_jobs':test_ids,
        'baseline_catalog':str(catalog),'baseline_catalog_sha256':sha256_file(catalog),
        'reference_training_report':str(reference_run/'report.json'),'reference_training_report_sha256':sha256_file(reference_run/'report.json'),
        'mer_reference':{'task':'mer6','test_report':str(reference_test),'training_report':str(reference_run/'report.json'),'expected_test_report_sha256':sha256_file(reference_test)},
        'output':str(base/'comparison.json'),'estimated_seconds':10},list(test_ids.values()),priority=210,resource='cpu');jobs.append(comparison)
    doc={'format_version':1,'status':'waiting','phase':'per_job_dependencies','host':'joy','allowed_gpus':list(range(8)),
        'description':'Matched winning MER recipe on MIntRec2/20, official dual-mode tests and fixed-cohort comparison',
        'jobs':[],'smoke_jobs':[],'preparation_jobs':[],'experiment_plan':plans,'checkpoint_selection_uses_test':False,
        'mer_reference_training_report':str(reference_run/'report.json'),'mer_reference_test_report':str(reference_test),
        'parameter_limits':{'language':2_000_000_000,'audio_encoder':1_000_000_000,'vision_encoder':1_000_000_000},
        'estimated_wall_hours':[2,3.5],
        'note':'Same LM4352/originalencoders,rank64/alpha128,broaddecoder+encoderLoRA,LR1e-4/1e-5,KD.75/T2,feature.1/.1,seed42,fiveepochs,batch4 as MER winner. Existing official AVT MIntRec input/splits retained; independent task weights/teachers. Each run tests immediately. Comparison reads official results only after checkpoint selection/training.'}
    for j in jobs:doc[j['location']['field']].append(copy.deepcopy(j['payload']))
    return {'schema_version':1,'jobs':jobs,'documents':{str(manifest):doc}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/joy');p.add_argument('--source-queue',type=Path,default=ROOT/'artifacts/queue/suzy');p.add_argument('--submit',action='store_true');args=p.parse_args()
    state=read_json(args.queue_root/'state.json')
    if state.get('host')!='joy' or state['allowed_gpus']!=list(range(8)):raise ValueError('Joy0–7 resource contract required')
    path=args.queue_root/'submissions'/(FAMILY+'.json');spec=read_json(path) if path.exists() else build_submission(read_json(args.source_queue/'state.json'),args.queue_root)
    validate_submission(state,spec)
    if args.submit:atomic_json(path,spec);atomic_json(args.queue_root/'inbox'/(fingerprint(spec)+'.json'),spec)
    print(json.dumps({'submitted':args.submit,'family':FAMILY,'jobs':len(spec['jobs']),'training_runs':2,'official_tests':2,'comparison_report':str(ROOT/'artifacts/full'/FAMILY/'comparison.json')},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
