#!/usr/bin/env python3
"""Two waves of MER recovery with original encoders and LM4352/4608."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from project_queue import validate_submission
from project_queue_jobs import entry,validate_export
from queue_ffn_expansion import training_chain
from queue_mer_focus import build_submission as validate_reference_contract,REFERENCES
from launch_pruning_followup import checkpoint_identity,pin_best
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.queueing import read_json,atomic_json,fingerprint
FAMILY='mer_native_encoders_20261106'
RECIPES={'STANDARD':('decoder_encoder_ffn',1e-4,1e-5),'BROAD':('decoder_encoder_linear',1e-4,1e-5),'GENTLE':('decoder_encoder_ffn',3e-5,3e-6)}


def definitions():
    rows=[]
    for width in (4352,4608):
        for recipe in RECIPES:
            for rank in (16,64):
                rows.append({'arm':f'W{width}_{recipe}_R{rank}_SEED42','width':width,'recipe':recipe,'rank':rank,'seed':42,'wave':1 if width==4352 else 2})
    for rank in (16,64):rows.append({'arm':f'W4352_STANDARD_R{rank}_SEED43','width':4352,'recipe':'STANDARD','rank':rank,'seed':43,'wave':2})
    return rows


def build_submission(legacy,queue_root,root=ROOT):
    root=Path(root);queue_root=Path(queue_root);validate_reference_contract(legacy,queue_root,root)
    by={j['id']:j for j in legacy['jobs']}
    cfg=copy.deepcopy(by[REFERENCES[16]]['payload']['config'])
    if cfg.get('mer_partition') or any(k in cfg for k in ('distillation_alpha_final','distillation_disagreement_alpha')):raise ValueError('Original MER constant-KD protocol required')
    source_job=by['recovery_controls_20261019/train:KD075_mer6'];source_run=source_job['payload']['run_dir'];source=pin_best(source_run)
    if source_job['status']!='completed' or not source:raise ValueError('LM4352 recovery is incomplete')
    source_report=read_json(Path(source_run)/'report.json');source_cfg=source_report['config']
    if (source_cfg['lora_target']!='decoder_linear' or source_cfg['distillation_alpha']!=.75
            or source_cfg['teacher_checkpoint']!=cfg['teacher_checkpoint']
            or source_report['train_count']!=2698 or source_report['dev_count']!=675
            or source_report['test_set_used'] is not False):raise ValueError('Unexpected LM4352 recovery source')
    counts={w:expected_counts(w,5120,3420) for w in (4352,4608)}
    if read_json(Path(source_cfg['model'])/'compression.json')['parameters_after']!=counts[4352]:raise ValueError('LM4352 source encoders are not original')
    wide_export=by['ffn_expansion_20261020/export:mer6_ffn4608_n128'];validate_export(wide_export['payload'])
    if wide_export['status']!='completed':raise ValueError('LM4608 initial export is incomplete')
    wide_model=wide_export['payload']['output'];wide_side=read_json(Path(wide_model)/'compression.json')
    if (wide_side['parameters_after']!=counts[4608] or wide_side['source_checkpoint']!=cfg['teacher_checkpoint'] or wide_side['calibration_samples']!=128):raise ValueError('Unexpected wider LM calibration/source')
    class_hash=sha256_file(Path(cfg['teacher_cache']));feature_hash=sha256_file(Path(cfg['feature_teacher_cache'])/'cache.json')
    if source_report.get('teacher_cache_sha256')!=class_hash:raise ValueError('LM recovery used different targets')
    base=root/'artifacts/full'/FAMILY;manifest=base/'workflow.json';stress=queue_root/'stress_cases/mer6_native_encoders_80_20.json'
    def located(key,kind,payload,deps=(),priority=100,resource='gpu'):
        return entry(FAMILY+'/'+key,kind,dict(payload,id=key,task='mer6'),{'manifest':str(manifest),'field':'preparation_jobs','key_field':'id','key':key},deps,priority=priority,resource=resource)
    prepare=located('prepare:mer6','prepare',{'config':cfg,'output':str(stress),'estimated_seconds':180},priority=180,resource='cpu')
    jobs=[prepare];plans=[]
    merged={w:str(root/'artifacts/compact'/FAMILY/f'mer6_lm{w}_recovered_original_encoders') for w in counts}
    def merge(width,training=None):
        p={'output':merged[width],'expected_parameters':counts[width],'estimated_seconds':240,
           'training_job':training,'source_run':None if training else source_run}
        if not training:p.update(source_identity=checkpoint_identity(source),source_base_manifest_sha256=sha256_file(Path(source_cfg['model'])/'compression.json'))
        return located(f'merge:mer6_lm{width}','merge_export',p,[training] if training else [],priority=160)
    merge4352=merge(4352);jobs.append(merge4352)
    # Prepare the wider LM while the six first-wave recoveries run.
    wide=copy.deepcopy(cfg)
    for key in ('feature_teacher_cache','feature_audio_weight','feature_vision_weight','encoder_learning_rate'):wide.pop(key,None)
    wide.update(model=wide_model,lora_rank=16,lora_alpha=32,lora_target='decoder_linear',seed=42,epochs=5,
                objective='sft_kd',distillation_alpha=.75,distillation_temperature=2.,learning_rate=1e-4,
                train_limit=None,dev_limit=None,eval_modes=['logit'],selection_mode='logit',eval_every_fraction=.5,
                micro_batch_size=2,gradient_accumulation_steps=2,gradient_checkpointing=True)
    chain=training_chain(FAMILY,manifest,queue_root,root,'mer6','W4608_LM_PREPARATION',wide,cfg['teacher_checkpoint'],[prepare['id']])
    wide_train=next(j for j in chain if j['kind']=='recovery_train')
    for j in chain:
        if j['kind']=='recovery_test':continue
        j['priority']=155 if j['kind']=='smoke_test' else 150 if j['kind']=='smoke_train' else 140
        j['payload'].update(architecture='LM4608_ORIGINAL_ENCODERS',budget_scope='per_modality',expected_parameters=counts[4608],wave='preparation',purpose='Intermediate LM-only KD recovery; no official test required')
        if j['kind'] in ('smoke_train','recovery_train'):
            j['payload'].update(expected_teacher_cache_sha256=class_hash,estimated_seconds=180 if j['kind']=='smoke_train' else 4*3600)
        if j['kind']=='smoke_train':j['payload']['memory_cases']=str(stress)
        jobs.append(j)
    merge4608=merge(4608,wide_train['id']);jobs.append(merge4608)
    anchors={}
    for d in definitions():
        width,rank,seed=d['width'],d['rank'],d['seed'];scope,lm_lr,enc_lr=RECIPES[d['recipe']]
        new=copy.deepcopy(cfg);new.update(model=merged[width],lora_rank=rank,lora_alpha=2*rank,lora_target=scope,lora_dropout=0.,
            seed=seed,epochs=5,learning_rate=lm_lr,encoder_learning_rate=enc_lr,
            objective='sft_kd',distillation_alpha=.75,distillation_temperature=2.,feature_audio_weight=.1,feature_vision_weight=.1,
            train_limit=None,dev_limit=None,eval_modes=['logit'],selection_mode='logit',eval_every_fraction=.5,
            micro_batch_size=2,gradient_accumulation_steps=2,gradient_checkpointing=True)
        deps=[prepare['id'],(merge4352 if width==4352 else merge4608)['id']]
        if seed==43:deps.append(anchors[rank])
        chain=training_chain(FAMILY,manifest,queue_root,root,'mer6',d['arm'],new,cfg['teacher_checkpoint'],deps)
        full=next(j for j in chain if j['kind']=='recovery_train')
        if width==4352 and d['recipe']=='STANDARD' and seed==42:anchors[rank]=full['id']
        priority=120 if d['wave']==1 else 70 if seed==43 else 80
        for j in chain:
            p=j['payload'];p.update(architecture=f'LM{width}_ORIGINAL_ENCODERS',budget_scope='per_modality',expected_parameters=counts[width],wave=d['wave'],original_encoders=True,
                initialization='Fresh final-recovery adapters on validation-selected merged LM recovery; original encoders retained')
            if j['kind'] in ('smoke_train','recovery_train'):
                j['priority']=priority+(5 if j['kind']=='smoke_train' else 0)
                p.update(expected_teacher_cache_sha256=class_hash,expected_feature_cache_sha256=feature_hash,
                         estimated_seconds=240 if j['kind']=='smoke_train' else (5.5 if scope=='decoder_encoder_linear' else 5)*3600)
            elif j['kind']=='smoke_test':j['priority']=priority+10
            elif j['kind']=='recovery_test':j['priority']=200;p['estimated_seconds']=1000
            if j['kind']=='smoke_train':p['memory_cases']=str(stress)
        jobs.extend(chain);plans.append({**d,'scope':scope,'learning_rate':lm_lr,'encoder_learning_rate':enc_lr,'epochs':5,'parameters':counts[width]})
    document={'format_version':1,'status':'waiting','phase':'per_job_dependencies','host':'joy','allowed_gpus':list(range(8)),
        'description':'MER original encoder adaptation: six LM4352 recoveries, six LM4608 recoveries, two seed checks and one LM preparation',
        'jobs':[],'smoke_jobs':[],'preparation_jobs':[],'experiment_plan':plans,
        'checkpoint_selection_uses_test':False,'official_tests':'Each of14 final recoveries is tested immediately after training; no batch-wide shortlist',
        'estimated_wall_hours':[10,12],'data_counts':{'train':2698,'dev':675,'official_test':834},
        'parameter_limits':{'language':2_000_000_000,'audio_encoder':1_000_000_000,'vision_encoder':1_000_000_000},
        'note':'No encoder pruning. Original fixed teacher,80/20split,input protocol,caches; classKD.75/T2+featureKD.1/.1. Two waves are priorities/dependencies, not a global barrier. Existing active control continues. Wider LM source selected by675-example validation only. Shared method adoption requires both MIntRec tasks.'}
    for j in jobs:document[j['location']['field']].append(copy.deepcopy(j['payload']))
    spec={'schema_version':1,'jobs':jobs,'documents':{str(manifest):document}};return spec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/joy');p.add_argument('--source-queue',type=Path,default=ROOT/'artifacts/queue/suzy');p.add_argument('--submit',action='store_true');args=p.parse_args()
    state=read_json(args.queue_root/'state.json')
    if state.get('host')!='joy' or state['allowed_gpus']!=list(range(8)):raise ValueError('Joy0–7 resource contract required')
    path=args.queue_root/'submissions'/(FAMILY+'.json');spec=read_json(path) if path.exists() else build_submission(read_json(args.source_queue/'state.json'),args.queue_root)
    validate_submission(state,spec)
    if args.submit:atomic_json(path,spec);atomic_json(args.queue_root/'inbox'/(fingerprint(spec)+'.json'),spec)
    print(json.dumps({'submitted':args.submit,'family':FAMILY,'jobs':len(spec['jobs']),'training_runs':sum(j['kind']=='recovery_train' for j in spec['jobs']),'final_recoveries':14,'intermediate_LM_recovery':1,'official_tests':14,'plan':next(iter(spec['documents'].values()))['experiment_plan']},indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
