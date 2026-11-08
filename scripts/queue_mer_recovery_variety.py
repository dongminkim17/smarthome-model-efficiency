#!/usr/bin/env python3
"""32 fixed-architecture MER recoveries plus exact checkpoint averaging on Joy."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from launch_pruning_followup import checkpoint_identity,pin_best
from project_queue import validate_submission
from project_queue_jobs import entry
from queue_ffn_expansion import training_chain
from queue_mer_focus import REFERENCES,build_submission as validate_reference_contract
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.queueing import atomic_json,fingerprint,read_json

FAMILY='mer_recovery_variety_20261104'


def definitions():
    recipes=[
      ('CONTROL',{},90),
      ('BROAD_ENC1E5',{'lora_target':'decoder_encoder_linear'},110),
      ('BROAD_ENC3E5',{'lora_target':'decoder_encoder_linear','encoder_learning_rate':3e-5},110),
      ('DECAY025',{'distillation_alpha_final':.25},105),
      ('DECAY000',{'distillation_alpha_final':0.},100),
      ('DROP005',{'lora_dropout':.05},105),
      ('DROP010',{'lora_dropout':.1},95),
      ('BROAD_DECAY025',{'lora_target':'decoder_encoder_linear','distillation_alpha_final':.25},110),
      ('BROAD_DROP005',{'lora_target':'decoder_encoder_linear','lora_dropout':.05},110),
      ('DECAY025_DROP005',{'distillation_alpha_final':.25,'lora_dropout':.05},105),
      ('DECAY025_FEATURE003',{'distillation_alpha_final':.25,'feature_audio_weight':.03,'feature_vision_weight':.03},100),
      ('GOLD_FEATURE_CONTROL',{'distillation_alpha':0.},90),
    ]
    variants=[]
    for rank in (16,64):
        for label,changes,priority in recipes:
            variants.append((f'FRESH_R{rank}_{label}',rank,False,dict(changes,epochs=5),priority))
        for label,changes in [
          ('KD025',{'distillation_alpha':.25}),
          ('DECAY000',{'distillation_alpha_final':0.}),
          ('GOLD_FEATURE_CONTROL',{'distillation_alpha':0.}),
          ('DECAY025_FEATURE003',{'distillation_alpha_final':.25,'feature_audio_weight':.03,'feature_vision_weight':.03}),
        ]:
            variants.append((f'CONT_R{rank}_{label}',rank,True,
                             dict(changes,epochs=3,learning_rate=1e-5,encoder_learning_rate=1e-6),40))
    return variants


def build_submission(legacy,queue_root,root=ROOT):
    root=Path(root);queue_root=Path(queue_root)
    validate_reference_contract(legacy,queue_root,root)
    by_id={j['id']:j for j in legacy['jobs']};refs={rank:by_id[key] for rank,key in REFERENCES.items()}
    cfg=copy.deepcopy(refs[16]['payload']['config']);teacher=cfg['teacher_checkpoint']
    if cfg.get('mer_partition') or cfg['task']!='mer6':raise ValueError('Use the original80/20 MER protocol')
    feature_hash=sha256_file(Path(cfg['feature_teacher_cache'])/'cache.json');class_hash=sha256_file(Path(cfg['teacher_cache']))
    counts=read_json(Path(cfg['model'])/'compression.json')['parameters_after']
    base=root/'artifacts/full'/FAMILY;manifest=base/'workflow.json'
    stress=queue_root/'stress_cases/mer6_variety_80_20.json'
    prepare=FAMILY+'/prepare:mer6'
    jobs=[entry(prepare,'prepare',{'id':'prepare:mer6','kind':'prepare','task':'mer6',
                 'output':str(stress),'config':copy.deepcopy(cfg),'estimated_seconds':180},
                 {'manifest':str(manifest),'field':'preparation_jobs','key_field':'id','key':'prepare:mer6'},
                 resource='cpu',priority=150)]
    averages=[];plans=[]
    def average_job(arm,config,training=None,source_run=None,priority=20):
        key='average:mer6_'+arm.lower();path=base/'selection'/('mer6_'+arm.lower())
        payload={'id':key,'name':'mer6_'+arm.lower(),'arm':arm,'kind':'checkpoint_average','task':'mer6',
                 'output':str(path),'training_job':training,'source_run':source_run,'dev_count':675,
                 'estimated_seconds':480,'architecture':'BALANCED','expected_parameters':counts}
        if source_run:payload['source_identity']=checkpoint_identity(pin_best(source_run))
        avg=entry(FAMILY+'/'+key,'checkpoint_average',payload,
                  {'manifest':str(manifest),'field':'selection_jobs','key_field':'id','key':key},
                  [training] if training else [],priority=priority)
        averages.append(avg['id']);jobs.append(avg)
        return avg
    for arm,rank,continuation,changes,priority in definitions():
        new=copy.deepcopy(cfg)
        new.update(lora_rank=rank,lora_alpha=2*rank,lora_dropout=0.,lora_target='decoder_encoder_ffn',
          seed=42,epochs=5,learning_rate=1e-4,encoder_learning_rate=1e-5,
          objective='sft_kd',distillation_alpha=.75,distillation_temperature=2.,
          feature_audio_weight=.1,feature_vision_weight=.1,
          train_limit=None,dev_limit=None,eval_modes=['logit'],selection_mode='logit',eval_every_fraction=.5,
          micro_batch_size=2,gradient_accumulation_steps=2,gradient_checkpointing=True)
        new.update(changes)
        if 'distillation_alpha_final' in new:new['distillation_decay_start']=.2
        chain=training_chain(FAMILY,manifest,queue_root,root,'mer6',arm,new,teacher,[prepare])
        parent=pin_best(refs[rank]['payload']['run_dir']) if continuation else None
        tests=[j for j in chain if j['kind']=='recovery_test'];train=next(j for j in chain if j['kind']=='recovery_train')
        for job in chain:
            p=job['payload'];p.update(architecture='BALANCED',budget_scope='per_modality',expected_parameters=counts,
                                     group='continuation' if continuation else 'fresh',
                                     reference_two_stage_train=REFERENCES[rank])
            if job['kind'] in ('smoke_train','recovery_train'):
                job['priority']=priority+(10 if job['kind']=='smoke_train' else 0)
                p.update(expected_feature_cache_sha256=feature_hash,expected_teacher_cache_sha256=class_hash,
                  estimated_seconds=180 if job['kind']=='smoke_train' else
                      (2.4 if continuation else 4.3 if new['lora_target']=='decoder_encoder_linear' else 3.9)*3600,
                  initialization='Original balanced best-dev adapter; fresh optimizer/schedule' if continuation else
                    'Fresh final-recovery adapters on the fixed sequential balanced export')
                if job['kind']=='smoke_train':p['memory_cases']=str(stress)
                if parent:
                    meta=read_json(Path(parent)/'metadata.json')
                    p.update(init_adapter=str(Path(parent)/'adapter'),parent_checkpoint=parent,parent_step=meta['global_step'],
                             parent_epoch=meta['global_step']/675,init_checkpoint_identity=checkpoint_identity(parent))
            elif job['kind']=='smoke_test':job['priority']=priority+13
        jobs.extend(j for j in chain if j['kind']!='recovery_test')
        avg=average_job(arm,new,training=train['id'])
        for test in tests:
            test['deps']=[avg['id'],FAMILY+'/shortlist:mer6'];test['payload']['shortlist_job']=FAMILY+'/shortlist:mer6'
            jobs.append(test)
        plans.append({'arm':arm,'rank':rank,'continuation':continuation,'epochs':new['epochs'],
                      'scope':new['lora_target'],'dropout':new['lora_dropout'],
                      'class_kd_initial':new['distillation_alpha'],'class_kd_final':new.get('distillation_alpha_final'),
                      'feature_weights':[new['feature_audio_weight'],new['feature_vision_weight']],
                      'learning_rate':new['learning_rate'],'encoder_learning_rate':new['encoder_learning_rate']})
    for rank,ref in refs.items():
        arm=f'HISTORICAL_R{rank}_AVERAGE'
        avg=average_job(arm,ref['payload']['config'],source_run=ref['payload']['run_dir'],priority=120)
        key='test:mer6_'+arm.lower()
        jobs.append(entry(FAMILY+'/'+key,'recovery_test',{'id':key,'kind':'test','name':'mer6_'+arm.lower(),
            'arm':arm,'task':'mer6','output':str(base/'test'/('mer6_'+arm.lower())),
            'split':'test','limit':None,'modes':['logit','greedy'],'shortlist_job':FAMILY+'/shortlist:mer6'},
            {'manifest':str(manifest),'field':'jobs','key_field':'id','key':key},
            [avg['id'],FAMILY+'/shortlist:mer6'],priority=35))
    jobs.append(entry(FAMILY+'/shortlist:mer6','shortlist',{'id':'shortlist:mer6','kind':'shortlist',
        'task':'mer6','top_k':8,'dev_count':675,'output':str(base/'validation_shortlist.json')},
        {'manifest':str(manifest),'field':'preparation_jobs','key_field':'id','key':'shortlist:mer6'},
        averages,resource='inline',priority=150))
    document={'format_version':1,'status':'waiting','phase':'per_job_dependencies','host':'joy',
        'allowed_gpus':list(range(8)),'description':'32 MER recovery runs and34 validation-only checkpoint-average selections',
        'jobs':[],'smoke_jobs':[],'selection_jobs':[],'preparation_jobs':[],
        'checkpoint_selection_uses_test':False,'shortlist_uses_test':False,'official_test_top_k':8,
        'experiment_plan':plans,'data_counts':{'train':2698,'dev':675,'official_test':834},
        'estimated_wall_hours':[15,18],
        'note':'Same balanced architecture and two-stage pipeline. MER screens shared recovery-recipe variants; adoption requires checking both MIntRec settings. Class-KD-off arms are labelled ablations and retain feature guidance. Exact update averaging changes no merged dense dimensions.'}
    for job in jobs:document[job['location']['field']].append(copy.deepcopy(job['payload']))
    return {'schema_version':1,'jobs':jobs,'documents':{str(manifest):document}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/joy')
    parser.add_argument('--source-queue',type=Path,default=ROOT/'artifacts/queue/suzy')
    parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    state=read_json(args.queue_root/'state.json')
    if state.get('host')!='joy' or state['allowed_gpus']!=list(range(8)):raise ValueError('Joy0–7 resource contract required')
    path=args.queue_root/'submissions'/(FAMILY+'.json')
    spec=read_json(path) if path.exists() else build_submission(read_json(args.source_queue/'state.json'),args.queue_root)
    validate_submission(state,spec)
    if args.submit:
        atomic_json(path,spec);atomic_json(args.queue_root/'inbox'/(fingerprint(spec)+'.json'),spec)
    print(json.dumps({'submitted':args.submit,'family':FAMILY,'jobs':len(spec['jobs']),
        'training_runs':sum(j['kind']=='recovery_train' for j in spec['jobs']),
        'average_selection_jobs':sum(j['kind']=='checkpoint_average' for j in spec['jobs']),
        'plan':next(iter(spec['documents'].values()))['experiment_plan']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
