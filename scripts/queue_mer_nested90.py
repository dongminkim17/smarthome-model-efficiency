#!/usr/bin/env python3
"""Twelve balanced MER recoveries, nested 90/10 holdout, about eight hours on Joy."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.cache_extension import seed_extension
from smarthome_efficiency.data import load_records
from smarthome_efficiency.queueing import atomic_json, fingerprint, read_json
from smarthome_efficiency.training import config_fingerprint
from launch_pruning_followup import checkpoint_identity, pin_best
from project_queue import validate_submission
from project_queue_jobs import entry
from queue_ffn_expansion import training_chain
from queue_mer_focus import REFERENCES, build_submission as validate_original_contract

FAMILY='mer_nested90_20261104'
PARTITION='nested_90_10'


def definitions():
    variants=[]
    for rank in (16,64):
        for kd in (.75,.5):
            for enc_lr,label in ((1e-5,'ENC1E5'),(3e-5,'ENC3E5')):
                variants.append((f'FRESH_R{rank}_KD{int(kd*100):03d}_{label}',rank,False,
                                 {'epochs':5,'learning_rate':1e-4,'encoder_learning_rate':enc_lr,
                                  'distillation_alpha':kd},100))
    for rank in (16,64):
        for lm_lr,enc_lr,label in ((3e-5,1e-5,'LM3E5_ENC1E5'),(1e-5,3e-6,'LM1E5_ENC3E6')):
            variants.append((f'CONT_R{rank}_{label}',rank,True,
                             {'epochs':3,'learning_rate':lm_lr,'encoder_learning_rate':enc_lr,
                              'distillation_alpha':.75},40))
    return variants


def split_manifest(cfg):
    args={'seed':cfg['split_seed'],'data_root':cfg['data_root']}
    old_train=load_records('mer6','train',**args)
    old_dev=load_records('mer6','dev',**args)
    train=load_records('mer6','train',mer_partition=PARTITION,**args)
    dev=load_records('mer6','dev',mer_partition=PARTITION,**args)
    ids=lambda rows:{r['id'] for r in rows}
    if ((len(old_train),len(old_dev),len(train),len(dev))!=(2698,675,3035,338)
            or not ids(old_train)<=ids(train) or not ids(dev)<=ids(old_dev)
            or ids(train)&ids(dev) or ids(train)|ids(dev)!=ids(old_train)|ids(old_dev)):
        raise ValueError('Nested MER partition violated historical holdout isolation')
    transferred=ids(train)-ids(old_train)
    doc={'partition':PARTITION,'split_seed':cfg['split_seed'],'test_set_used':False,
         'selection':'Stratified fixed-seed subset of historical development IDs, independent of predictions',
         'parent_train_ids':sorted(ids(old_train)),'parent_dev_ids':sorted(ids(old_dev)),
         'train_ids':sorted(ids(train)),'dev_ids':sorted(ids(dev)),
         'transferred_to_train_ids':sorted(transferred),
         'counts':{'parent_train':2698,'parent_dev':675,'train':3035,'dev':338,'transferred':337},
         'class_counts':{'train':dict(Counter(r['label'] for r in train)),
                         'dev':dict(Counter(r['label'] for r in dev))},
         'parent_rule':'sorted official Train&Val, sklearn stratified test_size=.2, seed42',
         'retained_rule':'sorted parent dev, stratified test_size=ceil(3373*.1)=338, seed42'}
    doc['partition_sha256']=config_fingerprint(doc)
    return doc,train


def build_submission(legacy,queue_root,root=ROOT):
    # Read-only reuse of the original batch's full export/teacher/reference checks.
    validate_original_contract(legacy,queue_root,root)
    by_id={j['id']:j for j in legacy['jobs']}
    refs={rank:by_id[key] for rank,key in REFERENCES.items()}
    cfg=copy.deepcopy(refs[16]['payload']['config'])
    split,_=split_manifest(cfg)
    base=Path(root)/'artifacts/full'/FAMILY
    manifest=base/'workflow.json'
    cache_root=Path(root)/'artifacts/teacher_cache'/FAMILY
    teacher=cfg['teacher_checkpoint']
    cfg.update(mer_partition=PARTITION,teacher_cache=str(cache_root/'class/cache.json'),
               feature_teacher_cache=str(cache_root/'features'))
    jobs=[]
    def location(key):return {'manifest':str(manifest),'field':'preparation_jobs','key_field':'id','key':key}
    for kind,label,output in (('followup','class',cache_root/'class'),('encoder_cache','features',cache_root/'features')):
        key='cache:'+label
        payload={'id':key,'task':'mer6','kind':'teacher' if label=='class' else 'encoder_cache',
                 'output':str(output),'teacher_checkpoint':teacher,'source_identity':checkpoint_identity(teacher),
                 'mer_partition':PARTITION,'expected_examples':3035,'estimated_seconds':900}
        jobs.append(entry(FAMILY+'/'+key,kind,payload,location(key),priority=150))
    stress=Path(queue_root)/'stress_cases'/('mer6_'+PARTITION+'.json')
    prep_id=FAMILY+'/prepare:mer6'
    jobs.append(entry(prep_id,'prepare',{'id':'prepare:mer6','task':'mer6','kind':'prepare',
        'output':str(stress),'config':copy.deepcopy(cfg),'estimated_seconds':900},location('prepare:mer6'),
        resource='cpu',priority=150))
    summaries=[]
    counts=read_json(Path(cfg['model'])/'compression.json')['parameters_after']
    for arm,rank,continuation,changes,priority in definitions():
        new=copy.deepcopy(cfg)
        new.update(lora_rank=rank,lora_alpha=2*rank,seed=42,use_transcript=False,
                   eval_modes=['logit'],selection_mode='logit',eval_every_fraction=.5,
                   train_limit=None,dev_limit=None,micro_batch_size=2,gradient_accumulation_steps=2,
                   gradient_checkpointing=True,distillation_temperature=2.,
                   feature_audio_weight=.1,feature_vision_weight=.1)
        new.update(changes)
        chain=training_chain(FAMILY,manifest,queue_root,root,'mer6',arm,new,teacher,
                             [prep_id,FAMILY+'/cache:class',FAMILY+'/cache:features'])
        parent=pin_best(refs[rank]['payload']['run_dir']) if continuation else None
        if continuation and not parent:raise ValueError('Warm start needs a pinned validation-selected adapter')
        metadata=read_json(Path(parent)/'metadata.json') if parent else None
        for job in chain:
            p=job['payload']
            p.update(architecture='BALANCED',budget_scope='per_modality',expected_parameters=counts,
                     group='nested90_continuation' if continuation else 'nested90_fresh',
                     reference_two_stage_train=REFERENCES[rank],recovery_epoch_budget=new['epochs'],
                     data_partition_sha256=split['partition_sha256'])
            if job['kind'] in ('smoke_train','recovery_train'):
                job['priority']=priority+(10 if job['kind']=='smoke_train' else 0)
                p.update(pin_generated_caches=True,expected_cache_examples=3035,
                         initialization='Historical balanced validation-selected adapter; new optimizer/schedule'
                            if continuation else 'Fresh adapter on fixed sequential balanced export',
                         estimated_seconds=240 if job['kind']=='smoke_train' else (2.8 if continuation else 4.3)*3600)
                if job['kind']=='smoke_train':p['memory_cases']=str(stress)
                if parent:
                    p.update(init_adapter=str(Path(parent)/'adapter'),parent_checkpoint=parent,
                             parent_step=metadata['global_step'],parent_epoch=metadata['global_step']/675,
                             init_checkpoint_identity=checkpoint_identity(parent))
            elif job['kind']=='smoke_test':job['priority']=priority+13
            else:
                p['shortlist_job']=FAMILY+'/shortlist:mer6'
                job['deps'].append(FAMILY+'/shortlist:mer6')
        jobs.extend(chain)
        summaries.append({'arm':arm,'rank':rank,'continuation':continuation,'epochs':new['epochs'],
            'learning_rate':new['learning_rate'],'encoder_learning_rate':new['encoder_learning_rate'],
            'class_kd':new['distillation_alpha'],'audio_feature_kd':.1,'vision_feature_kd':.1,
            'parent_checkpoint':parent,'train_count':3035,'dev_count':338})
    jobs.append(entry(FAMILY+'/shortlist:mer6','shortlist',
        {'id':'shortlist:mer6','task':'mer6','kind':'shortlist','top_k':4,'dev_count':338,
         'output':str(base/'validation_shortlist.json')},location('shortlist:mer6'),
        [j['id'] for j in jobs if j['kind']=='recovery_train'],resource='inline',priority=150))
    doc={'format_version':1,'status':'waiting','phase':'per_job_dependencies','host':'joy',
         'description':'12 balanced MER-only recoveries; nested 3035/338 train/dev; original 834 test',
         'allowed_gpus':list(range(8)),'jobs':[],'smoke_jobs':[],'preparation_jobs':[],
         'checkpoint_selection_uses_test':False,'shortlist_uses_test':False,'official_test_top_k':4,
         'data_partition':split,'experiment_plan':summaries,
         'estimated_wall_hours':[7,9],
         'note':'Fixed balanced geometry. Added337 old dev IDs join training. Original teacher/base unchanged. Historical models never trained on retained338 dev IDs. Fresh runs scheduled before shorter continuations.'}
    for job in jobs:doc[job['location']['field']].append(copy.deepcopy(job['payload']))
    return {'schema_version':1,'jobs':jobs,'documents':{str(manifest):doc}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/joy')
    parser.add_argument('--source-queue',type=Path,default=ROOT/'artifacts/queue/suzy')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    legacy=read_json(args.source_queue/'state.json')
    spec=build_submission(legacy,args.queue_root)
    state=read_json(args.queue_root/'state.json');validate_submission(state,spec)
    if state.get('host')!='joy' or state['allowed_gpus']!=list(range(8)):
        raise ValueError('This batch requires the authorized Joy0-7 queue')
    if args.submit:
        by_id={j['id']:j for j in legacy['jobs']}
        cfg=by_id[REFERENCES[16]]['payload']['config']
        split,records=split_manifest(cfg)
        base=ROOT/'artifacts/full'/FAMILY
        path=base/'partition.json'
        if path.exists() and read_json(path)!=split:raise ValueError('Cannot replace a saved data partition')
        atomic_json(path,split)
        caches=ROOT/'artifacts/teacher_cache'/FAMILY
        print(json.dumps({'stage':'seeding_training_targets','class':seed_extension(cfg['teacher_cache'],caches/'class',records,PARTITION)},indent=2),flush=True)
        print(json.dumps({'stage':'seeding_training_targets','features':seed_extension(cfg['feature_teacher_cache'],caches/'features',records,PARTITION,features=True)},indent=2),flush=True)
        saved=args.queue_root/'submissions'/(FAMILY+'.json')
        if saved.exists() and read_json(saved)!=spec:raise ValueError('Cannot replace an existing immutable submission')
        atomic_json(saved,spec)
        atomic_json(args.queue_root/'inbox'/(fingerprint(spec)+'.json'),spec)
    print(json.dumps({'submitted':args.submit,'family':FAMILY,'training_runs':12,
        'counts':next(iter(spec['documents'].values()))['data_partition']['counts'],
        'plans':next(iter(spec['documents'].values()))['experiment_plan']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
