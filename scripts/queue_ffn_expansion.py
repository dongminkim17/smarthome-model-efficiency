#!/usr/bin/env python3
"""Six approved activation-FFN configurations x three tasks, submitted additively."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from smarthome_efficiency.queueing import atomic_json,read_json,fingerprint
from smarthome_efficiency.training import config_fingerprint
from smarthome_efficiency.pruning import sha256_file
from launch_pruning_followup import checkpoint_identity
from project_queue_jobs import entry
from project_queue import validate_submission

FAMILY='ffn_expansion_20261020'
TASKS=('mintrec2','mer6','mintrec20')
VARIANTS=(
    ('W4352_SFT_LR3E5',4352,128,{'learning_rate':3e-5}),
    ('W4352_SFT_DROP005',4352,128,{'lora_dropout':.05}),
    ('W4352_KD_T4',4352,128,{'objective':'sft_kd','distillation_temperature':4.}),
    ('W4608_SFT_LR3E5',4608,128,{'learning_rate':3e-5}),
    ('W4608_SFT_DROP005',4608,128,{'lora_dropout':.05}),
    ('W4608_SFT_CAL512',4608,512,{}),
)


def training_chain(family,manifest,queue_root,root,task,arm,cfg,source,base_deps):
    """One immutable config -> own stress/reload gate -> full train -> official test."""
    output=Path(root)/'artifacts/full'/family;smoke=Path(root)/'artifacts/smoke'/family
    name=task+'_'+arm.lower()
    def location(field,key):
        return {'manifest':str(manifest),'field':field,'key_field':'id','key':key}
    train={'id':'train:'+name,'name':name,'task':task,'arm':arm,'kind':'train','status':'waiting',
           'output':str(output/'train'/name),'config':copy.deepcopy(cfg),
           'config_sha256':config_fingerprint(cfg),'init_adapter':None,'parent_checkpoint':None,
           'source_name':Path(cfg['model']).name,'source_checkpoint':source,
           'compact_model':cfg['model'],'training_epochs':cfg['epochs'],
           'initialization':'fresh rank-16 adapter on task-specific activation-pruned base; original teacher adapter merged before pruning'}
    probe=copy.deepcopy(train)
    probe.update(output=str(smoke/'train'/name),memory_cases=str(Path(queue_root)/'stress_cases'/f'{task}.json'),training_epochs=2)
    probe['config'].update(epochs=2,train_limit=24,dev_limit=20)
    probe['config_sha256']=config_fingerprint(probe['config'])
    probe_id=family+'/smoke_train:'+name
    jobs=[entry(probe_id,'smoke_train',probe,location('smoke_jobs',probe['id']),base_deps,priority=4)]
    for is_smoke in (True,False):
        parent=probe_id if is_smoke else family+'/train:'+name
        value={'id':'test:'+name,'name':name,'task':task,'arm':arm,'kind':'test','status':'waiting',
               'output':str((smoke if is_smoke else output)/'test'/name),
               'split':'dev' if is_smoke else 'test','limit':6 if is_smoke else None,'modes':['logit','greedy']}
        jobs.append(entry(family+('/smoke_test:' if is_smoke else '/test:')+name,
                    'smoke_test' if is_smoke else 'recovery_test',value,
                    location('smoke_jobs' if is_smoke else 'jobs',value['id']),[parent],priority=45 if is_smoke else 35))
    jobs.append(entry(family+'/train:'+name,'recovery_train',train,location('jobs',train['id']),
                      [family+'/smoke_test:'+name],priority=4))
    return jobs


def export_job(reference,sidecar,task,samples,root,manifest):
    """Reuse pinned 128-clip scores, or explicitly collect a new 512-clip cache."""
    root=Path(root)
    request={k:sidecar[k] for k in ('method','source_checkpoint','source_adapter_sha256',
             'source_metadata_sha256','calibration_samples','calibration_seed','calibration_cache')}
    request.update(ffn_width=4608,keep_layers=None,calibration_samples=samples)
    generate=samples!=128
    if generate:
        request['calibration_cache']=str(root/'artifacts/compact'/FAMILY/'calibration'/f'{task}_n{samples}_seed42.json')
    name=f'{task}_ffn4608_n{samples}';key='export:'+name
    value={'id':key,'task':task,'output':str(root/'artifacts/compact'/FAMILY/name),'request':request,
           'source_identity':checkpoint_identity(request['source_checkpoint']),
           'calibration_sha256':None if generate else sha256_file(Path(request['calibration_cache'])),
           'generate_calibration':generate,'expected_language_parameters':1_981_526_016,
           'estimated_seconds':1800 if generate else 180}
    return entry(FAMILY+'/'+key,'prune_export',value,
                 {'manifest':str(manifest),'field':'preparation_jobs','key_field':'id','key':key},priority=4)


def build_submission(state,root=ROOT):
    root=Path(root);manifest=root/'artifacts/full'/FAMILY/'workflow.json'
    by_id={j['id']:j for j in state['jobs']};jobs=[]
    for task in TASKS:
        reference_id=f'calibrated_pruning_20261019/sft:{task}_ffn_activation'
        reference=by_id[reference_id]['payload']
        sidecar=read_json(Path(reference['compact_model'])/'compression.json')
        if (sidecar['task']!=task or sidecar['method']!='ffn_activation' or sidecar['ffn_width']!=4352
                or sidecar['calibration_samples']!=128 or sidecar['calibration_seed']!=42
                or sidecar['parameters_after']['language']!=1_924_902_912 or not sidecar['verification']['passed']):
            raise ValueError('Expected the verified task-specific FFN-4352 reference with 128 calibration clips')
        teacher_id='calibrated_pruning_20261019/teacher:'+task
        teacher=by_id[teacher_id]
        cache_path=Path(teacher['payload']['output'])/'cache.json';cache=read_json(cache_path)
        if (teacher['status']!='completed' or cache['status']!='completed' or cache['spec']['split']!='train'
                or cache['spec']['test_set_used'] is not False
                or cache['spec']['teacher_checkpoint']!=sidecar['source_checkpoint']):
            raise ValueError('Original teacher cache is not valid for '+task)
        models={(4352,128):(reference['compact_model'],reference_id)}
        wide_id='ffn_width_4608_20261019/export:mintrec2_ffn_activation_4608'
        if task=='mintrec2' and wide_id in by_id:
            wide=by_id[wide_id]
            if wide['payload']['request']['source_checkpoint']!=sidecar['source_checkpoint']:
                raise ValueError('Existing wider export uses a different teacher')
            models[(4608,128)]=(wide['payload']['output'],wide_id)
        else:
            wide=export_job(reference,sidecar,task,128,root,manifest);jobs.append(wide)
            models[(4608,128)]=(wide['payload']['output'],wide['id'])
        calibrated=export_job(reference,sidecar,task,512,root,manifest);jobs.append(calibrated)
        models[(4608,512)]=(calibrated['payload']['output'],calibrated['id'])
        for arm,width,samples,changes in VARIANTS:
            model,parent=models[(width,samples)]
            cfg=copy.deepcopy(reference['config'])
            cfg.update(model=model,epochs=5,seed=42,lora_rank=16,lora_alpha=32,lora_dropout=0.,
                       lora_target='decoder_linear',micro_batch_size=2,gradient_accumulation_steps=2,
                       learning_rate=1e-4,train_limit=None,dev_limit=None,objective='sft',
                       teacher_checkpoint=None,teacher_cache=None,distillation_alpha=.5,distillation_temperature=2.,
                       eval_modes=['logit','greedy'],selection_mode='logit')
            cfg.update(changes)
            deps=[parent,'prepare:'+task]
            if cfg['objective']=='sft_kd':
                cfg.update(teacher_checkpoint=sidecar['source_checkpoint'],teacher_cache=str(cache_path));deps.append(teacher_id)
            jobs.extend(training_chain(FAMILY,manifest,state['queue_root'],root,task,arm,cfg,sidecar['source_checkpoint'],deps))
    document={'format_version':1,'status':'waiting','phase':'per_job_dependencies',
              'description':'Six activation-FFN configurations on each of three tasks; 18 fresh rank-16 five-epoch runs',
              'jobs':[],'smoke_jobs':[],'preparation_jobs':[],'checkpoint_selection_uses_test':False,
              'existing_source_selections_unchanged':True}
    for j in jobs:document[j['location']['field']].append(copy.deepcopy(j['payload']))
    spec={'schema_version':1,'jobs':jobs,'documents':{str(manifest):document}}
    validate_submission(state,spec)
    return spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/suzy')
    parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    state=read_json(args.queue_root/'state.json');saved=args.queue_root/'submissions'/f'{FAMILY}.json'
    if saved.exists():
        spec=read_json(saved);validate_submission(state,spec)
    else:
        spec=build_submission(state);atomic_json(saved,spec)
    if args.submit:
        path=args.queue_root/'inbox'/(fingerprint(spec)+'.json');atomic_json(path,spec)
        print('Submitted 18 training experiments and their preparation/smoke/test jobs: '+str(path))
    else:print('Validated 18-run submission: '+str(saved))
    return 0


if __name__=='__main__':raise SystemExit(main())
