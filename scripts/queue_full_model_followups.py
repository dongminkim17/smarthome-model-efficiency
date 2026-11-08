#!/usr/bin/env python3
"""Approved 27-run A/D factorial recovery comparison and rank/LR controls."""
from __future__ import annotations
import argparse,copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import read_json,atomic_json,fingerprint
from smarthome_efficiency.training import config_fingerprint
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.hybrid_pruning import D_COUNTS
from project_queue_jobs import entry
from project_queue import validate_submission
from queue_ffn_expansion import training_chain
from launch_pruning_followup import checkpoint_identity

FAMILY='full_model_followups_20261020'
TASKS=('mintrec2','mer6','mintrec20')
# arch, broad encoder scope, feature KD, rank, LR. A/default is the existing batch.
ARMS=(
 ('A_WIDE_R16','A',True,False,16,1e-4),
 ('A_FEATURE_R16','A',False,True,16,1e-4),
 ('A_WIDE_FEATURE_R16','A',True,True,16,1e-4),
 ('D_BASE_R16','D',False,False,16,1e-4),
 ('D_WIDE_R16','D',True,False,16,1e-4),
 ('D_FEATURE_R16','D',False,True,16,1e-4),
 ('D_WIDE_FEATURE_R16','D',True,True,16,1e-4),
 ('A_WIDE_R64','A',True,False,64,1e-4),
 ('A_WIDE_LR3E5','A',True,False,16,3e-5),
)


def build_submission(state,root=ROOT):
    root=Path(root);manifest=root/'artifacts/full'/FAMILY/'workflow.json';by={j['id']:j for j in state['jobs']};jobs=[]
    def location(field,key):return {'manifest':str(manifest),'field':field,'key_field':'id','key':key}
    for task in TASKS:
        a_id='full_model_2b_20261020/export:'+task+'_a_lm1536_a1024_v1024'
        a=by[a_id];p=a['payload'];side=read_json(Path(p['output'])/'compression.json')
        if (a['status']!='completed' or side['budget_scope']!='total' or side['method']!='multimodal_ffn_activation'
                or side['parameters_after']['total']!=1978160384 or not side['verification']['passed']):
            raise ValueError('Expected verified complete-model A export: '+task)
        source=side['source_checkpoint'];source_cfg=read_json(Path(source)/'metadata.json')['config']
        teacher=source_cfg['teacher_checkpoint'];cache=source_cfg['teacher_cache']
        teacher_id='calibrated_pruning_20261019/teacher:'+task
        oldcache=read_json(Path(cache))
        if (by[teacher_id]['status']!='completed' or oldcache['spec']['teacher_checkpoint']!=teacher
                or oldcache['spec']['split']!='train' or oldcache['spec']['test_set_used'] is not False):
            raise ValueError('Original class-logit teacher cache is not valid')
        feature_id=FAMILY+'/encoder_cache:'+task;feature_path=root/'artifacts/cache'/FAMILY/'encoder_features'/task
        payload={'id':'encoder_cache:'+task,'task':task,'output':str(feature_path),'teacher_checkpoint':teacher,
                 'source_identity':checkpoint_identity(teacher),'expected_examples':2698 if task=='mer6' else 1334,
                 'estimated_seconds':1800 if task=='mer6' else 900}
        jobs.append(entry(feature_id,'encoder_cache',payload,location('preparation_jobs',payload['id']),[teacher_id],priority=2))
        d_name=task+'_d_lm2048_a24x1536_v24x1280';d_id=FAMILY+'/export:'+d_name;d_path=root/'artifacts/compact'/FAMILY/d_name
        request={k:side[k] for k in ['source_checkpoint','source_adapter_sha256','source_metadata_sha256']}
        request.update(method='multimodal_depth_ffn_activation',budget_scope='total',keep_layers=None,
                       ffn_width=2048,audio_ffn_width=1536,vision_ffn_width=1280,audio_layers=24,vision_layers=24,
                       calibration_samples=128,calibration_seed=42,
                       calibration_cache=str(root/'artifacts/compact'/FAMILY/'calibration'/f'{task}_hybrid_n128.json'))
        payload={'id':'export:'+d_name,'task':task,'output':str(d_path),'request':request,
                 'source_identity':checkpoint_identity(source),'source_base_manifest_sha256':sha256_file(Path(source_cfg['model'])/'compression.json'),
                 'generate_calibration':True,'calibration_sha256':None,'expected_parameters':D_COUNTS,
                 'expected_language_parameters':D_COUNTS['language'],'estimated_seconds':600}
        jobs.append(entry(d_id,'prune_export',payload,location('preparation_jobs',payload['id']),[a_id],priority=2))
        reference=by['full_model_2b_20261020/train:'+task+'_a_lm1536_a1024_v1024']['payload']['config']
        for arm,arch,broad,feature,rank,lr in ARMS:
            cfg=copy.deepcopy(reference)
            cfg.update(model=p['output'] if arch=='A' else str(d_path),epochs=5,seed=42,
                       lora_rank=rank,lora_alpha=rank*2,lora_dropout=0.,
                       lora_target='decoder_encoder_linear' if broad else 'decoder_encoder_ffn',
                       learning_rate=lr,micro_batch_size=2,gradient_accumulation_steps=2,gradient_checkpointing=True,
                       train_limit=None,dev_limit=None,objective='sft_kd',teacher_checkpoint=teacher,teacher_cache=cache,
                       distillation_alpha=.75,distillation_temperature=2.,eval_modes=['logit','greedy'],selection_mode='logit')
            deps=[a_id if arch=='A' else d_id,'prepare:'+task,teacher_id]
            if feature:
                cfg.update(feature_teacher_cache=str(feature_path),feature_audio_weight=.1,feature_vision_weight=.1)
                deps.append(feature_id)
            chain=training_chain(FAMILY,manifest,state['queue_root'],root,task,arm,cfg,source,deps)
            for job in chain:
                if job['kind'] in ('smoke_train','recovery_train'):
                    job['priority']=2
                    job['payload'].update(architecture=arch,budget_scope='total',expected_parameters=side['parameters_after'] if arch=='A' else D_COUNTS,
                        initialization=f'initial verified {arch} base from merged task KD075 source; fresh rank-{rank} adapter; original teacher',
                        estimated_seconds=240 if job['kind']=='smoke_train' else (22000 if task=='mer6' else 10800))
            jobs.extend(chain)
    document={'format_version':1,'status':'waiting','phase':'per_job_dependencies',
              'description':'27 approved five-epoch task runs: A/D factorial encoder-scope x feature-KD comparison, plus A rank/LR controls',
              'jobs':[],'smoke_jobs':[],'preparation_jobs':[],'checkpoint_selection_uses_test':False,
              'existing_source_selections_unchanged':True,'feature_cache_split':'train',
              'parameter_scope':'complete deployed Thinker after LoRA merge; LM + audio + vision; no speech generator'}
    for job in jobs:document[job['location']['field']].append(copy.deepcopy(job['payload']))
    spec={'schema_version':1,'jobs':jobs,'documents':{str(manifest):document}}
    validate_submission(state,spec);return spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/suzy');parser.add_argument('--submit',action='store_true')
    args=parser.parse_args();state=read_json(args.queue_root/'state.json');saved=args.queue_root/'submissions'/f'{FAMILY}.json'
    if saved.exists():spec=read_json(saved);validate_submission(state,spec)
    else:spec=build_submission(state);atomic_json(saved,spec)
    if args.submit:
        path=args.queue_root/'inbox'/(fingerprint(spec)+'.json');atomic_json(path,spec)
        print('Submitted 27 training runs and 87 supporting preparation/smoke/test stages: '+str(path))
    else:print('Validated 27-run submission: '+str(saved))
    return 0

if __name__=='__main__':raise SystemExit(main())
