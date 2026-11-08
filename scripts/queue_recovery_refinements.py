#!/usr/bin/env python3
"""Approved balanced-base recovery controls; six variants on all three tasks."""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import atomic_json,read_json,fingerprint
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.multimodal_pruning import expected_counts
from project_queue_jobs import validate_export
from project_queue import validate_submission
from queue_ffn_expansion import training_chain

FAMILY='recovery_refinements_20261022'
TASKS=('mintrec2','mer6','mintrec20')
# Highest-probability checks first. Probes precede full training, so rank64 is
# checked under the same stress/reload gate before its lower-priority full run.
VARIANTS=(
 ('BALANCED_SEED43',False,{'seed':43},30),
 ('BALANCED_SEED44',False,{'seed':44},30),
 ('BALANCED_R32',False,{'lora_rank':32,'lora_alpha':64},30),
 ('BALANCED_R64',False,{'lora_rank':64,'lora_alpha':128},25),
 ('BALANCED_ENCLR3E5',False,{'encoder_learning_rate':3e-5},25),
 ('JOINT_R32_E10',True,{'lora_rank':32,'lora_alpha':64,'epochs':10},20),
)

def build_submission(state,root=ROOT,*,family=FAMILY,variants=VARIANTS,binary_eval_fraction=None):
 root=Path(root);manifest=root/'artifacts/full'/family/'workflow.json'
 by={j['id']:j for j in state['jobs']};jobs=[];counts=expected_counts(4352,3328,2048)
 for task in TASKS:
  ref_id='encoder_budget_20261021/train:'+task+'_balanced_feature_splitlr'
  ref=by[ref_id];cfg=ref['payload']['config'];source=Path(cfg['teacher_checkpoint'])
  teacher_id='calibrated_pruning_20261019/teacher:'+task
  feature_id='full_model_followups_20261020/encoder_cache:'+task
  seq_id='encoder_budget_20261021/export:'+task+'_balanced'
  joint_id='joint_pruning_20261022/export:'+task
  export_ids=(seq_id,joint_id) if any(v[1] for v in variants) else (seq_id,)
  for key in (ref_id,teacher_id,feature_id,*export_ids):
   if by[key]['status']!='completed':raise ValueError('Required source is not completed: '+key)
  expected={'epochs':5,'seed':42,'split_seed':42,'lora_target':'decoder_encoder_ffn',
   'lora_rank':16,'lora_alpha':32,'lora_dropout':0.,'learning_rate':1e-4,'encoder_learning_rate':1e-5,
   'objective':'sft_kd','distillation_alpha':.75,'distillation_temperature':2.,
   'feature_audio_weight':.1,'feature_vision_weight':.1,'selection_mode':'logit',
   'micro_batch_size':2,'gradient_accumulation_steps':2,'use_transcript':task!='mer6'}
  if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('Successful recovery recipe changed: '+task)
  if set(cfg['eval_modes'])!={'logit','greedy'}:raise ValueError('Both evaluation modes required')
  report=read_json(Path(ref['payload']['run_dir'])/'report.json')
  if report['status']!='completed' or report['config']!=cfg or report.get('test_set_used') is not False:
   raise ValueError('Reference must be complete validation-only training')
  for key in export_ids:
   validate_export(by[key]['payload'])
   side=read_json(Path(by[key]['payload']['output'])/'compression.json')
   if side['parameters_after']!=counts:raise ValueError('Matched balanced architecture required')
  if cfg['model']!=by[seq_id]['payload']['output']:raise ValueError('Reference base differs from balanced export')
  metadata=read_json(source/'metadata.json');base=Path(metadata['config']['model'])
  feature_path=Path(cfg['feature_teacher_cache'])/'cache.json'
  cache=read_json(Path(cfg['teacher_cache']));features=read_json(feature_path)
  for value,spec in ((cache,cache['spec']),(features,features['spec']['teacher'])):
   if (value['status']!='completed' or spec['split']!='train' or spec['test_set_used'] is not False
       or spec['teacher_checkpoint']!=str(source)
       or spec['teacher_metadata_sha256']!=sha256_file(source/'metadata.json')
       or spec['teacher_adapter_sha256']!=sha256_file(source/'adapter/adapter_model.safetensors')
       or spec['teacher_base_config_sha256']!=sha256_file(base/'config.json')):
    raise ValueError('Original training-only teacher cache changed')
  for arm,joint,changes,priority in variants:
   export_id=joint_id if joint else seq_id
   newcfg=copy.deepcopy(cfg);newcfg.update(model=by[export_id]['payload']['output']);newcfg.update(changes)
   if task=='mintrec2' and binary_eval_fraction is not None:newcfg['eval_every_fraction']=binary_eval_fraction
   extra_eval_seconds=0
   if newcfg['eval_every_fraction']!=cfg['eval_every_fraction']:
    import math,statistics
    events=[__import__('json').loads(line) for line in (Path(ref['payload']['run_dir'])/'metrics.jsonl').read_text().splitlines()]
    times=[e['seconds'] for e in events if e.get('event')=='validation' and e.get('seconds')]
    if not times:raise ValueError('Validation timings required for denser evaluation ETA')
    extra_eval_seconds=max(0,math.ceil(newcfg['epochs']/newcfg['eval_every_fraction'])-math.ceil(newcfg['epochs']/cfg['eval_every_fraction']))*statistics.median(times)
   chain=training_chain(family,manifest,state['queue_root'],root,task,arm,newcfg,str(source),
    [export_id,teacher_id,feature_id,'prepare:'+task])
   for job in chain:
    job['payload'].update(architecture='BALANCED',budget_scope='separate',expected_parameters=counts,
     reference_two_stage_train=ref_id,base_export=export_id,recovery_epoch_budget=newcfg['epochs'])
    if job['kind'] in ('smoke_train','recovery_train'):
     job['priority']=42 if priority==30 else 41 if priority==25 else 40
     if job['kind']=='recovery_train':job['priority']=priority
     job['payload'].update(expected_feature_cache_sha256=sha256_file(feature_path),
      initialization=('original teacher joint-pruned base' if joint else 'LM-recovered then encoder-pruned base')+
       '; fresh rank'+str(newcfg['lora_rank'])+' decoder+encoder-FFN LoRA; no previous recovery adapter loaded',
      estimated_seconds=240 if job['kind']=='smoke_train' else report['total_elapsed_seconds']*newcfg['epochs']/5+extra_eval_seconds)
   jobs.extend(chain)
 doc={'format_version':1,'status':'waiting','phase':'per_job_dependencies',
  'description':'18 approved recovery refinements: seed43/44, rank32/64, encoderLR3e-5; joint rank32 ten epochs; all three tasks',
  'jobs':[],'smoke_jobs':[],'preparation_jobs':[],'checkpoint_selection_uses_test':False,
  'existing_source_selections_unchanged':True,'parameter_limits':{'language':2_000_000_000,'combined_encoders':1_000_000_000},
  'note':'Seeds vary only final recovery; split/calibration/source remain fixed. Independent fresh adapters; no new exports or teacher targets.'}
 for job in jobs:doc[job['location']['field']].append(copy.deepcopy(job['payload']))
 spec={'schema_version':1,'jobs':jobs,'documents':{str(manifest):doc}}
 validate_submission(state,spec);return spec

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/suzy');p.add_argument('--submit',action='store_true');args=p.parse_args()
 state=read_json(args.queue_root/'state.json');saved=args.queue_root/'submissions'/f'{FAMILY}.json'
 if saved.exists():spec=read_json(saved);validate_submission(state,spec)
 else:spec=build_submission(state);atomic_json(saved,spec)
 if args.submit:
  path=args.queue_root/'inbox'/(fingerprint(spec)+'.json');atomic_json(path,spec)
  print('Submitted18 recovery runs with36 stress/reload gates and18 official tests: '+str(path))
 else:print('Validated18-run submission: '+str(saved))
 return 0
if __name__=='__main__':raise SystemExit(main())
