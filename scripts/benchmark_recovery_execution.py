#!/usr/bin/env python3
"""Bounded training-only recovery throughput probes with class and encoder-feature KD."""
import argparse,gc,json,random,statistics,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import read_json,atomic_json
from smarthome_efficiency.data import load_records,LABELS
from smarthome_efficiency.training import configure_determinism,lora_targets,optimizer_groups
from smarthome_efficiency.model import PilotModel
from smarthome_efficiency.distillation import TeacherLogitCache,teacher_spec,distillation_kl
from smarthome_efficiency.feature_distillation import TeacherFeatureCache,feature_spec

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True,choices=['mintrec2','mintrec20','mer6']);p.add_argument('--output',type=Path,required=True);p.add_argument('--stress-only',action='store_true');args=p.parse_args()
 state=read_json(ROOT/'artifacts/queue/suzy/state.json')
 job=next(j for j in state['jobs'] if j['id']=='gentle_recovery_20261023/train:'+args.task+'_feature003')
 cfg=job['payload']['config'];configure_determinism(cfg);torch.set_num_threads(2)
 runner=PilotModel(cfg['model'],args.task,LABELS[args.task],cfg['cache_dir'],max_frames=cfg['max_frames'],max_pixels=cfg['max_pixels'],use_transcript=cfg['use_transcript'],gradient_checkpointing=True)
 records=load_records(args.task,'train',seed=cfg['split_seed'],data_root=cfg['data_root'],mer_partition=cfg.get('mer_partition'))
 cls=TeacherLogitCache(cfg['teacher_cache'],teacher_spec(cfg,cfg['teacher_checkpoint'],runner.labels,runner.label_token_ids,runner.versions),records)
 features=TeacherFeatureCache(cfg['feature_teacher_cache'],feature_spec(cfg,cfg['teacher_checkpoint'],runner.labels,runner.label_token_ids,runner.versions),records)
 ids=read_json(ROOT/'artifacts/queue/suzy/stress_cases'/f'{args.task}.json')['ids'];by_id={r['id']:r for r in records};stress=[by_id[key] for key in ids]
 assert len(stress)==8
 typical=random.Random(42).sample(records,8)
 groups={'typical_a':typical[:4],'typical_b':typical[4:],'largest_a':stress[:4],'largest_b':stress[4:]}
 if args.stress_only:groups={k:v for k,v in groups.items() if k.startswith('largest')}
 prepared={r['id']:runner.prepare(r) for group in groups.values() for r in group}
 runner.load_model();runner.enable_lora(rank=cfg['lora_rank'],alpha=cfg['lora_alpha'],dropout=cfg['lora_dropout'],target_modules=lora_targets(cfg['lora_target']))
 parameters=[p for p in runner.model.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(optimizer_groups(runner.model,cfg),lr=0,weight_decay=0)
 for group in optimizer.param_groups:group['lr']=0.
 result={'status':'running','task':args.task,'test_set_used':False,'gpu':torch.cuda.get_device_name(),'effective_batch':4,'cases':[],'groups':{k:[r['id'] for r in v] for k,v in groups.items()}}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 def save():atomic_json(args.output,result)
 def update(rows,micro):
  optimizer.zero_grad(set_to_none=True);total=0.;predictions=[]
  for pos in range(0,len(rows),micro):
   rr=rows[pos:pos+micro];b=[prepared[r['id']] for r in rr];targets=[runner.labels.index(r['label']) for r in rr]
   loss=(runner.loss_batch(b[0],targets[0],return_class_gradients=True,return_encoder_features=True) if len(rr)==1 else runner.loss_microbatch(b,targets,return_class_gradients=True,return_encoder_features=True))
   student=loss.pop('class_logits_for_kd');kd=distillation_kl(student,cls.batch(rr,runner.model.device),cfg['distillation_temperature'])
   fl=features.losses(loss.pop('encoder_features_for_kd'),rr)
   objective=(1-cfg['distillation_alpha'])*loss['sft_loss']+cfg['distillation_alpha']*kd+cfg['feature_audio_weight']*fl['audio']+cfg['feature_vision_weight']*fl['vision']
   if not torch.isfinite(objective):raise RuntimeError('Nonfinite objective')
   (objective*len(rr)/len(rows)).backward();total+=float(objective.detach())*len(rr)/len(rows);predictions.extend(student.detach().argmax(-1).cpu().tolist())
   del loss,objective,kd,fl,student,b
  from smarthome_efficiency.multimodal_pruning import adapter_gradient_coverage
  coverage=adapter_gradient_coverage(runner.model)
  if any(v['gradient_l2']<=0 for v in coverage.values()):raise RuntimeError('Missing component gradients')
  torch.nn.utils.clip_grad_norm_(parameters,1.)
  optimizer.step();torch.cuda.synchronize()
  return total,predictions
 save();reference={}
 for checkpointing,micro in ([(True,2),(True,4)] if args.stress_only else [(True,2),(True,4),(False,2),(False,4)]):
  if checkpointing:runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  else:runner.model.gradient_checkpointing_disable()
  optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  case={'micro_batch_size':micro,'gradient_accumulation_steps':4//micro,'gradient_checkpointing':checkpointing,'status':'running','groups':{}};result['cases'].append(case);save()
  try:
   for name,rows in groups.items():
    loss,pred=update(rows,micro)
    if checkpointing and micro==2:reference[name]=(loss,pred)
    ref=reference[name];times=[]
    for repeat in range(2):
     begin=time.perf_counter();update(rows,micro);times.append(time.perf_counter()-begin)
    case['groups'][name]={'mean_update_seconds':statistics.mean(times),'loss_difference':abs(loss-ref[0]),'prediction_agreement':sum(a==b for a,b in zip(pred,ref[1]))/len(pred),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    print(args.task,checkpointing,micro,name,case['groups'][name],flush=True);save()
   case['status']='passed' if all(v['loss_difference']<.03 and v['prediction_agreement']==1. for v in case['groups'].values()) else 'numerical_mismatch'
  except torch.cuda.OutOfMemoryError:
   case['status']='oom';print(args.task,checkpointing,micro,'OOM',flush=True)
  except Exception as e:case.update(status='failed',error=f'{type(e).__name__}: {e}');print(case,flush=True)
  optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();save()
 result['status']='completed';save()
 return 0
if __name__=='__main__':raise SystemExit(main())
