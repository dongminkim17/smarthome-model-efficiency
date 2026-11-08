#!/usr/bin/env python3
"""Bounded batch/checkpointing benchmarks on training clips; no experiment updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import av
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from smarthome_efficiency.data import LABELS,load_records
from smarthome_efficiency.model import PilotModel
from smarthome_efficiency.training import load_config,configure_determinism


def duration(record):
    with av.open(record['video']) as c:
        values=[float(c.duration/av.time_base)] if c.duration else []
        for stream in c.streams:
            if stream.duration and stream.time_base:
                values.append(float(stream.duration*stream.time_base))
        return max(values) if values else 0


def grad_vector(parameters):
    return torch.cat([p.grad.detach().float().cpu().flatten() for p in parameters if p.grad is not None])


def update(runner,optimizer,parameters,batches,targets,micro):
    optimizer.zero_grad(set_to_none=True)
    values=[]; logits=[]
    for start in range(0,len(batches),micro):
        group=batches[start:start+micro]; labels=targets[start:start+micro]
        r=(runner.loss_batch(group[0],labels[0]) if len(group)==1 else runner.loss_microbatch(group,labels))
        (r['loss']*len(group)/len(batches)).backward()
        values.append(float(r['loss'].detach())*len(group)/len(batches))
        logits.append(r['class_logits'].reshape(len(group),-1).cpu())
    gradients=grad_vector(parameters)
    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.0))
    optimizer.step()  # lr=0: allocate/use optimizer state without changing weights
    torch.cuda.synchronize()
    return {'loss':sum(values),'logits':torch.cat(logits),'gradient':gradients,'grad_norm':norm}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    assert not args.output.exists(),args.output
    torch.set_num_threads(2)
    summary={'status':'running','gpu':torch.cuda.get_device_name(),'tasks':{},'test_set_used':False}
    def save(): args.output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    save(); shared=None
    for task in ['mintrec20','mer6']:
        cfg=load_config(ROOT/'configs'/f'{task}.yaml');configure_determinism(cfg)
        random.seed(42);torch.manual_seed(42)
        runner=PilotModel(cfg['model'],task,LABELS[task],cfg['cache_dir'],max_frames=cfg['max_frames'],max_pixels=cfg['max_pixels'],use_transcript=cfg['use_transcript'])
        if shared is None:
            runner.load_model()
            runner.enable_lora(rank=16,alpha=32,dropout=0,target_modules=r'model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))')
            shared=runner.model
        else:
            runner.model=shared;runner.lora_enabled=True
        parameters=[p for p in shared.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=0,weight_decay=0)
        records=load_records(task,'train',data_root=cfg['data_root'])
        print(task,'scanning training clip durations',flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            lengths=list(pool.map(duration,records))
        longest=sorted(range(len(records)),key=lambda i:lengths[i],reverse=True)[:4]
        typical=random.Random(42).sample(range(len(records)),8)
        batches={};prep_seconds={}
        for i in sorted(set(longest+typical)):
            start=time.perf_counter();batches[i]=runner.prepare(records[i]);prep_seconds[i]=time.perf_counter()-start
        groups={'typical_a':typical[:4],'typical_b':typical[4:],'longest':longest}
        task_result={'longest_seconds':[lengths[i] for i in longest],
                     'prepared_shapes':{records[i]['id']:{k:list(v.shape) for k,v in b.items()} for i,b in batches.items()},
                     'prepare_seconds':list(prep_seconds.values()),'cases':[]}
        summary['tasks'][task]=task_result;save()
        reference={}
        for checkpointing,micro in [(True,1),(True,2),(True,4),(False,1),(False,2)]:
            if checkpointing:
                shared.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            else: shared.gradient_checkpointing_disable()
            optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            case={'micro_batch_size':micro,'accumulation':4//micro,'checkpointing':checkpointing,'status':'running','groups':{}}
            task_result['cases'].append(case);save()
            try:
                for group_name,indices in groups.items():
                    group=[batches[i] for i in indices]
                    targets=[list(LABELS[task]).index(records[i]['label']) for i in indices]
                    # First pass checks numerics and warms kernels/state; next two are timed.
                    result=update(runner,optimizer,parameters,group,targets,micro)
                    if checkpointing and micro==1:
                        reference[group_name]=result
                    ref=reference[group_name]
                    gradient_relative_l2=float((result['gradient']-ref['gradient']).norm()/ref['gradient'].norm().clamp_min(1e-12))
                    logit_max_abs=float((result['logits']-ref['logits']).abs().max())
                    loss_difference=abs(result['loss']-ref['loss'])
                    pred_agreement=float((result['logits'].argmax(-1)==ref['logits'].argmax(-1)).float().mean())
                    times=[]
                    for repeat in range(2):
                        begin=time.perf_counter();timed=update(runner,optimizer,parameters,group,targets,micro)
                        times.append(time.perf_counter()-begin);del timed
                    numerical_ok=gradient_relative_l2<0.05 and loss_difference<0.03 and logit_max_abs<0.2 and pred_agreement==1.0
                    case['groups'][group_name]={'mean_update_seconds':statistics.mean(times),
                        'gradient_relative_l2':gradient_relative_l2,'logit_max_abs':logit_max_abs,
                        'loss_difference':loss_difference,'prediction_agreement':pred_agreement,
                        'numerical_ok':numerical_ok,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
                    print(task,checkpointing,micro,group_name,case['groups'][group_name],flush=True)
                    del result
                    save()
                case['status']='passed' if all(x['numerical_ok'] for x in case['groups'].values()) else 'numerical_mismatch'
            except torch.cuda.OutOfMemoryError as error:
                case.update(status='oom',error=str(error)[:400]);print(task,checkpointing,micro,'OOM',flush=True)
            except Exception as error:
                case.update(status='failed',error=f'{type(error).__name__}: {error}');print(case,flush=True)
            optimizer.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();save()
        shared.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        del optimizer,reference,batches;gc.collect();torch.cuda.empty_cache()
    summary['status']='completed';save()


if __name__=='__main__':main()
