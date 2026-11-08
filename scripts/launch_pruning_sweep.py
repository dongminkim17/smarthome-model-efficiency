#!/usr/bin/env python3
"""Queue structured pruning + five-epoch recovery jobs on free Suzy GPUs 1-7.

Existing jobs are never interrupted. Each task/method exports an immutable
compact base, then the existing trainer evaluates it at step zero and recovers
with a fresh adapter. Full runs automatically continue with best-validation test
evaluation and fresh distillation recovery; --followup none keeps SFT only.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import yaml

from gpu_queue_coordination import AdoptedProcess, allocation_lock, peer_reservations

from launch_experiments import PROJECT_ENV, ROOT, full_checkpoint, stamp, stop_owned_processes
from launch_suzy_sweep import child_environment, completion, prepare_resume, write_json
from evaluate import resolve_evaluation_checkpoint
from smarthome_efficiency.training import load_config

# Keep the three established baseline sources fixed across the two methods.
SOURCES = {
    'mintrec2': ROOT / 'artifacts/full/baseline_5epochs/mintrec2',
    'mintrec20': ROOT / 'artifacts/full/mintrec20_5epochs/train',
    'mer6': ROOT / 'artifacts/full/baseline_5epochs/mer6',
}
METHOD_SETTINGS = {'depth_uniform': {'keep_layers':17}, 'ffn_magnitude': {'ffn_width':4352},
                   'depth_bi': {'keep_layers':17}, 'ffn_activation': {'ffn_width':4352},
                   'depth_ffn_activation': {'keep_layers':28, 'ffn_width':6144}}
DEFAULT_METHODS = ('depth_uniform','ffn_magnitude')
CALIBRATED_METHODS = ('depth_bi','ffn_activation','depth_ffn_activation')


def available_gpus(indices, *, allowed_indices=None):
    """Return only idle requested GPUs. Busy devices are left alone and retried."""
    allowed = set(range(1, 8) if allowed_indices is None else allowed_indices)
    if not indices or len(indices) != len(set(indices)) or any(type(i) is not int or i not in allowed for i in indices):
        raise ValueError('Requested physical GPU indices are outside the approved pool')
    query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=20)
    devices={}
    for row in csv.reader(io.StringIO(query.stdout)):
        if row:
            index,uuid,used,util=[v.strip() for v in row]
            devices[int(index)]={'index':int(index),'uuid':uuid,'used_mib':int(used),'utilization_percent':int(util)}
    if any(index not in devices for index in indices):
        raise RuntimeError('Requested GPU is absent from this host')
    query=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                         check=True,capture_output=True,text=True,timeout=20)
    occupied={row[0].strip() for row in csv.reader(io.StringIO(query.stdout)) if row}
    return [devices[i] for i in indices if devices[i]['uuid'] not in occupied
            and devices[i]['used_mib'] <= 512 and devices[i]['utilization_percent'] <= 5]



def coexist_allocation(indices, external):
    """Use a disjoint spare device until the legacy allocator terminates."""
    terminal=external['status'] in ['completed','failed','interrupted']
    effective=list(indices) if terminal or external.get('shared_gpu_allocation') else [i for i in indices if i==7]
    reserved={j['gpu'] for j in external['jobs'] if j['status']=='running' and j.get('gpu') is not None}
    return effective,reserved

def fresh_plans(output, compact_root, tasks, epochs=5, smoke=False, methods=None, calibration_samples=128, calibration_seed=42):
    methods=list(DEFAULT_METHODS if methods is None else methods)
    if not methods or len(set(methods))!=len(methods) or any(m not in METHOD_SETTINGS for m in methods):
        raise ValueError('Choose unique supported pruning methods')
    if type(calibration_samples) is not int or calibration_samples<1:raise ValueError('Positive calibration count required')
    plans=[]
    for task in tasks:
        source=resolve_evaluation_checkpoint(SOURCES[task])
        teacher=json.loads((source/'metadata.json').read_text())['config']
        if teacher['task'] != task:
            raise ValueError('Source checkpoint task mismatch')
        for method in methods:
            settings=dict(METHOD_SETTINGS[method])
            if method in CALIBRATED_METHODS:
                settings.update(calibration_samples=calibration_samples,calibration_seed=calibration_seed,
                                calibration_cache=str(compact_root/'calibration'/f'{task}_n{calibration_samples}_seed{calibration_seed}.json'))
            name=f'{task}_{method}'
            compact=compact_root/name
            config=load_config(ROOT/'configs'/f'{task}.yaml')
            # Keep teacher media/prompt/split protocol; recovery changes capacity only.
            for key in ['data_root','cache_dir','split_seed','max_frames','max_pixels','use_transcript']:
                config[key]=teacher[key]
            config.update(model=str(compact),epochs=epochs,seed=42,
                          micro_batch_size=2,gradient_accumulation_steps=2,
                          gradient_checkpointing=task=='mer6',
                          lora_rank=16,lora_alpha=32,lora_dropout=0.0,lora_target='decoder_linear',
                          learning_rate=1e-4,eval_modes=['logit','greedy'],selection_mode='logit',
                          eval_every_fraction=0.5,train_limit=None,dev_limit=None)
            if smoke:
                config.update(epochs=1,train_limit=24,dev_limit=20)
            plans.append({'name':name,'task':task,'method':method,'pruning_settings':settings,
                          'source_checkpoint':str(source),'compact_model':str(compact),
                          'config':config,'output':str(output/name),'status':'pending','stage':'export'})
    return plans


def prepare_plans_for_resume(plans, epochs):
    for plan in plans:
        plan.pop('pid',None)
        output=Path(plan['output'])
        if (output/'checkpoints/latest').exists():
            prepare_resume([plan],epochs)
            plan['stage']='train'
        elif output.exists() and any(output.iterdir()):
            raise ValueError(f'{output}: incomplete run without resume checkpoint; inspect it and choose a new output')
        else:
            plan.update(status='pending',stage='export')
            if epochs is not None: plan['config']['epochs']=epochs
        for key in ['error','returncode','finished_at_utc']:
            plan.pop(key,None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--compact-root',type=Path,required=True)
    parser.add_argument('--methods',help='Comma-separated methods; default: depth_uniform,ffn_magnitude')
    parser.add_argument('--calibration-samples',type=int,default=128)
    parser.add_argument('--calibration-seed',type=int,default=42)
    parser.add_argument('--coexist-followup',type=Path,help='Coordinate with this follow-up manifest; legacy launchers use GPU 7 only')
    parser.add_argument('--adopt-running',action='store_true',help='Coordinator handoff: retain identified live workers; requires --resume')
    parser.add_argument('--wait-for-pool',action='store_true',help='Queue behind an existing pruning coordinator without stopping it')
    parser.add_argument('--teacher-cache-root',type=Path,help='Reuse completed original-teacher caches in the automatic follow-up')
    parser.add_argument('--tasks',default='mer6,mintrec2,mintrec20')
    parser.add_argument('--gpus',default='1,2,3,4,5,6,7')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--poll-seconds',type=int,default=15)
    parser.add_argument('--followup',choices=['none','test','test-and-distill'],help='Full-run default: test-and-distill; smoke default: none')
    parser.add_argument('--followup-output',type=Path,help='Default: OUTPUT_ROOT/followup')
    args=parser.parse_args()
    followup_mode=args.followup or ('none' if args.smoke else 'test-and-distill')
    if args.adopt_running and not args.resume:parser.error('--adopt-running requires --resume')
    indices=[int(v) for v in args.gpus.split(',')]
    tasks=args.tasks.split(',')
    methods=args.methods.split(',') if args.methods else list(DEFAULT_METHODS)
    if not methods or len(set(methods))!=len(methods) or any(m not in METHOD_SETTINGS for m in methods) or args.calibration_samples<1:
        parser.error('Choose unique supported methods and positive calibration count')
    if (not tasks or len(set(tasks))!=len(tasks) or any(t not in SOURCES for t in tasks)
            or not indices or len(set(indices))!=len(indices) or any(i not in range(1,8) for i in indices)):
        parser.error('Choose unique known tasks and Suzy GPU indices 1-7')
    if args.poll_seconds < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error('Epochs and polling interval must be positive')
    if args.smoke and args.resume:
        parser.error('Smoke runs cannot be resumed as full experiments')
    output=args.output_root.expanduser().resolve()
    compact=args.compact_root.expanduser().resolve()
    if args.resume:
        manifest=json.loads((output/'sweep.json').read_text())
        if manifest.get('controller') == 'project_queue':
            raise RuntimeError('This batch is managed by scripts/project_queue.py; resume the project queue instead')
        if manifest['smoke']: raise ValueError('Cannot resume smoke as full experiment')
        if manifest['compact_root'] != str(compact): raise ValueError('Cannot change compact bases on resume')
        plans=manifest['jobs']
        if args.methods and set(methods)!={p['method'] for p in plans}:raise ValueError('Cannot change methods on resume')
        if args.adopt_running:
            if args.epochs is not None:raise ValueError('Cannot change epochs during a live handoff')
            prepare_plans_for_resume([p for p in plans if p['status'] not in ['running','completed']],None)
        else:
            prepare_plans_for_resume(plans,args.epochs)
    else:
        if output.exists(): raise FileExistsError(f'{output}: use --resume or choose a new output')
        plans=fresh_plans(output,compact,tasks,args.epochs or 5,args.smoke,methods,args.calibration_samples,args.calibration_seed)
        manifest={'format_version':1,'created_at_utc':stamp(),'smoke':args.smoke,'jobs':plans,
                  'compact_root':str(compact),'official_test_evaluation':False,'recovery_objective':'SFT labels + EOS',
                  'distillation_enabled':False,'world_size_per_job':1,'effective_batch_size':4}
    print(json.dumps({'output':str(output),'compact_root':str(compact),'allowed_physical_gpus':indices,
                      'jobs':[{k:p[k] for k in ['name','source_checkpoint','pruning_settings','status','stage']} for p in plans]},indent=2),flush=True)
    if args.dry_run: return 0
    if socket.gethostname().split('.')[0] != 'suzy': raise RuntimeError('This launcher must run on Suzy')
    output.mkdir(parents=True,exist_ok=args.resume)
    compact.mkdir(parents=True,exist_ok=True)
    lock=(output/'.launcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Also prevent two coordinators from assigning the same free GPUs during startup.
    pool_lock=(ROOT/'artifacts/.pruning_gpu_pool.lock').open('a')
    while True:
        try:
            fcntl.flock(pool_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if not args.wait_for_pool:raise
            manifest.update(status='waiting_for_pool',host=socket.gethostname(),launcher_pid=os.getpid())
            write_json(output/'sweep.json',manifest)
            print('Waiting for the existing pruning coordinator to finish; its jobs continue unchanged',flush=True)
            time.sleep(args.poll_seconds)
    launch_dir=output/'launches'/stamp(); launch_dir.mkdir(parents=True)
    manifest['followup_mode']=followup_mode
    manifest.update(status='running',host=socket.gethostname(),launcher_pid=os.getpid(),
                    latest_launch=str(launch_dir),allowed_physical_gpus=indices,
                    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__).resolve(),ROOT/'scripts/prune_model.py',ROOT/'scripts/train.py']})
    manifest.pop('error',None); manifest.pop('finished_at_utc',None)
    manifest['shared_gpu_allocation']=True
    active=[{'process':AdoptedProcess(p),'plan':p} for p in plans if p['status']=='running'] if args.adopt_running else []
    for owned in active:owned['plan']['adopted_worker_exit_code_unavailable']=True

    def save():
        write_json(output/'sweep.json',manifest)

    def interrupted(signum,frame): raise KeyboardInterrupt(f'Received signal {signum}')
    handlers={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}

    def start(plan,gpu):
        for key in ['returncode','error','finished_at_utc']: plan.pop(key,None)
        environment=child_environment(plan['config'],gpu['uuid'])
        environment['PYTHONDONTWRITEBYTECODE']='1'
        if plan['stage']=='export':
            command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/prune_model.py'),
                     '--source-checkpoint',plan['source_checkpoint'],'--method',plan['method'],
                     '--output',plan['compact_model'],'--reuse']
            for key,value in plan['pruning_settings'].items():
                command += ['--'+key.replace('_','-'),str(value)]
        else:
            path=launch_dir/f"{plan['name']}.yaml"; path.write_text(yaml.safe_dump(plan['config'],sort_keys=False))
            command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/train.py'),
                     '--config',str(path),'--output',plan['output']]
            if plan.get('resume_checkpoint'): command += ['--resume',plan['resume_checkpoint']]
            if args.smoke: command += ['--max-steps','2']
        log_path=launch_dir/f"{plan['name']}.{plan['stage']}.log"
        with log_path.open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        plan.update(status='running',gpu=gpu['index'],gpu_uuid=gpu['uuid'],pid=process.pid,
                    command=command,log=str(log_path),started_at_utc=stamp())
        active.append({'process':process,'plan':plan})
        save(); print(f"Started {plan['name']} {plan['stage']} on GPU {gpu['index']}; PID {process.pid}",flush=True)

    try:
        save()
        while active or any(p['status']=='pending' for p in plans):
            for job in list(active):
                code=job['process'].poll()
                if code is None: continue
                plan=job['plan']; active.remove(job)
                if code==0 and plan['stage']=='export':
                    sidecar=json.loads((Path(plan['compact_model'])/'compression.json').read_text())
                    if not sidecar.get('verification',{}).get('passed'):
                        plan.update(status='failed',error='Export did not pass verification',returncode=code)
                    else:
                        plan.update(status='pending',stage='train',parameters=sidecar['parameters_after'])
                else:
                    result=(completion(plan,code,args.smoke) if plan['stage']=='train'
                            else {'status':'failed','returncode':code})
                    plan.update(result,finished_at_utc=stamp())
                save(); print(f"{plan['name']}: {plan['status']}, stage={plan['stage']}, exit={code}",flush=True)
            # One pruning coordinator owns the pool for its lifetime, including
            # the gap before a newly spawned process creates its CUDA context.
            with allocation_lock(ROOT):
                reserved={job['plan']['gpu'] for job in active}
                effective_indices=indices
                if args.coexist_followup:
                    external=json.loads(args.coexist_followup.expanduser().resolve().read_text())
                    effective_indices,external_reserved=coexist_allocation(indices,external)
                    reserved|=external_reserved
                manifest['effective_allocation_gpus']=effective_indices
                free=available_gpus(effective_indices) if effective_indices else []
                free=[gpu for gpu in free if gpu['index'] not in reserved]
                pending=sorted([p for p in plans if p['status']=='pending'],key=lambda p:p['stage']!='train')
                for plan,gpu in zip(pending,free): start(plan,gpu)
            if active or any(p['status']=='pending' for p in plans): time.sleep(args.poll_seconds)
        manifest.update(status='completed' if all(p['status']=='completed' for p in plans) else 'failed',finished_at_utc=stamp())
        save(); result_code=0 if manifest['status']=='completed' else 1
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                        error=f'{type(error).__name__}: {error}')
        raise
    finally:
        stop_owned_processes(active)
        for job in active: job['plan'].update(status='interrupted',returncode=job['process'].returncode)
        save()
        for sig,handler in handlers.items(): signal.signal(sig,handler)
        pool_lock.close(); lock.close()
    if result_code==0 and followup_mode!='none':
        followup_output=(args.followup_output or output/'followup').expanduser().resolve()
        path=followup_output/'workflow.json'
        command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/launch_pruning_followup.py'),
                 '--source-sweep',str(output),'--output-root',str(followup_output),
                 '--mode',followup_mode,'--gpus',args.gpus]
        if args.smoke:command+=['--smoke']
        if args.teacher_cache_root:command+=['--teacher-cache-root',str(args.teacher_cache_root.expanduser().resolve())]
        if args.coexist_followup:
            while json.loads(args.coexist_followup.expanduser().resolve().read_text())['status'] not in ['completed','failed','interrupted']:
                print('SFT finished; waiting for the older follow-up coordinator before starting this follow-up',flush=True)
                time.sleep(args.poll_seconds)
        if path.exists():
            state=json.loads(path.read_text());pid=state.get('launcher_pid')
            process=Path(f'/proc/{pid}/cmdline')
            if state.get('status')=='running' and process.exists():
                contents=process.read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if 'launch_pruning_followup.py' in contents and str(followup_output) in contents:
                    print(f'Follow-up queue already running: {followup_output}',flush=True)
                    return 0
            command+=['--resume']
        print(f'Continuing with {followup_mode}: {followup_output}',flush=True)
        return subprocess.call(command,cwd=ROOT)
    return result_code


if __name__=='__main__':
    try: raise SystemExit(main())
    except KeyboardInterrupt: raise SystemExit(130)
