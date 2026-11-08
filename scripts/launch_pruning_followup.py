#!/usr/bin/env python3
"""Observe pruning SFT runs, test validation winners, then distill and test again.

Per configuration: completed SFT -> pinned best-val test -> fresh compact KD ->
pinned best-val KD test. A training-only teacher cache is shared per task.
Existing SFT processes are observed, never restarted or signalled.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
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

from launch_experiments import ROOT,PROJECT_ENV,stamp,stop_owned_processes
from launch_pruning_sweep import available_gpus
from launch_suzy_sweep import child_environment,write_json
from evaluate import resolve_evaluation_checkpoint
from smarthome_efficiency.training import DEFAULTS,config_fingerprint,validate_resume


def source_definition(sweep):
    return [{key:job[key] for key in ['name','task','source_checkpoint','compact_model','config','output']}
            for job in sweep['jobs']]


def build_jobs(source,output,smoke=False,mode='test-and-distill',epochs=None,alpha=.5,temperature=2.,teacher_cache_root=None):
    jobs=[];teachers={}
    teacher_root=Path(teacher_cache_root) if teacher_cache_root is not None else output/'teachers'
    for original in source['jobs']:
        name=original['name'];task=original['task']
        if task in teachers and teachers[task]!=original['source_checkpoint']:
            raise ValueError('One task has differing source teachers; use separate follow-up workflows')
        teachers[task]=original['source_checkpoint']
        split='dev' if smoke else 'test'
        jobs.append({'id':f'sft_test:{name}','kind':'sft_test','name':name,'task':task,'deps':[],
                     'status':'waiting','output':str(output/'sft_test'/name),'source_run':original['output'],
                     'split':split,'limit':6 if smoke else None,'modes':'both'})
        if mode=='test':continue
        cfg=DEFAULTS|dict(original['config'])
        cfg.update(objective='sft_kd',model=original['compact_model'],epochs=epochs or cfg['epochs'],
                   teacher_checkpoint=original['source_checkpoint'],teacher_cache=str(teacher_root/task/'cache.json'),
                   distillation_alpha=alpha,distillation_temperature=temperature,eval_modes=['logit','greedy'],selection_mode='logit')
        if smoke:cfg.update(epochs=1,train_limit=24,dev_limit=20)
        jobs.append({'id':f'kd_train:{name}','kind':'kd_train','name':name,'task':task,
                     'deps':[f'sft_test:{name}',f'teacher:{task}'],'status':'waiting',
                     'output':str(output/'kd'/name),'config':cfg,'compact_model':original['compact_model']})
        jobs.append({'id':f'kd_test:{name}','kind':'kd_test','name':name,'task':task,
                     'deps':[f'kd_train:{name}'],'status':'waiting','output':str(output/'kd_test'/name),
                     'split':split,'limit':6 if smoke else None,'modes':'both'})
    if mode!='test':
        for task,checkpoint in teachers.items():
            jobs.append({'id':f'teacher:{task}','kind':'teacher','name':task,'task':task,'deps':[],
                         'status':'waiting','output':str(teacher_root/task),
                         'teacher_checkpoint':checkpoint,'limit':24 if smoke else None})
    return jobs


def pin_best(run,allow_paused=False):
    """Only a finished training run's validation-selected best is eligible."""
    run=Path(run);path=run/'report.json'
    if not path.exists():return None
    report=json.loads(path.read_text())
    if report['status'] not in (['completed','paused'] if allow_paused else ['completed']):return None
    if report.get('test_set_used') is not False:raise ValueError('Training run used test data')
    checkpoint=resolve_evaluation_checkpoint(run)
    metadata=json.loads((checkpoint/'metadata.json').read_text())
    if metadata['global_step']!=report['best_step']:raise ValueError('Best pointer disagrees with validation selection')
    return str(checkpoint)


def checkpoint_identity(checkpoint):
    checkpoint=Path(checkpoint)
    return {name:hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
            for name in ['metadata.json','adapter/adapter_model.safetensors']}


def refresh_jobs(jobs,external,smoke):
    by_id={job['id']:job for job in jobs};external_by_name={j['name']:j for j in external['jobs']}
    for job in jobs:
        if job['status'] not in ['waiting','ready']:continue
        dependencies=[by_id[d] for d in job['deps']]
        if any(d['status'] in ['failed','blocked'] for d in dependencies):
            job.update(status='blocked',error='A prerequisite failed');continue
        if any(d['status']!='completed' for d in dependencies):continue
        if job['kind']=='sft_test':
            source=external_by_name[job['name']]
            if source['status'] in ['failed','interrupted']:
                job.update(status='blocked',error='Source SFT did not complete');continue
            if source['status']!='completed':continue
            selected=pin_best(job['source_run'],allow_paused=smoke)
            if selected is None:continue
            if job.get('checkpoint') is not None and job['checkpoint']!=selected:
                raise ValueError('Source best checkpoint changed; use a new follow-up output')
            job['checkpoint']=selected
            job['checkpoint_identity']=checkpoint_identity(selected)
        elif job['kind']=='kd_test':
            training=by_id[job['deps'][0]]
            selected=pin_best(training['run_dir'],allow_paused=smoke)
            if selected is None:raise ValueError('Finished distillation has no best checkpoint')
            job['checkpoint']=selected;job['checkpoint_identity']=checkpoint_identity(selected)
        job['status']='ready'


def completed_result(job,smoke):
    if job['kind']=='teacher':
        path=Path(job['output'])/'cache.json';payload=json.loads(path.read_text())
        if (payload['status']!='completed' or payload['spec']['split']!='train'
                or payload['spec']['test_set_used'] is not False
                or payload['spec']['teacher_checkpoint']!=job['teacher_checkpoint']):
            raise ValueError('Invalid teacher cache completion')
        return {'cache':str(path),'examples':payload['num_examples'],
                'cache_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    report_path=Path(job['run_dir'])/'report.json';report=json.loads(report_path.read_text())
    if job['kind']=='kd_train':
        if (report['status']!=('paused' if smoke else 'completed') or report['global_step']<(2 if smoke else 1)
                or report.get('test_set_used') is not False or report.get('objective')!='sft_kd'):
            raise ValueError('Distillation training did not complete as requested')
        if not smoke and report['global_step']!=report['requested_steps']:raise ValueError('Incomplete epoch target')
        return {'report':str(report_path),'best_checkpoint':pin_best(job['run_dir'],allow_paused=smoke),
                'best_score':report['best_score'],'best_step':report['best_step'],'global_step':report['global_step']}
    if (report['status']!='completed' or report['split']!=job['split']
            or report['checkpoint']!=job['checkpoint'] or report['eval_modes']!=['logit','greedy']
            or report['test_set_used']!=(job['split']=='test')):
        raise ValueError('Evaluation does not match pinned checkpoint/split/modes')
    if job['limit'] is None and (report['partial_split'] or report['evaluated_count']!=report['full_split_count']):
        raise ValueError('Official evaluation must cover the complete test split')
    return {'report':str(report_path),'evaluated_count':report['evaluated_count'],
            'checkpoint':job['checkpoint'],'logit':report['logit'],'greedy':report['greedy']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sweep',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--teacher-cache-root',type=Path,help='Reuse compatible completed original-teacher caches')
    parser.add_argument('--mode',choices=['test','test-and-distill'],default='test-and-distill')
    parser.add_argument('--gpus',default='1,2,3,4,5,6,7')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--alpha',type=float)
    parser.add_argument('--temperature',type=float)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--adopt-running',action='store_true',help='Retain identified workers during coordinator handoff')
    parser.add_argument('--peer-manifest',type=Path,help='Reserve running/loading GPUs from this other queue')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--poll-seconds',type=int,default=15)
    args=parser.parse_args()
    if args.adopt_running and not args.resume:parser.error('--adopt-running requires --resume')
    indices=[int(v) for v in args.gpus.split(',')]
    if not indices or len(set(indices))!=len(indices) or any(i not in range(1,8) for i in indices):parser.error('Use unique Suzy GPU indices 1-7')
    if args.epochs is not None and args.epochs<1:parser.error('epochs must be positive')
    if args.alpha is not None and not 0<=args.alpha<=1:parser.error('alpha must be in [0,1]')
    if args.temperature is not None and not 0<args.temperature<float('inf'):parser.error('temperature must be finite and positive')
    if args.poll_seconds<1:parser.error('poll-seconds must be positive')
    source_root=args.source_sweep.expanduser().resolve();output=args.output_root.expanduser().resolve()
    source=json.loads((source_root/'sweep.json').read_text())
    definition_hash=config_fingerprint(source_definition(source))
    if args.resume:
        manifest=json.loads((output/'workflow.json').read_text())
        if manifest.get('controller') == 'project_queue':
            raise RuntimeError('This batch is managed by scripts/project_queue.py; resume the project queue instead')
        if (manifest['source_definition_sha256']!=definition_hash or manifest['smoke']!=args.smoke
                or manifest['mode']!=args.mode):raise ValueError('Cannot change follow-up inputs or mode on resume')
        for key,value in [('epochs',args.epochs),('alpha',args.alpha),('temperature',args.temperature)]:
            if value is not None and value!=manifest[key]:raise ValueError('Use a new output for changed distillation settings')
        jobs=manifest['jobs']
        for job in jobs:
            if job['status']!='completed' and not (args.adopt_running and job['status']=='running'):job.update(status='waiting')
            elif job['kind'] in ['sft_test','kd_test'] and checkpoint_identity(job['checkpoint'])!=job['checkpoint_identity']:
                raise ValueError('Previously evaluated checkpoint was modified')
    else:
        if output.exists():raise FileExistsError(output)
        alpha=.5 if args.alpha is None else args.alpha;temperature=2. if args.temperature is None else args.temperature
        jobs=build_jobs(source,output,args.smoke,args.mode,args.epochs,alpha,temperature,
                        args.teacher_cache_root.expanduser().resolve() if args.teacher_cache_root else None)
        manifest={'format_version':1,'created_at_utc':stamp(),'source_sweep':str(source_root),
                  'source_definition_sha256':definition_hash,'mode':args.mode,'smoke':args.smoke,
                  'epochs':args.epochs or (1 if args.smoke else 5),'alpha':alpha,'temperature':temperature,'jobs':jobs,
                  'test_results_used_for_training':False,'distillation_initialization':'immutable original pruned base + fresh zero-update LoRA'}
    print(json.dumps({'output':str(output),'jobs':[(j['id'],j['status']) for j in jobs],'smoke':args.smoke},indent=2),flush=True)
    if args.dry_run:return 0
    if socket.gethostname().split('.')[0]!='suzy':raise RuntimeError('Run this queue on Suzy')
    output.mkdir(parents=True,exist_ok=args.resume)
    lock=(output/'.workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pool=(ROOT/'artifacts/.pruning_followup_gpu_pool.lock').open('a');fcntl.flock(pool,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch_dir=output/'launches'/stamp();launch_dir.mkdir(parents=True)
    manifest.update(status='running',host=socket.gethostname(),launcher_pid=os.getpid(),latest_launch=str(launch_dir),allowed_physical_gpus=indices)
    manifest.pop('error',None);manifest.pop('finished_at_utc',None)
    manifest['shared_gpu_allocation']=True
    manifest['peer_manifest']=str(args.peer_manifest) if args.peer_manifest else None
    active=[{'process':AdoptedProcess(j),'plan':j} for j in jobs if j['status']=='running'] if args.adopt_running else []
    for owned in active:owned['plan']['adopted_worker_exit_code_unavailable']=True
    by_id={j['id']:j for j in jobs}

    def save():write_json(output/'workflow.json',manifest)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Received signal {signum}')
    handlers={sig:signal.signal(sig,interrupted) for sig in [signal.SIGINT,signal.SIGTERM]}

    def start(job,gpu):
        for key in ['error','returncode']:job.pop(key,None)
        kind=job['kind'];base=Path(job['output'])
        if kind=='teacher':
            cfg=json.loads((Path(job['teacher_checkpoint'])/'metadata.json').read_text())['config']
            command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/cache_teacher_logits.py'),
                     '--teacher-checkpoint',job['teacher_checkpoint'],'--output',str(base),'--resume']
            if job['limit'] is not None:command+=['--limit',str(job['limit'])]
        else:
            if kind=='kd_train':
                cfg=job['config'];run=Path(job.get('run_dir','')) if job.get('run_dir') else None
                resume=None
                if run is not None and (run/'checkpoints/latest').exists():
                    resume=(run/'checkpoints/latest').resolve()
                    saved=json.loads((resume/'metadata.json').read_text())
                    validate_resume(saved['config'],cfg,saved['world_size'],1)
                else:
                    attempt=job.get('attempt',0)+1;job['attempt']=attempt
                    run=base/f'attempt-{attempt:03d}'
                path=launch_dir/f"{job['name']}.kd.yaml";path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/train.py'),'--config',str(path),'--output',str(run)]
                if resume is not None:command+=['--resume',str(resume)]
                if args.smoke:command+=['--max-steps','2']
            else:
                if checkpoint_identity(job['checkpoint'])!=job['checkpoint_identity']:raise ValueError('Pinned checkpoint changed')
                cfg=json.loads((Path(job['checkpoint'])/'metadata.json').read_text())['config']
                attempt=job.get('attempt',0)+1;job['attempt']=attempt;run=base/f'attempt-{attempt:03d}'
                command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/evaluate.py'),
                         '--checkpoint',job['checkpoint'],'--split',job['split'],'--modes','both','--output',str(run)]
                if job['limit'] is not None:command+=['--limit',str(job['limit'])]
            job['run_dir']=str(run)
        log_path=launch_dir/(job['id'].replace(':','__')+'.log')
        env=child_environment(cfg,gpu['uuid']);env['PYTHONDONTWRITEBYTECODE']='1'
        with log_path.open('w') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        job.update(status='running',gpu=gpu['index'],gpu_uuid=gpu['uuid'],pid=process.pid,
                   command=command,log=str(log_path),started_at_utc=stamp())
        active.append({'process':process,'plan':job});save()
        print(f"Started {job['id']} on GPU {gpu['index']}; PID {process.pid}",flush=True)

    try:
        save()
        while True:
            source=json.loads((source_root/'sweep.json').read_text())
            if config_fingerprint(source_definition(source))!=definition_hash:raise ValueError('Source scientific configuration changed')
            for owned in list(active):
                code=owned['process'].poll()
                if code is None:continue
                active.remove(owned);job=owned['plan']
                try:
                    if code!=0:raise RuntimeError(f'Process exited {code}; inspect {job["log"]}')
                    result=completed_result(job,args.smoke)
                    job.update(status='completed',result=result,finished_at_utc=stamp(),returncode=code)
                except (OSError,ValueError,KeyError,RuntimeError) as error:
                    job.update(status='failed',error=str(error),returncode=code)
                save();print(f"{job['id']}: {job['status']}",flush=True)
            refresh_jobs(jobs,source,args.smoke)
            unfinished=[j for j in jobs if j['status'] in ['waiting','ready','running']]
            if not unfinished:break
            # The old SFT coordinator owns its running/loading GPUs. While it has
            # pending jobs, let it finish allocation before using remaining GPUs.
            if not any(j['status']=='pending' for j in source['jobs']):
                with allocation_lock(ROOT):
                    reserved={j['gpu'] for j in source['jobs'] if j['status']=='running' and j.get('gpu') is not None}
                    reserved|={owned['plan']['gpu'] for owned in active}
                    reserved|=peer_reservations(args.peer_manifest)
                    free=[gpu for gpu in available_gpus(indices) if gpu['index'] not in reserved]
                    ready=sorted([j for j in jobs if j['status']=='ready'],key=lambda j:{'sft_test':0,'kd_test':0,'teacher':1,'kd_train':2}[j['kind']])
                    for job,gpu in zip(ready,free):start(job,gpu)
            save()
            time.sleep(args.poll_seconds)
        manifest.update(status='completed' if all(j['status']=='completed' for j in jobs) else 'failed',finished_at_utc=stamp())
        save();write_json(output/'results.json',{j['id']:j.get('result',{'status':j['status'],'error':j.get('error')}) for j in jobs})
        return 0 if manifest['status']=='completed' else 1
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        stop_owned_processes(active)
        for owned in active:owned['plan'].update(status='interrupted')
        save()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        pool.close();lock.close()


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
