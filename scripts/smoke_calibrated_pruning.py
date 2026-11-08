#!/usr/bin/env python3
"""Bounded nine-config GPU checks; compact exports are reusable by the full sweep."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import yaml
from launch_experiments import ROOT,PROJECT_ENV,stamp,stop_owned_processes
from launch_pruning_sweep import CALIBRATED_METHODS,available_gpus,fresh_plans
from launch_suzy_sweep import child_environment,write_json


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output-root',type=Path,required=True);p.add_argument('--compact-root',type=Path,required=True)
 p.add_argument('--gpu',type=int,default=7);p.add_argument('--resume',action='store_true')
 a=p.parse_args()
 if a.gpu not in range(1,8):p.error('Suzy GPUs 1-7 only')
 if socket.gethostname().split('.')[0]!='suzy':raise RuntimeError('Run on Suzy')
 output=a.output_root.resolve();compact=a.compact_root.resolve()
 if a.resume:state=json.loads((output/'sweep.json').read_text());plans=state['jobs']
 else:
  output.mkdir(parents=True,exist_ok=False)
  plans=fresh_plans(output,compact,['mintrec2','mintrec20','mer6'],smoke=True,methods=CALIBRATED_METHODS)
  state={'smoke':True,'status':'running','created_at_utc':stamp(),'jobs':plans,'compact_root':str(compact)}
 active=[]
 def save():write_json(output/'sweep.json',state)
 def interrupt(signum,frame):raise KeyboardInterrupt
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,interrupt)
 try:
  for plan in plans:
   if plan['status']=='completed':continue
   for stage in ['export','train']:
    while True:
     free=available_gpus([a.gpu])
     if free:break
     print('Waiting for free smoke GPU',a.gpu,flush=True);time.sleep(15)
    gpu=free[0];env=child_environment(plan['config'],gpu['uuid']);env['PYTHONDONTWRITEBYTECODE']='1';env['TMPDIR']='/tmp'
    if stage=='export':
     command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/prune_model.py'),'--source-checkpoint',plan['source_checkpoint'],
              '--method',plan['method'],'--output',plan['compact_model'],'--reuse']
     for key,value in plan['pruning_settings'].items():command+=['--'+key.replace('_','-'),str(value)]
    else:
     config=output/(plan['name']+'.yaml');config.write_text(yaml.safe_dump(plan['config'],sort_keys=False))
     command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/train.py'),'--config',str(config),'--output',plan['output'],'--max-steps','2']
     if (Path(plan['output'])/'checkpoints/latest').exists():command+=['--resume',str((Path(plan['output'])/'checkpoints/latest').resolve())]
    log=output/(plan['name']+'.'+stage+'.log')
    with log.open('w') as stream:proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    plan.update(status='running',stage=stage,gpu=a.gpu,pid=proc.pid,log=str(log));active=[{'process':proc,'plan':plan}];save()
    print('Started',plan['name'],stage,'GPU',a.gpu,'PID',proc.pid,flush=True)
    code=proc.wait();active=[]
    if code:plan.update(status='failed',returncode=code);save();raise RuntimeError(f'{plan["name"]} {stage} failed; inspect {log}')
    if stage=='export':
     manifest=json.loads((Path(plan['compact_model'])/'compression.json').read_text())
     assert manifest['verification']['passed'] and manifest['parameters_after']['language']<2_000_000_000
     plan['parameters']=manifest['parameters_after']
    else:
     report=json.loads((Path(plan['output'])/'report.json').read_text())
     assert report['status']=='paused' and report['global_step']==2 and report['test_set_used'] is False
     plan.update(status='completed',global_step=2,best_score=report['best_score'],finished_at_utc=stamp())
    save();print('Passed',plan['name'],stage,flush=True)
  state.update(status='completed',finished_at_utc=stamp());save()
 except BaseException as error:
  state.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise
 finally:stop_owned_processes(active)

if __name__=='__main__':main()
