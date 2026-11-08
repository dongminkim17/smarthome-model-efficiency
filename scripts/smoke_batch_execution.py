#!/usr/bin/env python3
"""Exercise actual optimized training and a batch-preserving checkpoint resume."""
from pathlib import Path
import json,math,os,subprocess,sys
import yaml
from launch_suzy_sweep import check_gpus,child_environment
from launch_experiments import ROOT,PROJECT_ENV
from smarthome_efficiency.training import load_config
root=ROOT/'artifacts/optimization/suzy_batching_20261019/trainer_smoke'
root.mkdir(parents=True,exist_ok=False)
state=check_gpus([7])[7]
checks=[]
for task,checkpointing in [('mintrec20',False),('mer6',True)]:
 config=load_config(ROOT/'configs'/f'{task}.yaml')
 config.update(epochs=1,train_limit=24,dev_limit=20,micro_batch_size=2,
               gradient_accumulation_steps=2,lora_target='decoder_linear',
               gradient_checkpointing=checkpointing)
 path=root/f'{task}.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
 output=root/task;log=root/f'{task}.log'
 command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/train.py'),'--config',str(path),'--output',str(output),'--max-steps','2']
 with log.open('w') as f:
  subprocess.run(command,cwd=ROOT,env=child_environment(config,state['uuid']),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
 report=json.loads((output/'report.json').read_text())
 rows=[json.loads(l) for l in (output/'metrics.jsonl').read_text().splitlines()]
 trains=[r for r in rows if r['event']=='train']
 assert report['global_step']==2 and report['global_batch_size']==4 and report['test_set_used'] is False
 assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in trains)
 checks.append({'task':task,'status':'passed','micro_batch_size':2,'checkpointing':checkpointing,'peak_allocated_gib':max(r['peak_allocated_gib'] for r in trains)})
 print(checks[-1],flush=True)
 if task=='mintrec20':
  config.update(micro_batch_size=1,gradient_accumulation_steps=4)
  resume_config=root/'resume.yaml';resume_config.write_text(yaml.safe_dump(config,sort_keys=False))
  command=[str(PROJECT_ENV/'bin/python'),'-u',str(ROOT/'scripts/train.py'),'--config',str(resume_config),'--output',str(output),'--resume',str(output/'checkpoints/latest'),'--max-steps','3']
  with (root/'resume.log').open('w') as f:
   subprocess.run(command,cwd=ROOT,env=child_environment(config,state['uuid']),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
  resumed=json.loads((output/'report.json').read_text())
  assert resumed['global_step']==3 and resumed['global_batch_size']==4
  checks.append({'task':'mintrec20','status':'passed','check':'resume optimizer/schedule across microbatch 2 to 1'})
  print(checks[-1],flush=True)
(root/'verification.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n')
