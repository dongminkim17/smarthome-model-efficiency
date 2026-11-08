#!/usr/bin/env python3
"""Apply measured, batch-preserving execution tuning with the controller stopped."""
import argparse,copy,json,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import read_json,atomic_json,StateStore,same_process,scientific_fingerprint,now
from smarthome_efficiency.training import config_fingerprint,validate_resume
from smarthome_efficiency.pruning import sha256_file
FAMILY='gentle_recovery_20261023'
OUT=ROOT/'artifacts/optimization/gentle_batching_20261023'

def choose_execution(measured,packed,max_allocated_gib=21.5):
 if measured.get('status')!='completed' or packed.get('status')!='completed':raise ValueError('Benchmarks must finish')
 if measured.get('task')!=packed.get('task') or measured.get('test_set_used') is not False or packed.get('test_set_used') is not False:raise ValueError('Task/training-only provenance mismatch')
 base=next(c for c in measured['cases'] if c['micro_batch_size']==2 and c['gradient_checkpointing'])
 if base['status']!='passed':raise ValueError('Baseline execution did not pass')
 baseline_time=sum(base['groups'][k]['mean_update_seconds'] for k in ('typical_a','typical_b'))/2
 choice={'micro_batch_size':2,'gradient_accumulation_steps':2,'gradient_checkpointing':True}
 reason='Retain verified baseline';speedup=1.
 candidate=next(c for c in measured['cases'] if c['micro_batch_size']==4 and c['gradient_checkpointing'])
 stress=next(c for c in packed['cases'] if c['micro_batch_size']==4 and c['gradient_checkpointing'])
 if candidate['status']=='passed' and stress['status']=='passed' and set(stress['groups'])=={'largest_a','largest_b'}:
  peak=max(v['peak_allocated_gib'] for c in (candidate,stress) for v in c['groups'].values())
  candidate_time=sum(candidate['groups'][k]['mean_update_seconds'] for k in ('typical_a','typical_b'))/2
  if peak<=max_allocated_gib and candidate_time<=baseline_time*.95:
   choice.update(micro_batch_size=4,gradient_accumulation_steps=1);speedup=baseline_time/candidate_time;reason='Faster typical updates; packed-largest-clips and numerical checks passed'
  else:reason='Larger batch lacks sufficient memory margin or measured speed benefit'
 else:reason='Larger batch failed numerical or packed-largest-clips checks'
 return {'execution':choice,'typical_update_speedup':speedup,'reason':reason,'baseline_typical_update_seconds':baseline_time}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');args=p.parse_args()
 q=ROOT/'artifacts/queue/suzy';store=StateStore(q);s=store.load()
 if socket.gethostname().split('.')[0]!='suzy':raise ValueError('Must apply on Suzy')
 if same_process(s.get('scheduler_pid'),s.get('scheduler_identity')):raise ValueError('Controller must be detached before applying')
 if any(j['status'] in ('running','starting') for j in s['jobs']):raise ValueError('Active jobs must be explicitly detached first')
 selected={}
 for task in ('mintrec2','mer6','mintrec20'):
  measured=read_json(OUT/(task+'.json'));packed=read_json(OUT/(task+'_worst_packed.json'))
  ids=read_json(q/'stress_cases'/f'{task}.json')['ids']
  if packed['groups']['largest_a']!=ids[:4]:raise ValueError('Packed probe did not exercise the four largest selected clips')
  selected[task]=choose_execution(measured,packed)
  selected[task]['benchmark_sha256']={str(path):sha256_file(path) for path in (OUT/(task+'.json'),OUT/(task+'_worst_packed.json'))}
 updates=[]
 for j in s['jobs']:
  if not j['id'].startswith(FAMILY+'/') or j['kind']!='recovery_train':continue
  if j['status'] not in ('waiting','ready'):raise ValueError('Only unstarted or explicitly interrupted training can change execution')
  before=copy.deepcopy(j['payload']['config']);after=before|selected[j['task']]['execution']
  if scientific_fingerprint(before)!=scientific_fingerprint(after):raise ValueError('Scientific recipe changed')
  validate_resume(before,after,1,1)
  j['payload'].update(config=after,config_sha256=config_fingerprint(after))
  j.update(execution_override=selected[j['task']]['execution'],performance_verification=selected[j['task']],reason='User-approved measured execution tuning')
  updates.append({'id':j['id'],'before':{k:before[k] for k in selected[j['task']]['execution']},'after':selected[j['task']]['execution'],'attempts_retained':len(j['attempts'])})
 if len(updates)!=9:raise ValueError('Expected exactly nine scientific training runs')
 result={'at':now(),'tasks':selected,'updates':updates,'effective_batch_unchanged':4,'test_set_used':False}
 if args.apply:
  atomic_json(OUT/'state_before_apply.json',store.load());store.save(s);store.event('operator_execution_tuning_applied',family=FAMILY,verification=str(OUT/'applied.json'),runs=len(updates));atomic_json(OUT/'applied.json',result)
 print(json.dumps(result,indent=2))
 return 0
if __name__=='__main__':raise SystemExit(main())
