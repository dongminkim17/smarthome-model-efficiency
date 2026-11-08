#!/usr/bin/env python3
"""Queue only a two-update MER AVT SFT/reload preflight; no full training."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import atomic_json,read_json,fingerprint
from smarthome_efficiency.data import load_records
from mer_avt_pipeline import source_config,chain,finish_spec
from project_queue import validate_submission
FAMILY='mer_avt_20261023_preflight'

def build_submission(state,root=ROOT):
 root=Path(root);cfg=source_config(root);rows=load_records('mer6','train',seed=cfg['split_seed'],data_root=cfg['data_root']);available={r['id'] for r in rows}
 previous=read_json(Path(state['queue_root'])/'stress_cases/mer6.json')
 if any(k not in available for k in previous['ids']):raise ValueError('Smoke IDs must remain in the training split')
 path=root/'artifacts/smoke'/FAMILY/'training_cases.json'
 atomic_json(path,{'split':'train','test_set_used':False,'ids':previous['ids'],
  'note':'Bounded AVT smoke on known large media. Full pipeline separately indexes all AVT training token sizes.'})
 jobs=[j for j in chain(FAMILY,state,root,'SFT_QV16',cfg,None,[]) if j['kind'] in ('smoke_train','smoke_test')]
 for j in jobs:
  j['priority']=80
  if j['kind']=='smoke_train':j['payload']['memory_cases']=str(path)
 return finish_spec(state,FAMILY,jobs,root,'Bounded AVT training-only two-update SFT plus updated-adapter reload; no official test or full run')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--submit',action='store_true');args=p.parse_args()
 q=ROOT/'artifacts/queue/suzy';s=read_json(q/'state.json');saved=q/'submissions'/(FAMILY+'.json')
 if saved.exists():spec=read_json(saved);validate_submission(s,spec)
 else:spec=build_submission(s);atomic_json(saved,spec)
 if args.submit:
  path=q/'inbox'/(fingerprint(spec)+'.json');atomic_json(path,spec);print('Submitted bounded AVT preflight only: '+str(path))
 else:print('Validated bounded AVT preflight: '+str(saved))
 return 0
if __name__=='__main__':raise SystemExit(main())
