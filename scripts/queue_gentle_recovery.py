#!/usr/bin/env python3
"""Nine approved gentler recovery runs, additive submission to Suzy1–7."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import atomic_json,read_json,fingerprint
from project_queue import validate_submission
from queue_recovery_refinements import build_submission as build_refinements
FAMILY='gentle_recovery_20261023'
VARIANTS=(
 ('LM3E5_ENC1E5',False,{'learning_rate':3e-5},30),
 ('LM1E5_ENC1E6',False,{'learning_rate':1e-5,'encoder_learning_rate':1e-6},30),
 ('FEATURE003',False,{'feature_audio_weight':.03,'feature_vision_weight':.03},30),
)
def build_submission(state,root=ROOT):
 spec=build_refinements(state,root,family=FAMILY,variants=VARIANTS,binary_eval_fraction=.1)
 doc=next(iter(spec['documents'].values()))
 doc['description']='Nine approved runs: two gentler LR settings and weaker featureKD, each on all three tasks'
 doc['note']='Fresh rank16 adapters on fixed sequential balanced exports; original training-only teachers unchanged. Five epochs; binary validation every0.1epoch, other tasks every0.5epoch. Dense merged parameter counts unchanged. Validation-only selection; official dual-mode tests follow automatically.'
 validate_submission(state,spec);return spec

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/suzy');p.add_argument('--submit',action='store_true');args=p.parse_args()
 state=read_json(args.queue_root/'state.json');saved=args.queue_root/'submissions'/f'{FAMILY}.json'
 if saved.exists():spec=read_json(saved);validate_submission(state,spec)
 else:spec=build_submission(state);atomic_json(saved,spec)
 if args.submit:
  path=args.queue_root/'inbox'/(fingerprint(spec)+'.json');atomic_json(path,spec);print('Submitted nine training runs with18 stress/reload gates and9 official tests: '+str(path))
 else:print('Validated nine-run submission: '+str(saved))
 return 0
if __name__=='__main__':raise SystemExit(main())
