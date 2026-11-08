#!/usr/bin/env python3
"""Cache frozen original teacher class logits on the training split only."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evaluate import resolve_evaluation_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Cache directory; final payload is cache.json')
    parser.add_argument('--limit',type=int,help='Training-only subset for smoke checks')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--mer-partition', choices=['nested_90_10'], help='Training partition for target extraction; teacher weights remain frozen')
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    import torch
    from smarthome_efficiency.checkpointing import write_json
    from smarthome_efficiency.data import LABELS,load_records,select_records
    from smarthome_efficiency.distillation import TeacherLogitCache,sample_fingerprint,teacher_spec
    from smarthome_efficiency.model import PilotModel
    from smarthome_efficiency.training import config_fingerprint,configure_determinism
    source=resolve_evaluation_checkpoint(args.teacher_checkpoint)
    cfg=json.loads((source/'metadata.json').read_text())['config']
    if args.mer_partition:
        if cfg['task'] != 'mer6': raise ValueError('MER partition requires a MER teacher')
        cfg['mer_partition'] = args.mer_partition
    configure_determinism(cfg);torch.set_num_threads(cfg.get('num_threads',2))
    labels=list(LABELS[cfg['task']])
    records=select_records(load_records(cfg['task'],'train',seed=cfg['split_seed'],data_root=cfg['data_root'],mer_partition=cfg.get('mer_partition')),args.limit,seed=cfg['split_seed'])
    runner=PilotModel(cfg['model'],cfg['task'],labels,cfg['cache_dir'],max_frames=cfg['max_frames'],
                      max_pixels=cfg['max_pixels'],use_transcript=cfg['use_transcript'],gradient_checkpointing=False)
    spec=teacher_spec(cfg,source,labels,runner.label_token_ids,runner.versions)
    request={'spec':spec,'record_fingerprints':{r['id']:sample_fingerprint(r) for r in records}}
    output=args.output.expanduser().resolve();payload_path=output/'cache.json'
    if payload_path.exists():
        if not args.resume:raise FileExistsError(payload_path)
        cached=TeacherLogitCache(payload_path,spec,records)
        print(json.dumps({'event':'cache_reused','examples':len(records),'sha256':cached.sha256}),flush=True)
        return 0
    if output.exists() and not args.resume:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    request_path=output/'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text())!=request:raise ValueError('Partial teacher cache input provenance changed')
    else:write_json(request_path,request)
    progress=output/'progress.jsonl';rows={};valid_lines=[]
    if progress.exists():
        lines=progress.read_text().splitlines()
        for index,line in enumerate(lines):
            try:value=json.loads(line)
            except json.JSONDecodeError:
                if index==len(lines)-1:break
                raise
            identifier=value['id']
            if identifier not in request['record_fingerprints'] or value['sample_sha256']!=request['record_fingerprints'][identifier]:
                raise ValueError('Partial teacher cache contains stale records')
            if identifier in rows:raise ValueError('Duplicate teacher cache records')
            if len(value['logits'])!=len(labels) or not all(math.isfinite(v) for v in value['logits']):
                raise ValueError('Invalid partial teacher logits')
            rows[identifier]={k:value[k] for k in ['sample_sha256','logits']};valid_lines.append(json.dumps(value,ensure_ascii=False))
        progress.write_text('\n'.join(valid_lines)+('\n' if valid_lines else ''))
    runner.load_adapter(source/'adapter',trainable=False)
    if any(p.requires_grad for p in runner.model.parameters()):raise RuntimeError('Teacher must be frozen')
    started=time.perf_counter()
    with progress.open('a') as stream:
        for record in records:
            identifier=record['id']
            if identifier in rows:continue
            # score() gets only the prompt/media; no gold answer is appended.
            values=runner.score(runner.prepare(record))
            row={'sample_sha256':request['record_fingerprints'][identifier],'logits':values}
            rows[identifier]=row
            stream.write(json.dumps({'id':identifier,**row},ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            if len(rows)%25==0 or len(rows)==len(records):
                print(json.dumps({'event':'teacher_cache_progress','completed':len(rows),'total':len(records),
                                  'seconds':time.perf_counter()-started}),flush=True)
        os.fsync(stream.fileno())
    payload={'status':'completed','spec':spec,'rows':rows,'num_examples':len(rows),
             'created_at_utc':datetime.now(timezone.utc).isoformat(),
             'cache_id':config_fingerprint({'spec':spec,'rows':rows})}
    write_json(payload_path,payload)
    verified=TeacherLogitCache(payload_path,spec,records)
    print(json.dumps({'event':'teacher_cache_complete','examples':len(rows),'path':str(payload_path),
                      'bytes':payload_path.stat().st_size,'sha256':verified.sha256,'test_set_used':False}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
