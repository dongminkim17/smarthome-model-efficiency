#!/usr/bin/env python3
"""Cache original frozen audio/vision outputs on the training split, resumably."""
from __future__ import annotations
import argparse,json,os,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--train-ids',type=Path,help='Eight training-only memory-smoke IDs; requires --limit 8')
    parser.add_argument('--mer-partition', choices=['nested_90_10'], help='Training partition for target extraction; teacher weights remain frozen')
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error('limit must be positive')
    import torch
    from evaluate import resolve_evaluation_checkpoint
    from smarthome_efficiency.checkpointing import write_json
    from smarthome_efficiency.data import LABELS,load_records,select_records
    from smarthome_efficiency.distillation import sample_fingerprint
    from smarthome_efficiency.feature_distillation import feature_spec,feature_filename,validate_features,TeacherFeatureCache
    from smarthome_efficiency.model import PilotModel
    from smarthome_efficiency.pruning import sha256_file
    from smarthome_efficiency.training import configure_determinism
    from smarthome_efficiency.queueing import exclusive_lock
    source=Path(resolve_evaluation_checkpoint(args.teacher_checkpoint));cfg=json.loads((source/'metadata.json').read_text())['config']
    if args.mer_partition:
        if cfg['task'] != 'mer6': raise ValueError('MER partition requires a MER teacher')
        cfg['mer_partition'] = args.mer_partition
    # Existing original teachers use decoder-only q/v LoRA. Encoder targets must
    # never be silently discarded by this encoder-only extraction path.
    adapter=json.loads((source/'adapter/adapter_config.json').read_text())
    targets=adapter['target_modules']
    if not isinstance(targets,str) or not targets.startswith('model\\.layers\\.'):
        raise ValueError('Feature teacher must have decoder-only adapters; use the established original teacher')
    configure_determinism(cfg);torch.set_num_threads(cfg.get('num_threads',2))
    all_records=load_records(cfg['task'],'train',seed=cfg['split_seed'],data_root=cfg['data_root'],mer_partition=cfg.get('mer_partition'))
    records=select_records(all_records,args.limit,seed=cfg['split_seed'])
    if args.train_ids is not None:
        if args.limit!=8:parser.error('--train-ids requires --limit 8')
        from train import memory_probe_records
        records=memory_probe_records(all_records,args.train_ids)
    labels=list(LABELS[cfg['task']]);runner=PilotModel(cfg['model'],cfg['task'],labels,cfg['cache_dir'],max_frames=cfg['max_frames'],max_pixels=cfg['max_pixels'],use_transcript=cfg['use_transcript'],gradient_checkpointing=False)
    spec=feature_spec(cfg,source,labels,runner.label_token_ids,runner.versions)
    request={'spec':spec,'record_fingerprints':{r['id']:sample_fingerprint(r) for r in records}}
    output=args.output.expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(output/'.cache.lock',blocking=False):
        final=output/'cache.json'
        if final.exists():
            if not args.resume:raise FileExistsError(final)
            cached=TeacherFeatureCache(output,spec,records)
            print(json.dumps({'event':'encoder_cache_reused','examples':len(records),'sha256':cached.sha256}),flush=True);return 0
        request_path=output/'request.json'
        if request_path.exists():
            if not args.resume or json.loads(request_path.read_text())!=request:raise ValueError('Partial encoder cache provenance changed')
        else:write_json(request_path,request)
        folder=output/'features';folder.mkdir(exist_ok=True);progress=output/'progress.jsonl';rows={};valid_lines=[]
        if progress.exists():
            lines=progress.read_text().splitlines()
            for index,line in enumerate(lines):
                try:row=json.loads(line)
                except ValueError:
                    if index==len(lines)-1:break
                    raise
                key=row['id'];path=folder/feature_filename(key)
                if (key not in request['record_fingerprints'] or key in rows
                        or row['sample_sha256']!=request['record_fingerprints'][key]
                        or row['file']!=feature_filename(key) or not path.is_file()
                        or row['sha256']!=sha256_file(path)):
                    raise ValueError('Partial encoder feature cache is stale/corrupt')
                value=torch.load(path,map_location='cpu',weights_only=True);validate_features(value)
                if {k:list(v.shape) for k,v in value.items()}!=row['shapes']:raise ValueError('Partial feature shape mismatch')
                rows[key]={k:v for k,v in row.items() if k!='id'};valid_lines.append(json.dumps(row))
            progress.write_text('\n'.join(valid_lines)+('\n' if valid_lines else ''))
        runner.load_model();runner.model.requires_grad_(False).eval()
        started=time.perf_counter()
        with progress.open('a') as stream:
            for record in records:
                key=record['id']
                if key in rows:continue
                features=runner.encode_media(runner.prepare(record));filename=feature_filename(key)
                path=folder/filename;tmp=path.with_name('.'+filename+'.'+uuid.uuid4().hex+'.tmp')
                try:
                    with tmp.open('wb') as f:torch.save(features,f);f.flush();os.fsync(f.fileno())
                    tmp.replace(path)
                finally:tmp.unlink(missing_ok=True)
                row={'sample_sha256':sample_fingerprint(record),'file':filename,'sha256':sha256_file(path),
                     'shapes':{k:list(v.shape) for k,v in features.items()},'bytes':path.stat().st_size}
                rows[key]=row;stream.write(json.dumps({'id':key,**row})+'\n');stream.flush();os.fsync(stream.fileno())
                if len(rows)%25==0 or len(rows)==len(records):
                    print(json.dumps({'event':'encoder_cache_progress','completed':len(rows),'total':len(records),'seconds':time.perf_counter()-started}),flush=True)
        write_json(final,{'status':'completed','spec':spec,'rows':rows,'num_examples':len(rows),
                          'feature_bytes':sum(r['bytes'] for r in rows.values()),'test_set_used':False})
        verified=TeacherFeatureCache(output,spec,records)
        print(json.dumps({'event':'encoder_cache_complete','examples':len(records),'sha256':verified.sha256,
                          'feature_bytes':sum(r['bytes'] for r in rows.values()),'path':str(output),'test_set_used':False}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
