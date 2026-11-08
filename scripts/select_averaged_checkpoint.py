#!/usr/bin/env python3
"""Compare a best-dev checkpoint and two exact LoRA averages on full dev only."""
from __future__ import annotations
import argparse,copy,json,os,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.adapter_averaging import average_factors,averaging_windows
from smarthome_efficiency.checkpointing import atomic_link,write_json
from smarthome_efficiency.data import LABELS,load_records
from smarthome_efficiency.evaluation import classification_metrics
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.queueing import read_json
from launch_pruning_followup import checkpoint_identity,pin_best


def make_average(checkpoints,output):
    from safetensors.torch import load_file,save_file
    output=Path(output)
    metadata=[read_json(Path(c)/'metadata.json') for c in checkpoints]
    if any(m['config']!=metadata[0]['config'] or m.get('data_fingerprint')!=metadata[0].get('data_fingerprint')
           or m.get('base_model_manifest_sha256')!=metadata[0].get('base_model_manifest_sha256') for m in metadata):
        raise ValueError('Cannot average adapters from differing bases, datasets or recipes')
    request={'sources':[{'checkpoint':str(Path(c).resolve()),'identity':checkpoint_identity(c)} for c in checkpoints],
             'weights':[1/len(checkpoints)]*len(checkpoints),'method':'exact average of effective scaled B@A updates',
             'dense_architecture_unchanged':True,'optimizer_state_available':False}
    if output.exists():
        saved=read_json(output/'averaging.json')
        if saved['request']!=request or saved['identity']!=checkpoint_identity(output):raise ValueError('Saved average changed')
        return output
    states=[load_file(str(Path(c)/'adapter/adapter_model.safetensors')) for c in checkpoints]
    configs=[read_json(Path(c)/'adapter/adapter_config.json') for c in checkpoints]
    factors,config=average_factors(states,configs)
    output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_name('.averaging-'+uuid.uuid4().hex)
    (tmp/'adapter').mkdir(parents=True)
    save_file(factors,str(tmp/'adapter/adapter_model.safetensors'),metadata={'format':'pt'})
    write_json(tmp/'adapter/adapter_config.json',config)
    m=copy.deepcopy(metadata[0]);m['config'].update(lora_rank=config['r'],lora_alpha=config['lora_alpha'])
    m['checkpoint_averaging']=request
    write_json(tmp/'metadata.json',m)
    write_json(tmp/'averaging.json',{'request':request,'identity':checkpoint_identity(tmp)})
    tmp.rename(output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source_run.resolve();output=args.output.resolve()
    report=read_json(source/'report.json');cfg=report['config']
    if (report['status']!='completed' or report.get('test_set_used') is not False
            or cfg.get('train_limit') is not None or cfg.get('dev_limit') is not None or cfg['selection_mode']!='logit'):
        raise ValueError('Averaging requires completed full-data validation-only training')
    if (output/'report.json').exists():
        final=read_json(output/'report.json')
        if final['status']!='completed' or final['source_run']!=str(source):raise ValueError('Stale average selection')
        pin_best(output);print(json.dumps({'event':'averaging_reused','output':str(output)}),flush=True);return 0
    dev=load_records(cfg['task'],'dev',seed=cfg['split_seed'],data_root=cfg['data_root'],mer_partition=cfg.get('mer_partition'))
    if report['dev_count']!=len(dev):raise ValueError('Development count changed')
    selected=pin_best(source);best_step=report['best_step']
    predictions=[json.loads(line) for line in (source/'predictions'/f'dev-step-{best_step:08d}.jsonl').read_text().splitlines()]
    expected={r['id']:r['label'] for r in dev}
    if (len(predictions)!=len(dev) or {r['id']:r['true_label'] for r in predictions}!=expected):
        raise ValueError('Saved original predictions do not match full development labels/IDs')
    metric='weighted_f1' if cfg['task']=='mer6' else 'accuracy'
    score=classification_metrics([r['true_label'] for r in predictions],[r['logit_predicted_label'] for r in predictions],LABELS[cfg['task']])[metric]
    if abs(score-report['best_score'])>1e-12:raise ValueError('Saved best-dev score cannot be reproduced')
    checkpoints={int(p.name[5:]):p for p in (source/'checkpoints').glob('step-*') if p.name[5:].isdigit()}
    windows=averaging_windows(checkpoints,best_step)
    request={'source_run':str(source),'source_best_identity':checkpoint_identity(selected),
             'windows':windows,'dev_ids':sorted(expected),'test_set_used':False}
    output.mkdir(parents=True,exist_ok=True)
    if (output/'request.json').exists() and read_json(output/'request.json')!=request:raise ValueError('Averaging request changed')
    write_json(output/'request.json',request)
    candidates=[{'name':'single_best','score':score,'checkpoint':selected,'identity':checkpoint_identity(selected)}]
    for label,steps in windows.items():
        checkpoint=make_average([checkpoints[s] for s in steps],output/'averages'/label)
        evaluations=output/'validation'/label
        evaluations.mkdir(parents=True,exist_ok=True)
        completed=[p for p in sorted(evaluations.glob('attempt-*')) if (p/'report.json').exists()
                   and read_json(p/'report.json').get('status')=='completed']
        evaluation=completed[-1] if completed else evaluations/f'attempt-{len(list(evaluations.glob("attempt-*")))+1:03d}'
        if not completed:
            subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate.py'),'--checkpoint',str(checkpoint),
                            '--split','dev','--modes','logit','--output',str(evaluation)],check=True,cwd=ROOT)
        result=read_json(evaluation/'report.json')
        if (result['status']!='completed' or result['split']!='dev' or result['test_set_used'] is not False
                or result['partial_split'] or result['evaluated_count']!=len(dev) or result['checkpoint']!=str(checkpoint)):
            raise ValueError('Averaged candidate must cover full development only')
        candidates.append({'name':label,'score':result['logit'][metric],'checkpoint':str(checkpoint),
                           'identity':checkpoint_identity(checkpoint),'steps':steps,'report':str(evaluation/'report.json')})
    # Prefer the original checkpoint on ties; no unnecessary change without improvement.
    winner=max(candidates,key=lambda c:c['score'])
    chosen_metadata=read_json(Path(winner['checkpoint'])/'metadata.json')
    (output/'checkpoints').mkdir(exist_ok=True);atomic_link(output/'checkpoints/best',Path(winner['checkpoint']))
    write_json(output/'report.json',{'status':'completed','test_set_used':False,'source_run':str(source),
        'config':chosen_metadata['config'],'dev_count':len(dev),'train_count':report['train_count'],
        'best_score':winner['score'],'best_step':chosen_metadata['global_step'],
        'selection':'full development logit metric; single best or two predeclared exact-update averages',
        'selected_candidate':winner['name'],'candidates':candidates,
        'original_config':cfg,'original_best_score':report['best_score'],
        'source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in
            (Path(__file__).resolve(),ROOT/'src/smarthome_efficiency/adapter_averaging.py')},
        'dense_architecture_unchanged':True,'checkpoint_identity':checkpoint_identity(winner['checkpoint'])})
    print(json.dumps({'event':'averaging_complete','winner':winner['name'],'score':winner['score'],
                      'original_score':score,'output':str(output)}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
