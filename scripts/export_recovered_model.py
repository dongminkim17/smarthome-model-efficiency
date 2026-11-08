#!/usr/bin/env python3
"""Merge a validation-selected LM recovery into a native, unchanged-geometry base."""
from __future__ import annotations
import argparse,copy,gc,json,shutil,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from launch_pruning_followup import checkpoint_identity,pin_best
from smarthome_efficiency.pruning import PROCESSOR_FILES,parameter_counts,processor_cache_identity,processor_hashes,sha256_file
from smarthome_efficiency.queueing import atomic_json,read_json,now


def validate_merge_export(output, source, expected_counts=None):
    output,source=Path(output),Path(source)
    report=read_json(output/'compression.json');meta=read_json(source/'metadata.json')
    upstream=Path(meta['config']['model'])
    if (report.get('method')!='merged_lm_recovery' or report.get('source_checkpoint')!=str(source.resolve())
            or report.get('source_identity')!=checkpoint_identity(source)
            or report.get('upstream_compression_sha256')!=sha256_file(upstream/'compression.json')
            or report.get('parameters_after')!=report.get('parameters_before')
            or report.get('encoder_tensors_unchanged') is not True
            or report.get('verification_split')!='train' or report.get('test_set_used') is not False
            or report.get('parameter_limits')!={'language':2_000_000_000,'audio_encoder':1_000_000_000,'vision_encoder':1_000_000_000}):
        raise ValueError('Merged recovery export provenance or geometry differs')
    counts=report['parameters_after']
    if (expected_counts is not None and counts!=expected_counts) or any(not 0<counts[k]<=cap for k,cap in report['parameter_limits'].items()):
        raise ValueError('Merged model violates expected component counts')
    cfg=read_json(output/'config.json');old=read_json(upstream/'config.json')
    if cfg!=old:raise ValueError('Merge changed the native configuration')
    if counts!=read_json(upstream/'compression.json')['parameters_after']:
        raise ValueError('Merge changed upstream parameter counts')
    v=report.get('verification',{})
    if not all(v.get(k) is True for k in ('passed','save_reload_class_logits_match','save_reload_generation_match')):
        raise ValueError('Merge export lacks native save/reload verification')
    if not (output/'model.safetensors').is_file() and not (output/'model.safetensors.index.json').is_file():
        raise ValueError('Merged export weights missing')
    if processor_cache_identity(output)!=processor_cache_identity(upstream):raise ValueError('Merge changed processor identity')
    return {'report':str(output/'compression.json'),'model':str(output),'language_parameters':counts['language'],'total_parameters':counts['total'],'parameters':counts}



def validate_source_config(cfg):
    """Support the frozen AV MER and AVT MIntRec protocols with LM-only LoRA."""
    task=cfg.get('task')
    if (task not in ('mer6','mintrec2','mintrec20') or cfg.get('lora_target')!='decoder_linear'
            or cfg.get('mer_partition') or cfg.get('train_limit') is not None or cfg.get('dev_limit') is not None
            or cfg.get('use_transcript') is not (task!='mer6')):
        raise ValueError('Expected a full-data decoder-only recovery with the established task input protocol')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();source=args.source_checkpoint.resolve();output=args.output.resolve()
    # Resolve a completed training run, then pin its ordinary best-dev checkpoint.
    if pin_best(source.parent.parent)!=str(source):raise ValueError('Merge requires the completed run\'s validation winner')
    metadata=read_json(source/'metadata.json');cfg=metadata['config'];upstream=Path(cfg['model']);parent=read_json(upstream/'compression.json')
    validate_source_config(cfg)
    if parent['parameters_after']['audio_encoder']!=637676544 or parent['parameters_after']['vision_encoder']!=668684288:
        raise ValueError('Source does not retain the original encoders')
    if output.exists():print(json.dumps(validate_merge_export(output,source,parent['parameters_after'])));return 0
    import torch
    from smarthome_efficiency.model import PilotModel
    from smarthome_efficiency.data import LABELS,load_records,select_records
    from smarthome_efficiency.training import configure_determinism
    configure_determinism(cfg);torch.set_num_threads(cfg['num_threads']);started=time.perf_counter()
    temporary=output.with_name('.'+output.name+'.merging-'+uuid.uuid4().hex);temporary.mkdir(parents=True)
    try:
        labels=list(LABELS[cfg['task']]);records=select_records(load_records(cfg['task'],'train',seed=cfg['split_seed'],data_root=cfg['data_root']),2,seed=cfg['split_seed'])
        runner=PilotModel(cfg['model'],cfg['task'],labels,cfg['cache_dir'],max_frames=cfg['max_frames'],max_pixels=cfg['max_pixels'],use_transcript=cfg['use_transcript'],gradient_checkpointing=False)
        batches=[runner.prepare(r) for r in records];runner.load_adapter(source/'adapter',trainable=False)
        base=runner.model.get_base_model()
        encoder_before=[(n,p.data_ptr(),p._version) for n,p in base.named_parameters() if n.startswith(('audio_tower.','visual.'))]
        prior=[runner.score(b) for b in batches]
        runner.model=runner.model.merge_and_unload(safe_merge=True);runner.lora_enabled=False
        if hasattr(runner.model,'peft_config'):del runner.model.peft_config
        runner.model.requires_grad_(False).eval()
        if encoder_before!=[(n,p.data_ptr(),p._version) for n,p in runner.model.named_parameters() if n.startswith(('audio_tower.','visual.'))]:raise RuntimeError('Merge modified encoders')
        counts=parameter_counts(runner.model)
        if counts!=parent['parameters_after']:raise RuntimeError('Merge changed parameter counts')
        merged=[runner.evaluate_batch(b,labels.index(r['label']),modes='both',max_new_tokens=8) for r,b in zip(records,batches)]
        difference=max(abs(a-b) for old,new in zip(prior,merged) for a,b in zip(old,new['logit_scores']))
        runner.model.save_pretrained(temporary,safe_serialization=True,max_shard_size='2GB')
        # Preserve source JSON exactly: geometry and processor did not change.
        shutil.copyfile(upstream/'config.json',temporary/'config.json')
        for name in (*PROCESSOR_FILES,'LICENSE','LICENSE.txt','NOTICE'):
            if (upstream/name).is_file():shutil.copyfile(upstream/name,temporary/name)
        manifest={'format_version':1,'method':'merged_lm_recovery','task':cfg['task'],'created_at_utc':now(),
            'source_checkpoint':str(source),'source_identity':checkpoint_identity(source),'source_model':str(upstream),
            'upstream_compression_sha256':sha256_file(upstream/'compression.json'),'pruning_method':parent.get('pruning_method',parent['method']),
            'ffn_width':parent['ffn_width'],'parameters_before':counts,'parameters_after':counts,
            'encoder_tensors_unchanged':True,'architecture_unchanged':True,'existing_task_adapter_merged':True,
            'processor_cache_identity':processor_cache_identity(upstream),'processor_files_sha256':processor_hashes(upstream),
            'verification_split':'train','test_set_used':False,'parameter_limits':{'language':2_000_000_000,'audio_encoder':1_000_000_000,'vision_encoder':1_000_000_000},
            'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha256_file(Path(__file__))}}
        atomic_json(temporary/'compression.json',manifest)
        runner.model=None;del base,runner;gc.collect();torch.cuda.empty_cache()
        reload=PilotModel(temporary,cfg['task'],labels,cfg['cache_dir'],max_frames=cfg['max_frames'],max_pixels=cfg['max_pixels'],use_transcript=cfg['use_transcript'],gradient_checkpointing=False)
        reload.load_model()
        if parameter_counts(reload.model)!=counts:raise RuntimeError('Reload changed parameters')
        for record,batch,expected in zip(records,batches,merged):
            actual=reload.evaluate_batch(batch,labels.index(record['label']),modes='both',max_new_tokens=8)
            torch.testing.assert_close(torch.tensor(actual['logit_scores']),torch.tensor(expected['logit_scores']),atol=1e-5,rtol=1e-5)
            if actual['generated_token_ids']!=expected['generated_token_ids']:raise RuntimeError('Reload changed generation')
        manifest['verification']={'passed':True,'save_reload_class_logits_match':True,'save_reload_generation_match':True,'examples':[r['id'] for r in records],'bf16_merge_max_abs_class_logit_difference':difference}
        manifest['export_seconds']=time.perf_counter()-started;atomic_json(temporary/'compression.json',manifest)
        temporary.rename(output);print(json.dumps(validate_merge_export(output,source,counts)));return 0
    finally:
        if temporary.exists():shutil.rmtree(temporary)

if __name__=='__main__':raise SystemExit(main())
