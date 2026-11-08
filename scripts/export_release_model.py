#!/usr/bin/env python3
"""Merge a final task adapter and export a portable, verified classification model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from evaluate import resolve_evaluation_checkpoint
from launch_pruning_followup import pin_best
from smarthome_efficiency.checkpointing import write_json
from smarthome_efficiency.pruning import PROCESSOR_FILES, parameter_counts, sha256_file
from smarthome_efficiency.release import (
    BASE_MODEL, BASE_REVISION, PARAMETERS, task_settings, validate_package,
)


def validate_source(config):
    expected = {'epochs': 5, 'seed': 42, 'split_seed': 42, 'lora_rank': 64, 'lora_alpha': 128,
                'lora_target': 'decoder_encoder_linear', 'objective': 'sft_kd',
                'distillation_alpha': .75, 'distillation_temperature': 2.,
                'feature_audio_weight': .1, 'feature_vision_weight': .1,
                'learning_rate': 1e-4, 'encoder_learning_rate': 1e-5}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Source does not match the chosen final recovery recipe')
    if config.get('train_limit') is not None or config.get('dev_limit') is not None or config.get('mer_partition'):
        raise ValueError('Release requires the established full-data task protocol')
    task_settings(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verification-examples', type=int, default=2)
    args = parser.parse_args()
    if args.verification_examples < 1:
        parser.error('--verification-examples must be positive')
    source = resolve_evaluation_checkpoint(args.checkpoint)
    if pin_best(source.parent.parent) != str(source):
        raise ValueError('Release requires the completed run validation winner')
    metadata = json.loads((source/'metadata.json').read_text())
    config = metadata['config']
    validate_source(config)
    adapter_hash = sha256_file(source/'adapter/adapter_model.safetensors')
    output = args.output.expanduser().resolve()
    if output.exists():
        manifest, _ = validate_package(output, verify_hashes=True)
        if manifest['source_adapter_sha256'] != adapter_hash or manifest['repo_id'] != args.repo_id:
            raise ValueError('Existing release has a different source or repository')
        print(json.dumps({'status': 'reused', 'output': str(output)}), flush=True)
        return
    import torch
    from smarthome_efficiency.data import LABELS, load_records, select_records
    from smarthome_efficiency.model import PilotModel
    from smarthome_efficiency.training import configure_determinism
    configure_determinism(config)
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['num_threads'])
    started = time.perf_counter()
    temporary = output.with_name('.'+output.name+'.export-'+uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    try:
        labels = list(LABELS[config['task']])
        records = select_records(load_records(config['task'], 'train', data_root=config['data_root'],
                                             seed=config['split_seed']), args.verification_examples, seed=config['split_seed'])
        runner = PilotModel(config['model'], config['task'], labels, config['cache_dir'],
                            max_frames=config['max_frames'], max_pixels=config['max_pixels'],
                            use_transcript=config['use_transcript'], gradient_checkpointing=False)
        batches = [runner.prepare(record) for record in records]
        runner.load_adapter(source/'adapter', trainable=False)
        before = [runner.score(batch) for batch in batches]
        runner.model = runner.model.merge_and_unload(safe_merge=True)
        runner.lora_enabled = False
        if hasattr(runner.model, 'peft_config'):
            del runner.model.peft_config
        runner.model.requires_grad_(False).eval()
        if parameter_counts(runner.model) != PARAMETERS:
            raise ValueError('Merged model component counts differ')
        exported = [runner.evaluate_batch(batch, labels.index(record['label']), modes='both', max_new_tokens=8)
                  for record, batch in zip(records, batches)]
        export_difference = max(abs(a-b) for old, new in zip(before, exported) for a, b in zip(old, new['logit_scores']))
        runner.model.save_pretrained(temporary, safe_serialization=True, max_shard_size='2GB')
        base = Path(config['model'])
        model_config = json.loads((base/'config.json').read_text())
        model_config.pop('_name_or_path', None)
        model_config.pop('_commit_hash', None)
        write_json(temporary/'config.json', model_config)
        for filename in PROCESSOR_FILES:
            if (base/filename).is_file():
                shutil.copyfile(base/filename, temporary/filename)
        shutil.copyfile(base/'LICENSE', temporary/'LICENSE')
        (temporary/'NOTICE').write_text(
            'Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, '
            'Copyright (c) Alibaba Cloud. All Rights Reserved.\n\n'
            'Built with Qwen. Modified by Sung-Ju Lee, Dong Min Kim, Ryuhaerang Choi, '
            'Hyungjun Yoon and Seohyeon Yoo.\n'
            'Changes include task fine-tuning, decoder FFN channel pruning, '
            'knowledge distillation with final decoder/encoder LoRA adapters.\n')
        write_json(temporary/'task_config.json', task_settings(config))
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                 for p in sorted(temporary.iterdir()) if p.is_file()}
        manifest = {'format_version': 1, 'format': 'merged_thinker', 'task': config['task'],
                    'repo_id': args.repo_id, 'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
                    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'parameters': PARAMETERS,
                    'pruning': {'method': 'ffn_activation', 'decoder_layers': 36, 'ffn_width': 4352,
                                'calibration_samples': 128, 'calibration_seed': 42},
                    'source_adapter_sha256': adapter_hash,
                    'source_metadata_sha256': sha256_file(source/'metadata.json'),
                    'source_global_step': metadata['global_step'], 'files': files,
                    'verification': {'split': 'train', 'examples': len(records),
                                     'export_max_abs_logit_difference': export_difference}}
        write_json(temporary/'release.json', manifest)
        runner.model = None
        del runner
        gc.collect()
        torch.cuda.empty_cache()
        reload = PilotModel(temporary, config['task'], labels, config['cache_dir'],
                            max_frames=config['max_frames'], max_pixels=config['max_pixels'],
                            use_transcript=config['use_transcript'], gradient_checkpointing=False)
        reload.load_model()
        if parameter_counts(reload.model) != PARAMETERS:
            raise ValueError('Reloaded model component counts differ')
        for record, batch, expected in zip(records, batches, exported):
            actual = reload.evaluate_batch(batch, labels.index(record['label']), modes='both', max_new_tokens=8)
            torch.testing.assert_close(torch.tensor(actual['logit_scores']), torch.tensor(expected['logit_scores']), atol=1e-5, rtol=1e-5)
            if actual['generated_token_ids'] != expected['generated_token_ids']:
                raise ValueError('Save/reload generated tokens differ')
        manifest['verification'].update(save_reload_logits_match=True, save_reload_generation_match=True)
        write_json(temporary/'release.json', manifest)
        validate_package(temporary)
        temporary.rename(output)
        print(json.dumps({'status': 'completed', 'task': config['task'], 'output': str(output),
                          'seconds': time.perf_counter()-started, 'export_difference': export_difference}), flush=True)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == '__main__':
    main()
