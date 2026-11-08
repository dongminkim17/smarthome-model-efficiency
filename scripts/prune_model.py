#!/usr/bin/env python3
"""Merge a saved task adapter, prune language weights, and export a verified Thinker.

Calibration/verification opens training data only. Full pre-recovery validation
is performed automatically at step zero of scripts/train.py. No test data here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evaluate import resolve_evaluation_checkpoint
from smarthome_efficiency.pruning import (
    FORMAT_VERSION, METHODS, PROCESSOR_FILES, apply_pruning, parameter_counts,
    processor_cache_identity, processor_hashes, sha256_file,
)


def log(event, **values):
    print(json.dumps({'event': event, **values}, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--keep-layers', type=int)
    parser.add_argument('--ffn-width', type=int)
    parser.add_argument('--audio-ffn-width', type=int)
    parser.add_argument('--vision-ffn-width', type=int)
    parser.add_argument('--audio-layers', type=int)
    parser.add_argument('--vision-layers', type=int)
    parser.add_argument('--budget-scope', choices=['language', 'total', 'separate'], default='language')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verification-examples', type=int, default=2)
    parser.add_argument('--calibration-samples', type=int, default=128)
    parser.add_argument('--calibration-seed', type=int, default=42)
    parser.add_argument('--calibration-cache', type=Path)
    parser.add_argument('--reuse', action='store_true', help='Accept an existing verified matching export')
    args = parser.parse_args()
    if args.verification_examples < 1:
        parser.error('--verification-examples must be positive')
    if args.method == 'depth_uniform' and (args.keep_layers is None or args.ffn_width is not None):
        parser.error('depth_uniform requires --keep-layers only')
    if args.method == 'ffn_magnitude' and (args.ffn_width is None or args.keep_layers is not None):
        parser.error('ffn_magnitude requires --ffn-width only')
    hybrid = args.method == 'multimodal_depth_ffn_activation'
    encoder_only = args.method == 'encoder_ffn_activation'
    multimodal = hybrid or encoder_only or args.method == 'multimodal_ffn_activation'
    joint = args.method == 'multimodal_ffn_activation' and args.budget_scope == 'separate'
    if hybrid:
        if any(type(v) is not int or v<2 for v in (args.audio_layers,args.vision_layers)):
            parser.error('Hybrid pruning requires --audio-layers and --vision-layers')
    elif args.audio_layers is not None or args.vision_layers is not None:
        parser.error('Encoder layers require multimodal_depth_ffn_activation')
    if multimodal:
        if (args.budget_scope not in (('separate',) if encoder_only else ('total', 'separate') if not hybrid else ('total',)) or args.keep_layers is not None
                or any(type(v) is not int or v < 1 for v in (args.ffn_width, args.audio_ffn_width, args.vision_ffn_width))):
            parser.error('Multimodal FFN pruning requires all three positive widths and the matching total/separate budget scope')
    elif args.audio_ffn_width is not None or args.vision_ffn_width is not None or args.budget_scope != 'language':
        parser.error('Encoder widths/total budget require multimodal_ffn_activation')
    calibrated = multimodal or args.method in ('depth_bi','ffn_activation','depth_ffn_activation')
    if args.method == 'depth_bi' and (args.keep_layers is None or args.ffn_width is not None):
        parser.error('depth_bi requires --keep-layers only')
    if args.method == 'ffn_activation' and (args.ffn_width is None or args.keep_layers is not None):
        parser.error('ffn_activation requires --ffn-width only')
    if args.method == 'depth_ffn_activation' and (args.keep_layers is None or args.ffn_width is None):
        parser.error('depth_ffn_activation requires both --keep-layers and --ffn-width')
    if args.calibration_samples < 1 or (calibrated and args.calibration_cache is None):
        parser.error('Calibrated methods require positive --calibration-samples and --calibration-cache')
    source = resolve_evaluation_checkpoint(args.source_checkpoint)
    metadata = json.loads((source / 'metadata.json').read_text())
    cfg = metadata['config']
    output = args.output.expanduser().resolve()
    request = {'method': args.method, 'keep_layers': args.keep_layers, 'ffn_width': args.ffn_width,
               'source_checkpoint': str(source),
               'source_adapter_sha256': sha256_file(source / 'adapter/adapter_model.safetensors'),
               'source_metadata_sha256': sha256_file(source / 'metadata.json')}
    if multimodal:
        request.update(audio_ffn_width=args.audio_ffn_width, vision_ffn_width=args.vision_ffn_width, budget_scope=args.budget_scope)
    if hybrid:
        request.update(audio_layers=args.audio_layers, vision_layers=args.vision_layers)
    if calibrated:
        request.update(calibration_samples=args.calibration_samples, calibration_seed=args.calibration_seed,
                       calibration_cache=str(args.calibration_cache.expanduser().resolve()))
    if output.exists():
        if not args.reuse:
            raise FileExistsError(output)
        manifest = json.loads((output / 'compression.json').read_text())
        if any(manifest.get(k) != v for k, v in request.items()) or not manifest.get('verification', {}).get('passed'):
            raise ValueError('Existing export does not match this pruning request or was not verified')
        processor_cache_identity(output)
        if calibrated:
            calibration = manifest.get('calibration', {})
            if calibration.get('sha256') != sha256_file(args.calibration_cache):
                raise ValueError('Calibration cache changed since compact export')
        if not (output / 'model.safetensors').exists() and not (output / 'model.safetensors.index.json').exists():
            raise ValueError('Existing export has no model weights')
        log('reuse_verified_export', output=str(output), parameters=manifest['parameters_after'])
        return 0

    import torch
    from smarthome_efficiency.checkpointing import write_json
    from smarthome_efficiency.data import LABELS, load_records, select_records
    from smarthome_efficiency.model import PilotModel
    from smarthome_efficiency.training import configure_determinism

    configure_determinism(cfg)
    torch.set_num_threads(cfg.get('num_threads', 2))
    torch.manual_seed(cfg.get('seed', 42))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.exporting-{uuid.uuid4().hex}')
    temporary.mkdir()
    started = time.perf_counter()
    try:
        labels = list(LABELS[cfg['task']])
        records = select_records(load_records(cfg['task'], 'train', seed=cfg['split_seed'], data_root=cfg['data_root'], mer_partition=cfg.get('mer_partition')),
                                 args.verification_examples, seed=cfg['split_seed'])
        runner = PilotModel(cfg['model'], cfg['task'], labels, cfg['cache_dir'],
                            max_frames=cfg['max_frames'], max_pixels=cfg['max_pixels'],
                            use_transcript=cfg['use_transcript'], gradient_checkpointing=False)
        batches = [runner.prepare(record) for record in records]
        runner.load_adapter(source / 'adapter', trainable=False)
        before_merge = [runner.score(batch) for batch in batches]
        runner.model = runner.model.merge_and_unload(safe_merge=True)
        runner.lora_enabled = False
        if hasattr(runner.model, 'peft_config'):
            del runner.model.peft_config
        runner.model.requires_grad_(False).eval()
        after_merge = [runner.score(batch) for batch in batches]
        merge_difference = max(abs(a-b) for row_a, row_b in zip(before_merge, after_merge) for a,b in zip(row_a,row_b))
        log('adapter_merged', source=str(source), bf16_max_abs_class_logit_difference=merge_difference)
        before = parameter_counts(runner.model)
        from smarthome_efficiency.multimodal_pruning import is_prunable_ffn_tensor
        if encoder_only:
            from smarthome_efficiency.encoder_pruning import is_prunable_encoder_tensor as is_prunable_ffn_tensor
        protected_before = [(n, p.data_ptr(), p._version) for n, p in runner.model.named_parameters()
                            if multimodal and not hybrid and not is_prunable_ffn_tensor(n)]
        encoders_before = [(n, p.data_ptr(), p._version) for n,p in runner.model.named_parameters()
                           if n.startswith(('audio_tower.', 'visual.'))]
        importance = None
        calibration = None
        if calibrated:
            if hybrid:
                from smarthome_efficiency.hybrid_pruning import obtain_hybrid_calibration as obtain_calibration
            elif multimodal:
                from smarthome_efficiency.multimodal_pruning import obtain_multimodal_calibration as obtain_calibration
            else:
                from smarthome_efficiency.pruning_calibration import obtain_calibration
            calibration_records = select_records(load_records(cfg['task'], 'train', seed=cfg['split_seed'], data_root=cfg['data_root'], mer_partition=cfg.get('mer_partition')),
                                                 args.calibration_samples, seed=args.calibration_seed)
            importance, calibration = obtain_calibration(runner, cfg, source, calibration_records, args.calibration_seed,
                                                        args.calibration_cache.expanduser().resolve(),
                                                        progress=lambda done,total:log('calibration_progress',completed=done,total=total))
            log('calibration_complete',examples=len(calibration_records),reused=calibration['reused'],sha256=calibration['sha256'])
        selection = apply_pruning(runner.model, args.method, keep_layers=args.keep_layers, ffn_width=args.ffn_width, importance=importance,
                                  audio_ffn_width=args.audio_ffn_width, vision_ffn_width=args.vision_ffn_width,
                                  audio_layers=args.audio_layers, vision_layers=args.vision_layers)
        if calibrated:
            selection['calibration'] = calibration
        after = parameter_counts(runner.model)
        encoders_after = [(n, p.data_ptr(), p._version) for n,p in runner.model.named_parameters()
                          if n.startswith(('audio_tower.', 'visual.'))]
        if not multimodal and encoders_before != encoders_after:
            raise RuntimeError('Pruning modified audiovisual encoder weights')
        if multimodal and not hybrid:
            protected_after = [(n, p.data_ptr(), p._version) for n, p in runner.model.named_parameters()
                               if not is_prunable_ffn_tensor(n)]
            if protected_before != protected_after:
                raise RuntimeError('Pruning modified a non-FFN tensor or AV interface')
        if joint:
            from smarthome_efficiency.joint_pruning import validate_joint_budget
            validate_joint_budget(before, after)
        elif encoder_only:
            from smarthome_efficiency.encoder_pruning import validate_separate_budget
            validate_separate_budget(before, after)
        elif not 0 < after[args.budget_scope] < 2_000_000_000 or after[args.budget_scope] >= before[args.budget_scope]:
            raise ValueError(f'Export must reduce {args.budget_scope} parameters strictly below 2B: {after}')
        log('pruned', method=args.method, parameters_before=before, parameters_after=after)
        expected = [runner.evaluate_batch(batch, labels.index(record['label']), modes='both', max_new_tokens=8)
                    for batch, record in zip(batches, records)]
        manifest = {'format_version': FORMAT_VERSION, **request, 'task': cfg['task'],
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'source_model': cfg['model'], 'parameters_before': before, 'parameters_after': after,
                    'language_parameter_scope': 'input embeddings + decoder + final norm + output head; AV encoders excluded',
                    'existing_task_adapter_merged': True, 'recovery_updates': 0,
                    'processor_cache_identity': runner.model_fingerprint,
                    'processor_files_sha256': processor_hashes(Path(cfg['model'])),
                    'encoder_tensors_unchanged': not multimodal,
                    'non_ffn_tensors_unchanged': (not hybrid) if multimodal else None,
                    'total_parameter_scope': 'Thinker language body/head + audio encoder/projection + vision encoder/merger; no speech generator',
                    'verification_split': 'train', 'test_set_used': False,
                    'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                                      [Path(__file__).resolve(), ROOT/'src/smarthome_efficiency/pruning.py', ROOT/'src/smarthome_efficiency/pruning_calibration.py',
                                       ROOT/'src/smarthome_efficiency/multimodal_pruning.py',
                                       ROOT/'src/smarthome_efficiency/hybrid_pruning.py',
                                       ROOT/'src/smarthome_efficiency/encoder_pruning.py',
                                       ROOT/'src/smarthome_efficiency/joint_pruning.py']},
                    **selection}
        if joint:
            manifest['joint_language_encoder_pruning'] = True
            manifest['language_tensors_unchanged'] = False
        if encoder_only or joint:
            manifest['parameter_limits'] = {'language': 2_000_000_000, 'combined_encoders': 1_000_000_000}
            manifest['combined_encoder_parameters'] = after['audio_encoder'] + after['vision_encoder']
        runner.model.save_pretrained(temporary, safe_serialization=True, max_shard_size='2GB')
        # Byte-identical assets preserve the existing processor input cache safely.
        for name in PROCESSOR_FILES:
            path = Path(cfg['model']) / name
            if path.is_file():
                shutil.copyfile(path, temporary / name)
        for name in ('LICENSE', 'LICENSE.txt', 'NOTICE'):
            path = Path(cfg['model']) / name
            if path.is_file():
                shutil.copyfile(path, temporary / name)
        write_json(temporary / 'compression.json', manifest)
        del encoders_before, encoders_after
        runner.model = None
        del runner
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = PilotModel(temporary, cfg['task'], labels, cfg['cache_dir'],
                              max_frames=cfg['max_frames'], max_pixels=cfg['max_pixels'],
                              use_transcript=cfg['use_transcript'], gradient_checkpointing=False)
        reloaded.load_model()
        if parameter_counts(reloaded.model) != after:
            raise RuntimeError('Reload changed compact model parameter counts')
        actual = []
        for batch, record, reference in zip(batches, records, expected):
            value = reloaded.evaluate_batch(batch, labels.index(record['label']), modes='both', max_new_tokens=8)
            torch.testing.assert_close(torch.tensor(value['logit_scores']), torch.tensor(reference['logit_scores']),
                                       atol=1e-5, rtol=1e-5)
            if value['generated_token_ids'] != reference['generated_token_ids']:
                raise RuntimeError('Compact checkpoint reload changed greedy generation')
            actual.append({'id': record['id'], 'label': record['label'], **value})
        manifest['verification'] = {'passed': True, 'examples': actual,
                                    'bf16_merge_max_abs_class_logit_difference': merge_difference,
                                    'save_reload_class_logits_match': True, 'save_reload_generation_match': True,
                                    'native_av_forward_and_cached_generation': True}
        manifest['export_seconds'] = time.perf_counter() - started
        write_json(temporary / 'compression.json', manifest)
        temporary.rename(output)
        log('export_complete', output=str(output), parameters=after, seconds=manifest['export_seconds'])
        return 0
    except BaseException as error:
        write_json(output.with_suffix('.failure.json'), {'error': f'{type(error).__name__}: {error}',
                   'request': request, 'traceback': traceback.format_exc()})
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == '__main__':
    raise SystemExit(main())
