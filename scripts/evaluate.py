#!/usr/bin/env python3
"""Evaluate a training checkpoint or released task model on a dev or test split.

Use plain Python for one GPU or torchrun --nproc_per_node=4 for four GPUs. A run
root resolves to checkpoints/best; an exact checkpoint (including latest) is also
accepted. All examples in the chosen split are evaluated unless --limit is set.
This command never trains, selects checkpoints, or loads optimizer pickle state.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def resolve_evaluation_checkpoint(path: Path) -> Path:
    """Resolve an explicit checkpoint or a run's best checkpoint, never silently latest."""
    path = Path(path).expanduser().resolve()
    if (path / 'metadata.json').is_file():
        selected = path
    elif (path / 'checkpoints/best').exists():
        selected = (path / 'checkpoints/best').resolve()
    elif (path / 'best').exists():
        selected = (path / 'best').resolve()
    else:
        raise ValueError(f'No explicit checkpoint or best-checkpoint pointer at {path}')
    for name in ('metadata.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'):
        if not (selected / name).is_file():
            raise FileNotFoundError(selected / name)
    return selected


def checked_prediction_rows(shards, records):
    """Gathered evaluation must contain each requested example exactly once."""
    rows = sorted((row for shard in shards for row in shard), key=lambda row: row['id'])
    expected = {record['id']: record['label'] for record in records}
    actual = [row['id'] for row in rows]
    if len(expected) != len(records) or len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise RuntimeError('Distributed evaluation lost, duplicated, or substituted examples')
    if any(row['true_label'] != expected[row['id']] for row in rows):
        raise RuntimeError('Evaluation ground truth disagrees with the selected manifest')
    return rows


def source_hashes():
    paths = [Path(__file__).resolve(), *sorted((ROOT / 'src/smarthome_efficiency').glob('*.py'))]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path,
                        help='Exact full checkpoint or run root containing checkpoints/best')
    source.add_argument('--model', help='Released model directory or Hugging Face model ID')
    parser.add_argument('--revision', help='Hugging Face model revision for --model')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data', help='Dataset root for --model')
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'artifacts/pilot_cache', help='Input cache for --model')
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for this evaluation')
    parser.add_argument('--modes', choices=('logit', 'greedy', 'both'), help='Override saved evaluation modes')
    parser.add_argument('--limit', type=int, help='Optional stratified subset for a bounded smoke test')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    return args


def run(args):
    # Resolve a requested Hub snapshot before switching inference to local files.
    released = getattr(args, 'model', None) is not None
    if released:
        from smarthome_efficiency.release import resolve_model, evaluation_config
        checkpoint = resolve_model(args.model, args.revision)
        release, cfg = evaluation_config(checkpoint, args.data_root, args.cache_dir)
        metadata = {'global_step': release['source_global_step'], 'config': cfg}
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import torch.distributed as dist
    from smarthome_efficiency.checkpointing import write_json
    from smarthome_efficiency.data import LABELS, load_records, select_records
    from smarthome_efficiency.evaluation import (
        PARSER_PROTOCOL, classification_metrics, normalize_modes, prediction_agreement,
    )
    from smarthome_efficiency.model import PilotModel
    from smarthome_efficiency.training import config_fingerprint, configure_determinism

    if not released:
        checkpoint = resolve_evaluation_checkpoint(args.checkpoint)
        metadata = json.loads((checkpoint / 'metadata.json').read_text())
        cfg = dict(metadata['config'])
        compact_sidecar = Path(cfg['model']) / 'compression.json'
        compact_hash = hashlib.sha256(compact_sidecar.read_bytes()).hexdigest() if compact_sidecar.is_file() else None
        if metadata.get('base_model_manifest_sha256') != compact_hash:
            raise ValueError('Compact base checkpoint differs from the saved evaluation checkpoint')
    cfg.setdefault('deterministic', True)
    configure_determinism(cfg)
    modes = normalize_modes(args.modes if args.modes is not None else cfg['eval_modes'])
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world not in (1, 4):
        raise ValueError('This evaluation entry point supports one or four GPU ranks')
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError('Each rank requires its own visible CUDA GPU')
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(cfg.get('num_threads', 2))
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    output = args.output.expanduser().resolve()
    started = time.perf_counter()
    output_created = False
    report = {
        'status': 'initializing', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(checkpoint), 'checkpoint_global_step': metadata.get('global_step'),
        'split': args.split, 'test_set_used': args.split == 'test', 'task': cfg['task'],
        'eval_modes': list(modes), 'world_size': world, 'config': cfg,
        'deterministic': cfg['deterministic'],
        'requested_limit': args.limit, 'selection_mode_used_during_training': cfg.get('selection_mode'),
        'purpose': 'Fixed saved-checkpoint evaluation; no training or checkpoint selection',
    }

    def gather(value):
        if world == 1:
            return [value]
        values = [None] * world
        dist.all_gather_object(values, value)
        return values

    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            output_created = True
        if world > 1:
            dist.barrier(device_ids=[local_rank])
            output_created = True
        labels = list(LABELS[cfg['task']])
        label_to_index = {label: index for index, label in enumerate(labels)}
        # No training split or other evaluation split is opened here.
        full_records = load_records(cfg['task'], args.split, seed=cfg['split_seed'], data_root=cfg['data_root'], mer_partition=cfg.get('mer_partition'))
        records = select_records(full_records, args.limit, seed=cfg['split_seed'])
        report.update({
            'labels': labels, 'full_split_count': len(full_records), 'evaluated_count': len(records),
            'partial_split': len(records) != len(full_records),
            'input_protocol': 'audio+video+provided transcript' if cfg['use_transcript'] else 'audio+video',
            'greedy_parser_protocol': PARSER_PROTOCOL,
            'dev_protocol': 'official MIntRec dev' if cfg['task'].startswith('mintrec') else 'stratified 20% of MER Train&Val',
        })
        manifest = {'split': args.split, 'split_seed': cfg['split_seed'], 'records': records}
        if rank == 0:
            write_json(output / 'data_manifest.json', manifest)
            write_json(output / 'provenance.json', {
                'source_sha256': source_hashes(), 'python': sys.version, 'torch': str(torch.__version__),
                **({'release_manifest_sha256': hashlib.sha256((checkpoint/'release.json').read_bytes()).hexdigest()}
                   if released else {
                       'checkpoint_metadata_sha256': hashlib.sha256((checkpoint/'metadata.json').read_bytes()).hexdigest(),
                       'checkpoint_adapter_sha256': hashlib.sha256((checkpoint/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
                   }),
                'labels': labels, 'data_fingerprint': config_fingerprint(manifest),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'world_size': world,
                'checkpoint': str(checkpoint), 'split': args.split,
            })
        torch.manual_seed(cfg.get('seed', 42) + rank)
        runner = PilotModel(
            cfg['model'], cfg['task'], labels, cfg['cache_dir'], max_frames=cfg['max_frames'],
            max_pixels=cfg['max_pixels'], use_transcript=cfg['use_transcript'], device_index=local_rank,
        )
        if released:
            runner.load_model()
            runner.model.requires_grad_(False)
        else:
            runner.load_adapter(checkpoint / 'adapter', trainable=False)
        if any(parameter.requires_grad for parameter in runner.model.parameters()):
            raise RuntimeError('Evaluation adapter unexpectedly has trainable parameters')
        torch.cuda.reset_peak_memory_stats(local_rank)
        if world > 1:
            dist.barrier(device_ids=[local_rank])
        evaluation_started = time.perf_counter()
        local_rows = []
        for index in range(rank, len(records), world):
            record = records[index]
            try:
                result = runner.evaluate_batch(
                    runner.prepare(record), label_to_index[record['label']], modes=modes,
                    max_new_tokens=cfg['max_new_tokens'],
                )
            except Exception as error:
                raise RuntimeError(f'Evaluation failed on {record["id"]}, rank {rank}') from error
            local_rows.append({'id': record['id'], 'true_label': record['label'], **result})
            if rank == 0 and len(local_rows) % 25 == 0:
                print(json.dumps({'event': 'evaluation_progress', 'rank': rank,
                                  'local_done': len(local_rows), 'local_total': len(records[rank::world])}), flush=True)
        torch.cuda.synchronize(local_rank)
        local_stats = {
            'rank': rank, 'gpu': torch.cuda.get_device_name(local_rank), 'examples': len(local_rows),
            'evaluation_seconds': time.perf_counter() - evaluation_started,
            'model_load_seconds': runner.load_seconds,
            'peak_allocated_gib': torch.cuda.max_memory_allocated(local_rank) / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(local_rank) / 2**30,
            'cache_hits': runner.cache_hits, 'cache_misses': runner.cache_misses,
        }
        rows = checked_prediction_rows(gather(local_rows), records)
        rank_stats = gather(local_stats)
        truth = [row['true_label'] for row in rows]
        report.update({key: statistics.mean(row[key] for row in rows)
                       for key in ('sft_loss', 'class_loss', 'eos_loss')})
        for mode in modes:
            report[mode] = classification_metrics(truth, [row[f'{mode}_predicted_label'] for row in rows], labels)
        if len(modes) == 2:
            report['agreement'] = prediction_agreement(
                [row['logit_predicted_label'] for row in rows], [row['greedy_predicted_label'] for row in rows],
            )
        report.update({'status': 'completed', 'ranks': rank_stats,
                       'evaluation_seconds': max(item['evaluation_seconds'] for item in rank_stats),
                       'peak_allocated_gib_per_gpu': max(item['peak_allocated_gib'] for item in rank_stats),
                       'peak_reserved_gib_per_gpu': max(item['peak_reserved_gib'] for item in rank_stats),
                       'total_seconds': time.perf_counter() - started})
        if rank == 0:
            (output / 'predictions.jsonl').write_text(''.join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
            write_json(output / 'report.json', report)
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    except Exception as error:
        report.update({'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc(), 'total_seconds': time.perf_counter() - started})
        if output_created:
            write_json(output / f'failure-rank-{rank}.json', report)
        raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    run(parse_args())
