#!/usr/bin/env python3
"""Bounded audiovisual LoRA feasibility runs; official test sets are never loaded."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from smarthome_efficiency.data import LABELS, load_records, select_records
from smarthome_efficiency.model import PilotModel


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def memory():
    torch.cuda.synchronize()
    return {'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(LABELS), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'artifacts/pilot_cache')
    parser.add_argument('--train-limit', type=int, default=256)
    parser.add_argument('--dev-limit', type=int, default=96)
    parser.add_argument('--train-eval-limit', type=int, default=32)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--max-train-seconds', type=int, default=1800)
    parser.add_argument('--max-frames', type=int, default=8)
    parser.add_argument('--max-pixels', type=int, default=224**2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-transcript', action='store_true')
    parser.add_argument('--balanced-train', action='store_true', help='Class-balanced training subset for overfit diagnostics')
    args = parser.parse_args()
    if min(args.train_limit, args.dev_limit, args.train_eval_limit, args.steps,
           args.grad_accum, args.rank, args.max_train_seconds) < 1:
        parser.error('Limits, steps, accumulation, rank and time budget must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('Learning rate must be finite and positive')
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    labels = LABELS[args.task]
    label_to_index = {label: i for i, label in enumerate(labels)}
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Training feasibility and development results; not an official benchmark score',
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'labels': list(labels), 'test_set_used': False,
              'input_protocol': 'audio+video+provided transcript' if args.task.startswith('mintrec') and not args.no_transcript else 'audio+video',
              'prediction_protocol': 'Argmax of single-token answer-letter logits among the fixed allowed choices',
              'training_objective': 'Causal language-model loss on answer letter and EOS; prompt and media tokens masked',
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__).resolve(), ROOT/'src/smarthome_efficiency/data.py', ROOT/'src/smarthome_efficiency/model.py']}}
    write_json(args.output / 'config.json', report)
    runner = None

    def event(kind, **fields):
        value = {'event': kind, 'elapsed_seconds': round(time.perf_counter()-started, 3), **fields}
        with (args.output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        print(json.dumps(value, allow_nan=False), flush=True)

    def evaluate(records, name):
        begin = time.perf_counter()
        predictions = []
        for i, record in enumerate(records):
            try:
                scores = runner.score(runner.prepare(record))
            except Exception as error:
                raise RuntimeError(f'{name} failed on {record["id"]}') from error
            if len(scores) != len(labels) or not np.isfinite(scores).all():
                raise RuntimeError('Invalid class score vector')
            predicted = int(np.argmax(scores))
            predictions.append({'id': record['id'], 'true_label': record['label'],
                                'predicted_label': labels[predicted], 'scores': list(scores)})
            if (i+1) % 32 == 0:
                event('evaluation_progress', name=name, done=i+1, total=len(records))
        truth = [label_to_index[x['true_label']] for x in predictions]
        pred = [label_to_index[x['predicted_label']] for x in predictions]
        result = {'n': len(records), 'accuracy': accuracy_score(truth, pred),
                  'weighted_f1': f1_score(truth, pred, labels=list(range(len(labels))), average='weighted', zero_division=0),
                  'macro_f1': f1_score(truth, pred, labels=list(range(len(labels))), average='macro', zero_division=0),
                  'confusion_matrix': confusion_matrix(truth, pred, labels=list(range(len(labels)))).tolist(),
                  'predicted_counts': dict(Counter(x['predicted_label'] for x in predictions)),
                  'seconds': time.perf_counter()-begin}
        with (args.output / f'{name}_predictions.jsonl').open('w') as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + '\n')
        write_json(args.output / f'{name}_metrics.json', result)
        event('evaluation_complete', name=name, **result)
        return result

    try:
        if not torch.cuda.is_available():
            raise RuntimeError('A visible CUDA GPU is required')
        full_train = load_records(args.task, 'train', seed=args.seed)
        full_dev = load_records(args.task, 'dev', seed=args.seed)
        if {r['id'] for r in full_train} & {r['id'] for r in full_dev}:
            raise RuntimeError('Training/development ID overlap')
        train = select_records(full_train, args.train_limit, seed=args.seed, balanced=args.balanced_train)
        dev = select_records(full_dev, args.dev_limit, seed=args.seed)
        train_eval = select_records(train, args.train_eval_limit, seed=args.seed)
        write_json(args.output / 'data_manifest.json', {'train': train, 'dev': dev,
                    'train_eval_ids': [r['id'] for r in train_eval],
                    'full_train_ids': [r['id'] for r in full_train],
                    'full_dev_ids': [r['id'] for r in full_dev]})
        report['data'] = {'full_train_count': len(full_train), 'full_dev_count': len(full_dev),
                          'train_count': len(train), 'dev_count': len(dev),
                          'train_label_counts': dict(Counter(r['label'] for r in train)),
                          'dev_label_counts': dict(Counter(r['label'] for r in dev)),
                          'train_selection': 'balanced' if args.balanced_train else 'stratified proportional',
                          'dev_selection': 'stratified proportional',
                          'dev_split': 'official' if args.task.startswith('mintrec') else 'stratified 20% of official Train&Val, seed42 by default'}
        report['gpu'] = torch.cuda.get_device_name(0)
        event('data_ready', **report['data'])
        runner = PilotModel(args.model, args.task, labels, args.cache_dir,
                            max_frames=args.max_frames, max_pixels=args.max_pixels,
                            use_transcript=not args.no_transcript)
        report['baseline'] = {'train': evaluate(train_eval, 'baseline_train'),
                              'dev': evaluate(dev, 'baseline_dev')}
        write_json(args.output / 'report.json', report | {'status': 'baseline_complete'})
        report['base_parameters'] = dict(runner.parameters_info)
        runner.enable_lora(rank=args.rank)
        parameters = [p for p in runner.model.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError('No trainable LoRA parameters')
        report['trainable_parameters'] = sum(p.numel() for p in parameters)
        report['total_parameters_with_lora'] = sum(p.numel() for p in runner.model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        rng = random.Random(args.seed)
        order = list(range(len(train)))
        cursor = len(order)
        losses, step_times = [], []
        trained_examples = 0
        completed_steps = 0
        parameter_updated = False
        train_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        event('training_start', trainable_parameters=report['trainable_parameters'])
        stopping_reason = 'requested_steps_complete'
        for step in range(args.steps):
            if time.perf_counter()-train_start >= args.max_train_seconds:
                stopping_reason = 'training_time_budget_reached'
                break
            step_start = time.perf_counter()
            micro_losses = []
            for _ in range(args.grad_accum):
                if cursor == len(order):
                    rng.shuffle(order)
                    cursor = 0
                record = train[order[cursor]]
                cursor += 1
                try:
                    batch = runner.prepare(record)
                    loss = runner.train_batch(batch, label_to_index[record['label']])
                except Exception as error:
                    raise RuntimeError(f'Training failed on {record["id"]}') from error
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Nonfinite training loss')
                micro_losses.append(float(loss.detach().item()))
                (loss / args.grad_accum).backward()
                trained_examples += 1
                del loss, batch
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).item())
            if not math.isfinite(grad_norm):
                raise RuntimeError('Nonfinite gradient norm')
            changed = None
            if completed_steps == 0:
                changed = next((p for p in parameters if p.grad is not None and p.grad.abs().max().item() > 0), None)
                if changed is None:
                    raise RuntimeError('No nonzero adapter gradient')
                before = changed.detach().clone()
            optimizer.step()
            if changed is not None:
                parameter_updated = not torch.equal(before, changed.detach())
                if not parameter_updated:
                    raise RuntimeError('Optimizer did not update the sampled adapter parameter')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_seconds = time.perf_counter()-step_start
            step_times.append(step_seconds)
            loss_value = statistics.mean(micro_losses)
            losses.append(loss_value)
            completed_steps += 1
            if completed_steps % 10 == 0 or completed_steps == 1 or completed_steps == args.steps:
                event('training_progress', step=completed_steps, requested_steps=args.steps,
                      loss=loss_value, recent_mean_loss=statistics.mean(losses[-10:]),
                      grad_norm=grad_norm, step_seconds=step_seconds,
                      training_elapsed_seconds=time.perf_counter()-train_start)
        training_seconds = time.perf_counter()-train_start
        report['training'] = {'completed_steps': completed_steps, 'requested_steps': args.steps,
            'examples_processed': trained_examples, 'effective_epochs': trained_examples/len(train),
            'seconds': training_seconds, 'stopping_reason': stopping_reason,
            'mean_step_seconds': statistics.mean(step_times) if step_times else None,
            'mean_first_10_losses': statistics.mean(losses[:10]) if losses else None,
            'mean_last_10_losses': statistics.mean(losses[-10:]) if losses else None,
            'parameter_updated': parameter_updated, 'memory': memory(),
            'estimated_full_training_epoch_minutes': training_seconds/max(trained_examples, 1)*len(full_train)/60,
            'estimate_note': 'Single-GPU extrapolation at current frame/audio settings; includes observed media preparation/cache misses and is not a full-run measurement'}
        runner.model.save_pretrained(args.output / 'adapter', safe_serialization=True)
        event('adapter_saved', **report['training'])
        # Parameter gradients are no longer needed during the held-out development evaluation.
        del optimizer
        report['final'] = {'train': evaluate(train_eval, 'final_train'),
                           'dev': evaluate(dev, 'final_dev')}
        report['processor_cache'] = {'hits': runner.cache_hits, 'misses': runner.cache_misses}
        report['status'] = 'completed'
        event('run_complete', task=args.task, final_dev=report['final']['dev'])
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['total_seconds'] = time.perf_counter()-started
        write_json(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
