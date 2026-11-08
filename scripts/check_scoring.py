#!/usr/bin/env python3
"""Check prompt-only versus teacher-forced scoring on a few development clips.

Predicted labels must agree and changing the teacher-forced gold label must not
change first-answer logits. Actual prompt/teacher numerical differences are
reported; --atol optionally adds an explicit numerical-equality requirement.
The default does not invent a permissive BF16 tolerance. No test data are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate import resolve_evaluation_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path, help='Completed pilot run with adapter and data_manifest.json')
    source.add_argument('--checkpoint', type=Path, help='Full checkpoint or run root containing checkpoints/best')
    parser.add_argument('--output', type=Path, help='New JSON file; default SOURCE/scoring_verification.json')
    parser.add_argument('--examples', type=int, default=2)
    parser.add_argument('--atol', type=float, help='Optional absolute prompt/teacher numerical tolerance')
    parser.add_argument('--causal-atol', type=float, default=1e-6,
                        help='Tolerance when changing only the gold target; default 1e-6')
    args = parser.parse_args()
    if not 1 <= args.examples <= 16:
        parser.error('--examples must be between 1 and 16')
    for name in ('atol', 'causal_atol'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f'--{name.replace("_", "-")} must be finite and nonnegative')
    return args


def run(args):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one GPU for this bounded diagnostic, not multiple torchrun ranks')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from smarthome_efficiency.checkpointing import write_json
    from smarthome_efficiency.data import LABELS, load_records
    from smarthome_efficiency.model import PilotModel
    from smarthome_efficiency.training import configure_determinism

    if args.run_dir:
        source = args.run_dir.expanduser().resolve()
        saved = json.loads((source / 'report.json').read_text())
        if saved.get('status') != 'completed':
            raise ValueError('Pilot run must be completed')
        cfg = saved['config']
        labels = saved['labels']
        records = json.loads((source / 'data_manifest.json').read_text())['dev']
        adapter = source / 'adapter'
        use_transcript = not cfg.get('no_transcript', False)
    else:
        source = resolve_evaluation_checkpoint(args.checkpoint)
        saved = json.loads((source / 'metadata.json').read_text())
        cfg = saved['config']
        labels = list(LABELS[cfg['task']])
        records = load_records(cfg['task'], 'dev', seed=cfg['split_seed'], data_root=cfg['data_root'], mer_partition=cfg.get('mer_partition'))
        adapter = source / 'adapter'
        use_transcript = cfg['use_transcript']
    cfg = dict(cfg)
    cfg.setdefault('deterministic', True)
    configure_determinism(cfg)
    if not records:
        raise ValueError('No development examples available')
    records = records[:args.examples]
    output = (args.output if args.output is not None else source / 'scoring_verification.json').expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; choose a new --output')
    if not torch.cuda.is_available():
        raise RuntimeError('A visible CUDA GPU is required')
    torch.set_num_threads(cfg.get('num_threads', 2))
    torch.manual_seed(cfg.get('seed', 42))
    started = time.perf_counter()
    report = {
        'status': 'initializing', 'source': str(source), 'task': cfg['task'], 'labels': labels,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'test_set_used': False,
        'deterministic': cfg['deterministic'],
        'prompt_teacher_atol': args.atol, 'gold_label_invariance_atol': args.causal_atol,
        'numerical_policy': 'Report actual prompt/teacher differences; prediction equality and gold-label invariance are mandatory',
        'model_source_sha256': hashlib.sha256((ROOT / 'src/smarthome_efficiency/model.py').read_bytes()).hexdigest(),
        'examples': [],
    }
    try:
        runner = PilotModel(
            cfg['model'], cfg['task'], labels, cfg['cache_dir'],
            max_frames=cfg.get('max_frames', 8), max_pixels=cfg.get('max_pixels', 224**2),
            use_transcript=use_transcript,
        )
        runner.load_adapter(adapter, trainable=False)
        for record in records:
            correct_index = labels.index(record['label'])
            other_index = (correct_index + 1) % len(labels)
            batch = runner.prepare(record)
            prompt_scores = runner.score(batch)
            correct = runner.evaluate_batch(batch, correct_index, modes='logit')
            different = runner.evaluate_batch(batch, other_index, modes='logit')
            prompt_prediction = labels[max(range(len(labels)), key=prompt_scores.__getitem__)]
            prompt_difference = max(abs(a - b) for a, b in zip(prompt_scores, correct['logit_scores']))
            causal_difference = max(abs(a - b) for a, b in zip(correct['logit_scores'], different['logit_scores']))
            item = {
                'id': record['id'], 'true_label': record['label'], 'alternative_gold_label': labels[other_index],
                'prompt_scores': prompt_scores, 'teacher_correct_gold_scores': correct['logit_scores'],
                'teacher_other_gold_scores': different['logit_scores'],
                'prompt_prediction': prompt_prediction, 'teacher_prediction': correct['logit_predicted_label'],
                'other_gold_prediction': different['logit_predicted_label'],
                'predictions_match': prompt_prediction == correct['logit_predicted_label'] == different['logit_predicted_label'],
                'prompt_teacher_max_absolute_difference': prompt_difference,
                'gold_label_change_max_absolute_difference': causal_difference,
                'gold_label_invariant': causal_difference <= args.causal_atol,
                'prompt_teacher_within_requested_atol': None if args.atol is None else prompt_difference <= args.atol,
            }
            report['examples'].append(item)
            print(json.dumps(item, allow_nan=False), flush=True)
        report['max_prompt_teacher_difference'] = max(row['prompt_teacher_max_absolute_difference'] for row in report['examples'])
        report['max_gold_label_change_difference'] = max(row['gold_label_change_max_absolute_difference'] for row in report['examples'])
        if not all(row['predictions_match'] and row['gold_label_invariant'] and
                   row['prompt_teacher_within_requested_atol'] is not False for row in report['examples']):
            raise RuntimeError('Scoring predictions, gold-label invariance, or the requested numerical tolerance failed')
        report['status'] = 'passed'
    except Exception as error:
        report.update({'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc()})
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        write_json(output, report)
        print(json.dumps({'status': report['status'], 'output': str(output), 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    run(parse_args())
