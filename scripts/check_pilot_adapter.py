#!/usr/bin/env python3
"""Reload a saved pilot adapter and reproduce a few recorded dev predictions.

No training or official test data are used. Run with the project environment and
one explicitly selected idle GPU; --validate-only performs metadata checks on CPU.
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
import warnings

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_run(run_dir: Path, count: int):
    """Validate saved metadata and choose examples without importing CUDA libraries."""
    report = read_json(run_dir / 'report.json')
    if report.get('status') != 'completed':
        raise ValueError(f"Expected completed run, got status {report.get('status')!r}")
    saved_config = read_json(run_dir / 'config.json')
    config = report['config']
    if config != saved_config['config']:
        raise ValueError('report.json and config.json disagree on run configuration')
    labels = report['labels']
    if labels != saved_config['labels'] or len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError('Missing, duplicate, or inconsistent label order')
    if report.get('test_set_used') is not False:
        raise ValueError('Expected a development-only pilot')
    adapter = run_dir / 'adapter'
    for required in ('adapter_config.json', 'adapter_model.safetensors'):
        if not (adapter / required).is_file():
            raise FileNotFoundError(adapter / required)
    if not Path(config['model']).is_dir():
        raise FileNotFoundError(config['model'])
    manifest = read_json(run_dir / 'data_manifest.json')
    dev_records = {record['id']: record for record in manifest['dev']}
    if len(dev_records) != len(manifest['dev']):
        raise ValueError('Duplicate development IDs')
    if set(dev_records) & {record['id'] for record in manifest['train']}:
        raise ValueError('Training/development overlap')
    predictions = [json.loads(line) for line in
                   (run_dir / 'final_dev_predictions.jsonl').read_text().splitlines() if line.strip()]
    if not predictions:
        raise ValueError('No final development predictions')
    selected = predictions[:count]
    if len({prediction['id'] for prediction in predictions}) != len(predictions):
        raise ValueError('Duplicate recorded prediction IDs')
    if {prediction['id'] for prediction in predictions} != set(dev_records):
        raise ValueError('Recorded final predictions do not match the development manifest')
    for prediction in selected:
        record = dev_records[prediction['id']]
        if record['label'] != prediction['true_label']:
            raise ValueError(f"Ground-truth mismatch for {record['id']}")
        scores = prediction['scores']
        if len(scores) != len(labels) or not all(math.isfinite(x) for x in scores):
            raise ValueError(f"Invalid recorded logits for {record['id']}")
        if labels[max(range(len(scores)), key=scores.__getitem__)] != prediction['predicted_label']:
            raise ValueError(f"Recorded logits disagree with predicted label for {record['id']}")
    source_warnings = []
    for name, expected in report.get('source_sha256', {}).items():
        source = ROOT / name
        actual = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        if actual != expected:
            message = f'Source differs from the recorded run: {name}'
            warnings.warn(message)
            source_warnings.append({'path': name, 'recorded_sha256': expected, 'current_sha256': actual})
    return report, config, labels, adapter, dev_records, selected, source_warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='Verification JSON inside the run directory')
    parser.add_argument('--examples', type=int, default=4, help='First N final development predictions (1-64)')
    parser.add_argument('--atol', type=float, default=1e-3, help='Maximum absolute difference in saved/reloaded label logits')
    parser.add_argument('--validate-only', action='store_true', help='Check metadata on CPU without loading model weights or writing a GPU result')
    args = parser.parse_args()
    if not 1 <= args.examples <= 64:
        parser.error('--examples must be between 1 and 64')
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error('--atol must be finite and nonnegative')
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error('--run-dir must be an existing directory')
    output = (args.output if args.output is not None else run_dir / 'adapter_verification.json').resolve()
    if not output.is_relative_to(run_dir) or output.suffix != '.json':
        parser.error('--output must be a JSON file within the run directory')
    protected = {'report.json', 'config.json', 'data_manifest.json'}
    if output.name in protected or output.name.endswith('_metrics.json') or output.is_relative_to(run_dir / 'adapter'):
        parser.error('--output must not overwrite original run reports or adapter files')
    if not output.parent.is_dir():
        parser.error('--output parent directory must already exist')
    started = time.perf_counter()
    result = {
        'purpose': 'Saved-adapter round-trip verification against recorded development logits',
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'run_dir': str(run_dir), 'atol': args.atol, 'requested_examples': args.examples,
        'test_set_used': False, 'examples': [],
    }
    try:
        report, config, labels, adapter, records, selected, source_warnings = validate_run(run_dir, args.examples)
        result.update({'task': config['task'], 'labels': labels,
                       'source_warnings': source_warnings,
                       'selected_ids': [prediction['id'] for prediction in selected]})
        if args.validate_only:
            print(json.dumps(result | {'status': 'metadata_validated', 'gpu_used': False}, indent=2))
            return
        # All model and adapter inputs are local; forbid implicit network fallback.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path.insert(0, str(ROOT / 'src'))
        import torch
        from peft import PeftModel
        from smarthome_efficiency.model import PilotModel

        if not torch.cuda.is_available():
            raise RuntimeError('A visible CUDA GPU is required for adapter verification')
        torch.manual_seed(config.get('seed', 42))
        result['gpu'] = torch.cuda.get_device_name(0)
        runner = PilotModel(
            config['model'], config['task'], labels, config['cache_dir'],
            max_frames=config.get('max_frames', 8), max_pixels=config.get('max_pixels', 224**2),
            use_transcript=not config.get('no_transcript', False),
        )
        runner.load_model()
        runner.model = PeftModel.from_pretrained(
            runner.model, str(adapter), is_trainable=False, local_files_only=True,
        )
        runner.lora_enabled = True
        runner.model.eval()
        result['adapter_trainable_parameters'] = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
        if result['adapter_trainable_parameters']:
            raise RuntimeError('Reloaded inference adapter unexpectedly has trainable parameters')
        for prediction in selected:
            scores = runner.score(runner.prepare(records[prediction['id']]))
            if len(scores) != len(labels) or not all(math.isfinite(x) for x in scores):
                raise RuntimeError(f"Invalid reloaded logits for {prediction['id']}")
            predicted_label = labels[max(range(len(scores)), key=scores.__getitem__)]
            max_difference = max(abs(actual - expected) for actual, expected in zip(scores, prediction['scores']))
            item = {
                'id': prediction['id'], 'recorded_label': prediction['predicted_label'],
                'reloaded_label': predicted_label, 'recorded_scores': prediction['scores'],
                'reloaded_scores': scores, 'max_absolute_logit_difference': max_difference,
                'label_matches': predicted_label == prediction['predicted_label'],
                'logits_match': max_difference <= args.atol,
            }
            result['examples'].append(item)
            print(json.dumps(item, allow_nan=False), flush=True)
        result['max_absolute_logit_difference'] = max(item['max_absolute_logit_difference'] for item in result['examples'])
        if not all(item['label_matches'] and item['logits_match'] for item in result['examples']):
            raise RuntimeError('Reloaded adapter does not reproduce recorded labels/logits within tolerance')
        result['status'] = 'passed'
    except Exception as error:
        result.update({'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc()})
        raise
    finally:
        if not args.validate_only:
            result['seconds'] = time.perf_counter() - started
            write_json(output, result)
            print(json.dumps({'status': result.get('status'), 'output': str(output),
                              'seconds': result['seconds']}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
