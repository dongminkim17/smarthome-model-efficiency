#!/usr/bin/env python3
"""Extract verified experiment data for the self-contained IITP year-two report.

This script reads saved training/evaluation artifacts; it does not run models.
The final three-task comparison is independently recomputed from predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from compare_project_benchmarks import EXPECTED_COUNTS, compare

TASKS = ('mintrec2', 'mintrec20', 'mer6')
TASK_NAMES = {'mintrec2': 'MIntRec-2', 'mintrec20': 'MIntRec-20', 'mer6': 'MER2023-SEMI'}
TEST_N = {'mintrec2': 445, 'mintrec20': 445, 'mer6': 834}
METHOD_DEFS = [
    ('depth_uniform', 'Evenly spaced whole-layer pruning', 'naive_pruning_20261019', 17, 11008,
     'Keep evenly spaced original decoder layers, including the first and last.'),
    ('ffn_magnitude', 'Weight-magnitude FFN channel pruning', 'naive_pruning_20261019', 36, 4352,
     'Rank channels by the sum of squared gate/up row and down-projection column weights.'),
    ('depth_bi', 'Block-influence layer pruning', 'calibrated_pruning_20261019', 17, 11008,
     'ShortGPT-style mean valid-token cosine change across each decoder block; remove low-influence blocks.'),
    ('ffn_activation', 'Activation-ranked FFN channel pruning', 'calibrated_pruning_20261019', 36, 4352,
     'Minitron/Model Optimizer-style mean absolute gated FFN activation over tokens, then L2 across examples.'),
    ('depth_ffn_activation', 'Combined layer and activation-FFN pruning', 'calibrated_pruning_20261019', 28, 6144,
     'Apply block-influence layer selection and activation-ranked channel selection from the same intact source.'),
]


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


class ReportExtractor:
    def __init__(self, index):
        self.index = Path(index)
        with self.index.open() as source:
            self.rows = list(csv.DictReader(source))

    def select(self, family, task, arm=None, name=None, test_report=None):
        found = [row for row in self.rows
                 if row['family'] == family and row['task'] == task
                 and (arm is None or row['arm'] == arm)
                 and (name is None or row['name'] == name)
                 and (test_report is None or Path(row['test_report']).resolve() == Path(test_report).resolve())]
        if len(found) != 1:
            raise ValueError(f'Expected one source row: {(family, task, arm, name)}; found {len(found)}')
        return found[0]

    def run(self, row, include_curves=False):
        report = read(row['report'])
        test = read(row['test_report'])
        task = row['task']
        assert row['training_status'] == 'completed' and row['test_status'] == 'completed'
        assert report['status'] == 'completed' and test['status'] == 'completed'
        assert report.get('test_set_used') is False
        assert test['task'] == task and test['split'] == 'test'
        assert test['evaluated_count'] == TEST_N[task] and test['full_split_count'] == TEST_N[task]
        assert test.get('partial_split') is False
        assert Path(test['checkpoint']).resolve() == Path(row['best_checkpoint']).resolve()
        metric = 'weighted_f1' if task == 'mer6' else 'accuracy'
        for mode in ('logit', 'greedy'):
            for key in ('accuracy', 'weighted_f1'):
                assert abs(float(row[f'{mode}_test_{key}']) - test[mode][key]) < 1e-12
        base = report['base_parameters']
        components = base['components']
        counts = {
            'language': base.get('language_parameters', components['model'] + components['lm_head']),
            'audio': components['audio_tower'], 'vision': components['visual'],
            'combined_encoders': components['audio_tower'] + components['visual'], 'total': base['total'],
        }
        assert counts['language'] + counts['audio'] + counts['vision'] == counts['total']
        cfg = report['config']
        metrics_path = Path(row['report']).parent / 'metrics.jsonl'
        events = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        validation = [event for event in events if event.get('event') == 'validation']
        train = [event for event in events if event.get('event') == 'train']
        assert train and validation
        winner = next(event for event in validation if event['global_step'] == report['best_step'])
        assert abs(winner['logit'][metric] - report['best_score']) < 1e-12
        source_paths = {'training_report': row['report'], 'official_test_report': row['test_report'], 'metrics': metrics_path}
        model_manifest = Path(cfg['model']) / 'compression.json'
        if model_manifest.exists():
            source_paths['compression_manifest'] = model_manifest
        result = {
            'id': row['id'], 'family': row['family'], 'name': row['name'], 'arm': row['arm'],
            'task': task, 'task_name': TASK_NAMES[task], 'primary_metric': metric,
            'parameters': counts,
            'test': {mode: {key: test[mode][key] for key in
                           ('n', 'accuracy', 'weighted_f1', 'macro_f1', 'invalid_count', 'invalid_rate')}
                     for mode in ('logit', 'greedy')},
            'test_primary': test['logit'][metric],
            'train_count': report['train_count'], 'validation_count': report['dev_count'],
            'test_count': test['evaluated_count'], 'labels': report['labels'],
            'input_protocol': test.get('input_protocol'),
            'best_validation': {'step': report['best_step'], 'epoch': winner['epoch'],
                                'score': report['best_score'], 'sft_loss': winner['sft_loss']},
            'initial_validation': {'epoch': validation[0]['epoch'], 'score': validation[0]['logit'][metric],
                                   'sft_loss': validation[0]['sft_loss']},
            'last_validation': {'epoch': validation[-1]['epoch'], 'score': validation[-1]['logit'][metric],
                                'sft_loss': validation[-1]['sft_loss']},
            'config': cfg,
            'training': {key: cfg.get(key) for key in (
                'epochs', 'seed', 'split_seed', 'learning_rate', 'encoder_learning_rate', 'scheduler',
                'warmup_ratio', 'weight_decay', 'max_grad_norm', 'lora_rank', 'lora_alpha', 'lora_dropout',
                'lora_target', 'objective', 'distillation_alpha', 'distillation_temperature',
                'feature_audio_weight', 'feature_vision_weight', 'max_frames', 'max_pixels',
                'use_transcript', 'eval_modes', 'selection_mode', 'eval_every_fraction', 'max_new_tokens',
                'gradient_checkpointing', 'micro_batch_size', 'gradient_accumulation_steps')},
            'execution': {
                'world_size': report['world_size'], 'effective_batch_size': report['global_batch_size'],
                'steps_per_epoch': report['steps_per_epoch'], 'total_steps': report['global_step'],
                'trainable_parameters': report['trainable_parameters'],
                'elapsed_seconds': report['total_elapsed_seconds'],
                'official_test_seconds': test.get('total_seconds'),
                'peak_allocated_gib': max((event.get('peak_allocated_gib', 0) for event in train), default=0),
            },
            'sources': {**{key: rel(value) for key, value in source_paths.items()},
                        'checkpoint': rel(row['best_checkpoint']), 'curve_dashboard': rel(row['curves']),
                        'sha256': {key: sha256(value) for key, value in source_paths.items()}},
        }
        if include_curves:
            points = [[event['epoch'], event.get('sft_loss', event['loss'])] for event in train]
            assert all(math.isfinite(x) and math.isfinite(y) for x, y in points)
            assert all(points[i][0] <= points[i + 1][0] for i in range(len(points) - 1))
            result['curves'] = {
                'train_sft_columns': ['epoch', 'minibatch_sft_loss'], 'train_sft': points,
                'validation_columns': ['epoch', 'full_dev_sft_loss', 'logit_primary'],
                'validation': [[event['epoch'], event['sft_loss'], event['logit'][metric]] for event in validation],
                'train_components_columns': ['epoch', 'total_objective', 'supervised_loss', 'class_kd_loss',
                                             'audio_cosine_loss', 'vision_cosine_loss', 'lm_learning_rate',
                                             'encoder_learning_rate'],
                'train_components': [[event['epoch'], event['loss'], event.get('sft_loss', event['loss']),
                                      event.get('distillation_loss'), event.get('audio_feature_loss'),
                                      event.get('vision_feature_loss'), event.get('learning_rate'),
                                      event.get('encoder_learning_rate')] for event in train],
                'feature_columns': ['epoch', 'audio_cosine_loss', 'vision_cosine_loss'],
                'feature': [[event['epoch'], event['audio_feature_loss'], event['vision_feature_loss']]
                            for event in train if 'audio_feature_loss' in event],
                'raw_train_point_count': len(train), 'validation_point_count': len(validation),
                'sft_loss_source_key': 'sft_loss' if 'sft_loss' in train[0] else 'loss',
            }
        return result

    def extract(self, comparison_path, request_path):
        saved = read(comparison_path)
        request = read(request_path)
        recomputed = compare(request)
        assert saved['status'] == 'completed' and saved['request_sha256'] == sha256(request_path)
        for key in ('results', 'checks', 'main_project_performance_targets_met',
                    'requested_verification_targets_met', 'scope'):
            assert saved[key] == recomputed[key], f'Comparison verification failed: {key}'
        assert saved['requested_verification_targets_met'] and all(saved['checks'].values())
        primary = []
        for stage, family, arm in [('original', 'original_baselines', 'SFT'),
                                   ('lm_only', 'recovery_controls_20261019', 'KD075')]:
            primary.append({'id': stage, 'runs': {task: self.run(self.select(family, task, arm), True)
                                                 for task in TASKS}})
        final = {'id': 'final', 'runs': {}}
        for case in saved['results']:
            task = case['task']
            family = 'mer_native_encoders_20261106' if task == 'mer6' else 'mintrec_shared_recipe_20261106'
            row = self.select(family, task, test_report=case['test_report'])
            current = self.run(row, True)
            first = primary[1]['runs'][task]
            manifest = read(Path(current['config']['model']) / 'compression.json')
            raw_pruning = read(Path(first['config']['model']) / 'compression.json')
            assert manifest['method'] == 'merged_lm_recovery' and manifest['pruning_method'] == 'ffn_activation'
            assert manifest['ffn_width'] == 4352 and manifest['encoder_tensors_unchanged'] is True
            assert manifest['architecture_unchanged'] is True and manifest['parameters_after'] == EXPECTED_COUNTS
            assert Path(manifest['source_checkpoint']).resolve() == Path(first['sources']['checkpoint']).resolve()
            assert current['config']['teacher_checkpoint'] == first['config']['teacher_checkpoint']
            assert first['config']['lora_target'] == 'decoder_linear' and first['config']['lora_rank'] == 16
            assert first['config']['lora_alpha'] == 32 and first['config']['distillation_alpha'] == .75
            assert first['config']['distillation_temperature'] == 2 and first['config']['epochs'] == 5
            assert first['config']['seed'] == 42 and first['config']['split_seed'] == 42
            assert raw_pruning['method'] == 'ffn_activation' and raw_pruning['ffn_width'] == 4352
            assert raw_pruning['calibration_samples'] == 128 and raw_pruning['calibration_seed'] == 42
            assert raw_pruning['encoder_tensors_unchanged'] is True
            current['ranking'] = case['modes']
            current['pruning'] = {
                'method': 'ffn_activation', 'decoder_layers': 36, 'original_ffn_width': 11008,
                'retained_ffn_width': 4352, 'calibration_train_clips': 128, 'calibration_seed': 42,
                'parameters_before': raw_pruning['parameters_before'],
                'parameters_after': raw_pruning['parameters_after'],
                'original_encoder_geometry_retained': True,
                'initial_recovery_checkpoint': rel(manifest['source_checkpoint']),
                'calibration_manifest': rel(raw_pruning['calibration_cache']),
                'raw_pruning_manifest': rel(Path(first['config']['model']) / 'compression.json'),
                'raw_pruning_manifest_sha256': sha256(Path(first['config']['model']) / 'compression.json'),
            }
            final['runs'][task] = current
        primary.append(final)
        methods = []
        for method_id, title, family, depth, width, description in METHOD_DEFS:
            methods.append({
                'id': method_id, 'title': title, 'decoder_layers': depth, 'lm_ffn_width': width,
                'description': description, 'calibration_train_clips': 128 if family.startswith('calibrated') else 0,
                'runs': {task: {arm: self.run(self.select(family, task, arm, f'{task}_{method_id}'))
                                for arm in ('SFT', 'KD')} for task in TASKS},
            })
        training_status = Counter(row['training_status'] for row in self.rows)
        test_status = Counter(row['test_status'] for row in self.rows)
        return {
            'schema_version': 2, 'report_date': '2026-11-07', 'results_through': '2026-11-06', 'root': str(ROOT),
            'inventory': {
                'tracked': len(self.rows), 'completed_training': training_status['completed'],
                'completed_official_tests': test_status['completed'], 'cancelled': training_status['cancelled'],
                'training_status_counts': dict(training_status), 'test_status_counts': dict(test_status),
                'definition': 'Rows in experiments/results.csv: tracked training configurations, including teacher/preparatory training and cancelled configurations. Counts are not unique method counts and exclude private smoke checks, exports and CPU queue jobs.',
            },
            'source_index': {'path': rel(self.index), 'sha256': sha256(self.index)},
            'chosen_verification': {
                'report': rel(comparison_path), 'sha256': sha256(comparison_path),
                'request': rel(request_path), 'request_sha256': sha256(request_path),
                'recomputed_from_full_predictions': True, 'verified_checks': saved['checks'],
                'results': saved['results'], 'scope': saved['scope'],
            },
            'tasks': [{'id': task, 'name': TASK_NAMES[task], 'train': 1334 if task != 'mer6' else 2698,
                       'validation': 445 if task != 'mer6' else 675, 'test': TEST_N[task],
                       'primary_metric': 'weighted_f1' if task == 'mer6' else 'accuracy',
                       'classes': {'mintrec2': 2, 'mintrec20': 20, 'mer6': 6}[task],
                       'input_modalities': 'audio+video' if task == 'mer6' else 'audio+video+provided transcript'}
                      for task in TASKS],
            'primary': primary, 'methods': methods,
            'notes': [
                'Test scores use complete official test splits; individual checkpoints are selected by full validation.',
                'The final task checkpoints have separate weights and teachers using one shared architecture and numerical recovery recipe; they are not one multitask checkpoint.',
                'Curves use epoch within each independent training/recovery stage, not a cumulative training timeline.',
                'Train SFT losses are stochastic optimizer-update minibatch means, not complete-training-set evaluations.',
                'Validation losses are complete-development-set means. Supervised losses use the recorded sft_loss field when present, otherwise the supervised-only loss field.',
                'Composite SFT/KD/feature objectives are retained separately from the comparable supervised-loss curves.',
                'No test learning curves exist; official test metrics remain separate tables or points.',
                'MER validation and official SEMI test have different class distributions; their F1 levels are not interchangeable.',
                'Original task-SFT runs use four GPUs; LM recovery and final recovery runs use one GPU. All use effective batch four.',
                'Elapsed training times reflect different GPU counts/objectives and are not inference-efficiency benchmarks.',
                'Confirmed year-two encoder parameter caps apply independently per modality.',
                'Published ranks refer to the fixed named comparison cohort, not an exhaustive literature leaderboard.',
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT / 'experiments/results.csv')
    parser.add_argument('--comparison', type=Path, default=ROOT / 'artifacts/full/mintrec_shared_recipe_20261106/comparison.json')
    parser.add_argument('--request', type=Path, default=ROOT / 'artifacts/queue/joy/attempts/mintrec_shared_recipe_20261106__compare_shared_recipe/001/comparison_request.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/analysis/report_20261107/report_data.json')
    args = parser.parse_args()
    data = ReportExtractor(args.index).extract(args.comparison, args.request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    print(json.dumps({'output': str(args.output), 'bytes': args.output.stat().st_size,
                      'schema_version': data['schema_version'], 'inventory': data['inventory'],
                      'primary_runs': sum(len(stage['runs']) for stage in data['primary']),
                      'method_comparison_runs': sum(len(runs) for method in data['methods'] for runs in method['runs'].values()),
                      'final_verification': data['chosen_verification']['verified_checks']}, indent=2))


if __name__ == '__main__':
    main()
