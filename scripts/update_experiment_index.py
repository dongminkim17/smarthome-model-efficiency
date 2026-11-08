#!/usr/bin/env python3
"""Build a timestamped experiment overview from authoritative saved artifacts.

Standard-library only; uses no GPUs, imports no model code, and never changes
queue state. Generated views are reporting outputs, never training inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import io
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def tail_progress(path):
    """Read a small suffix; ignore an in-progress final JSONL write."""
    if not path.is_file():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65536))
        rows = stream.read().splitlines()
    for row in reversed(rows):
        try:
            value = json.loads(row)
        except (ValueError, UnicodeDecodeError):
            continue
        if value.get('global_step') is not None:
            return {k: value.get(k) for k in ('event', 'global_step', 'epoch', 'recorded_at', 'elapsed_seconds')}
    return {}


def finite(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def compact_status(value):
    if value in ('completed', 'failed', 'interrupted', 'blocked', 'running', 'cancelled'):
        return value
    return 'queued'


def absolute(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def normalize_run(root, family, name, task, arm, run_dir, state, job=None,
                  test_job=None, test_report_path=None, tests_scheduled=True):
    job = job or {}
    run = absolute(root, run_dir) if run_dir else None
    report_path = run / 'report.json' if run else None
    report = read_json(report_path) if report_path else None
    cfg = (report or {}).get('config') or job.get('config') or {}
    status = compact_status(state or (report or {}).get('status'))
    best = None
    if run and (run / 'checkpoints/best').exists():
        best = str((run / 'checkpoints/best').resolve())
    progress = tail_progress(run / 'metrics.jsonl') if run and status == 'running' else {}
    steps = progress.get('global_step', (report or {}).get('global_step'))
    expected = (report or {}).get('requested_steps')
    epoch = progress.get('epoch')
    if epoch is None and expected and cfg.get('epochs'):
        epoch = steps / expected * cfg['epochs'] if steps is not None else None
    components = (report or {}).get('base_parameters', {}).get('components', {})
    language_parameters = components.get('model', 0) + components.get('lm_head', 0) or None
    sidecar = read_json(absolute(root, cfg['model'])/'compression.json') if cfg.get('model') else None
    compact_counts = (sidecar or {}).get('parameters_after', {})
    if not language_parameters:
        language_parameters = compact_counts.get('language')
    expected_counts = job.get('expected_parameters', {})
    if not language_parameters:
        language_parameters = expected_counts.get('language')
    total_parameters = ((report or {}).get('base_parameters', {}).get('total')
                        or compact_counts.get('total') or expected_counts.get('total'))
    encoder_parameters = (components.get('audio_tower', 0) + components.get('visual', 0)
                          or sum(compact_counts.get(k, 0) for k in ('audio_encoder','vision_encoder'))
                          or sum(expected_counts.get(k, 0) for k in ('audio_encoder','vision_encoder')) or None)
    budget_scope = job.get('budget_scope') or (sidecar or {}).get('budget_scope')
    test_state = 'not_scheduled' if not tests_scheduled else 'queued'
    if test_job:
        test_state = compact_status(test_job['status'])
        if test_job.get('run_dir'):
            test_report_path = Path(test_job['run_dir']) / 'report.json'
    test_report_path = absolute(root, test_report_path) if test_report_path else None
    test = read_json(test_report_path) if test_report_path else None
    warnings = []
    metrics = {}
    if test is not None:
        if test.get('status') == 'completed':
            valid = (test.get('split') == 'test' and test.get('test_set_used') is True
                     and test.get('partial_split') is False
                     and test.get('evaluated_count') == test.get('full_split_count')
                     and test.get('evaluated_count', 0) > 0
                     and best is not None and test.get('checkpoint') == best)
            if valid:
                test_state = 'completed'
                metrics = {f'{mode}_test_{key}': finite(test.get(mode, {}).get(key))
                           for mode in ('logit', 'greedy') for key in ('accuracy', 'weighted_f1', 'macro_f1', 'invalid_rate')}
            else:
                test_state = 'unverified'
                warnings.append('Test report is partial, mismatched or not tied to the current best checkpoint')
        elif test_state == 'queued':
            test_state = compact_status(test.get('status'))
    metric = 'weighted_f1' if task == 'mer6' else 'accuracy'
    source = job.get('init_adapter') or cfg.get('teacher_checkpoint') or job.get('source_checkpoint')
    return {
        'id': f'{family}/{name}/{arm}', 'family': family, 'name': name, 'task': task, 'arm': arm,
        'training_status': status, 'test_status': test_state, 'epoch': epoch,
        'epochs_requested': cfg.get('epochs', 5), 'global_step': steps,
        'primary_metric': metric, 'best_val_score': finite((report or {}).get('best_score')),
        'language_parameters': language_parameters, 'total_parameters': total_parameters,
        'combined_encoder_parameters': encoder_parameters,
        'budget_scope': budget_scope, 'lora_rank': cfg.get('lora_rank'),
        'lora_alpha': cfg.get('lora_alpha'), 'lora_target': cfg.get('lora_target'),
        'cancellation_reason': job.get('cancellation_reason'),
        'objective': cfg.get('objective', 'sft') if cfg else None,
        'learning_rate': cfg.get('learning_rate'), 'encoder_learning_rate': cfg.get('encoder_learning_rate'), 'seed': cfg.get('seed'),
        'modalities': 'audio+video+transcript' if cfg.get('use_transcript') else 'audio+video' if cfg else None,
        'gpu': job.get('gpu') if status == 'running' else None,
        'pid': job.get('pid') if status == 'running' else None,
        'source': source, 'initialization': job.get('initialization'),
        'selected_source': job.get('source_name'), 'config': cfg,
        'parent_checkpoint': job.get('parent_checkpoint'), 'parent_step': job.get('parent_step'),
        'parent_epoch': job.get('parent_epoch'),
        'updates_including_parent_at_latest': job.get('parent_step', 0) + steps if steps is not None else None,
        'run_dir': str(run) if run else None, 'report': str(report_path) if report_path else None,
        'test_report': str(test_report_path) if test_report_path else None,
        'best_checkpoint': best, 'curves': str(run / 'curves.png') if run and (run / 'curves.png').is_file() else None,
        'log': job.get('log'), 'warnings': warnings, **metrics,
    }


def collect(root, catalog):
    records = []
    families = []
    for family in catalog['families']:
        group = []
        manifest_path = absolute(root, family['manifest']) if family.get('manifest') else None
        manifest = read_json(manifest_path) if manifest_path else None
        followup = read_json(absolute(root, family['followup'])) if family.get('followup') else None
        if family['kind'] == 'baseline':
            for run in family['runs']:
                group.append(normalize_run(root, family['id'], run['name'], run['task'], 'SFT',
                                           run['path'], None, test_report_path=run['test_report']))
        elif family['kind'] in ('sweep', 'pruning'):
            follow_jobs = {j['id']: j for j in (followup or {}).get('jobs', [])}
            for job in (manifest or {}).get('jobs', []):
                task = job.get('task') or job['config']['task']
                group.append(normalize_run(root, family['id'], job['name'], task, 'SFT',
                                           job['output'], job['status'], job,
                                           test_job=follow_jobs.get('sft_test:' + job['name']),
                                           tests_scheduled=family['kind'] == 'pruning'))
                if family['kind'] == 'pruning':
                    kd = follow_jobs.get('kd_train:' + job['name'])
                    planned = {'config': dict(job['config'], objective='sft_kd'),
                               'source_checkpoint': job.get('source_checkpoint')}
                    group.append(normalize_run(root, family['id'], job['name'], task, 'KD',
                                               (kd or {}).get('run_dir'), (kd or {}).get('status', 'waiting'), kd or planned,
                                               test_job=follow_jobs.get('kd_test:' + job['name'])))
        elif family['kind'] == 'recovery':
            jobs = {j['id']: j for j in (manifest or {}).get('jobs', [])}
            for job in jobs.values():
                if job['kind'] != 'train':
                    continue
                group.append(normalize_run(root, family['id'], job['name'], job['task'], job['arm'],
                                           job.get('run_dir'), job['status'], job,
                                           test_job=jobs.get('test:' + job['name']),
                                           tests_scheduled=('test:' + job['name']) in jobs))
        else:
            raise ValueError('Unknown catalog family type')
        records.extend(group)
        families.append({**family, 'status': (manifest or {}).get('status', 'completed' if group and all(r['training_status'] == 'completed' for r in group) else 'not_started'),
                         'followup_status': (followup or {}).get('status'),
                         'training_counts': dict(Counter(r['training_status'] for r in group)),
                         'test_counts': dict(Counter(r['test_status'] for r in group)),
                         'run_count': len(group), 'manifest_absolute': str(manifest_path) if manifest_path else None,
                         'launcher_pid': (manifest or {}).get('launcher_pid')})
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate logical experiment IDs')
    return records, families


def score(value):
    return f'{100 * value:.2f}%' if value is not None else '—'


def link(path, label):
    return f'[{label}](<{path}>)' if path else '—'


def render_status(snapshot):
    lines = ['# Experiment status', '', f"Snapshot: **{snapshot['generated_at_kst']}**. Refresh before relying on runtime state.",
             '', 'Generated from queue manifests and run reports. This view does not control experiments.',
             'Validation and official test scores are separate. MER uses weighted F1; MIntRec uses accuracy.',
             '', 'See [QUEUE.md](QUEUE.md) for active GPU assignments, phases and exact waiting prerequisites.',
             '', '## Queue overview', '', '| Family | Training: done / active / queued / other | Official tests done | Protocol |',
             '| --- | --- | ---: | --- |']
    for family in snapshot['families']:
        counts = family['training_counts']
        done, active, queued = [counts.get(k, 0) for k in ('completed', 'running', 'queued')]
        other = family['run_count'] - done - active - queued
        lines.append(f"| {family['label']} | {done} / {active} / {queued} / {other} | {family['test_counts'].get('completed', 0)} | {link(str(ROOT / family['protocol']), 'details')} |")
    lines += ['', 'Smoke jobs, teacher-cache creation and separate test jobs are not counted as training experiments.',
              'Queued KD rows are included even before the follow-up coordinator starts.',
              '', '## Full experiment table', '',
              '| Experiment | Task | Arm | Train | Epoch | Best val | Official test: logit / greedy | Artifacts |',
              '| --- | --- | --- | --- | ---: | ---: | ---: | --- |']
    for run in snapshot['runs']:
        metric = run['primary_metric']
        logit = score(run.get(f'logit_test_{metric}'))
        greedy = score(run.get(f'greedy_test_{metric}'))
        result = f'{logit} / {greedy}' if run['test_status'] == 'completed' else run['test_status']
        epoch = f"{run['epoch']:.2f}" if run['epoch'] is not None else '—'
        artifacts = link(run['run_dir'], 'run') + ' ' + link(run['test_report'], 'test')
        lines.append(f"| {run['family']}/{run['name']} | {run['task']} | {run['arm']} | {run['training_status']} | {epoch} | {score(run['best_val_score'])} | {result} | {artifacts} |")
    warnings = [(r['id'], w) for r in snapshot['runs'] for w in r['warnings']]
    if warnings:
        lines += ['', '## Verification notes', ''] + [f'- {name}: {warning}' for name, warning in warnings]
    lines += ['', 'Detailed configurations and lineage: [runs.json](runs.json). Spreadsheet-friendly measurements: [results.csv](results.csv).', '']
    return '\n'.join(lines)


def atomic_write(path, text):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'experiments/catalog.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments')
    args = parser.parse_args()
    catalog = read_json(args.catalog)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.refresh.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records, families = collect(ROOT, catalog)
        snapshot = {'schema_version': 1, 'generated_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
                    'source_catalog': str(args.catalog.resolve()), 'families': families, 'runs': records,
                    'note': 'Reporting snapshot only. Read linked queue manifests for authoritative live state.'}
        atomic_write(output / 'runs.json', json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
        columns = ['id', 'family', 'name', 'task', 'arm', 'training_status', 'test_status', 'epoch',
                   'epochs_requested', 'global_step', 'language_parameters', 'total_parameters', 'combined_encoder_parameters', 'budget_scope', 'lora_rank', 'lora_alpha', 'lora_target',
                   'objective', 'cancellation_reason', 'learning_rate', 'encoder_learning_rate', 'seed', 'modalities', 'primary_metric', 'best_val_score',
                   'logit_test_accuracy', 'greedy_test_accuracy', 'logit_test_weighted_f1', 'greedy_test_weighted_f1',
                   'selected_source', 'parent_checkpoint', 'parent_step', 'parent_epoch',
                   'updates_including_parent_at_latest', 'source', 'run_dir', 'report', 'test_report', 'best_checkpoint', 'curves']
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
        atomic_write(output / 'results.csv', stream.getvalue())
        atomic_write(output / 'STATUS.md', render_status(snapshot))
        print(f"Indexed {len(records)} training experiments across {len(families)} families at {snapshot['generated_at_kst']}")


if __name__ == '__main__':
    main()
