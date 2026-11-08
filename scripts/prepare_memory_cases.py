#!/usr/bin/env python3
"""Index prepared training inputs once and choose large memory-smoke examples."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from smarthome_efficiency.data import LABELS, load_records
from smarthome_efficiency.model import PilotModel
from smarthome_efficiency.training import load_config
from smarthome_efficiency.queueing import atomic_json, now


def choose_cases(rows, count=8):
    selected = []
    for key, take in [('tokens', 4), ('audio_frames', 2), ('vision_elements', 2)]:
        ranked = sorted(rows, key=lambda r: (-r[key], r['id']))
        for row in ranked:
            if row['id'] not in selected:
                selected.append(row['id'])
                take -= 1
                if not take:
                    break
    for row in sorted(rows, key=lambda r: (-r['tokens'], r['id'])):
        if len(selected) >= count:
            break
        if row['id'] not in selected:
            selected.append(row['id'])
    return selected[:count]



def ensure_teacher_cases(ids, rows, disagreement_ids):
    """Preserve memory extremes while covering both teacher branches in eight clips."""
    selected = list(ids)
    all_ids = {row['id'] for row in rows}
    wrong = set(disagreement_ids)
    if len(selected) != 8 or len(set(selected)) != 8 or not set(selected) <= all_ids or not wrong <= all_ids:
        raise ValueError('Invalid training-case selection')
    if not wrong or wrong == all_ids:
        raise ValueError('Teacher-gating preflight requires both agreeing and disagreeing training clips')
    for wants_wrong in (True, False):
        if any((key in wrong) == wants_wrong for key in selected):
            continue
        choices = sorted((row for row in rows if (row['id'] in wrong) == wants_wrong),
                         key=lambda row: (-row['tokens'], row['id']))
        # The first four cases are the largest token sequences. Replace the last
        # lower-priority case only if branch coverage was missing.
        selected[-1] = choices[0]['id']
    return selected

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-teacher-disagreement', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = load_records(cfg['task'], 'train', seed=cfg['split_seed'], data_root=cfg['data_root'], mer_partition=cfg.get('mer_partition'))
    runner = PilotModel(cfg['model'], cfg['task'], LABELS[cfg['task']], cfg['cache_dir'],
                        max_frames=cfg['max_frames'], max_pixels=cfg['max_pixels'], use_transcript=cfg['use_transcript'])
    rows = []
    for index, record in enumerate(records):
        batch = runner.prepare(record)
        rows.append({'id': record['id'], 'tokens': int(batch['attention_mask'].sum()),
                     'audio_frames': int(batch.get('feature_attention_mask').sum()) if 'feature_attention_mask' in batch else 0,
                     'vision_elements': sum(t.numel() for key, t in batch.items() if key.startswith('pixel_values'))})
        if (index + 1) % 250 == 0:
            print(json.dumps({'event': 'preparing_inputs', 'completed': index+1, 'total': len(records)}), flush=True)
    ids = choose_cases(rows)
    coverage = {}
    if args.require_teacher_disagreement:
        from smarthome_efficiency.distillation import TeacherLogitCache, teacher_spec
        labels = list(LABELS[cfg['task']])
        spec = teacher_spec(cfg, cfg['teacher_checkpoint'], labels, runner.label_token_ids, runner.versions)
        teacher = TeacherLogitCache(cfg['teacher_cache'], spec, records)
        truth = {record['id']: labels.index(record['label']) for record in records}
        wrong = {key for key in truth if max(range(len(labels)), key=lambda c: teacher.rows[key]['logits'][c]) != truth[key]}
        ids = ensure_teacher_cases(ids, rows, wrong)
        coverage = {'teacher_coverage': {'teacher_cache_sha256': teacher.sha256,
                    'disagreement_ids': [key for key in ids if key in wrong],
                    'agreement_ids': [key for key in ids if key not in wrong],
                    'all_training_disagreements': len(wrong)}}
    atomic_json(args.output, {'status': 'completed', 'split': 'train', 'test_set_used': False,
                              'task': cfg['task'], 'ids': ids, 'created_at': now(), 'examples_indexed': len(rows), **coverage,
                              'selected': [r for r in rows if r['id'] in ids], 'input_sizes': rows,
                              'note': 'Large training cases across prompt/audio/vision; empirical preflight, not a formal worst-case bound.'})


if __name__ == '__main__':
    main()
