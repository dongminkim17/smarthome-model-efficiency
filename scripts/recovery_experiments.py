"""Definitions and validation-only source selection for recovery ablations A1–A4."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from launch_pruning_followup import checkpoint_identity, pin_best
from smarthome_efficiency.training import DEFAULTS, config_fingerprint

TASKS = ('mer6', 'mintrec2', 'mintrec20')
ARMS = ('A1', 'A2', 'A3', 'A4')
CONTROL_ARMS = ('SFT_CONT', 'KD025', 'KD075', 'R64_LOWLR')
DESCRIPTIONS = {
    'A1': 'Initial compact base + fresh rank-64 SFT',
    'A2': 'Initial compact base + fresh rank-64 SFT/KD; original teacher',
    'A3': 'Best rank-16 compact SFT adapter + five new SFT/KD epochs at LR 3e-5',
    'A4': 'Original unpruned pretrained Omni + fresh rank-64 SFT; original teacher adapter scope',
    'SFT_CONT': 'Same rank-16 SFT parent as A3; five new SFT epochs at LR 3e-5',
    'KD025': 'Initial compact base + fresh rank-16 SFT/KD; original teacher; alpha 0.25',
    'KD075': 'Initial compact base + fresh rank-16 SFT/KD; original teacher; alpha 0.75',
    'R64_LOWLR': 'Same initial compact base and fresh rank-64 SFT as A1; LR 3e-5',
}


def queued_jobs(output, arms=ARMS, tasks=TASKS):
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS + CONTROL_ARMS for a in arms):
        raise ValueError('Choose unique supported recovery arms')
    return [{'id': f'{kind}:{arm}_{task}', 'name': f'{arm}_{task}', 'arm': arm,
             'task': task, 'kind': kind, 'status': 'waiting',
             'deps': [] if kind == 'train' else [f'train:{arm}_{task}'],
             'output': str(Path(output) / kind / f'{arm}_{task}')}
            for arm in arms for task in tasks for kind in ('train', 'test')]


def select_sources(sweeps, tasks=TASKS):
    """Read completed SFT validation reports only; never read test/KD scores."""
    candidates = {task: [] for task in tasks}
    for sweep in sweeps:
        if sweep['status'] != 'completed':
            raise ValueError('All candidate SFT sweeps must finish before source selection')
        for job in sweep['jobs']:
            if job['task'] not in candidates:
                continue
            if job['status'] != 'completed':
                raise ValueError('A candidate did not finish')
            run = Path(job['output'])
            report = json.loads((run / 'report.json').read_text())
            cfg = DEFAULTS | report['config']
            if (report['status'] != 'completed' or report.get('test_set_used') is not False
                    or cfg['objective'] != 'sft' or cfg['selection_mode'] != 'logit'
                    or cfg['task'] != job['task'] or cfg['model'] != job['compact_model']
                    or cfg['lora_rank'] != 16 or cfg['lora_alpha'] != 32):
                raise ValueError('Expected completed rank-16 SFT source with validation-only selection')
            score = report['best_score']
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid source validation score')
            checkpoint = pin_best(run)
            metadata = json.loads((Path(checkpoint) / 'metadata.json').read_text())
            if metadata['config'] != report['config']:
                raise ValueError('Source checkpoint configuration differs from report')
            sidecar = json.loads((Path(job['compact_model']) / 'compression.json').read_text())
            if not sidecar.get('verification', {}).get('passed'):
                raise ValueError('Compact export was not verified')
            count = sidecar['parameters_after']['language']
            if not 0 < count < 2_000_000_000:
                raise ValueError('Student exceeds the language parameter budget')
            candidates[job['task']].append({
                'name': job['name'], 'method': job['method'], 'sft_run': str(run),
                'validation_score': score, 'validation_metric': 'weighted_f1' if job['task'] == 'mer6' else 'accuracy',
                'checkpoint': checkpoint, 'checkpoint_identity': checkpoint_identity(checkpoint),
                'compact_model': job['compact_model'], 'language_parameters': count,
                'teacher_checkpoint': str(Path(job['source_checkpoint']).resolve()), 'config': cfg,
            })
    selected = {}
    for task, choices in candidates.items():
        if not choices or len({c['name'] for c in choices}) != len(choices):
            raise ValueError('Missing or duplicated task candidates')
        if len({c['teacher_checkpoint'] for c in choices}) != 1:
            raise ValueError('Candidate methods must share the established original teacher')
        ranked = sorted(choices, key=lambda c: (-c['validation_score'], c['name']))
        selected[task] = {'winner': ranked[0], 'ranking': ranked,
                          'rule': 'highest completed rank-16 SFT best validation score; ties by name',
                          'test_results_used': False, 'distillation_results_used': False}
    return selected


def materialize_jobs(output, selected, teacher_cache_root, smoke=False, arms=ARMS, tasks=TASKS):
    jobs = queued_jobs(output, arms, tasks)
    for job in jobs:
        winner = selected[job['task']]['winner']
        if job['kind'] == 'test':
            job.update(split='dev' if smoke else 'test', limit=6 if smoke else None,
                       modes=['logit', 'greedy'])
            continue
        cfg = copy.deepcopy(winner['config'])
        cfg.update(epochs=5, seed=42, micro_batch_size=2, gradient_accumulation_steps=2,
                   train_limit=None, dev_limit=None, learning_rate=1e-4,
                   eval_modes=['logit', 'greedy'], selection_mode='logit',
                   objective='sft', teacher_checkpoint=None, teacher_cache=None)
        job['init_adapter'] = None
        if job['arm'] == 'A4':
            teacher_metadata = json.loads((Path(winner['teacher_checkpoint']) / 'metadata.json').read_text())
            cfg = copy.deepcopy(DEFAULTS | teacher_metadata['config'])
            cfg.update(epochs=5, seed=42, lora_rank=64, lora_alpha=128, objective='sft',
                       teacher_checkpoint=None, teacher_cache=None,
                       micro_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True,
                       train_limit=None, dev_limit=None, eval_modes=['logit', 'greedy'], selection_mode='logit')
            job['initialization'] = 'original pretrained base; fresh zero-update rank-64 adapter'
        else:
            cfg['model'] = winner['compact_model']
            if job['arm'] in ('A1', 'A2', 'R64_LOWLR'):
                cfg.update(lora_rank=64, lora_alpha=128)
                if job['arm'] == 'R64_LOWLR':
                    cfg['learning_rate'] = 3e-5
                job['initialization'] = 'immutable initial compact base; fresh zero-update rank-64 adapter'
            elif job['arm'] in ('A3', 'SFT_CONT'):
                cfg.update(learning_rate=3e-5)
                job['init_adapter'] = str(Path(winner['checkpoint']) / 'adapter')
                job['init_checkpoint_identity'] = winner['checkpoint_identity']
                job['initialization'] = 'best rank-16 SFT adapter; fresh optimizer/schedule; five additional epochs'
            else:
                job['initialization'] = 'immutable initial compact base; fresh zero-update rank-16 adapter'
            if job['arm'] in ('A2', 'A3', 'KD025', 'KD075'):
                cfg.update(objective='sft_kd', teacher_checkpoint=winner['teacher_checkpoint'],
                           teacher_cache=str(Path(teacher_cache_root) / job['task'] / 'cache.json'),
                           distillation_alpha={'KD025': .25, 'KD075': .75}.get(job['arm'], .5),
                           distillation_temperature=2.)
                cache = json.loads(Path(cfg['teacher_cache']).read_text())
                if (cache['status'] != 'completed' or cache['spec']['split'] != 'train'
                        or cache['spec']['test_set_used'] is not False
                        or cache['spec']['teacher_checkpoint'] != cfg['teacher_checkpoint']):
                    raise ValueError('Original teacher cache is unavailable or changed')
        if smoke:
            cfg.update(epochs=1, train_limit=24, dev_limit=20)
        job.update(config=cfg, config_sha256=config_fingerprint(cfg), source_name=winner['name'],
                   status='ready', training_epochs=cfg['epochs'])
    return jobs
