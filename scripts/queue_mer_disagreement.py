#!/usr/bin/env python3
"""Nine MER runs: teacher-disagreement KD with repeated-seed group selection."""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from launch_pruning_followup import checkpoint_identity, pin_best
from project_queue import validate_submission
from project_queue_jobs import entry
from queue_ffn_expansion import training_chain
from queue_mer_focus import build_submission as validate_reference_contract
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.queueing import atomic_json, fingerprint, read_json

FAMILY = 'mer_disagreement_20261105'
CONTROL_REFERENCES = {
    (16, 42): 'encoder_budget_20261021/train:mer6_balanced_feature_splitlr',
    (16, 43): 'recovery_refinements_20261022/train:mer6_balanced_seed43',
    (64, 42): 'recovery_refinements_20261022/train:mer6_balanced_r64',
}


def definitions():
    result = []
    for rank in (16, 64):
        for alpha in (.25, 0.):
            setting = f'R{rank}_DISAGREE{int(alpha*100):03d}'
            for seed in (42, 43):
                result.append({'arm': f'{setting}_SEED{seed}', 'rank': rank, 'seed': seed,
                               'disagreement_alpha': alpha, 'setting': setting})
    result.append({'arm': 'CONTROL_R64_SEED43', 'rank': 64, 'seed': 43,
                   'disagreement_alpha': None, 'setting': 'CONTROL_R64'})
    return result


def validate_controls(legacy):
    """Reuse only exact matched controls, without inspecting official-test scores."""
    by_id = {j['id']: j for j in legacy['jobs']}
    cfg = copy.deepcopy(by_id[CONTROL_REFERENCES[(16, 42)]]['payload']['config'])
    class_hash = sha256_file(Path(cfg['teacher_cache']))
    feature_hash = sha256_file(Path(cfg['feature_teacher_cache'])/'cache.json')
    base_hash = sha256_file(Path(cfg['model'])/'compression.json')
    controls, manifest_hash = [], None
    for (rank, seed), key in CONTROL_REFERENCES.items():
        job = by_id[key];run = Path(job['payload']['run_dir']);report = read_json(run/'report.json')
        expected = dict(cfg, lora_rank=rank, lora_alpha=2*rank, seed=seed)
        actual = report['config']
        normalize = lambda c: {k: v for k, v in c.items() if k != 'eval_modes'}
        if (job['status'] != 'completed' or report['status'] != 'completed'
                or report.get('test_set_used') is not False or report.get('train_count') != 2698
                or report.get('dev_count') != 675 or report.get('world_size') != 1
                or job['payload']['config'] != actual or normalize(actual) != normalize(expected)
                or 'logit' not in actual['eval_modes']
                or report.get('teacher_cache_sha256') != class_hash
                or report.get('feature_cache_sha256') != feature_hash):
            raise ValueError('Historical control is not an exact matching full-data recovery: '+key)
        digest = sha256_file(run/'data_manifest.json')
        if manifest_hash is not None and digest != manifest_hash:
            raise ValueError('Historical controls have different training/development records')
        manifest_hash = digest
        best = pin_best(run)
        if not best:
            raise ValueError('Historical control lacks its best-validation checkpoint')
        metadata = read_json(Path(best)/'metadata.json')
        if (metadata['config'] != actual or metadata.get('base_model_manifest_sha256') != base_hash
                or metadata.get('teacher_cache_sha256') != class_hash
                or metadata.get('feature_cache_sha256') != feature_hash):
            raise ValueError('Historical control checkpoint provenance changed')
        controls.append({'job': key, 'rank': rank, 'seed': seed, 'run_dir': str(run),
                         'checkpoint': best, 'checkpoint_identity': checkpoint_identity(best),
                         'data_manifest_sha256': digest, 'base_model_manifest_sha256': base_hash,
                         'teacher_cache_sha256': class_hash, 'feature_cache_sha256': feature_hash,
                         'reuse_note': 'Same recovery recipe; historical validation also generated greedy outputs, but selected by logit F1.'})
    return cfg, controls


def build_submission(legacy, queue_root, root=ROOT):
    root, queue_root = Path(root), Path(queue_root)
    validate_reference_contract(legacy, queue_root, root)
    cfg, controls = validate_controls(legacy)
    if cfg.get('mer_partition') or cfg['task'] != 'mer6':
        raise ValueError('Disagreement experiment requires the original MER80/20 protocol')
    if any(k in cfg for k in ('distillation_disagreement_alpha', 'distillation_alpha_final', 'distillation_decay_start')):
        raise ValueError('Reference must use the original constant class distillation')
    counts = read_json(Path(cfg['model'])/'compression.json')['parameters_after']
    base = root/'artifacts/full'/FAMILY;manifest = base/'workflow.json'
    stress = queue_root/'stress_cases/mer6_disagreement_80_20.json'
    prepare = FAMILY+'/prepare:mer6';shortlist = FAMILY+'/shortlist:mer6'
    jobs = [entry(prepare, 'prepare', {'id': 'prepare:mer6', 'kind': 'prepare', 'task': 'mer6',
                  'config': copy.deepcopy(cfg), 'output': str(stress), 'estimated_seconds': 180,
                  'require_teacher_disagreement': True},
                  {'manifest': str(manifest), 'field': 'preparation_jobs', 'key_field': 'id', 'key': 'prepare:mer6'},
                  resource='cpu', priority=150)]
    groups, plan = {}, []
    for variant in definitions():
        arm, rank, seed, alpha = (variant[k] for k in ('arm', 'rank', 'seed', 'disagreement_alpha'))
        new = copy.deepcopy(cfg)
        new.update(lora_rank=rank, lora_alpha=2*rank, seed=seed, epochs=5,
                   eval_modes=['logit'], selection_mode='logit', eval_every_fraction=.5,
                   train_limit=None, dev_limit=None, micro_batch_size=2, gradient_accumulation_steps=2,
                   gradient_checkpointing=True)
        if alpha is not None:
            new['distillation_disagreement_alpha'] = alpha
        chain = training_chain(FAMILY, manifest, queue_root, root, 'mer6', arm, new,
                               cfg['teacher_checkpoint'], [prepare])
        train = next(j for j in chain if j['kind'] == 'recovery_train')
        if alpha is not None:
            group = groups.setdefault(variant['setting'], {'name': variant['setting'], 'config': new, 'members': []})
            group['members'].append({'job': train['id'], 'seed': seed})
        priority = 100 if alpha is not None else 80
        for job in chain:
            payload = job['payload']
            payload.update(architecture='BALANCED', budget_scope='per_modality', expected_parameters=counts,
                           group='teacher_disagreement' if alpha is not None else 'control',
                           setting=variant['setting'], recovery_epoch_budget=5,
                           reference_two_stage_train=CONTROL_REFERENCES[(rank, 42)])
            if job['kind'] in ('smoke_train', 'recovery_train'):
                job['priority'] = priority + (2 if job['kind'] == 'smoke_train' else 0)
                payload.update(expected_teacher_cache_sha256=controls[0]['teacher_cache_sha256'],
                               expected_feature_cache_sha256=controls[0]['feature_cache_sha256'],
                               initialization='Fresh final-recovery adapters on the fixed sequential balanced export',
                               estimated_seconds=180 if job['kind'] == 'smoke_train' else 4*3600)
                if job['kind'] == 'smoke_train':
                    payload['memory_cases'] = str(stress)
            elif job['kind'] == 'smoke_test':
                job['priority'] = priority + 3
            elif job['kind'] == 'recovery_test':
                payload['estimated_seconds'] = 900
                if alpha is not None:
                    payload['shortlist_job'] = shortlist
                    job['deps'].append(shortlist)
                else:
                    payload['mandatory_control_test'] = True
        jobs.extend(chain)
        plan.append({**variant, 'epochs': 5, 'class_kd_agreement': .75,
                     'teacher_temperature': 2., 'feature_weights': [.1, .1]})
    members = [m['job'] for g in groups.values() for m in g['members']]
    selection_payload = {'id': 'shortlist:mer6', 'kind': 'shortlist', 'task': 'mer6',
                         'top_k': 2, 'required_seeds': [42, 43], 'groups': list(groups.values()),
                         'dev_count': 675, 'train_count': 2698,
                         'expected_data_manifest_sha256': controls[0]['data_manifest_sha256'],
                         'expected_teacher_cache_sha256': controls[0]['teacher_cache_sha256'],
                         'expected_feature_cache_sha256': controls[0]['feature_cache_sha256'],
                         'output': str(base/'validation_shortlist.json')}
    jobs.append(entry(shortlist, 'shortlist', selection_payload,
                      {'manifest': str(manifest), 'field': 'preparation_jobs', 'key_field': 'id', 'key': 'shortlist:mer6'},
                      members, resource='inline', priority=150))
    document = {'format_version': 1, 'status': 'waiting', 'phase': 'per_job_dependencies', 'host': 'joy',
                'allowed_gpus': list(range(8)), 'jobs': [], 'smoke_jobs': [], 'preparation_jobs': [],
                'description': 'Nine MER recoveries: eight teacher-disagreement variants plus one matched control',
                'experiment_plan': plan, 'reused_controls': controls,
                'checkpoint_selection_uses_test': False, 'shortlist_uses_test': False,
                'official_test_top_k_groups': 2, 'seeds_per_group': [42, 43],
                'official_test_expected_count': 5, 'estimated_wall_hours': [8, 9],
                'data_counts': {'train': 2698, 'dev': 675, 'official_test': 834},
                'note': 'Fixed balanced pruning, inputs, original teacher, and feature KD. Five epochs. Select two complete rank+gating settings by mean best-dev F1 over seeds42/43, then test both seeds; control test is independent. No checkpoint averaging. Before adoption, verify the chosen recovery rule on both MIntRec tasks.'}
    for job in jobs:
        document[job['location']['field']].append(copy.deepcopy(job['payload']))
    return {'schema_version': 1, 'jobs': jobs, 'documents': {str(manifest): document}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root', type=Path, default=ROOT/'artifacts/queue/joy')
    parser.add_argument('--source-queue', type=Path, default=ROOT/'artifacts/queue/suzy')
    parser.add_argument('--submit', action='store_true');args = parser.parse_args()
    state = read_json(args.queue_root/'state.json')
    if state.get('host') != 'joy' or state['allowed_gpus'] != list(range(8)):
        raise ValueError('Joy0–7 resource contract required')
    path = args.queue_root/'submissions'/(FAMILY+'.json')
    spec = read_json(path) if path.exists() else build_submission(read_json(args.source_queue/'state.json'), args.queue_root)
    validate_submission(state, spec)
    if args.submit:
        atomic_json(path, spec)
        atomic_json(args.queue_root/'inbox'/(fingerprint(spec)+'.json'), spec)
    print(json.dumps({'submitted': args.submit, 'family': FAMILY, 'jobs': len(spec['jobs']),
                      'new_training_runs': sum(j['kind'] == 'recovery_train' for j in spec['jobs']),
                      'reused_controls': len(next(iter(spec['documents'].values()))['reused_controls']),
                      'official_tests': 'Two paired-seed settings plus independent control; at most five tests',
                      'plan': next(iter(spec['documents'].values()))['experiment_plan']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
