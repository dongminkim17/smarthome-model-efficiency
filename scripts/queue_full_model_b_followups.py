#!/usr/bin/env python3
"""Submit 18 approved B recovery / nearby-budget experiments, additively."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from smarthome_efficiency.queueing import atomic_json, read_json, fingerprint
from smarthome_efficiency.training import config_fingerprint
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.pruning import sha256_file
from launch_pruning_followup import checkpoint_identity
from project_queue import validate_submission
from project_queue_jobs import entry
from queue_ffn_expansion import training_chain

FAMILY = 'full_model_b_followups_20261021'
TASKS = ('mintrec2', 'mer6', 'mintrec20')
B_WIDTHS = (1280, 1536, 1024)
B_CONTROLS = (
    ('B_LOWLR', {'learning_rate': 3e-5}),
    ('B_SPLITLR', {'encoder_learning_rate': 1e-5}),
    ('B_KD050', {'distillation_alpha': .5}),
    ('B_SFT', {'objective': 'sft', 'teacher_checkpoint': None, 'teacher_cache': None}),
)
NEW_BUDGETS = (
    ('E_LM1408_A1536_V896', (1408, 1536, 896)),
    ('F_LM1280_A2048_V768', (1280, 2048, 768)),
)


def build_submission(state, root=ROOT):
    root = Path(root)
    manifest = root/'artifacts/full'/FAMILY/'workflow.json'
    by_id = {j['id']: j for j in state['jobs']}
    jobs = []
    for task in TASKS:
        # Binary-intent work takes the next free resource after older priority-2
        # work; other new task runs follow. No running worker is preempted.
        priority = 1 if task == 'mintrec2' else 0
        b_name = task+'_b_lm1280_a1536_v1024'
        b_export_id = 'full_model_2b_20261020/export:'+b_name
        b_export = by_id[b_export_id]
        side = read_json(Path(b_export['payload']['output'])/'compression.json')
        if (b_export['status'] != 'completed' or side.get('task') != task
                or side['budget_scope'] != 'total' or side['method'] != 'multimodal_ffn_activation'
                or side['parameters_after'] != expected_counts(*B_WIDTHS)
                or not side['verification']['passed']):
            raise ValueError('Expected verified B export: '+task)
        reference = by_id['full_model_2b_20261020/train:'+b_name]['payload']['config']
        if (reference['lora_target'] != 'decoder_encoder_ffn' or reference['lora_rank'] != 16
                or reference['objective'] != 'sft_kd' or reference['distillation_alpha'] != .75
                or reference['distillation_temperature'] != 2. or reference['learning_rate'] != 1e-4
                or reference.get('feature_teacher_cache') or reference.get('encoder_learning_rate')):
            raise ValueError('B control recipe differs from the approved reference')
        teacher_id = 'calibrated_pruning_20261019/teacher:'+task
        cache = read_json(Path(reference['teacher_cache']))
        if (by_id[teacher_id]['status'] != 'completed' or cache['status'] != 'completed'
                or cache['spec']['split'] != 'train' or cache['spec']['test_set_used'] is not False
                or cache['spec']['teacher_checkpoint'] != reference['teacher_checkpoint']):
            raise ValueError('Original training-only teacher cache is unavailable')
        source = side['source_checkpoint']
        source_config = read_json(Path(source)/'metadata.json')['config']
        calibration_path = Path(side['calibration_cache'])
        calibration_hash = sha256_file(calibration_path)
        if side['calibration']['sha256'] != calibration_hash:
            raise ValueError('Pinned whole-model calibration cache changed')
        variants = []
        for arm, changes in B_CONTROLS:
            variants.append((arm, 'B', b_export['payload']['output'], b_export_id, expected_counts(*B_WIDTHS), changes))
        for arm, widths in NEW_BUDGETS:
            counts = expected_counts(*widths)
            if counts['total'] >= 2_000_000_000:
                raise ValueError('New budget exceeds 2B')
            name = task+'_'+arm.lower()
            output = root/'artifacts/compact'/FAMILY/name
            request = {k: side[k] for k in (
                'method', 'budget_scope', 'source_checkpoint', 'source_adapter_sha256',
                'source_metadata_sha256', 'calibration_samples', 'calibration_seed', 'calibration_cache')}
            request.update(keep_layers=None, ffn_width=widths[0], audio_ffn_width=widths[1], vision_ffn_width=widths[2])
            payload = {'id': 'export:'+name, 'task': task, 'output': str(output), 'request': request,
                       'source_identity': checkpoint_identity(source),
                       'source_base_manifest_sha256': sha256_file(Path(source_config['model'])/'compression.json'),
                       'generate_calibration': False, 'calibration_sha256': calibration_hash,
                       'expected_parameters': counts, 'expected_language_parameters': counts['language'],
                       'estimated_seconds': 300}
            export_id = FAMILY+'/'+payload['id']
            jobs.append(entry(export_id, 'prune_export', payload,
                              {'manifest': str(manifest), 'field': 'preparation_jobs', 'key_field': 'id', 'key': payload['id']},
                              [b_export_id], priority=priority))
            variants.append((arm, arm[0], str(output), export_id, counts, {}))
        for arm, architecture, model, export_id, counts, changes in variants:
            cfg = copy.deepcopy(reference)
            cfg.update(model=model, epochs=5, seed=42, train_limit=None, dev_limit=None,
                       micro_batch_size=2, gradient_accumulation_steps=2, gradient_checkpointing=True,
                       eval_modes=['logit', 'greedy'], selection_mode='logit')
            cfg.update(changes)
            deps = [export_id, 'prepare:'+task]
            if cfg['objective'] == 'sft_kd':
                deps.append(teacher_id)
            chain = training_chain(FAMILY, manifest, state['queue_root'], root, task, arm, cfg, source, deps)
            for job in chain:
                if job['kind'] in ('smoke_train', 'recovery_train'):
                    job['priority'] = priority
                    job['payload'].update(architecture=architecture, budget_scope='total', expected_parameters=counts,
                        initialization='initial verified whole-model base from pinned merged KD075 source; fresh rank16 decoder+encoder-FFN LoRA',
                        estimated_seconds=240 if job['kind']=='smoke_train' else (22000 if task=='mer6' else 10800))
            jobs.extend(chain)
    document = {'format_version': 1, 'status': 'waiting', 'phase': 'per_job_dependencies',
                'description': '18 approved runs: four B recovery controls and two nearby FFN budgets on each of three tasks',
                'jobs': [], 'smoke_jobs': [], 'preparation_jobs': [], 'checkpoint_selection_uses_test': False,
                'existing_source_selections_unchanged': True,
                'parameter_scope': 'complete deployed Thinker after LoRA merge; LM + audio + vision; no unused speech generator'}
    for job in jobs:
        document[job['location']['field']].append(copy.deepcopy(job['payload']))
    spec = {'schema_version': 1, 'jobs': jobs, 'documents': {str(manifest): document}}
    validate_submission(state, spec)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root', type=Path, default=ROOT/'artifacts/queue/suzy')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    state = read_json(args.queue_root/'state.json')
    saved = args.queue_root/'submissions'/f'{FAMILY}.json'
    if saved.exists():
        spec = read_json(saved)
        validate_submission(state, spec)
    else:
        spec = build_submission(state)
        atomic_json(saved, spec)
    if args.submit:
        path = args.queue_root/'inbox'/(fingerprint(spec)+'.json')
        atomic_json(path, spec)
        print('Submitted 18 training runs and 60 export/smoke/test stages: '+str(path))
    else:
        print('Validated 18-run submission: '+str(saved))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
