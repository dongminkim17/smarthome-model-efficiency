#!/usr/bin/env python3
"""Submit three approved whole-Thinker FFN budgets x three tasks, additively."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from smarthome_efficiency.queueing import atomic_json, read_json, fingerprint
from smarthome_efficiency.training import config_fingerprint
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.multimodal_pruning import BUDGETS, expected_counts
from launch_pruning_followup import checkpoint_identity, pin_best
from project_queue_jobs import entry
from queue_ffn_expansion import training_chain
from project_queue import validate_submission

FAMILY = 'full_model_2b_20261020'
TASKS = ('mintrec2', 'mer6', 'mintrec20')


def build_submission(state, root=ROOT):
    root = Path(root)
    manifest = root/'artifacts/full'/FAMILY/'workflow.json'
    by_id = {j['id']: j for j in state['jobs']}
    jobs = []
    for task in TASKS:
        source_id = 'recovery_controls_20261019/train:KD075_' + task
        source_job = by_id[source_id]
        if source_job['status'] != 'completed':
            raise ValueError('KD075 source has not completed: ' + task)
        run = Path(source_job['payload']['run_dir'])
        report = read_json(run/'report.json')
        source = pin_best(run)
        metadata = read_json(Path(source)/'metadata.json')
        cfg = metadata['config']
        if (report['status'] != 'completed' or report.get('test_set_used') is not False
                or cfg != report['config'] or cfg['task'] != task or cfg['objective'] != 'sft_kd'
                or cfg['distillation_alpha'] != .75 or cfg['distillation_temperature'] != 2.
                or cfg['selection_mode'] != 'logit' or cfg['lora_rank'] != 16
                or cfg['lora_target'] != 'decoder_linear'):
            raise ValueError('Expected pinned KD075 validation-selected source: ' + task)
        sidecar_path = Path(cfg['model'])/'compression.json'
        sidecar = read_json(sidecar_path)
        if (sidecar['method'] != 'ffn_activation' or sidecar['ffn_width'] != 4352
                or not sidecar['verification']['passed']
                or sidecar['parameters_after']['total'] != 3_231_263_744
                or sidecar['parameters_after']['language'] != 1_924_902_912
                or metadata.get('base_model_manifest_sha256') != sha256_file(sidecar_path)):
            raise ValueError('Unexpected compact KD075 base: ' + task)
        cache_path = Path(cfg['teacher_cache']); cache = read_json(cache_path)
        teacher_id = 'calibrated_pruning_20261019/teacher:' + task
        if (by_id[teacher_id]['status'] != 'completed' or cache['status'] != 'completed'
                or cache['spec']['split'] != 'train' or cache['spec']['test_set_used'] is not False
                or cache['spec']['teacher_checkpoint'] != cfg['teacher_checkpoint']):
            raise ValueError('Original training-only teacher cache is unavailable: ' + task)
        first_export_id = None
        for arm, widths in BUDGETS.items():
            name = task + '_' + arm.lower()
            export_id = FAMILY + '/export:' + name
            model = root/'artifacts/compact'/FAMILY/name
            counts = expected_counts(*widths)
            if counts['total'] >= 2_000_000_000:
                raise ValueError('Requested architecture exceeds the total parameter budget')
            request = {'method': 'multimodal_ffn_activation', 'budget_scope': 'total',
                       'source_checkpoint': source, 'keep_layers': None, 'ffn_width': widths[0],
                       'audio_ffn_width': widths[1], 'vision_ffn_width': widths[2],
                       'source_adapter_sha256': sha256_file(Path(source)/'adapter/adapter_model.safetensors'),
                       'source_metadata_sha256': sha256_file(Path(source)/'metadata.json'),
                       'calibration_samples': 128, 'calibration_seed': 42,
                       'calibration_cache': str(root/'artifacts/compact'/FAMILY/'calibration'/f'{task}_kd075_n128.json')}
            payload = {'id': 'export:'+name, 'task': task, 'output': str(model), 'request': request,
                       'source_identity': checkpoint_identity(source), 'source_base_manifest_sha256': sha256_file(sidecar_path),
                       'generate_calibration': True, 'calibration_sha256': None,
                       'expected_language_parameters': counts['language'], 'expected_parameters': counts,
                       'estimated_seconds': 1800 if first_export_id is None else 300}
            # Collect once per task, then reuse that exact source/calibration for
            # both alternative budgets without occupying GPUs on a cache lock.
            dependencies = [source_id] + ([first_export_id] if first_export_id else [])
            jobs.append(entry(export_id, 'prune_export', payload,
                              {'manifest': str(manifest), 'field': 'preparation_jobs', 'key_field': 'id', 'key': payload['id']},
                              dependencies, priority=3))
            if first_export_id is None:
                first_export_id = export_id
            recovery_cfg = copy.deepcopy(cfg)
            recovery_cfg.update(model=str(model), epochs=5, seed=42, lora_rank=16, lora_alpha=32, lora_dropout=0.,
                                lora_target='decoder_encoder_ffn', learning_rate=1e-4,
                                micro_batch_size=2, gradient_accumulation_steps=2, gradient_checkpointing=True,
                                train_limit=None, dev_limit=None, objective='sft_kd',
                                distillation_alpha=.75, distillation_temperature=2.,
                                eval_modes=['logit','greedy'], selection_mode='logit')
            chain = training_chain(FAMILY, manifest, state['queue_root'], root, task, arm, recovery_cfg, source,
                                   [export_id, 'prepare:'+task, teacher_id])
            for job in chain:
                if job['kind'] in ('smoke_train','recovery_train'):
                    job['priority'] = 3
                    job['payload'].update(
                        initialization='merge best-validation task KD075 adapter, prune LM/audio/vision FFNs, fresh rank-16 decoder + encoder-FFN LoRA',
                        expected_parameters=counts, budget_scope='total',
                        estimated_seconds=(180 if job['kind']=='smoke_train' else (18000 if task=='mer6' else 9000)))
            jobs.extend(chain)
    document = {'format_version':1, 'status':'waiting', 'phase':'per_job_dependencies',
                'description':'Three total-Thinker <2B FFN budgets x three tasks; five-epoch KD075 + encoder-FFN LoRA recovery',
                'jobs':[], 'smoke_jobs':[], 'preparation_jobs':[], 'checkpoint_selection_uses_test':False,
                'existing_source_selections_unchanged':True,
                'parameter_scope':'all deployed Thinker learned parameters, including LM head/embeddings and both encoders/projections; merged LoRA; no unused speech generator'}
    for job in jobs:
        document[job['location']['field']].append(copy.deepcopy(job['payload']))
    spec = {'schema_version':1, 'jobs':jobs, 'documents':{str(manifest):document}}
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
        spec = read_json(saved); validate_submission(state, spec)
    else:
        spec = build_submission(state); atomic_json(saved, spec)
    if args.submit:
        path = args.queue_root/'inbox'/(fingerprint(spec)+'.json')
        atomic_json(path,spec)
        print('Submitted 9 training experiments, 9 verified exports and 27 smoke/test stages: '+str(path))
    else:
        print('Validated 9-run whole-model submission: '+str(saved))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
