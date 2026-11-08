#!/usr/bin/env python3
"""Submit the approved MIntRec-2 FFN-4608 SFT/KD extension, without editing live state."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from smarthome_efficiency.queueing import atomic_json, read_json, fingerprint
from smarthome_efficiency.training import config_fingerprint
from smarthome_efficiency.pruning import sha256_file
from launch_pruning_followup import checkpoint_identity
from project_queue_jobs import entry
from project_queue import validate_submission

FAMILY = 'ffn_width_4608_20261019'


def build_submission(state, root=ROOT):
    root = Path(root)
    reference = next(j for j in state['jobs'] if j['id'] == 'calibrated_pruning_20261019/sft:mintrec2_ffn_activation')['payload']
    reference_sidecar = read_json(Path(reference['compact_model'])/'compression.json')
    if (reference_sidecar['ffn_width'] != 4352 or reference_sidecar['method'] != 'ffn_activation'
            or reference_sidecar['task'] != 'mintrec2' or not reference_sidecar['verification']['passed']):
        raise ValueError('Expected the verified 4352-channel MIntRec-2 reference export')
    teacher_id = 'calibrated_pruning_20261019/teacher:mintrec2'
    teacher = next(j for j in state['jobs'] if j['id'] == teacher_id)
    if teacher['status'] != 'completed':
        raise ValueError('The existing original-teacher cache must be completed')
    output = root/'artifacts/full'/FAMILY
    smoke = root/'artifacts/smoke'/FAMILY
    model = root/'artifacts/compact'/FAMILY/'mintrec2_ffn_activation_4608'
    manifest = str(output/'workflow.json')
    jobs = []
    def location(field, key):
        return {'manifest': manifest, 'field': field, 'key_field': 'id', 'key': key}
    export_id = FAMILY+'/export:mintrec2_ffn_activation_4608'
    request = {k: reference_sidecar[k] for k in ('method', 'source_checkpoint', 'source_adapter_sha256',
               'source_metadata_sha256', 'calibration_samples', 'calibration_seed', 'calibration_cache')}
    request.update(ffn_width=4608, keep_layers=None)
    expected_count = reference_sidecar['parameters_after']['language'] + 36*3*2048*(4608-4352)
    if expected_count != 1_981_526_016 or expected_count >= 2_000_000_000:
        raise ValueError('Unexpected FFN-4608 language parameter count')
    export = {'id': 'export:mintrec2_ffn_activation_4608', 'task': 'mintrec2', 'output': str(model),
              'request': request, 'source_identity': checkpoint_identity(request['source_checkpoint']),
              'calibration_sha256': sha256_file(Path(request['calibration_cache'])),
              'expected_language_parameters': expected_count}
    jobs.append(entry(export_id, 'prune_export', export, location('preparation_jobs', export['id']), priority=30))
    for arm in ('SFT', 'KD'):
        name = 'mintrec2_ffn4608_'+arm.lower()
        cfg = copy.deepcopy(reference['config'])
        cfg.update(model=str(model), epochs=5, seed=42, micro_batch_size=2, gradient_accumulation_steps=2,
                   lora_rank=16, lora_alpha=32, lora_target='decoder_linear', learning_rate=1e-4,
                   train_limit=None, dev_limit=None, objective='sft', teacher_checkpoint=None, teacher_cache=None,
                   eval_modes=['logit','greedy'], selection_mode='logit')
        if arm == 'KD':
            cfg.update(objective='sft_kd', teacher_checkpoint=request['source_checkpoint'],
                       teacher_cache=str(Path(teacher['payload']['output'])/'cache.json'),
                       distillation_alpha=.5, distillation_temperature=2.)
            cache = read_json(cfg['teacher_cache'])
            if (cache['status'] != 'completed' or cache['spec']['split'] != 'train'
                    or cache['spec']['test_set_used'] is not False
                    or cache['spec']['teacher_checkpoint'] != request['source_checkpoint']):
                raise ValueError('Teacher cache does not match the fixed original teacher')
        train = {'id':'train:'+name, 'name':name, 'task':'mintrec2', 'arm':arm, 'kind':'train',
                 'status':'waiting', 'output':str(output/'train'/name), 'config':cfg,
                 'config_sha256':config_fingerprint(cfg), 'init_adapter':None, 'parent_checkpoint':None,
                 'source_name':'mintrec2_ffn_activation_4608', 'source_checkpoint':request['source_checkpoint'],
                 'compact_model':str(model), 'training_epochs':5,
                 'initialization':'fresh rank-16 adapter on immutable FFN-4608 export; original task adapter merged before pruning'}
        probe = copy.deepcopy(train)
        probe.update(output=str(smoke/'train'/name), memory_cases=str(Path(state['queue_root'])/'stress_cases/mintrec2.json'))
        probe['config'].update(epochs=2, train_limit=24, dev_limit=20)
        probe['config_sha256'] = config_fingerprint(probe['config'])
        probe['training_epochs'] = 2
        probe_id = FAMILY+'/smoke_train:'+name
        dependencies = [export_id, 'prepare:mintrec2'] + ([teacher_id] if arm == 'KD' else [])
        jobs.append(entry(probe_id, 'smoke_train', probe, location('smoke_jobs', probe['id']), dependencies, priority=10))
        for is_smoke in (True, False):
            key = FAMILY+('/smoke_test:' if is_smoke else '/test:')+name
            parent = probe_id if is_smoke else FAMILY+'/train:'+name
            payload = {'id':'test:'+name, 'name':name, 'task':'mintrec2', 'arm':arm, 'kind':'test',
                       'status':'waiting', 'output':str((smoke if is_smoke else output)/'test'/name),
                       'split':'dev' if is_smoke else 'test', 'limit':6 if is_smoke else None,
                       'modes':['logit','greedy']}
            jobs.append(entry(key, 'smoke_test' if is_smoke else 'recovery_test', payload,
                              location('smoke_jobs' if is_smoke else 'jobs', payload['id']), [parent],
                              priority=45 if is_smoke else 35))
        jobs.append(entry(FAMILY+'/train:'+name, 'recovery_train', train, location('jobs',train['id']),
                          [FAMILY+'/smoke_test:'+name], priority=5))
    document = {'format_version':1, 'status':'waiting', 'phase':'per_job_dependencies',
                'description':'Two approved MIntRec-2 FFN-4608 rank-16 SFT/KD runs; fixed original teacher',
                'jobs':[], 'smoke_jobs':[], 'preparation_jobs':[], 'expected_language_parameters':expected_count,
                'checkpoint_selection_uses_test':False, 'existing_source_selections_unchanged':True}
    for job in jobs:
        document[job['location']['field']].append(copy.deepcopy(job['payload']))
    spec = {'schema_version':1, 'jobs':jobs, 'documents':{manifest:document}}
    validate_submission(state, spec)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/suzy')
    parser.add_argument('--submit',action='store_true')
    args = parser.parse_args()
    state = read_json(args.queue_root/'state.json')
    saved = args.queue_root/'submissions'/f'{FAMILY}.json'
    if saved.exists():
        spec = read_json(saved)
        validate_submission(state,spec)
    else:
        spec = build_submission(state)
        atomic_json(saved,spec)
    if args.submit:
        path = args.queue_root/'inbox'/(fingerprint(spec)+'.json')
        atomic_json(path,spec)
        print('Submitted two five-epoch training experiments plus export/smoke/test dependencies: '+str(path))
    else:
        print('Validated submission: '+str(saved))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
