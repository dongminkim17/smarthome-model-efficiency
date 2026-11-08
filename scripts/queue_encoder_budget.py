#!/usr/bin/env python3
"""Keep the recovered 1.925B LM and submit 24 separate-budget encoder experiments."""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from smarthome_efficiency.queueing import atomic_json, read_json, fingerprint
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.encoder_pruning import encoder_parameters, validate_separate_budget
from launch_pruning_followup import checkpoint_identity
from project_queue_jobs import entry
from project_queue import validate_submission
from queue_ffn_expansion import training_chain

FAMILY = 'encoder_budget_20261021'
TASKS = ('mintrec2', 'mer6', 'mintrec20')
ARCHITECTURES = {
    'BALANCED': (4352, 3328, 2048),
    'AUDIO': (4352, 3584, 1920),
    'VISION': (4352, 3072, 2176),
}
# Six paired architecture comparisons, then two stability controls. All three
# tasks use the identical matrix; no selection from official test performance.
VARIANTS = (
    ('BALANCED_KD', 'BALANCED', 4, False, {}),
    ('BALANCED_FEATURE', 'BALANCED', 4, True, {}),
    ('AUDIO_KD', 'AUDIO', 2, False, {}),
    ('AUDIO_FEATURE', 'AUDIO', 2, True, {}),
    ('VISION_KD', 'VISION', 2, False, {}),
    ('VISION_FEATURE', 'VISION', 2, True, {}),
    ('BALANCED_FEATURE_LOWLR', 'BALANCED', 3, True, {'learning_rate': 3e-5}),
    ('BALANCED_FEATURE_SPLITLR', 'BALANCED', 3, True, {'encoder_learning_rate': 1e-5}),
)


def build_submission(state, root=ROOT):
    root = Path(root)
    manifest = root/'artifacts/full'/FAMILY/'workflow.json'
    by = {j['id']: j for j in state['jobs']}
    jobs = []
    for task in TASKS:
        ref = by['full_model_followups_20261020/train:'+task+'_a_feature_r16']
        cfg = ref['payload']['config']
        a_id = 'full_model_2b_20261020/export:'+task+'_a_lm1536_a1024_v1024'
        side = read_json(Path(cfg['model'])/'compression.json')
        teacher_id = 'calibrated_pruning_20261019/teacher:'+task
        feature_id = 'full_model_followups_20261020/encoder_cache:'+task
        source_job = by['recovery_controls_20261019/train:KD075_'+task]
        if (any(by[k]['status'] != 'completed' for k in (ref['id'], a_id, teacher_id, feature_id, source_job['id']))
                or cfg['objective'] != 'sft_kd' or cfg['distillation_alpha'] != .75
                or cfg['distillation_temperature'] != 2. or cfg['learning_rate'] != 1e-4
                or cfg['lora_target'] != 'decoder_encoder_ffn' or cfg['lora_rank'] != 16
                or cfg['lora_alpha'] != 32 or cfg.get('encoder_learning_rate') is not None
                or cfg['feature_audio_weight'] != .1 or cfg['feature_vision_weight'] != .1
                or side['parameters_after'] != expected_counts(1536, 1024, 1024)):
            raise ValueError('Expected completed original KD075 and A-feature reference for '+task)
        source = Path(side['source_checkpoint'])
        source_meta = read_json(source/'metadata.json')
        source_side = read_json(Path(source_meta['config']['model'])/'compression.json')
        before = expected_counts(4352, 5120, 3420)
        # The historical A export already pins the validation-selected KD075
        # source. Never reselect it based on any official test result.
        source_run = Path(source_job['payload']['run_dir']).resolve()
        if (source.resolve().parent.parent != source_run
                or source_side['parameters_after'] != before
                or source_meta['config']['objective'] != 'sft_kd'
                or source_meta['config']['distillation_alpha'] != .75
                or not source_side['verification']['passed']):
            raise ValueError('Encoder pruning must start from the pinned recovered FFN4352 LM')
        cache = read_json(Path(cfg['teacher_cache']))
        feature_path = Path(cfg['feature_teacher_cache'])/'cache.json'
        feature = read_json(feature_path)
        for spec in (cache['spec'], feature['spec']['teacher']):
            if (spec['split'] != 'train' or spec['test_set_used'] is not False
                    or spec['teacher_checkpoint'] != cfg['teacher_checkpoint']):
                raise ValueError('Original teacher caches must be training-only and match')
        if cache['status'] != 'completed' or feature['status'] != 'completed':
            raise ValueError('Both original teacher caches must be complete')
        calibration = Path(side['calibration_cache'])
        cal_hash = sha256_file(calibration)
        if cal_hash != side['calibration']['sha256']:
            raise ValueError('Pinned encoder calibration cache changed')
        exports = {}
        for architecture, widths in ARCHITECTURES.items():
            counts = expected_counts(*widths)
            validate_separate_budget(before, counts)
            name = task+'_'+architecture.lower()
            output = root/'artifacts/compact'/FAMILY/name
            request = {k: side[k] for k in ('source_checkpoint','source_adapter_sha256',
                'source_metadata_sha256','calibration_samples','calibration_seed','calibration_cache')}
            request.update(method='encoder_ffn_activation', budget_scope='separate', keep_layers=None,
                           ffn_width=widths[0], audio_ffn_width=widths[1], vision_ffn_width=widths[2])
            payload = {'id':'export:'+name, 'task':task, 'output':str(output), 'request':request,
                'source_identity':checkpoint_identity(source),
                'source_base_manifest_sha256':sha256_file(Path(source_meta['config']['model'])/'compression.json'),
                'generate_calibration':False, 'calibration_sha256':cal_hash,
                'expected_parameters':counts, 'expected_language_parameters':counts['language'],
                'expected_combined_encoder_parameters':encoder_parameters(counts), 'estimated_seconds':300}
            export = entry(FAMILY+'/'+payload['id'], 'prune_export', payload,
                {'manifest':str(manifest), 'field':'preparation_jobs', 'key_field':'id', 'key':payload['id']},
                [a_id], priority=26 if architecture=='BALANCED' else 21)
            jobs.append(export)
            exports[architecture] = (str(output), export['id'], counts)
        for arm, architecture, tier, features, changes in VARIANTS:
            model, export_id, counts = exports[architecture]
            config = copy.deepcopy(cfg)
            config.update(model=model, epochs=5, seed=42, train_limit=None, dev_limit=None,
                micro_batch_size=2, gradient_accumulation_steps=2, gradient_checkpointing=True,
                eval_modes=['logit','greedy'], selection_mode='logit')
            if not features:
                for key in ('feature_teacher_cache','feature_audio_weight','feature_vision_weight'):
                    config.pop(key, None)
            config.update(changes)
            deps = [export_id, teacher_id, 'prepare:'+task]
            if features:
                deps.append(feature_id)
            chain = training_chain(FAMILY, manifest, state['queue_root'], root, task, arm,
                                   config, str(source), deps)
            for job in chain:
                job['payload'].update(architecture=architecture, budget_scope='separate',
                    expected_parameters=counts,
                    expected_combined_encoder_parameters=encoder_parameters(counts))
                if job['kind'] in ('smoke_train','recovery_train'):
                    # Balanced pairs on all tasks first; binary first within a
                    # tier. A long MER run can overlap the short intent runs.
                    job['priority'] = 20+tier+(1 if task=='mintrec2' else 0)
                    job['payload'].update(
                        initialization='pinned recovered KD075 FFN4352 LM merged unchanged; encoder-only activation pruning; fresh rank16 decoder+encoder-FFN LoRA',
                        estimated_seconds=240 if job['kind']=='smoke_train' else (25200 if task=='mer6' else 12600))
                    if features:
                        job['payload']['expected_feature_cache_sha256'] = sha256_file(feature_path)
            jobs.extend(chain)
    document = {'format_version':1, 'status':'waiting', 'phase':'per_job_dependencies',
        'description':'24 five-epoch runs: three encoder allocations x class/feature KD, plus balanced low/split LR controls; all three tasks',
        'jobs':[], 'smoke_jobs':[], 'preparation_jobs':[], 'checkpoint_selection_uses_test':False,
        'existing_source_selections_unchanged':True, 'feature_cache_split':'train',
        'parameter_scope':'merged LM including embeddings/head <=2B; combined audio+vision including projection/merger <=1B; unused speech generator excluded',
        'parameter_limits':{'language':2_000_000_000,'combined_encoders':1_000_000_000},
        'overnight_target':'2026-10-22 12:00 Asia/Seoul; planning estimate, no running-job cutoff'}
    for job in jobs:
        document[job['location']['field']].append(copy.deepcopy(job['payload']))
    spec = {'schema_version':1, 'jobs':jobs, 'documents':{str(manifest):document}}
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
        spec = read_json(saved); validate_submission(state,spec)
    else:
        spec = build_submission(state); atomic_json(saved,spec)
    if args.submit:
        path = args.queue_root/'inbox'/(fingerprint(spec)+'.json')
        atomic_json(path,spec)
        print('Submitted 24 training runs, 9 exports, 48 stress/reload gates and 24 official tests: '+str(path))
    else:
        print('Validated 24-run submission: '+str(saved))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
