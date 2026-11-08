#!/usr/bin/env python3
"""Sixteen MER-only recovery runs on the fixed balanced architecture, Joy GPUs0-7."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from smarthome_efficiency.queueing import StateStore, atomic_json, exclusive_lock, fingerprint, now, read_json, refresh_readiness
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.training import config_fingerprint
from launch_pruning_followup import checkpoint_identity, pin_best
from project_queue import validate_submission
from project_queue_jobs import entry, validate_export
from queue_ffn_expansion import training_chain

FAMILY = 'mer_focus_20261103'
REFERENCES = {
    16: 'encoder_budget_20261021/train:mer6_balanced_feature_splitlr',
    64: 'recovery_refinements_20261022/train:mer6_balanced_r64',
}


def definitions():
    variants = []
    for rank in (16, 64):
        for lm_lr, encoder_lr, label in ((3e-5, 1e-5, 'LM3E5_ENC1E5'), (1e-5, 3e-6, 'LM1E5_ENC3E6')):
            variants.append((f'CONT_R{rank}_{label}', rank, True,
                             {'epochs': 3, 'learning_rate': lm_lr, 'encoder_learning_rate': encoder_lr}, 90))
    for rank in (16, 64):
        for kd in (.5, .25):
            for encoder_lr, label in ((1e-5, 'ENC1E5'), (3e-5, 'ENC3E5')):
                variants.append((f'FRESH_R{rank}_KD{int(kd*100):03d}_{label}', rank, False,
                                 {'distillation_alpha': kd, 'encoder_learning_rate': encoder_lr},
                                 80 if kd == .5 else 60))
    for rank in (16, 64):
        for audio, vision, label in ((.3, .03, 'AUDIO03_VISION003'), (.03, .3, 'AUDIO003_VISION03')):
            variants.append((f'FEATURE_R{rank}_{label}', rank, False,
                             {'distillation_alpha': .5, 'feature_audio_weight': audio,
                              'feature_vision_weight': vision}, 50))
    assert len(variants) == 16
    return variants


def build_submission(legacy, queue_root, root=ROOT):
    root, queue_root = Path(root), Path(queue_root)
    by_id = {job['id']: job for job in legacy['jobs']}
    refs = {rank: by_id[key] for rank, key in REFERENCES.items()}
    reference = refs[16]
    cfg = reference['payload']['config']
    export = by_id['encoder_budget_20261021/export:mer6_balanced']['payload']
    validate_export(export)
    counts = expected_counts(4352, 3328, 2048)
    sidecar = read_json(Path(cfg['model'])/'compression.json')
    if sidecar['parameters_after'] != counts or cfg['model'] != export['output']:
        raise ValueError('Fixed balanced export differs from the approved reference')
    if counts['language'] != 1_924_902_912 or any(counts[key] > 1_000_000_000 for key in ('audio_encoder', 'vision_encoder')):
        raise ValueError('Architecture violates confirmed year-two component limits')
    for rank, ref in refs.items():
        report = read_json(Path(ref['payload']['run_dir'])/'report.json')
        config = ref['payload']['config']
        if (ref['status'] != 'completed' or report['status'] != 'completed'
                or report['test_set_used'] is not False or report['config'] != config
                or config['model'] != cfg['model'] or config['lora_rank'] != rank
                or config['use_transcript'] or config['split_seed'] != 42):
            raise ValueError('Reference must be a completed, matching audiovisual balanced recovery')
    feature_path = Path(cfg['feature_teacher_cache'])/'cache.json'
    class_path = Path(cfg['teacher_cache'])
    features, teacher = read_json(feature_path), read_json(class_path)
    source = Path(cfg['teacher_checkpoint'])
    if (features['status'] != 'completed' or teacher['status'] != 'completed'
            or teacher['spec']['split'] != 'train' or teacher['spec']['test_set_used'] is not False
            or teacher['spec']['teacher_checkpoint'] != str(source)
            or features['spec']['teacher'] != teacher['spec']):
        raise ValueError('Original training-only class/feature caches must match')
    source_identity = checkpoint_identity(source)
    if (teacher['spec']['teacher_adapter_sha256'] != source_identity['adapter/adapter_model.safetensors']
            or teacher['spec']['teacher_metadata_sha256'] != source_identity['metadata.json']):
        raise ValueError('Pinned teacher cache provenance changed')
    manifest = root/'artifacts/full'/FAMILY/'workflow.json'
    prepare_id = 'prepare:mer6'
    prepare = entry(prepare_id, 'prepare',
                    {'id': 'prepare:mer6', 'kind': 'prepare', 'task': 'mer6',
                     'output': str(queue_root/'stress_cases/mer6.json'), 'config': copy.deepcopy(cfg)},
                    {'manifest': str(manifest), 'field': 'preparation_jobs', 'key_field': 'id', 'key': prepare_id},
                    resource='cpu', priority=100)
    jobs, summaries = [prepare], []
    for arm, rank, continuation, changes, priority in definitions():
        new = copy.deepcopy(cfg)
        new.update(lora_rank=rank, lora_alpha=2*rank, epochs=5, seed=42, use_transcript=False,
                   eval_modes=['logit'], selection_mode='logit', eval_every_fraction=.5,
                   train_limit=None, dev_limit=None, micro_batch_size=2, gradient_accumulation_steps=2,
                   gradient_checkpointing=True, learning_rate=1e-4, encoder_learning_rate=1e-5)
        new.update(changes)
        chain = training_chain(FAMILY, manifest, queue_root, root, 'mer6', arm, new, str(source), [prepare_id])
        parent = pin_best(refs[rank]['payload']['run_dir']) if continuation else None
        if continuation and parent is None:
            raise ValueError('Continuation parent has no validation-selected checkpoint')
        metadata = read_json(Path(parent)/'metadata.json') if parent else None
        for job in chain:
            payload = job['payload']
            payload.update(architecture='BALANCED', budget_scope='per_modality',
                           expected_parameters=counts, recovery_epoch_budget=new['epochs'],
                           reference_two_stage_train=REFERENCES[rank], group='continuation' if continuation else
                           'feature_balance' if arm.startswith('FEATURE') else 'class_kd')
            if job['kind'] in ('smoke_train', 'recovery_train'):
                job['priority'] = priority + (2 if job['kind'] == 'smoke_train' else 0)
                payload.update(expected_feature_cache_sha256=sha256_file(feature_path),
                               expected_teacher_cache_sha256=sha256_file(class_path),
                               initialization=('Validation-selected balanced adapter; new optimizer and cosine schedule'
                                               if continuation else 'Fresh adapter on the fixed sequential balanced export'),
                               estimated_seconds=240 if job['kind']=='smoke_train' else
                               read_json(Path(refs[rank]['payload']['run_dir'])/'report.json')['total_elapsed_seconds']*.9*new['epochs']/5)
                if parent:
                    payload.update(init_adapter=str(Path(parent)/'adapter'), parent_checkpoint=parent,
                                   parent_step=metadata['global_step'],
                                   parent_epoch=metadata['global_step']/675,
                                   init_checkpoint_identity=checkpoint_identity(parent))
            elif job['kind'] == 'smoke_test':
                job['priority'] = priority + 3
            else:
                payload['shortlist_job'] = FAMILY+'/shortlist:mer6'
                job['deps'].append(FAMILY+'/shortlist:mer6')
        jobs.extend(chain)
        summaries.append({'arm':arm,'rank':rank,'continuation':continuation,'epochs':new['epochs'],
                          'learning_rate':new['learning_rate'],'encoder_learning_rate':new['encoder_learning_rate'],
                          'class_kd':new['distillation_alpha'],'audio_feature_kd':new['feature_audio_weight'],
                          'vision_feature_kd':new['feature_vision_weight'],'parent_checkpoint':parent})
    selection = entry(FAMILY+'/shortlist:mer6', 'shortlist',
                      {'id':'shortlist:mer6','kind':'shortlist','task':'mer6','top_k':4,'dev_count':675,
                       'output':str(manifest.parent/'validation_shortlist.json')},
                      {'manifest':str(manifest),'field':'preparation_jobs','key_field':'id','key':'shortlist:mer6'},
                      [job['id'] for job in jobs if job['kind']=='recovery_train'], resource='inline',priority=100)
    jobs.append(selection)
    doc={'format_version':1,'status':'waiting','phase':'per_job_dependencies',
         'description':'16 approved MER-only runs, fixed balanced architecture; Joy GPUs0-7',
         'host':'joy','allowed_gpus':list(range(8)),'jobs':[],'smoke_jobs':[],'preparation_jobs':[],
         'checkpoint_selection_uses_test':False,'shortlist_uses_test':False,'official_test_top_k':4,
         'parameter_limits':{'language':2_000_000_000,'audio_encoder':1_000_000_000,'vision_encoder':1_000_000_000},
         'experiment_plan':summaries,
         'note':'No new pruning, architecture, split, media, transcripts or teacher. Logit-only validation; top4 complete validation winners receive official dual-mode tests. Continuations restart optimizer/schedule in new directories.'}
    for job in jobs:
        doc[job['location']['field']].append(copy.deepcopy(job['payload']))
    return {'schema_version':1,'jobs':jobs,'documents':{str(manifest):doc}}


def initialize(queue_root, legacy_root):
    queue_root=Path(queue_root)
    with exclusive_lock(queue_root/'scheduler.lock'):
        if (queue_root/'state.json').exists():
            state=StateStore(queue_root).load()
            if state['host']!='joy' or state['allowed_gpus']!=list(range(8)):
                raise ValueError('Existing queue has a different resource contract')
            if sum(j['kind']=='recovery_train' for j in state['jobs'])!=16:
                raise ValueError('Existing queue does not match this batch')
            return state
        state={'schema_version':1,'created_at':now(),'status':'ready','host':'joy','allowed_gpus':list(range(8)),
               'queue_root':str(queue_root.resolve()),'jobs':[],'documents':{},'source_entries':{},
               'selected_sources':{},'reporting_errors':[],
               'policy':{'max_retries':1,'memory_headroom_gib':1.,'cpu_workers':1,'A5_enabled':False}}
        spec=build_submission(StateStore(legacy_root).load(),queue_root)
        validate_submission(state,spec)
        atomic_json(queue_root/'submissions'/f'{FAMILY}.json',spec)
        state['jobs']=spec['jobs'];state['documents']=spec['documents']
        state['submissions']=[{'sha256':fingerprint(spec),'at':now()}]
        refresh_readiness(state['jobs'])
        StateStore(queue_root).save(state)
        for path,doc in spec['documents'].items(): atomic_json(path,doc)
        (queue_root/'dispatch.enabled').write_text('Authorized MER-only Joy dispatch '+now()+'\n')
        return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/joy')
    parser.add_argument('--source-queue',type=Path,default=ROOT/'artifacts/queue/suzy')
    parser.add_argument('--initialize',action='store_true')
    args=parser.parse_args()
    if args.initialize:
        state=initialize(args.queue_root,args.source_queue)
        print(json.dumps({'queue_root':str(args.queue_root),'jobs':len(state['jobs']),
                          'scientific_training_runs':sum(j['kind']=='recovery_train' for j in state['jobs']),
                          'host':state['host'],'gpus':state['allowed_gpus'],'official_tests':'validation top4'},indent=2))
    else:
        spec=build_submission(StateStore(args.source_queue).load(),args.queue_root)
        print(json.dumps({'validated_jobs':len(spec['jobs']),'training_runs':16,
                          'plans':next(iter(spec['documents'].values()))['experiment_plan']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
