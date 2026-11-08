#!/usr/bin/env python3
"""Prioritize 24 approved A + feature-KD follow-ups across all three tasks."""
from __future__ import annotations
import argparse,copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import atomic_json,read_json,fingerprint
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.multimodal_pruning import expected_counts
from project_queue_jobs import entry
from project_queue import validate_submission
from queue_ffn_expansion import training_chain

FAMILY='full_model_a_feature_20261021'
TASKS=('mintrec2','mer6','mintrec20')
# Priority tier: learning-rate / feature-strength comparisons, modality ablations,
# then rank / class-KD controls. Binary intent precedes all other new task runs.
VARIANTS=(
    ('FEATURE_WEAK',3,{'feature_audio_weight':.03,'feature_vision_weight':.03}),
    ('FEATURE_STRONG',3,{'feature_audio_weight':.3,'feature_vision_weight':.3}),
    ('FEATURE_LOWLR',3,{'learning_rate':3e-5}),
    ('FEATURE_SPLITLR',3,{'encoder_learning_rate':1e-5}),
    ('FEATURE_AUDIO',2,{'feature_audio_weight':.1,'feature_vision_weight':0.}),
    ('FEATURE_VISION',2,{'feature_audio_weight':0.,'feature_vision_weight':.1}),
    ('FEATURE_KD050',1,{'distillation_alpha':.5}),
    ('FEATURE_R32',1,{'lora_rank':32,'lora_alpha':64}),
)


def build_submission(state,root=ROOT):
    root=Path(root);manifest=root/'artifacts/full'/FAMILY/'workflow.json';by={j['id']:j for j in state['jobs']};jobs=[]
    for task in TASKS:
        reference=by['full_model_followups_20261020/train:'+task+'_a_feature_r16']
        if reference['status']!='completed':raise ValueError('A feature reference has not completed: '+task)
        cfg=reference['payload']['config'];base=Path(cfg['model']);side=read_json(base/'compression.json')
        expected=expected_counts(1536,1024,1024)
        if (side['method']!='multimodal_ffn_activation' or side['budget_scope']!='total'
                or side['parameters_after']!=expected or not side['verification']['passed']
                or cfg['lora_target']!='decoder_encoder_ffn' or cfg['lora_rank']!=16 or cfg['lora_alpha']!=32
                or cfg['learning_rate']!=1e-4 or cfg['objective']!='sft_kd' or cfg['distillation_alpha']!=.75
                or cfg['distillation_temperature']!=2. or cfg['feature_audio_weight']!=.1 or cfg['feature_vision_weight']!=.1
                or cfg.get('encoder_learning_rate') is not None):raise ValueError('Unexpected A feature reference recipe')
        feature=Path(cfg['feature_teacher_cache'])/'cache.json';cached=read_json(feature)
        teacher_id='calibrated_pruning_20261019/teacher:'+task
        feature_id='full_model_followups_20261020/encoder_cache:'+task
        a_id='full_model_2b_20261020/export:'+task+'_a_lm1536_a1024_v1024'
        if (any(by[key]['status']!='completed' for key in [teacher_id,feature_id,a_id])
                or cached['status']!='completed' or cached['spec']['teacher']['split']!='train'
                or cached['spec']['teacher']['test_set_used'] is not False
                or cached['spec']['teacher']['teacher_checkpoint']!=cfg['teacher_checkpoint']):
            raise ValueError('Completed training-only caches and A base required')
        source=side['source_checkpoint'];feature_hash=sha256_file(feature)
        for arm,tier,changes in VARIANTS:
            config=copy.deepcopy(cfg);config.update(epochs=5,seed=42,train_limit=None,dev_limit=None,
                micro_batch_size=2,gradient_accumulation_steps=2,gradient_checkpointing=True,
                eval_modes=['logit','greedy'],selection_mode='logit');config.update(changes)
            priority=20+tier if task=='mintrec2' else 10+tier
            chain=training_chain(FAMILY,manifest,state['queue_root'],root,task,arm,config,source,
                                 [a_id,teacher_id,feature_id,'prepare:'+task])
            for job in chain:
                if job['kind'] in ('smoke_train','recovery_train'):
                    job['priority']=priority
                    job['payload'].update(architecture='A',budget_scope='total',expected_parameters=expected,
                        expected_feature_cache_sha256=feature_hash,
                        initialization='same initial verified A base; fresh decoder+encoder-FFN LoRA; fixed original class and encoder-feature teachers',
                        estimated_seconds=240 if job['kind']=='smoke_train' else (24000 if task=='mer6' else 10800))
            jobs.extend(chain)
    document={'format_version':1,'status':'waiting','phase':'per_job_dependencies',
              'description':'24 approved A+feature-KD runs: eight one-variable controls on each of three tasks; binary intent and feature/LR controls prioritized',
              'jobs':[],'smoke_jobs':[],'preparation_jobs':[],'checkpoint_selection_uses_test':False,
              'existing_source_selections_unchanged':True,'feature_cache_split':'train',
              'parameter_scope':'complete deployed Thinker after LoRA merge; 1,978,160,384 parameters; speech generation excluded'}
    for job in jobs:document[job['location']['field']].append(copy.deepcopy(job['payload']))
    spec={'schema_version':1,'jobs':jobs,'documents':{str(manifest):document}}
    validate_submission(state,spec);return spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/suzy')
    parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    state=read_json(args.queue_root/'state.json');saved=args.queue_root/'submissions'/f'{FAMILY}.json'
    if saved.exists():spec=read_json(saved);validate_submission(state,spec)
    else:spec=build_submission(state);atomic_json(saved,spec)
    if args.submit:
        path=args.queue_root/'inbox'/(fingerprint(spec)+'.json');atomic_json(path,spec)
        print('Submitted 24 training runs and 72 stress/reload/official-test stages: '+str(path))
    else:print('Validated 24-run prioritized submission: '+str(saved))
    return 0


if __name__=='__main__':raise SystemExit(main())
