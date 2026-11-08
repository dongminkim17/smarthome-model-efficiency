#!/usr/bin/env python3
"""Direct joint pruning: matched architecture/recipe, independent 5/10-epoch recovery."""
from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from smarthome_efficiency.queueing import atomic_json,read_json,fingerprint
from smarthome_efficiency.pruning import sha256_file,processor_hashes
from smarthome_efficiency.multimodal_pruning import expected_counts
from smarthome_efficiency.joint_pruning import validate_joint_budget
from launch_pruning_followup import checkpoint_identity
from project_queue_jobs import entry
from project_queue import validate_submission
from queue_ffn_expansion import training_chain

FAMILY='joint_pruning_20261022'
TASKS=('mintrec2','mer6','mintrec20')
EPOCHS=(5,10)
WIDTHS=(4352,3328,2048)


def build_submission(state,root=ROOT):
    root=Path(root);manifest=root/'artifacts/full'/FAMILY/'workflow.json'
    by={j['id']:j for j in state['jobs']};jobs=[]
    counts=expected_counts(*WIDTHS);before=expected_counts(11008,5120,3420)
    validate_joint_budget(before,counts)
    for task in TASKS:
        reference_id='encoder_budget_20261021/train:'+task+'_balanced_feature_splitlr'
        reference=by[reference_id];cfg=reference['payload']['config']
        reference_test=by['encoder_budget_20261021/test:'+task+'_balanced_feature_splitlr']
        teacher_id='calibrated_pruning_20261019/teacher:'+task
        feature_id='full_model_followups_20261020/encoder_cache:'+task
        if any(by[k]['status']!='completed' for k in (reference_id,reference_test['id'],teacher_id,feature_id)):
            raise ValueError('Reference and original caches must be completed: '+task)
        if (cfg['epochs']!=5 or cfg['lora_target']!='decoder_encoder_ffn' or cfg['lora_rank']!=16
                or cfg['lora_alpha']!=32 or cfg['learning_rate']!=1e-4 or cfg['encoder_learning_rate']!=1e-5
                or cfg['objective']!='sft_kd' or cfg['distillation_alpha']!=.75
                or cfg['distillation_temperature']!=2. or cfg['feature_audio_weight']!=.1
                or cfg['feature_vision_weight']!=.1):
            raise ValueError('Expected fixed successful two-stage recovery recipe')
        reference_side=read_json(Path(cfg['model'])/'compression.json')
        if reference_side['parameters_after']!=counts or not reference_side['verification']['passed']:
            raise ValueError('Two-stage architecture differs from direct joint target')
        source=Path(cfg['teacher_checkpoint']);meta=read_json(source/'metadata.json');base=Path(meta['config']['model'])
        config=read_json(base/'config.json')
        config=config.get('thinker_config',config)
        if ((base/'compression.json').exists() or meta['config']['task']!=task
                or meta['config']['lora_target']!='qv' or meta['config']['epochs']!=5
                or meta['config'].get('objective','sft')!='sft'
                or config['text_config']['intermediate_size']!=11008
                or config['text_config']['num_hidden_layers']!=36
                or config['audio_config']['encoder_ffn_dim']!=5120
                or config['audio_config']['encoder_layers']!=32
                or config['vision_config']['intermediate_size']!=3420
                or config['vision_config']['depth']!=32):
            raise ValueError('Joint pruning must begin at the original unpruned task SFT teacher')
        if processor_hashes(base) != reference_side['processor_files_sha256']:
            raise ValueError('Original processor differs from the matched two-stage reference')
        cache=read_json(Path(cfg['teacher_cache']));feature_path=Path(cfg['feature_teacher_cache'])/'cache.json';features=read_json(feature_path)
        for value,spec in ((cache,cache['spec']),(features,features['spec']['teacher'])):
            if (value['status']!='completed' or spec['split']!='train' or spec['test_set_used'] is not False
                    or spec['teacher_checkpoint']!=str(source)
                    or spec['teacher_metadata_sha256']!=sha256_file(source/'metadata.json')
                    or spec['teacher_adapter_sha256']!=sha256_file(source/'adapter/adapter_model.safetensors')
                    or spec['teacher_base_config_sha256']!=sha256_file(base/'config.json')):
                raise ValueError('Teacher targets must remain pinned to original training-only caches')
        output=root/'artifacts/compact'/FAMILY/task
        request={'method':'multimodal_ffn_activation','budget_scope':'separate','source_checkpoint':str(source),
            'source_adapter_sha256':sha256_file(source/'adapter/adapter_model.safetensors'),
            'source_metadata_sha256':sha256_file(source/'metadata.json'),'keep_layers':None,
            'ffn_width':WIDTHS[0],'audio_ffn_width':WIDTHS[1],'vision_ffn_width':WIDTHS[2],
            'calibration_samples':128,'calibration_seed':42,
            'calibration_cache':str(root/'artifacts/compact'/FAMILY/'calibration'/f'{task}_original_teacher_n128.json')}
        p={'id':'export:'+task,'task':task,'output':str(output),'request':request,
            'source_identity':checkpoint_identity(source),'source_base_snapshot':str(base.resolve()),
            'source_base_config_sha256':sha256_file(base/'config.json'),'source_processor_files_sha256':processor_hashes(base),
            'generate_calibration':True,'calibration_sha256':None,
            'expected_language_parameters':counts['language'],'expected_parameters':counts,
            'estimated_seconds':900}
        export=entry(FAMILY+'/'+p['id'],'prune_export',p,
            {'manifest':str(manifest),'field':'preparation_jobs','key_field':'id','key':p['id']},[teacher_id],priority=30)
        jobs.append(export)
        report=read_json(Path(reference['payload']['run_dir'])/'report.json')
        if report['status']!='completed' or report['config']!=cfg or report.get('test_set_used') is not False:
            raise ValueError('Reference recovery report is not a complete validation-only training record')
        for epochs in EPOCHS:
            arm=f'JOINT_E{epochs:02d}';newcfg=copy.deepcopy(cfg);newcfg.update(model=str(output),epochs=epochs)
            chain=training_chain(FAMILY,manifest,state['queue_root'],root,task,arm,newcfg,str(source),
                [export['id'],teacher_id,feature_id,'prepare:'+task])
            for job in chain:
                job['payload'].update(architecture='BALANCED',budget_scope='separate',expected_parameters=counts,
                    comparison='joint_vs_two_stage',reference_two_stage_train=reference_id,
                    reference_two_stage_test_report=reference_test['result']['report'],recovery_epoch_budget=epochs)
                if job['kind'] in ('smoke_train','recovery_train'):
                    job['priority']=20
                    job['payload'].update(expected_feature_cache_sha256=sha256_file(feature_path),
                        initialization='original task-SFT teacher adapter merged; LM/audio/vision pruned together; fresh rank16 decoder+encoder-FFN LoRA; no intermediate recovery',
                        estimated_seconds=240 if job['kind']=='smoke_train' else report['total_elapsed_seconds']*epochs/5)
            jobs.extend(chain)
    doc={'format_version':1,'status':'waiting','phase':'per_job_dependencies',
        'description':'Six matched direct joint-pruning runs: independent five/ten-epoch recoveries on all three tasks',
        'jobs':[],'smoke_jobs':[],'preparation_jobs':[],'checkpoint_selection_uses_test':False,
        'existing_source_selections_unchanged':True,'parameter_limits':{'language':2_000_000_000,'combined_encoders':1_000_000_000},
        'parameter_scope':'same merged balanced Thinker as two-stage reference: LM1924902912, combined encoders990823680',
        'comparison_note':'Ten epochs match the sum of two five-epoch recovery stages; this is not exact FLOP/optimizer-trajectory matching. Five/ten arms are independent fresh adapters, with cosine schedules over their own horizons.'}
    for job in jobs:doc[job['location']['field']].append(copy.deepcopy(job['payload']))
    spec={'schema_version':1,'jobs':jobs,'documents':{str(manifest):doc}}
    validate_submission(state,spec);return spec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue-root',type=Path,default=ROOT/'artifacts/queue/suzy');p.add_argument('--submit',action='store_true');args=p.parse_args()
    state=read_json(args.queue_root/'state.json');saved=args.queue_root/'submissions'/f'{FAMILY}.json'
    if saved.exists():spec=read_json(saved);validate_submission(state,spec)
    else:spec=build_submission(state);atomic_json(saved,spec)
    if args.submit:
        path=args.queue_root/'inbox'/(fingerprint(spec)+'.json');atomic_json(path,spec)
        print('Submitted six joint-pruning recovery runs, three exports, twelve smoke/reload gates and six official tests: '+str(path))
    else:print('Validated six-run joint-pruning submission: '+str(saved))
    return 0

if __name__=='__main__':raise SystemExit(main())
