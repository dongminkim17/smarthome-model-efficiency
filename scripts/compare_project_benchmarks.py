#!/usr/bin/env python3
"""Recompute full official results for a shared recipe and rank a fixed cohort."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from smarthome_efficiency.evaluation import classification_metrics
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.queueing import atomic_json,read_json,now

RECIPE_KEYS=('epochs','seed','split_seed','lora_rank','lora_alpha','lora_dropout','lora_target',
    'learning_rate','encoder_learning_rate','objective','distillation_alpha','distillation_temperature',
    'feature_audio_weight','feature_vision_weight','max_frames','max_pixels','warmup_ratio','scheduler',
    'weight_decay','max_grad_norm','eval_every_fraction','selection_mode','gradient_checkpointing')
EXPECTED_COUNTS={'language':1924902912,'audio_encoder':637676544,'vision_encoder':668684288,'other':0,'total':3231263744}


def compare(request):
    catalog=read_json(request['baseline_catalog'])
    if sha256_file(Path(request['baseline_catalog']))!=request['baseline_catalog_sha256']:raise ValueError('Comparison baseline cohort changed')
    reference=read_json(request['reference_training_report'])
    if sha256_file(Path(request['reference_training_report']))!=request['reference_training_report_sha256']:raise ValueError('Reference training report changed')
    rows=[]
    if {c['task'] for c in request['cases']}!={'mintrec2','mintrec20','mer6'} or len(request['cases'])!=3:raise ValueError('Exactly three task settings required')
    for case in request['cases']:
        task=case['task'];path=Path(case['test_report']);test=read_json(path);train=read_json(case['training_report']);cfg=train['config']
        if case.get('expected_test_report_sha256') and sha256_file(path)!=case['expected_test_report_sha256']:raise ValueError('Pinned reference official result changed')
        expected_n=834 if task=='mer6' else 445
        if (test.get('status')!='completed' or test.get('split')!='test' or test.get('test_set_used') is not True
                or test.get('partial_split') is not False or test.get('evaluated_count')!=expected_n
                or test.get('full_split_count')!=expected_n or test.get('task')!=task or cfg.get('task')!=task
                or train.get('status')!='completed' or train.get('test_set_used') is not False
                or Path(test['checkpoint']).resolve()!=(Path(case['training_report']).parent/'checkpoints/best').resolve()
                or test['config']!=cfg or train.get('world_size')!=1):
            raise ValueError('Official result is incomplete or not tied to its best-validation checkpoint')
        if any(cfg.get(k)!=reference['config'].get(k) for k in RECIPE_KEYS):raise ValueError('Shared recovery recipe differs: '+task)
        if (cfg['micro_batch_size']*cfg['gradient_accumulation_steps']!=4
                or cfg.get('distillation_disagreement_alpha') is not None or cfg.get('distillation_alpha_final') is not None
                or cfg['use_transcript'] is not (task!='mer6')):raise ValueError('Unexpected batch/objective/input setting')
        counts=read_json(Path(cfg['model'])/'compression.json')['parameters_after']
        if counts!=EXPECTED_COUNTS:raise ValueError('Shared architecture/component counts differ')
        predpath=path.parent/'predictions.jsonl';pred=[json.loads(x) for x in predpath.read_text().splitlines() if x]
        manifest=read_json(path.parent/'data_manifest.json')['records']
        if (len(pred)!=expected_n or len({x['id'] for x in pred})!=expected_n
                or {x['id']:x['true_label'] for x in pred}!={x['id']:x['label'] for x in manifest}):raise ValueError('Official predictions do not cover the exact test manifest')
        metrics={};metric='weighted_f1' if task=='mer6' else 'accuracy'
        for mode in ('logit','greedy'):
            m=classification_metrics([x['true_label'] for x in pred],[x[mode+'_predicted_label'] for x in pred],test['labels'])
            if abs(m[metric]-test[mode][metric])>1e-12:raise ValueError('Saved metric does not match predictions')
            rank=1+sum(row['score_percent']>100*m[metric] for row in catalog['tasks'][task]['external'])
            metrics[mode]={'metric':metric,'score':m[metric],'rank':rank,'cohort_size':len(catalog['tasks'][task]['external'])+1,'within_top5':rank<=5,'invalid_count':m['invalid_count']}
        rows.append({'task':task,'modes':metrics,'parameters':counts,'test_report':str(path),'test_report_sha256':sha256_file(path),'predictions_sha256':sha256_file(predpath),'checkpoint':test['checkpoint']})
    by={r['task']:r for r in rows};checks={'binary_accuracy_ge_92':by['mintrec2']['modes']['logit']['score']>=.92,
        'mintrec2_top5':by['mintrec2']['modes']['logit']['within_top5'],'mintrec20_top5_supplementary':by['mintrec20']['modes']['logit']['within_top5'],
        'mer_top5':by['mer6']['modes']['logit']['within_top5'],'year2_component_limits':True,'same_recovery_recipe_and_architecture':True}
    return {'status':'completed','generated_at_utc':now(),'test_set_used':True,'checkpoint_selection_uses_test':False,
        'scope':catalog['scope'],'results':rows,'checks':checks,
        'main_project_performance_targets_met':checks['binary_accuracy_ge_92'] and checks['mintrec2_top5'] and checks['mer_top5'],
        'requested_verification_targets_met':checks['binary_accuracy_ge_92'] and all(checks[k] for k in ('mintrec2_top5','mintrec20_top5_supplementary','mer_top5')),
        'note':'Ranks within the fixed published comparison cohort; task-specific weights/teachers and established AV(MER)/AVT(MIntRec) inputs. A failed performance target is a scientific result, not an execution failure.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    request=read_json(args.request);report=compare(request);report['request_sha256']=sha256_file(args.request);atomic_json(args.output,report);print(json.dumps(report,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
