"""Cross-task recipe identity, official metric recomputation and target checks."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from export_recovered_model import validate_source_config
from compare_project_benchmarks import compare,RECIPE_KEYS,EXPECTED_COUNTS
from queue_mintrec_shared_recipe import build_submission
from smarthome_efficiency.training import DEFAULTS,load_config
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.evaluation import classification_metrics
from project_queue import validate_submission
import yaml


def recipe():
    return dict(DEFAULTS,task='mer6',epochs=5,seed=42,split_seed=42,lora_rank=64,lora_alpha=128,lora_dropout=0.,lora_target='decoder_encoder_linear',learning_rate=1e-4,encoder_learning_rate=1e-5,objective='sft_kd',teacher_checkpoint='teacher',teacher_cache='classcache',distillation_alpha=.75,distillation_temperature=2.,feature_teacher_cache='features',feature_audio_weight=.1,feature_vision_weight=.1,max_frames=8,max_pixels=224**2,use_transcript=False,gradient_checkpointing=True,micro_batch_size=2,gradient_accumulation_steps=2,eval_modes=['logit'],selection_mode='logit',eval_every_fraction=.5)

class SharedRecipeTests(unittest.TestCase):
    def test_merge_supports_both_mintrec_transcript_protocols_and_original_mer(self):
        for task in ['mintrec2','mintrec20','mer6']:
            cfg={'task':task,'lora_target':'decoder_linear','use_transcript':task!='mer6','train_limit':None,'dev_limit':None}
            validate_source_config(cfg)
            for change in [{'use_transcript':task=='mer6'},{'lora_target':'decoder_encoder_linear'},{'train_limit':8},{'mer_partition':'nested_90_10'}]:
                with self.assertRaises(ValueError):validate_source_config(cfg|change)

    def comparison_fixture(self,root,binary_correct=410,fine_correct=350):
        cases=[]
        for task,n,classes in [('mintrec2',445,2),('mintrec20',445,20),('mer6',834,6)]:
            model=root/(task+'_model');model.mkdir();(model/'compression.json').write_text(json.dumps({'parameters_after':EXPECTED_COUNTS}))
            run=root/(task+'_train');(run/'checkpoints/step-1').mkdir(parents=True);(run/'checkpoints/best').symlink_to('step-1',target_is_directory=True)
            testdir=root/(task+'_test');testdir.mkdir();cfg=recipe()|{'task':task,'model':str(model),'use_transcript':task!='mer6'}
            labels=[str(i) for i in range(classes)];correct=binary_correct if task=='mintrec2' else fine_correct if task=='mintrec20' else 750
            pred=[{'id':str(i),'true_label':labels[i%classes],'logit_predicted_label':labels[(i+(i>=correct))%classes],'greedy_predicted_label':labels[(i+(i>=correct))%classes]} for i in range(n)]
            (testdir/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pred))
            (testdir/'data_manifest.json').write_text(json.dumps({'records':[{'id':p['id'],'label':p['true_label']} for p in pred]}))
            metrics=classification_metrics([p['true_label'] for p in pred],[p['logit_predicted_label'] for p in pred],labels)
            test={'status':'completed','split':'test','test_set_used':True,'partial_split':False,'evaluated_count':n,'full_split_count':n,'task':task,'config':cfg,'labels':labels,'checkpoint':str((run/'checkpoints/best').resolve()),'logit':metrics,'greedy':metrics}
            (run/'report.json').write_text(json.dumps({'status':'completed','test_set_used':False,'world_size':1,'config':cfg}));(testdir/'report.json').write_text(json.dumps(test))
            cases.append({'task':task,'test_report':str(testdir/'report.json'),'training_report':str(run/'report.json')})
        cat=root/'catalog.json';cat.write_text(json.dumps({'scope':'Fixed test cohort','tasks':{task:{'external':[{'score_percent':v} for v in [90.,88.,86.,80.,73.]]} for task in ['mintrec2','mintrec20','mer6']}}))
        ref=Path(cases[2]['training_report']);return {'cases':cases,'baseline_catalog':str(cat),'baseline_catalog_sha256':sha256_file(cat),'reference_training_report':str(ref),'reference_training_report_sha256':sha256_file(ref)}

    def test_recomputes_all_scores_ranks_and_hard_threshold(self):
        with tempfile.TemporaryDirectory() as f:
            request=self.comparison_fixture(Path(f));r=compare(request)
            self.assertTrue(r['main_project_performance_targets_met']);self.assertTrue(r['checks']['mintrec20_top5_supplementary']);self.assertTrue(r['requested_verification_targets_met'])
            binary=next(row for row in r['results'] if row['task']=='mintrec2')
            self.assertAlmostEqual(binary['modes']['logit']['score'],410/445);self.assertEqual(binary['modes']['logit']['rank'],1)
        with tempfile.TemporaryDirectory() as f:
            r=compare(self.comparison_fixture(Path(f),409));self.assertEqual(r['status'],'completed');self.assertFalse(r['main_project_performance_targets_met']);self.assertFalse(r['checks']['binary_accuracy_ge_92'])

    def test_supplementary_twenty_class_rank_is_checked_separately(self):
        with tempfile.TemporaryDirectory() as f:
            r=compare(self.comparison_fixture(Path(f),fine_correct=320))
            self.assertTrue(r['main_project_performance_targets_met'])
            self.assertFalse(r['requested_verification_targets_met'])
            self.assertFalse(r['checks']['mintrec20_top5_supplementary'])

    def test_partial_mismatched_recipe_missing_prediction_or_metric_fails_closed(self):
        for mutation in ['partial','recipe','missing','wrong_metric','test_source','budget','reference_pin']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as f:
                root=Path(f);request=self.comparison_fixture(root);case=request['cases'][0];tp=Path(case['test_report']);test=json.loads(tp.read_text());rp=Path(case['training_report']);train=json.loads(rp.read_text())
                if mutation=='partial':test['partial_split']=True
                elif mutation=='recipe':train['config']['lora_rank']=16
                elif mutation=='missing':p=tp.parent/'predictions.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-1]))
                elif mutation=='wrong_metric':test['logit']['accuracy']=.99
                elif mutation=='test_source':test['checkpoint']='other'
                elif mutation=='budget':p=root/'mintrec2_model/compression.json';p.write_text(json.dumps({'parameters_after':EXPECTED_COUNTS|{'language':2100000000}}))
                elif mutation=='reference_pin':request['cases'][2]['expected_test_report_sha256']='wrong'
                tp.write_text(json.dumps(test));rp.write_text(json.dumps(train))
                with self.assertRaises(ValueError):compare(request)

    def builder_fixture(self,root):
        ref=root/'mer';ref.mkdir();(ref/'report.json').write_text('{}');test=root/'mer_test.json';test.write_text('{}');jobs=[]
        for task in ['mintrec2','mintrec20']:
            teacher=root/(task+'_teacher');teacher.mkdir();cache=root/(task+'_class.json');features=root/'artifacts/cache/full_model_followups_20261020/encoder_features'/task;features.mkdir(parents=True)
            spec={'split':'train','test_set_used':False,'teacher_checkpoint':str(teacher),'teacher_metadata_sha256':'m','teacher_adapter_sha256':'a'};rows={str(i):{} for i in range(1334)}
            cache.write_text(json.dumps({'status':'completed','spec':spec,'rows':rows}));(features/'cache.json').write_text(json.dumps({'status':'completed','spec':{'teacher':spec},'rows':rows}))
            model=root/(task+'_base');model.mkdir();(model/'compression.json').write_text(json.dumps({'parameters_after':EXPECTED_COUNTS,'method':'ffn_activation','ffn_width':4352,'calibration_samples':128,'source_checkpoint':str(teacher),'encoder_tensors_unchanged':True,'verification':{'passed':True}}))
            cfg=recipe()|{'task':task,'model':str(model),'use_transcript':True,'lora_rank':16,'lora_alpha':32,'lora_target':'decoder_linear','teacher_checkpoint':str(teacher),'teacher_cache':str(cache)}
            run=root/(task+'_source');run.mkdir();(run/'report.json').write_text(json.dumps({'status':'completed','test_set_used':False,'train_count':1334,'dev_count':445,'world_size':1,'config':cfg,'teacher_cache_sha256':sha256_file(cache)}))
            jobs.append({'id':'recovery_controls_20261019/train:KD075_'+task,'status':'completed','payload':{'run_dir':str(run)}})
        catalog=root/'artifacts/analysis/baselines_20261103/report_comparison_tables.json';catalog.parent.mkdir(parents=True);catalog.write_text('{}')
        return {'jobs':jobs},ref,test

    def test_two_matched_runs_independent_official_tests_and_comparison_dependency(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);legacy,ref,test=self.builder_fixture(root)
            with patch('queue_mintrec_shared_recipe.reference_recipe',return_value=recipe()),patch('queue_mintrec_shared_recipe.pin_best',side_effect=lambda p:str(Path(p)/'checkpoints/step-1')),patch('queue_mintrec_shared_recipe.checkpoint_identity',return_value={'metadata.json':'m','adapter/adapter_model.safetensors':'a'}):spec=build_submission(legacy,root/'queue',root,ref,test)
            validate_submission({'jobs':[],'documents':{}},spec);self.assertEqual(len(spec['jobs']),13)
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train'];self.assertEqual({j['task'] for j in trains},{'mintrec2','mintrec20'})
            for j in trains:
                cfg=j['payload']['config'];self.assertTrue(all(cfg.get(k)==recipe().get(k) for k in RECIPE_KEYS));self.assertTrue(cfg['use_transcript']);self.assertEqual(cfg['micro_batch_size']*cfg['gradient_accumulation_steps'],4)
                p=root/'cfg.yaml';p.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(p),cfg)
            tests=[j for j in spec['jobs'] if j['kind']=='recovery_test'];self.assertEqual(len(tests),2);self.assertTrue(all(len(j['deps'])==1 and j['payload']['limit'] is None and j['payload']['modes']==['logit','greedy'] for j in tests))
            comparison=next(j for j in spec['jobs'] if j['kind']=='benchmark_comparison');self.assertEqual(set(comparison['deps']),{j['id'] for j in tests})
            self.assertFalse(any(j['kind']=='shortlist' for j in spec['jobs']))

if __name__=='__main__':unittest.main()
