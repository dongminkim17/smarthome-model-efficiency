"""Reporting must distinguish validation, verified tests and queued work."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from update_experiment_index import normalize_run, collect, tail_progress


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class ExperimentIndexTests(unittest.TestCase):
    def fixture(self, root):
        run=root/'run';checkpoint=run/'checkpoints/step-00000004';checkpoint.mkdir(parents=True)
        (run/'checkpoints/best').symlink_to(checkpoint,target_is_directory=True)
        report={'status':'completed','config':{'task':'mintrec2','epochs':5,'lora_rank':16,'use_transcript':True},
                'global_step':10,'requested_steps':10,'best_score':.8,
                'base_parameters':{'components':{'model':100,'lm_head':20,'audio_tower':50}}}
        save(run/'report.json',report)
        test={'status':'completed','split':'test','test_set_used':True,'partial_split':False,
              'evaluated_count':445,'full_split_count':445,'checkpoint':str(checkpoint),
              'logit':{'accuracy':.75},'greedy':{'accuracy':.74}}
        save(root/'test.json',test)
        return run,test

    def test_completed_training_is_not_completed_test_and_scores_stay_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,test=self.fixture(root)
            row=normalize_run(root,'family','run','mintrec2','SFT',run,'completed')
            self.assertEqual(row['test_status'],'queued');self.assertNotIn('logit_test_accuracy',row)
            row=normalize_run(root,'family','run','mintrec2','SFT',run,'completed',test_report_path=root/'test.json')
            self.assertEqual(row['test_status'],'completed');self.assertEqual(row['best_val_score'],.8)
            self.assertEqual(row['logit_test_accuracy'],.75);self.assertEqual(row['greedy_test_accuracy'],.74)
            self.assertEqual(row['language_parameters'],120);self.assertEqual(row['epoch'],5.)

    def test_partial_or_wrong_checkpoint_tests_are_not_published_as_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,test=self.fixture(root)
            for change in [{'partial_split':True},{'checkpoint':'/wrong'},{'evaluated_count':20},{'split':'dev','test_set_used':False}]:
                save(root/'test.json',test|change)
                row=normalize_run(root,'family','run','mintrec2','SFT',run,'completed',test_report_path=root/'test.json')
                self.assertEqual(row['test_status'],'unverified');self.assertTrue(row['warnings'])
                self.assertNotIn('logit_test_accuracy',row)

    def test_progress_tolerates_partial_jsonl_and_uses_latest_update(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,test=self.fixture(root)
            path=run/'metrics.jsonl'
            path.write_text(json.dumps({'event':'train','global_step':5,'epoch':2.5})+'\n{"unfinished":')
            self.assertEqual(tail_progress(path)['global_step'],5)
            row=normalize_run(root,'family','run','mintrec2','SFT',run,'running',job={'gpu':4,'pid':12})
            self.assertEqual(row['epoch'],2.5);self.assertEqual(row['gpu'],4)

    def test_pruning_inventory_includes_future_kd_without_counting_exports_or_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,test=self.fixture(root)
            save(root/'sweep.json',{'status':'running','jobs':[{'name':'method','config':{'task':'mintrec2'},'output':str(run),'status':'completed'}]})
            catalog={'families':[{'id':'p','label':'pruning','kind':'pruning','manifest':'sweep.json','followup':'missing.json','protocol':'protocol.md'}]}
            rows,families=collect(root,catalog)
            self.assertEqual(len(rows),2);self.assertEqual({r['arm'] for r in rows},{'SFT','KD'})
            self.assertEqual(rows[1]['training_status'],'queued');self.assertEqual(rows[1]['objective'],'sft_kd')
            self.assertEqual(families[0]['training_counts'],{'completed':1,'queued':1})

    def test_unscheduled_tests_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,test=self.fixture(root)
            row=normalize_run(root,'family','run','mer6','SFT',run,'completed',tests_scheduled=False)
            self.assertEqual(row['test_status'],'not_scheduled');self.assertEqual(row['primary_metric'],'weighted_f1')
