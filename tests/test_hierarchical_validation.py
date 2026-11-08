"""Hierarchical probabilities are sums, and validation provenance cannot be mixed."""
import copy,json,math,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from smarthome_efficiency.data import LABELS
from smarthome_efficiency.hierarchical_validation import project_validation_rows
from smarthome_efficiency.queueing import atomic_json
from analyse_intent_hierarchy import analyse


def example(identifier='a'):
    labels=LABELS['mintrec20'];scores=[-100.]*20
    for label,p in [('Complain',.4),('Inform',.3),('Advise',.3)]:scores[labels.index(label)]=math.log(p)
    return {'id':identifier,'true_label':'Inform','logit_scores':scores,'logit_predicted_label':'Complain','greedy_predicted_label':None}


class HierarchicalValidationTests(unittest.TestCase):
    def test_probability_sum_can_differ_from_mapping_fine_argmax(self):
        rows,metrics=project_validation_rows([example()],[{'id':'a','label':'Inform'}],LABELS['mintrec20'])
        self.assertAlmostEqual(rows[0]['coarse_probabilities']['goals'],.6)
        self.assertEqual(metrics['probability_sum']['accuracy'],1.);self.assertEqual(metrics['hard_map']['accuracy'],0.)
        self.assertEqual(metrics['greedy_map']['invalid_rate'],1.)

    def test_mismatched_ids_gold_labels_or_logit_order_are_rejected(self):
        manifest=[{'id':'a','label':'Inform'}]
        for changes in [{'id':'test-id'},{'true_label':'Praise'},{'logit_scores':[1.]},
                        {'logit_scores':[float('nan')]*20},{'logit_predicted_label':'Inform'}]:
            with self.assertRaises(ValueError):project_validation_rows([example()|changes],manifest,LABELS['mintrec20'])
        with self.assertRaises(ValueError):project_validation_rows([example(),example()],manifest,LABELS['mintrec20'])
        with self.assertRaises(ValueError):project_validation_rows([example()],manifest,list(reversed(LABELS['mintrec20'])))

    def test_end_to_end_uses_only_original_best_validation_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'run';checkpoint=run/'checkpoints/best'
            atomic_json(checkpoint/'metadata.json',{'global_step':7,'config':{'task':'mintrec20'}})
            atomic_json(run/'report.json',{'status':'completed','test_set_used':False,'best_step':7,'best_score':0.})
            atomic_json(run/'provenance.json',{'labels':list(LABELS['mintrec20'])})
            atomic_json(run/'data_manifest.json',{'dev':[{'id':str(i),'label':'Inform'} for i in range(445)],'full_dev_count':445})
            (run/'predictions').mkdir();(run/'predictions/dev-step-00000007.jsonl').write_text(''.join(json.dumps(example(str(i)))+'\n' for i in range(445)))
            record={'id':'fixture','family':'fixture','name':'mintrec20','arm':'B','task':'mintrec20','training_status':'completed',
                    'total_parameters':1963496704,'run_dir':str(run),'test_report':'this_test_path_must_never_be_opened'}
            atomic_json(root/'index.json',{'runs':[record]});result=analyse(root/'index.json',root/'analysis')
            self.assertFalse(result['test_set_used']);self.assertFalse(result['threshold_fitted']);self.assertFalse(result['checkpoint_selection_changed'])
            self.assertEqual(result['results'][0]['coarse_validation']['probability_sum']['accuracy'],1.)


if __name__=='__main__':unittest.main()
