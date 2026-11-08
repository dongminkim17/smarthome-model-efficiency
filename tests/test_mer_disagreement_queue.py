"""Paired-seed selection and immutable nine-run MER experiment contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from project_queue import validate_submission
from project_queue_jobs import entry
from queue_mer_disagreement import CONTROL_REFERENCES, FAMILY, build_submission, definitions, validate_controls
from smarthome_efficiency.pruning import sha256_file
from smarthome_efficiency.queueing import refresh_readiness
from validation_shortlist import select_validation_shortlist


class DisagreementQueueTests(unittest.TestCase):
    def base_config(self):
        return {'task': 'mer6', 'model': 'balanced', 'data_root': 'data', 'split_seed': 42,
                'seed': 42, 'train_limit': None, 'dev_limit': None, 'selection_mode': 'logit',
                'eval_modes': ['logit'], 'micro_batch_size': 2, 'gradient_accumulation_steps': 2,
                'gradient_checkpointing': True, 'lora_rank': 16, 'lora_alpha': 32,
                'teacher_checkpoint': 'original_teacher', 'teacher_cache': 'class_cache',
                'feature_teacher_cache': 'feature_cache', 'distillation_alpha': .75,
                'distillation_disagreement_alpha': .25, 'max_frames': 8, 'max_pixels': 224**2,
                'epochs': 5, 'learning_rate': 1e-4, 'encoder_learning_rate': 1e-5,
                'feature_audio_weight': .1, 'feature_vision_weight': .1}

    def grouped_state(self, folder, scores):
        jobs, groups = [], []
        common = {'train': ['a'], 'dev': ['b']}
        for group_no, pair in enumerate(scores):
            name = f'setting{group_no}'
            cfg = dict(self.base_config(), lora_rank=16 if group_no % 2 == 0 else 64,
                       lora_alpha=32 if group_no % 2 == 0 else 128,
                       distillation_disagreement_alpha=.25 if group_no < 2 else 0.)
            group = {'name': name, 'config': cfg, 'members': []}
            for seed, score in zip((42, 43), pair):
                key = name+f'_seed{seed}';run = Path(folder)/key;run.mkdir()
                (run/'data_manifest.json').write_text(json.dumps(common))
                report = {'status': 'completed', 'test_set_used': False, 'train_count': 2698,
                          'dev_count': 675, 'world_size': 1, 'config': dict(cfg, seed=seed),
                          'best_score': score, 'teacher_cache_sha256': 'class_hash',
                          'feature_cache_sha256': 'feature_hash'}
                (run/'report.json').write_text(json.dumps(report))
                train = entry(key, 'recovery_train', {'task': 'mer6', 'name': key, 'run_dir': str(run)})
                train['status'] = 'completed'
                test = entry('test:'+key, 'recovery_test', {'task': 'mer6', 'shortlist_job': 'select'}, deps=[key, 'select'])
                jobs.extend([train, test]);group['members'].append({'job': key, 'seed': seed})
            groups.append(group)
        mandatory = entry('test:control', 'recovery_test', {'task': 'mer6'}, deps=['control'])
        control = entry('control', 'recovery_train', {'task': 'mer6'});control['status'] = 'completed'
        jobs.extend([control, mandatory])
        select = entry('select', 'shortlist', {'task': 'mer6', 'top_k': 2, 'groups': groups,
                       'required_seeds': [42, 43], 'dev_count': 675, 'train_count': 2698,
                       'expected_data_manifest_sha256': sha256_file(run/'data_manifest.json'),
                       'expected_teacher_cache_sha256': 'class_hash', 'expected_feature_cache_sha256': 'feature_hash',
                       'output': str(Path(folder)/'decision.json')},
                       deps=[m['job'] for g in groups for m in g['members']], resource='inline')
        jobs.append(select)
        return {'jobs': jobs}, select

    def select(self, state, job):
        with patch('launch_pruning_followup.pin_best', side_effect=lambda x: str(Path(x)/'checkpoints/best')), \
             patch('launch_pruning_followup.checkpoint_identity', return_value={'metadata.json': 'pinned'}):
            return select_validation_shortlist(state, job)

    def test_group_mean_not_lucky_single_seed_selects_two_complete_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            state, job = self.grouped_state(folder, [(.90, .50), (.76, .74), (.73, .73)])
            decision = self.select(state, job)
            self.assertEqual(decision['selected_groups'], ['setting1', 'setting2'])
            self.assertEqual(decision['selected'], ['setting1_seed42', 'setting1_seed43', 'setting2_seed42', 'setting2_seed43'])
            self.assertFalse(decision['test_results_used'])
            self.assertAlmostEqual(decision['group_ranking'][0]['mean_validation_score'], .75)
            self.assertEqual(next(j for j in state['jobs'] if j['id']=='test:setting0_seed42')['status'], 'cancelled')
            self.assertEqual(next(j for j in state['jobs'] if j['id']=='test:control')['status'], 'waiting')
            self.assertTrue(all(j['status']=='completed' for j in state['jobs'] if j['kind']=='recovery_train'))

    def test_failed_seed_excludes_its_group_and_does_not_block_other_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            state, job = self.grouped_state(folder, [(.90, .91), (.75, .75), (.74, .74)])
            next(j for j in state['jobs'] if j['id']=='setting0_seed43')['status'] = 'failed'
            refresh_readiness(state['jobs']);self.assertEqual(job['status'], 'ready')
            decision = self.select(state, job)
            self.assertEqual(decision['selected_groups'], ['setting1', 'setting2'])
            self.assertEqual(decision['excluded_incomplete_groups'][0]['group'], 'setting0')
            self.assertEqual(next(j for j in state['jobs'] if j['id']=='test:setting0_seed42')['status'], 'cancelled')

    def test_all_groups_incomplete_finishes_with_empty_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            state, job = self.grouped_state(folder, [(.90, .91), (.75, .75)])
            for j in state['jobs']:
                if j['id'] in ('setting0_seed43', 'setting1_seed43'):j['status'] = 'failed'
            decision = self.select(state, job)
            self.assertEqual(decision['selected'], [])
            self.assertEqual(len(decision['excluded_incomplete_groups']), 2)
            self.assertEqual(next(j for j in state['jobs'] if j['id']=='test:control')['status'], 'waiting')

    def test_ties_use_group_name_and_members_are_seed_ordered(self):
        with tempfile.TemporaryDirectory() as folder:
            state, job = self.grouped_state(folder, [(.75, .75)]*3)
            job['payload']['groups'].reverse();job['deps'].reverse()
            self.assertEqual(self.select(state, job)['selected_groups'], ['setting0', 'setting1'])

    def test_wrong_seed_recipe_data_or_targets_fail_before_selection(self):
        for mutation in ('seed', 'rank', 'gating', 'data', 'class_cache', 'feature_cache', 'train_count', 'test_used'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                state, job = self.grouped_state(folder, [(.75, .75)])
                run=Path(folder)/'setting0_seed43';report=json.loads((run/'report.json').read_text())
                if mutation=='seed':report['config']['seed']=44
                elif mutation=='rank':report['config']['lora_rank']=64
                elif mutation=='gating':report['config']['distillation_disagreement_alpha']=0.
                elif mutation=='data':(run/'data_manifest.json').write_text('{"dev":["other"]}')
                elif mutation=='class_cache':report['teacher_cache_sha256']='other'
                elif mutation=='feature_cache':report['feature_cache_sha256']='other'
                elif mutation=='train_count':report['train_count']=3035
                else:report['test_set_used']=True
                (run/'report.json').write_text(json.dumps(report))
                with self.assertRaises(ValueError):self.select(state, job)
                self.assertFalse(Path(job['payload']['output']).exists())

    def test_execution_fallback_preserves_group_but_effective_batch_change_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            state, job = self.grouped_state(folder, [(.75, .75)])
            p=Path(folder)/'setting0_seed43/report.json';r=json.loads(p.read_text())
            r['config'].update(micro_batch_size=1,gradient_accumulation_steps=4);p.write_text(json.dumps(r))
            self.assertEqual(len(self.select(state, job)['selected']), 2)
            r['config']['gradient_accumulation_steps']=8;p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):self.select(state, job)

    def test_duplicate_or_missing_declared_seeds_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            state, job = self.grouped_state(folder, [(.75, .75)])
            job['payload']['groups'][0]['members'][1]['seed']=42
            with self.assertRaises(ValueError):self.select(state, job)

    def test_matrix_and_builder_preserve_one_method_and_control_independence(self):
        defs=definitions();self.assertEqual(len(defs),9)
        self.assertEqual(sum(d['disagreement_alpha'] is None for d in defs),1)
        self.assertEqual({(d['rank'],d['seed'],d['disagreement_alpha']) for d in defs if d['disagreement_alpha'] is not None},
                         {(r,s,a) for r in (16,64) for s in (42,43) for a in (.25,0.)})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'model';model.mkdir()
            counts={'language':1924902912,'audio_encoder':490818560,'vision_encoder':500005120,'total':2915726592,'other':0}
            (model/'compression.json').write_text(json.dumps({'parameters_after':counts}))
            cfg=self.base_config();cfg.pop('distillation_disagreement_alpha');cfg['model']=str(model)
            controls=[{'data_manifest_sha256':'data_hash','teacher_cache_sha256':'class_hash','feature_cache_sha256':'feature_hash'}]*3
            with patch('queue_mer_disagreement.validate_reference_contract'), \
                 patch('queue_mer_disagreement.validate_controls',return_value=(cfg,controls)):
                spec=build_submission({},root/'queue',root)
            self.assertEqual(len(spec['jobs']),38)
            validate_submission({'jobs':[],'documents':{}},spec)
            trains=[j for j in spec['jobs'] if j['kind']=='recovery_train'];self.assertEqual(len(trains),9)
            self.assertTrue(all(j['payload']['config']['epochs']==5 for j in trains))
            self.assertTrue(all(j['payload']['config']['model']==str(model) for j in trains))
            self.assertTrue(all(j['payload']['config']['max_frames']==8 for j in trains))
            self.assertTrue(all(j['payload']['config']['max_pixels']==224**2 for j in trains))
            ctrl=next(j for j in trains if j['payload']['arm']=='CONTROL_R64_SEED43')
            self.assertNotIn('distillation_disagreement_alpha',ctrl['payload']['config'])
            controltest=next(j for j in spec['jobs'] if j['kind']=='recovery_test' and j['payload']['arm']=='CONTROL_R64_SEED43')
            self.assertEqual(controltest['deps'],[ctrl['id']]);self.assertNotIn('shortlist_job',controltest['payload'])
            shortlist=next(j for j in spec['jobs'] if j['kind']=='shortlist')
            self.assertEqual(len(shortlist['deps']),8);self.assertEqual(len(shortlist['payload']['groups']),4)
            self.assertNotIn(ctrl['id'],shortlist['deps'])
            self.assertTrue(next(j for j in spec['jobs'] if j['kind']=='prepare')['payload']['require_teacher_disagreement'])
            self.assertFalse(any(j['kind']=='checkpoint_average' for j in spec['jobs']))

    def test_reused_control_must_match_config_manifest_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=root/'model';model.mkdir();(model/'compression.json').write_text('{}')
            feature=root/'feature';feature.mkdir();(feature/'cache.json').write_text('{}')
            teacher=root/'cache.json';teacher.write_text('{}')
            cfg=self.base_config();cfg.pop('distillation_disagreement_alpha')
            cfg.update(model=str(model),teacher_cache=str(teacher),feature_teacher_cache=str(feature))
            jobs=[]
            for (rank,seed),key in CONTROL_REFERENCES.items():
                run=root/f'r{rank}s{seed}';run.mkdir();checkpoint=run/'checkpoints/best';checkpoint.mkdir(parents=True)
                conf=dict(cfg,lora_rank=rank,lora_alpha=2*rank,seed=seed)
                report={'status':'completed','test_set_used':False,'config':conf,'train_count':2698,'dev_count':675,'world_size':1,
                        'teacher_cache_sha256':sha256_file(teacher),'feature_cache_sha256':sha256_file(feature/'cache.json')}
                (run/'report.json').write_text(json.dumps(report));(run/'data_manifest.json').write_text('{"train":[],"dev":[]}')
                meta={'config':conf,'base_model_manifest_sha256':sha256_file(model/'compression.json'),
                      'teacher_cache_sha256':report['teacher_cache_sha256'],'feature_cache_sha256':report['feature_cache_sha256']}
                (checkpoint/'metadata.json').write_text(json.dumps(meta))
                jobs.append({'id':key,'status':'completed','payload':{'config':conf,'run_dir':str(run)}})
            with patch('queue_mer_disagreement.pin_best',side_effect=lambda run:str(Path(run)/'checkpoints/best')), \
                 patch('queue_mer_disagreement.checkpoint_identity',return_value={'metadata.json':'fixed'}):
                actual,controls=validate_controls({'jobs':jobs});self.assertEqual(actual,cfg);self.assertEqual(len(controls),3)
                (root/'r16s43/data_manifest.json').write_text('{"train":["changed"],"dev":[]}')
                with self.assertRaises(ValueError):validate_controls({'jobs':jobs})


if __name__=='__main__':unittest.main()
