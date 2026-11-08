"""Regression checks for distributed weighting and resume contracts."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from smarthome_efficiency.checkpointing import inherit_best_checkpoint, resolve_checkpoint, save_checkpoint, trim_metrics
from smarthome_efficiency.training import DEFAULTS, epoch_order, learning_rate, load_config, make_schedule, rank_update, validate_resume


class TrainingTests(unittest.TestCase):
    def test_sharding_and_weighted_gradient_match_real_example_mean(self):
        # Test partial updates, including ranks with no real examples and accumulation.
        for count in (1, 3, 10, 14, 23):
            for accumulation in (1, 3):
                order = epoch_order(count, 42, 0)
                visited = []
                for update in range((count+4*accumulation-1)//(4*accumulation)):
                    rank_gradients = []
                    expected = []
                    for rank in range(4):
                        indices, valid = rank_update(order, update, 4, rank, accumulation)
                        gradient = 0.0
                        for index in indices:
                            if index is not None:
                                visited.append(index)
                                expected.append(index+1.0)
                                gradient += (index+1.0) * 4 / valid
                        rank_gradients.append(gradient)
                    self.assertAlmostEqual(sum(rank_gradients)/4, sum(expected)/len(expected))
                self.assertEqual(sorted(visited), list(range(count)))

    def test_resume_iteration_starts_at_next_unseen_update(self):
        order = epoch_order(14, 42, 3)
        first = [i for rank in range(4) for i in rank_update(order, 0, 4, rank, 1)[0] if i is not None]
        rest = [i for update in range(1, 4) for rank in range(4) for i in rank_update(order, update, 4, rank, 1)[0] if i is not None]
        self.assertFalse(set(first) & set(rest))
        self.assertEqual(sorted(first+rest), list(range(14)))
        self.assertEqual(order, epoch_order(14, 42, 3))
        self.assertNotEqual(order, epoch_order(14, 42, 4))

    def test_schedule_resume_and_extension(self):
        cfg = DEFAULTS | {'learning_rate': 1e-4, 'warmup_ratio': .1}
        original = make_schedule(cfg, 20)
        saved = json.loads(json.dumps(original))
        self.assertEqual([learning_rate(saved, i) for i in range(7,20)], [learning_rate(original,i) for i in range(7,20)])
        extension = make_schedule(cfg, 40, start_step=20)
        self.assertGreater(learning_rate(extension, 20), 0)
        self.assertEqual(learning_rate(extension, 21), cfg['learning_rate'])
        self.assertLess(learning_rate(extension, 39), learning_rate(extension, 25))

    def test_resume_rejects_changed_data_or_effective_batch(self):
        cfg = dict(DEFAULTS)
        validate_resume(cfg, cfg | {'epochs': 10}, 4, 4)
        for changed in ({'split_seed':43}, {'gradient_accumulation_steps':2}, {'eval_modes':['greedy']}, {'epochs':4}):
            with self.assertRaises(ValueError):
                validate_resume(cfg, cfg | changed, 4, 4)
        with self.assertRaises(ValueError):
            validate_resume(cfg, cfg, 4, 2)

    def test_config_modes_and_selection_guard(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'config.yaml'
            path.write_text(yaml.safe_dump({'model':'model','eval_modes':'both'}))
            self.assertEqual(load_config(path)['eval_modes'], ['logit','greedy'])
            path.write_text(yaml.safe_dump({'model':'model','eval_modes':['greedy']}))
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text(yaml.safe_dump({'model':'model','eval_modes':['greedy'],'selection_mode':'greedy'}))
            self.assertEqual(load_config(path)['eval_modes'], ['greedy'])

    def test_best_checkpoint_survives_two_forked_continuations(self):
        class Adapter:
            def save_pretrained(self, path, **kwargs):
                Path(path).mkdir()
                (Path(path) / 'adapter_model.safetensors').write_bytes(b'fixture')

        with tempfile.TemporaryDirectory() as directory:
            first, second, third = [Path(directory) / name for name in ('first', 'second', 'third')]
            state = {'global_step': 2, 'next_epoch': 0, 'next_update': 2, 'world_size': 4,
                     'config': dict(DEFAULTS), 'data_fingerprint': 'test', 'schedule': {},
                     'best_score': .9, 'best_step': 2, 'elapsed_seconds': 3.0}
            original_best = save_checkpoint(first, Adapter(), state, is_best=True)
            inherit_best_checkpoint(first, second, state['best_step'])
            # The continuation never improves on the original run's best model.
            later = save_checkpoint(second, Adapter(), state | {'global_step': 4, 'next_update': 4})
            self.assertFalse((second / 'checkpoints' / original_best.name).exists())
            resumed_state = torch.load(later / 'training_state.pt', weights_only=False)
            inherit_best_checkpoint(second, third, resumed_state['best_step'])
            self.assertEqual((third / 'checkpoints/best').resolve(), original_best)
            self.assertEqual(json.loads((third / 'checkpoints/best/metadata.json').read_text())['best_score'], .9)

            # A newer best pointer must not replace the older checkpoint's saved best.
            save_checkpoint(second, Adapter(), state | {'global_step': 6, 'next_update': 6,
                                                       'best_step': 6, 'best_score': .95}, is_best=True)
            with self.assertRaisesRegex(ValueError, 'Best checkpoint at step 2 is unavailable'):
                inherit_best_checkpoint(second, Path(directory) / 'fourth', resumed_state['best_step'])

    def test_atomic_checkpoint_and_metrics_rollback(self):
        class Adapter:
            def save_pretrained(self, path, **kwargs):
                Path(path).mkdir()
                (Path(path)/'adapter_model.safetensors').write_bytes(b'fixture')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {'global_step':2, 'next_epoch':0, 'next_update':2, 'world_size':4,
                     'config':dict(DEFAULTS), 'data_fingerprint':'test', 'schedule':{},
                     'best_score':.5,'best_step':2,'elapsed_seconds':3.0,'optimizer':{'state':torch.tensor([2.0])}}
            checkpoint = save_checkpoint(root, Adapter(), state, is_best=True)
            self.assertEqual(resolve_checkpoint(root), checkpoint)
            self.assertEqual((root/'checkpoints/best').resolve(), checkpoint)
            loaded = torch.load(checkpoint/'training_state.pt',weights_only=False)
            self.assertTrue(torch.equal(loaded['optimizer']['state'], torch.tensor([2.0])))
            with self.assertRaises(FileExistsError):
                save_checkpoint(root, Adapter(), state)
            metrics = root/'metrics.jsonl'
            metrics.write_text('\n'.join(json.dumps({'event':'train','global_step':step}) for step in (1,2,3))+'\n{"partial":')
            trim_metrics(metrics,2)
            self.assertEqual([json.loads(line)['global_step'] for line in metrics.read_text().splitlines()],[1,2])
            self.assertEqual(len(list(root.glob('metrics.jsonl.before-resume-*'))),1)


if __name__ == '__main__':
    unittest.main()
