"""Safety and configuration checks for the seven independent GPU jobs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from launch_suzy_sweep import check_gpus, child_environment, completion, fresh_plans, prepare_resume, apply_execution_overrides


class SuzySweepTests(unittest.TestCase):
    def test_plan_preserves_batches_splits_and_scientific_controls(self):
        plans = fresh_plans(Path('/unused'))
        self.assertEqual([p['gpu'] for p in plans], list(range(1, 8)))
        self.assertEqual(len({p['output'] for p in plans}), 7)
        for p in plans:
            c = p['config']
            self.assertEqual(c['epochs'], 5)
            self.assertEqual(c['micro_batch_size'] * c['gradient_accumulation_steps'], 4)
            self.assertEqual(c['split_seed'], 42)
            self.assertIsNone(c['train_limit'])
            self.assertIsNone(c['dev_limit'])
            self.assertEqual(c['eval_modes'], ['logit', 'greedy'])
        self.assertEqual([p['config']['seed'] for p in plans], [42, 43, 42, 42, 43, 44, 42])
        self.assertEqual(plans[2]['config']['lora_target'], 'decoder_linear')
        self.assertEqual(plans[3]['config']['learning_rate'], 5e-5)
        smoke = fresh_plans(Path('/unused'), smoke=True)
        self.assertTrue(all(p['config']['epochs'] == 1 and p['config']['dev_limit'] == 20 for p in smoke))

    def test_execution_overrides_preserve_batch_and_reject_recipe_changes(self):
        plans = fresh_plans(Path('/unused'))
        apply_execution_overrides(plans, {'mer_qv_seed42': {'micro_batch_size': 2, 'gradient_checkpointing': False}})
        self.assertEqual(plans[0]['config']['micro_batch_size'], 2)
        self.assertEqual(plans[0]['config']['gradient_accumulation_steps'], 2)
        self.assertFalse(plans[0]['config']['gradient_checkpointing'])
        self.assertEqual(plans[1]['config']['micro_batch_size'], 1)
        for invalid in [{'unknown': {}}, {'mer_qv_seed42': {'learning_rate': 1}},
                        {'mer_qv_seed42': {'micro_batch_size': 3}},
                        {'mer_qv_seed42': {'gradient_checkpointing': 'off'}}]:
            with self.assertRaises(ValueError):
                apply_execution_overrides(plans, invalid)

    def test_busy_gpu_zero_is_ignored_but_selected_processes_are_rejected(self):
        devices = '0, GPU-zero, Blackwell, 97887, 37000, 99\n1, GPU-one, RTX3090, 24576, 16, 0\n'
        results = [subprocess.CompletedProcess([], 0, stdout=devices),
                   subprocess.CompletedProcess([], 0, stdout='GPU-zero, 123\n')]
        with patch('launch_suzy_sweep.subprocess.run', side_effect=results):
            self.assertEqual(check_gpus([1])[1]['uuid'], 'GPU-one')
        results[1] = subprocess.CompletedProcess([], 0, stdout='GPU-one, 456\n')
        with patch('launch_suzy_sweep.subprocess.run', side_effect=results):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                check_gpus([1])
        results[0] = subprocess.CompletedProcess([], 0, stdout=devices.replace('24576, 16, 0', '24576, 1000, 0'))
        results[1] = subprocess.CompletedProcess([], 0, stdout='')
        with patch('launch_suzy_sweep.subprocess.run', side_effect=results):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                check_gpus([1])

    def test_physical_uuid_and_single_process_override_inherited_ddp_environment(self):
        config = fresh_plans(Path('/unused'))[0]['config']
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0', 'WORLD_SIZE': '8', 'LOCAL_RANK': '7'}):
            env = child_environment(config, 'GPU-one')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-one')
        self.assertEqual(env['WORLD_SIZE'], '1')
        self.assertEqual(env['LOCAL_RANK'], '0')

    def test_exit_zero_requires_finished_training_and_no_test_use(self):
        with tempfile.TemporaryDirectory() as root:
            plan = {'output': root}
            path = Path(root) / 'report.json'
            report = {'status': 'paused', 'global_step': 2, 'test_set_used': False,
                      'best_score': .5, 'best_step': 2}
            path.write_text(json.dumps(report))
            self.assertEqual(completion(plan, 0, smoke=True)['status'], 'completed')
            self.assertEqual(completion(plan, 0, smoke=False)['status'], 'failed')
            report.update(status='completed', test_set_used=True)
            path.write_text(json.dumps(report))
            self.assertEqual(completion(plan, 0, smoke=False)['status'], 'failed')
            report['test_set_used'] = False
            path.write_text(json.dumps(report))
            self.assertEqual(completion(plan, 0, smoke=False)['status'], 'completed')
            self.assertEqual(completion(plan, 1, smoke=False)['status'], 'failed')

    def test_completed_runs_skip_and_extension_resumes_saved_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            plans = fresh_plans(Path(root))[:1]
            plan = plans[0]
            output = Path(plan['output'])
            cp = output / 'checkpoints/step-00000002'
            (cp / 'adapter').mkdir(parents=True)
            (cp / 'adapter/adapter_model.safetensors').touch()
            (cp / 'training_state.pt').touch()
            metadata = {'config': plan['config'], 'world_size': 1}
            (cp / 'metadata.json').write_text(json.dumps(metadata))
            (cp.parent / 'latest').symlink_to(cp.name)
            (output / 'report.json').write_text(json.dumps({'status': 'completed', 'config': plan['config']}))
            prepare_resume(plans, None)
            self.assertEqual(plan['status'], 'completed')
            prepare_resume(plans, 8)
            self.assertEqual(plan['status'], 'pending')
            self.assertEqual(plan['config']['epochs'], 8)
            self.assertEqual(Path(plan['resume_checkpoint']), cp)
            metadata['world_size'] = 4
            (cp / 'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'world size'):
                prepare_resume(plans, None)


if __name__ == '__main__':
    unittest.main()
