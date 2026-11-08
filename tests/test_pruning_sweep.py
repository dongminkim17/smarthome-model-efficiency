"""Pruning queue configuration, idle-GPU selection, and resume contracts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from launch_pruning_sweep import available_gpus, fresh_plans, prepare_plans_for_resume


class PruningSweepTests(unittest.TestCase):
    def setUp(self):
        self.source_directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.source_directory.cleanup)
        sources={}
        from smarthome_efficiency.training import load_config
        for task in ['mer6','mintrec2','mintrec20']:
            source=Path(self.source_directory.name)/task
            (source/'adapter').mkdir(parents=True)
            config=load_config(ROOT/'configs'/f'{task}.yaml')
            (source/'metadata.json').write_text(json.dumps({'config':config}))
            (source/'adapter/adapter_config.json').write_text('{}')
            (source/'adapter/adapter_model.safetensors').touch()
            sources[task]=source
        patcher=patch('launch_pruning_sweep.SOURCES',sources)
        patcher.start();self.addCleanup(patcher.stop)

    def test_six_jobs_share_recovery_recipe_and_pin_established_sources(self):
        plans=fresh_plans(Path('/unused/runs'),Path('/unused/compact'),['mer6','mintrec2','mintrec20'])
        self.assertEqual(len(plans),6)
        self.assertEqual(len({p['compact_model'] for p in plans}),6)
        for task in ['mer6','mintrec2','mintrec20']:
            pair=[p for p in plans if p['task']==task]
            self.assertEqual(pair[0]['source_checkpoint'],pair[1]['source_checkpoint'])
            for plan in pair:
                c=plan['config']
                self.assertEqual(c['epochs'],5)
                self.assertEqual(c['micro_batch_size']*c['gradient_accumulation_steps'],4)
                self.assertEqual(c['lora_target'],'decoder_linear')
                self.assertEqual(c['eval_modes'],['logit','greedy'])
                self.assertEqual(c['split_seed'],42)
                self.assertIsNone(c['train_limit'])
                self.assertIsNone(c['dev_limit'])
                self.assertEqual(c['gradient_checkpointing'],task=='mer6')
                self.assertEqual(c['model'],plan['compact_model'])
        smoke=fresh_plans(Path('/unused/runs'),Path('/unused/compact'),['mer6'],smoke=True)
        self.assertTrue(all(p['config']['train_limit']==24 and p['config']['dev_limit']==20 for p in smoke))

    def test_busy_devices_are_queued_and_gpu_zero_cannot_be_selected(self):
        devices='0, GPU-zero, 16, 0\n1, GPU-one, 16, 0\n2, GPU-two, 2000, 0\n3, GPU-three, 16, 0\n4, GPU-four, 16, 90\n'
        results=[subprocess.CompletedProcess([],0,stdout=devices),
                 subprocess.CompletedProcess([],0,stdout='GPU-three, 123\n')]
        with patch('launch_pruning_sweep.subprocess.run',side_effect=results):
            self.assertEqual([g['index'] for g in available_gpus([1,2,3,4])],[1])
        for invalid in [[0],[8],[1,1],[]]:
            with self.assertRaises(ValueError): available_gpus(invalid)

    def test_resume_uses_saved_optimizer_config_and_detects_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as root:
            plans=fresh_plans(Path(root)/'runs',Path(root)/'compact',['mer6'])[:1]
            plan=plans[0]; output=Path(plan['output'])
            cp=output/'checkpoints/step-00000002'; (cp/'adapter').mkdir(parents=True)
            (cp/'adapter/adapter_model.safetensors').touch(); (cp/'training_state.pt').touch()
            (cp/'metadata.json').write_text(json.dumps({'config':plan['config'],'world_size':1}))
            (cp.parent/'latest').symlink_to(cp.name)
            (output/'report.json').write_text(json.dumps({'status':'completed','config':plan['config']}))
            prepare_plans_for_resume(plans,None)
            self.assertEqual(plan['status'],'completed')
            self.assertEqual(plan['stage'],'train')
            prepare_plans_for_resume(plans,7)
            self.assertEqual(plan['status'],'pending')
            self.assertEqual(plan['config']['epochs'],7)
            self.assertEqual(plan['resume_checkpoint'],str(cp))
        with tempfile.TemporaryDirectory() as root:
            plans=fresh_plans(Path(root)/'runs',Path(root)/'compact',['mer6'])[:1]
            output=Path(plans[0]['output']); output.mkdir(parents=True)
            (output/'incomplete').touch()
            with self.assertRaisesRegex(ValueError,'incomplete'): prepare_plans_for_resume(plans,None)


if __name__=='__main__': unittest.main()
