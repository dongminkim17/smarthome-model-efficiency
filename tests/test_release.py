"""Portable model packages, final adapter merging and released-model evaluation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, TaskType, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
from evaluate import parse_args, resolve_evaluation_checkpoint
from export_release_model import validate_source
from test_pruning import tiny_model
from smarthome_efficiency.pruning import parameter_counts, sha256_file
from smarthome_efficiency.release import PARAMETERS, task_settings, validate_package, evaluation_config, resolve_model
from smarthome_efficiency.training import DEFAULTS, lora_targets


def recipe(task='mintrec2'):
    return DEFAULTS | {'task': task, 'use_transcript': task != 'mer6', 'objective': 'sft_kd',
                       'lora_rank': 64, 'lora_alpha': 128, 'lora_target': 'decoder_encoder_linear',
                       'distillation_alpha': .75, 'feature_audio_weight': .1, 'feature_vision_weight': .1,
                       'encoder_learning_rate': 1e-5}


def fixture(path):
    assets = {'config.json': json.dumps({'model_type': 'qwen2_5_omni_thinker', 'text_config': {'num_hidden_layers': 36, 'intermediate_size': 4352}}),
              'task_config.json': json.dumps(task_settings(recipe())), 'preprocessor_config.json': '{}',
              'tokenizer.json': '{}', 'LICENSE': 'license', 'NOTICE': 'notice', 'model.safetensors': 'tensor fixture'}
    for name, text in assets.items():
        (path/name).write_text(text)
    files = {name: {'bytes': (path/name).stat().st_size, 'sha256': sha256_file(path/name)} for name in assets}
    manifest = {'format_version': 1, 'format': 'merged_thinker', 'task': 'mintrec2',
                'parameters': PARAMETERS, 'files': files, 'source_global_step': 1}
    (path/'release.json').write_text(json.dumps(manifest))
    return manifest


class ReleaseTests(unittest.TestCase):
    def test_chosen_recipe_requires_native_inputs_and_final_adapter_scope(self):
        for task in ['mintrec2', 'mintrec20', 'mer6']:
            validate_source(recipe(task))
            for change in [{'lora_rank': 16}, {'feature_audio_weight': 0}, {'use_transcript': task == 'mer6'}, {'train_limit': 8}]:
                with self.assertRaises(ValueError):
                    validate_source(recipe(task) | change)

    def test_package_is_portable_and_validates_hashes_and_label_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            manifest, settings = validate_package(root, verify_hashes=True)
            self.assertEqual(manifest['parameters'], PARAMETERS)
            self.assertNotIn('/mnt/', json.dumps(settings))
            self.assertEqual(resolve_model(str(root)), root.resolve())
            _, config = evaluation_config(root, root/'data', root/'cache')
            self.assertEqual(config['model'], str(root.resolve()))
            settings['labels'].reverse()
            (root/'task_config.json').write_text(json.dumps(settings))
            with self.assertRaises(ValueError):
                validate_package(root)

    def test_file_tampering_and_escaping_manifest_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = fixture(root)
            (root/'model.safetensors').write_text('changed fixture')
            with self.assertRaises(ValueError):
                validate_package(root, verify_hashes=True)
            manifest['files']['../outside'] = {'bytes': 0, 'sha256': '0'*64}
            (root/'release.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_package(root)

    def test_model_and_checkpoint_cli_modes_are_mutually_exclusive(self):
        common = ['evaluate.py', '--split', 'test', '--output', 'out']
        with patch.object(sys, 'argv', common+['--model', 'user/model']):
            args = parse_args()
            self.assertEqual(args.model, 'user/model')
            self.assertIsNone(args.checkpoint)
        with patch.object(sys, 'argv', common+['--model', 'user/model', '--checkpoint', 'checkpoint']):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_merge_includes_decoder_audio_and_vision_adapters(self):
        torch.set_num_threads(2)
        model = tiny_model()
        original = {name: value.detach().clone() for name, value in model.named_parameters()}
        counts = parameter_counts(model)
        wrapped = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
                                 target_modules=lora_targets('decoder_encoder_linear'), task_type=TaskType.CAUSAL_LM))
        with torch.no_grad():
            for name, value in wrapped.named_parameters():
                if 'lora_A' in name or 'lora_B' in name:
                    value.fill_(.02)
        merged = wrapped.merge_and_unload(safe_merge=True)
        self.assertEqual(parameter_counts(merged), counts)
        current = dict(merged.named_parameters())
        for prefix in ['model.', 'audio_tower.', 'visual.']:
            self.assertTrue(any(not torch.equal(value, current[name]) for name, value in original.items() if name.startswith(prefix)))
