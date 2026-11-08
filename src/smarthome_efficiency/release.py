"""Portable metadata and validation for released task models."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from .data import LABELS
from .pruning import sha256_file

FORMAT_VERSION = 1
BASE_MODEL = 'Qwen/Qwen2.5-Omni-3B'
BASE_REVISION = 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'
CODE_URL = 'https://github.com/dongminkim17/smarthome-model-efficiency'
PARAMETERS = {'language': 1_924_902_912, 'audio_encoder': 637_676_544,
              'vision_encoder': 668_684_288, 'other': 0, 'total': 3_231_263_744}
INFERENCE_KEYS = ('task', 'seed', 'split_seed', 'max_frames', 'max_pixels', 'use_transcript',
                  'max_new_tokens', 'num_threads')


def task_settings(config):
    from .model import SYSTEM_PROMPT, build_prompt, build_input_prompt
    task = config['task']
    if task not in LABELS or config['use_transcript'] is not (task != 'mer6'):
        raise ValueError('Unexpected task or input protocol')
    settings = {key: config[key] for key in INFERENCE_KEYS}
    settings.update(format_version=FORMAT_VERSION, labels=list(LABELS[task]),
                    answer_letters=[chr(65+i) for i in range(len(LABELS[task]))],
                    system_prompt=SYSTEM_PROMPT, deterministic=True,
                    eval_modes=['logit', 'greedy'], selection_mode='logit',
                    prompt_sha256=hashlib.sha256(inspect.getsource(build_prompt).strip().encode()).hexdigest(),
                    input_prompt_sha256=hashlib.sha256(inspect.getsource(build_input_prompt).strip().encode()).hexdigest())
    return settings

def validate_package(path, verify_hashes=False):
    path = Path(path)
    manifest = json.loads((path/'release.json').read_text())
    settings = json.loads((path/'task_config.json').read_text())
    if manifest.get('format_version') != FORMAT_VERSION or manifest.get('format') != 'merged_thinker':
        raise ValueError('Unsupported released-model format')
    task = manifest.get('task')
    if task not in LABELS or settings.get('task') != task or settings.get('labels') != list(LABELS[task]):
        raise ValueError('Released model task/label mapping differs')
    if manifest.get('parameters') != PARAMETERS:
        raise ValueError('Released model parameter counts differ')
    if settings != task_settings(settings):
        raise ValueError('Released inference settings or prompt implementation differ')
    config = json.loads((path/'config.json').read_text())
    if (config.get('model_type') != 'qwen2_5_omni_thinker'
            or config['text_config']['num_hidden_layers'] != 36
            or config['text_config']['intermediate_size'] != 4352):
        raise ValueError('Released model architecture differs')
    files = manifest.get('files', {})
    required = {'config.json', 'task_config.json', 'preprocessor_config.json', 'tokenizer.json', 'LICENSE', 'NOTICE'}
    if not required <= set(files) or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Release is missing model, processor or licensing assets')
    for name, identity in files.items():
        if Path(name).name != name or not (path/name).is_file():
            raise ValueError('Invalid or missing release file '+name)
        if (path/name).stat().st_size != identity['bytes']:
            raise ValueError('Release file size differs '+name)
        if verify_hashes and sha256_file(path/name) != identity['sha256']:
            raise ValueError('Release file checksum differs '+name)
    index = path/'model.safetensors.index.json'
    if index.exists():
        shards = set(json.loads(index.read_text())['weight_map'].values())
        if not shards <= set(files):
            raise ValueError('Weight index references an untracked shard')
    return manifest, settings


def resolve_model(value, revision=None):
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(value), revision=revision))


def evaluation_config(path, data_root, cache_dir):
    manifest, settings = validate_package(path)
    config = {key: settings[key] for key in INFERENCE_KEYS}
    config.update(model=str(Path(path).resolve()), data_root=str(Path(data_root).expanduser().resolve()),
                  cache_dir=str(Path(cache_dir).expanduser().resolve()), deterministic=True,
                  eval_modes=settings['eval_modes'], selection_mode=settings['selection_mode'])
    return manifest, config
