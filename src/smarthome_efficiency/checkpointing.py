"""Atomic local checkpoints for adapter weights and complete training state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu())


def atomic_link(path, target):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.unlink(missing_ok=True)
    temp.symlink_to(os.path.relpath(target, path.parent))
    os.replace(temp, path)


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if (path / 'checkpoints/latest').exists():
        path = (path / 'checkpoints/latest').resolve()
    elif (path / 'latest').exists():
        path = (path / 'latest').resolve()
    for name in ('training_state.pt', 'metadata.json', 'adapter/adapter_model.safetensors'):
        if not (path / name).is_file():
            raise ValueError(f'Incomplete full checkpoint: missing {path/name}; plain LoRA adapters require --init-adapter')
    return path


def inherit_best_checkpoint(source_output, output, best_step):
    """Preserve a best checkpoint inherited through earlier continuation runs.

    Prefer the saved step directory over the mutable best pointer: a newer
    checkpoint may have become best since the resumed checkpoint was written.
    """
    if best_step is None:
        return None
    source_root = Path(source_output) / 'checkpoints'
    for candidate in (source_root / f'step-{best_step:08d}', source_root / 'best'):
        if not candidate.exists():
            continue
        checkpoint = resolve_checkpoint(candidate)
        metadata = json.loads((checkpoint / 'metadata.json').read_text())
        if metadata.get('global_step') != best_step:
            continue
        destination = Path(output) / 'checkpoints' / 'best'
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_link(destination, checkpoint)
        return checkpoint
    raise ValueError(f'Best checkpoint at step {best_step} is unavailable in {source_root}; '
                     'restore its checkpoint or best pointer before continuing')


def save_checkpoint(output, model, state, is_best=False):
    root = Path(output) / 'checkpoints'
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'step-{state["global_step"]:08d}'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {target}')
    temporary = root / f'.saving-{uuid.uuid4().hex}'
    temporary.mkdir()
    try:
        model.save_pretrained(temporary / 'adapter', safe_serialization=True)
        torch.save(state, temporary / 'training_state.pt')
        metadata = {key: state[key] for key in ('global_step', 'next_epoch', 'next_update', 'world_size',
                    'config', 'data_fingerprint', 'schedule', 'best_score', 'best_step', 'elapsed_seconds')}
        if 'base_model_manifest_sha256' in state:
            metadata['base_model_manifest_sha256'] = state['base_model_manifest_sha256']
        if 'teacher_cache_sha256' in state:
            metadata['teacher_cache_sha256'] = state['teacher_cache_sha256']
        if state.get('feature_cache_sha256') is not None:
            metadata['feature_cache_sha256'] = state['feature_cache_sha256']
        write_json(temporary / 'metadata.json', metadata)
        temporary.rename(target)
        atomic_link(root / 'latest', target)
        if is_best:
            atomic_link(root / 'best', target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def trim_metrics(path, step):
    """Discard post-checkpoint scalar events after interrupted work, keeping a backup."""
    path = Path(path)
    if not path.exists():
        return
    original = path.read_text()
    kept = []
    for line in original.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # A killed writer may leave a partial final line. Preserve it in backup.
            continue
        if event.get('global_step', 0) <= step:
            kept.append(json.dumps(event, ensure_ascii=False, allow_nan=False))
    replacement = '\n'.join(kept) + ('\n' if kept else '')
    if replacement != original:
        backup = path.with_name(path.name + f'.before-resume-{uuid.uuid4().hex[:8]}')
        backup.write_text(original)
        path.write_text(replacement)
