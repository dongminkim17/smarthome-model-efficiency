"""Configuration, deterministic update batches, and resumable LR schedules."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    'task': 'mintrec2', 'model': '', 'data_root': str(ROOT / 'data'),
    'cache_dir': str(ROOT / 'artifacts/pilot_cache'),
    'seed': 42, 'split_seed': 42, 'epochs': 5, 'deterministic': True, 'gradient_checkpointing': True,
    'learning_rate': 1e-4, 'warmup_ratio': 0.03, 'scheduler': 'cosine',
    'weight_decay': 0.0, 'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 1, 'micro_batch_size': 1,
    'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': 0.0, 'lora_target': 'qv',
    'max_frames': 8, 'max_pixels': 224**2, 'use_transcript': True,
    'eval_modes': ['logit', 'greedy'], 'selection_mode': 'logit',
    'eval_every_fraction': 0.5, 'max_new_tokens': 8,
    'train_limit': None, 'dev_limit': None, 'num_threads': 2,
    'objective': 'sft', 'teacher_checkpoint': None, 'teacher_cache': None,
    'distillation_alpha': 0.5, 'distillation_temperature': 2.0,
}
FEATURE_DEFAULTS = {'feature_teacher_cache': None, 'feature_audio_weight': 0., 'feature_vision_weight': 0.}
OPTIMIZER_DEFAULTS = {'encoder_learning_rate': None}
PARTITION_DEFAULTS = {'mer_partition': None}
KD_SCHEDULE_DEFAULTS = {'distillation_alpha_final': None, 'distillation_decay_start': 0.2}
# Optional: absent keys must not alter historical configuration fingerprints.
KD_DISAGREEMENT_DEFAULTS = {'distillation_disagreement_alpha': None}
RESUME_MUTABLE = {'epochs', 'num_threads', 'cache_dir', 'micro_batch_size', 'gradient_accumulation_steps', 'gradient_checkpointing'}


def load_config(path):
    with Path(path).open() as stream:
        supplied = yaml.safe_load(stream)
    if not isinstance(supplied, dict):
        raise ValueError('Configuration must be a YAML mapping')
    unknown = set(supplied) - (set(DEFAULTS) | set(FEATURE_DEFAULTS) | set(OPTIMIZER_DEFAULTS) | set(PARTITION_DEFAULTS) | set(KD_SCHEDULE_DEFAULTS) | set(KD_DISAGREEMENT_DEFAULTS))
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    config = DEFAULTS | supplied
    # Preserve old configs/fingerprints exactly when feature KD is not requested.
    if set(supplied) & set(FEATURE_DEFAULTS):
        config = FEATURE_DEFAULTS | config
    if set(supplied) & set(KD_SCHEDULE_DEFAULTS):
        config = KD_SCHEDULE_DEFAULTS | config
        if config['objective'] != 'sft_kd': raise ValueError('Scheduled KD requires sft_kd')
        for key in KD_SCHEDULE_DEFAULTS:
            value=config[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(key+' must be finite and numeric')
        if not 0 <= config['distillation_alpha_final'] <= 1 or not 0 <= config['distillation_decay_start'] < 1:
            raise ValueError('KD final weight must be in [0,1]; decay start in [0,1)')
    if 'distillation_disagreement_alpha' in supplied:
        value = config['distillation_disagreement_alpha']
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('distillation_disagreement_alpha must be finite and in [0,1]')
        if config['objective'] != 'sft_kd':
            raise ValueError('Teacher-disagreement KD requires sft_kd')
        if set(supplied) & set(KD_SCHEDULE_DEFAULTS):
            raise ValueError('Teacher-disagreement KD cannot be combined with a KD schedule')
    if 'encoder_learning_rate' in supplied:
        value = config['encoder_learning_rate']
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('encoder_learning_rate must be finite and positive when supplied')
        if config['lora_target'] not in ('decoder_encoder_ffn', 'decoder_encoder_linear'):
            raise ValueError('encoder_learning_rate requires trainable encoder adapters')
    if 'mer_partition' in supplied and (config['task'] != 'mer6' or config['mer_partition'] != 'nested_90_10'):
        raise ValueError('mer_partition must be nested_90_10 and requires task mer6')
    if config['task'] not in ('mintrec2', 'mintrec20', 'mer6'):
        raise ValueError('task must be mintrec2, mintrec20, or mer6')
    for key in ('epochs', 'gradient_accumulation_steps', 'micro_batch_size',
                'lora_rank', 'lora_alpha', 'max_frames', 'max_pixels', 'max_new_tokens', 'num_threads'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'split_seed'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if config['max_new_tokens'] > 128:
        raise ValueError('max_new_tokens must be at most 128')
    if config['max_frames'] % 2 or not 2 <= config['max_frames'] <= 32:
        raise ValueError('max_frames must be even and between 2 and 32')
    if config['max_pixels'] < 56**2:
        raise ValueError('max_pixels must be at least 56**2')
    for key in ('learning_rate', 'warmup_ratio', 'weight_decay', 'max_grad_norm', 'lora_dropout', 'eval_every_fraction'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite and numeric')
    if config['learning_rate'] <= 0 or config['max_grad_norm'] <= 0 or config['weight_decay'] < 0:
        raise ValueError('learning_rate/max_grad_norm must be positive; weight_decay nonnegative')
    if not 0 <= config['warmup_ratio'] < 1 or not 0 <= config['lora_dropout'] < 1:
        raise ValueError('warmup_ratio/lora_dropout must be in [0, 1)')
    if not 0 < config['eval_every_fraction'] <= 1:
        raise ValueError('eval_every_fraction must be in (0, 1]')
    if config['scheduler'] not in ('cosine', 'constant'):
        raise ValueError('scheduler must be cosine or constant')
    if config['lora_target'] not in ('qv', 'decoder_linear', 'decoder_encoder_ffn', 'decoder_encoder_linear'):
        raise ValueError('Unknown LoRA target scope')
    for key in ('use_transcript', 'deterministic', 'gradient_checkpointing'):
        if type(config[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    modes = config['eval_modes']
    if isinstance(modes, str):
        modes = ['logit', 'greedy'] if modes == 'both' else [modes]
    if not isinstance(modes, list) or not modes or any(m not in ('logit', 'greedy') for m in modes) or len(modes) != len(set(modes)):
        raise ValueError('eval_modes must select logit, greedy, or both without duplicates')
    config['eval_modes'] = modes
    if config['selection_mode'] not in modes:
        raise ValueError('selection_mode must be enabled in eval_modes')
    for key in ('train_limit', 'dev_limit'):
        if config[key] is not None and (type(config[key]) is not int or config[key] < 1):
            raise ValueError(f'{key} must be null or a positive integer')
    for key in ('model', 'data_root', 'cache_dir'):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f'{key} must be a nonempty path')
        path = Path(config[key]).expanduser()
        config[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    if config['objective'] not in ('sft', 'sft_kd'):
        raise ValueError('objective must be sft or sft_kd')
    for key in ('distillation_alpha', 'distillation_temperature'):
        value=config[key]
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite and numeric')
    if not 0 <= config['distillation_alpha'] <= 1 or config['distillation_temperature'] <= 0:
        raise ValueError('Distillation alpha must be in [0,1] and temperature positive')
    for key in ('teacher_checkpoint','teacher_cache'):
        value=config[key]
        if value is None and config['objective']=='sft':
            continue
        if not isinstance(value,str) or not value:
            raise ValueError(f'{key} is required for sft_kd')
        path=Path(value).expanduser()
        config[key]=str((ROOT/path).resolve() if not path.is_absolute() else path.resolve())
    if set(config) & set(FEATURE_DEFAULTS):
        for key in ('feature_audio_weight','feature_vision_weight'):
            v=config[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
                raise ValueError(key+' must be finite and nonnegative')
        enabled=config['feature_audio_weight']>0 or config['feature_vision_weight']>0
        if not enabled or config['lora_target'] not in ('decoder_encoder_ffn','decoder_encoder_linear'):
            raise ValueError('Feature KD requires positive weights and trainable encoder adapters')
        value=config['feature_teacher_cache']
        if not isinstance(value,str) or not value:raise ValueError('Feature cache path is required')
        path=Path(value).expanduser()
        config['feature_teacher_cache']=str((ROOT/path).resolve() if not path.is_absolute() else path.resolve())
    return config


def config_fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_resume(saved, current, saved_world_size, world_size):
    defaults = DEFAULTS | FEATURE_DEFAULTS | OPTIMIZER_DEFAULTS | PARTITION_DEFAULTS | KD_SCHEDULE_DEFAULTS | KD_DISAGREEMENT_DEFAULTS
    changed = [key for key in defaults if key not in RESUME_MUTABLE and saved.get(key, defaults[key]) != current.get(key, defaults[key])]
    if changed:
        raise ValueError(f'Full resume cannot change {changed}; use --init-adapter in a new run for a new recipe')
    if saved['micro_batch_size'] * saved['gradient_accumulation_steps'] != current['micro_batch_size'] * current['gradient_accumulation_steps']:
        raise ValueError('Full resume must preserve the effective batch size')
    if world_size != saved_world_size:
        raise ValueError('Full resume requires the original world size; use --init-adapter to change it')
    if current['epochs'] < saved['epochs']:
        raise ValueError('Cannot reduce the epoch budget on full resume')


def epoch_order(count, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def rank_update(order, update_index, world_size, rank, accumulation):
    """Unpadded data sharding; None marks a zero-weight synchronization forward."""
    start = update_index * world_size * accumulation
    valid_count = min(world_size * accumulation, max(0, len(order) - start))
    records = []
    for micro in range(accumulation):
        position = start + micro * world_size + rank
        records.append(order[position] if position < len(order) else None)
    return records, valid_count


def make_schedule(config, total_steps, start_step=0):
    return {'start_step': start_step, 'end_step': total_steps,
            'warmup_steps': math.ceil((total_steps-start_step)*config['warmup_ratio']),
            'kind': config['scheduler'], 'peak_lr': config['learning_rate']}


def learning_rate(schedule, step):
    offset = step - schedule['start_step']
    length = schedule['end_step'] - schedule['start_step']
    if length <= 0 or offset < 0:
        raise ValueError('Invalid learning-rate schedule phase')
    warmup = schedule['warmup_steps']
    if offset < warmup:
        multiplier = (offset + 1) / warmup
    elif schedule['kind'] == 'constant':
        multiplier = 1.0
    else:
        progress = min(1.0, (offset - warmup) / max(1, length - warmup))
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule['peak_lr'] * multiplier


def configure_determinism(config):
    """Set CUDA reproducibility options before CUDA contexts are initialized."""
    import os
    import torch

    deterministic = config.get('deterministic', True)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def lora_targets(scope):
    """Exact PEFT module scope, shared by fresh and resumed training."""
    decoder = r'model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
    if scope == 'qv':
        return r'model\.layers\.\d+\.self_attn\.(q_proj|v_proj)'
    if scope == 'decoder_linear':
        return decoder
    if scope == 'decoder_encoder_ffn':
        return (r'(' + decoder + r'|audio_tower\.layers\.\d+\.(fc1|fc2)'
                r'|visual\.blocks\.\d+\.mlp\.(gate_proj|up_proj|down_proj))')
    if scope == 'decoder_encoder_linear':
        return (r'(' + decoder
                + r'|audio_tower\.layers\.\d+\.(fc1|fc2|self_attn\.(q_proj|k_proj|v_proj|out_proj))'
                r'|audio_tower\.proj|visual\.blocks\.\d+\.(mlp\.(gate_proj|up_proj|down_proj)|attn\.(q|k|v|proj))'
                r'|visual\.merger\.mlp\.(0|2))')
    raise ValueError('Unknown LoRA scope: ' + str(scope))


def optimizer_groups(model, config):
    """Keep the legacy single group unless an encoder LR is explicitly requested.

    Group order and membership are stable across checkpoint reloads. Scales apply
    the same warmup/cosine multiplier to the specified component peak rates.
    """
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if config.get('encoder_learning_rate') is None:
        return [{'params': [p for _, p in named]}]
    groups = {name: [] for name in ('language', 'audio_encoder', 'vision_encoder')}
    for name, p in named:
        parts = name.split('.')
        component = ('audio_encoder' if 'audio_tower' in parts else
                     'vision_encoder' if 'visual' in parts else 'language')
        groups[component].append(p)
    if not all(groups.values()):
        raise ValueError('Split LR requires trainable LM, audio and vision parameters')
    ratio = config['encoder_learning_rate'] / config['learning_rate']
    return [{'params': params, 'component': name, 'lr_scale': 1. if name == 'language' else ratio}
            for name, params in groups.items()]


def set_optimizer_learning_rate(optimizer, rate):
    for group in optimizer.param_groups:
        group['lr'] = rate * group.get('lr_scale', 1.)


def distillation_weight(config, step, total_steps):
    """Hold initial KD weight, then linearly anneal by absolute optimizer step."""
    initial=config['distillation_alpha']
    final=config.get('distillation_alpha_final')
    if final is None: return initial
    if total_steps < 1 or step < 0: raise ValueError('Invalid KD schedule step')
    progress=min(1.,step/max(1,total_steps-1))
    start=config.get('distillation_decay_start',.2)
    fraction=max(0.,min(1.,(progress-start)/(1-start)))
    return initial+(final-initial)*fraction
