"""Structured language pruning: selectors are separate from physical surgery.

Naive and calibrated selectors share validated physical layer/channel surgery.
Future gradient scores can use the same operations and checkpoint exporter.
Only the Thinker's language decoder is modified; its AV interfaces stay fixed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

FORMAT_VERSION = 1
METHODS = ('depth_uniform', 'ffn_magnitude', 'depth_bi', 'ffn_activation', 'depth_ffn_activation', 'multimodal_ffn_activation', 'multimodal_depth_ffn_activation', 'encoder_ffn_activation')
PROCESSOR_FILES = (
    'preprocessor_config.json', 'video_preprocessor.json', 'feature_extractor_config.json',
    'processor_config.json', 'tokenizer.json',
    'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
    'vocab.json', 'merges.txt', 'chat_template.json', 'chat_template.jinja',
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def processor_hashes(path: Path) -> dict[str, str]:
    path = Path(path)
    return {name: sha256_file(path / name) for name in PROCESSOR_FILES if (path / name).is_file()}


def processor_cache_identity(path: Path) -> dict:
    """Reuse original prepared inputs only for byte-identical processor assets.

    Pruning changes decoder weights/config, not tokenization or media processing.
    Old snapshots retain their exact previous cache identity and cache format.
    The exporter copies processor files and records their hashes explicitly.
    """
    path = Path(path).resolve()
    original = {'snapshot_path': str(path), 'config_sha256': sha256_file(path / 'config.json')}
    sidecar = path / 'compression.json'
    if not sidecar.is_file():
        return original
    manifest = json.loads(sidecar.read_text())
    if manifest.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported compact checkpoint manifest version')
    expected = manifest.get('processor_files_sha256')
    identity = manifest.get('processor_cache_identity')
    if not expected or processor_hashes(path) != expected:
        raise ValueError('Compact checkpoint processor files differ from the source processor')
    if not isinstance(identity, dict) or set(identity) != {'snapshot_path', 'config_sha256'}:
        raise ValueError('Missing original processor cache identity')
    return identity


def _language(model):
    if any(hasattr(module, 'lora_A') for module in model.modules()):
        raise ValueError('Merge and unload existing LoRA adapters before structural pruning')
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers') or not hasattr(model, 'lm_head'):
        raise TypeError('Expected a Qwen Thinker with model.layers and lm_head')
    return model.model


def parameter_counts(model) -> dict[str, int]:
    _language(model)
    components = {'language': 0, 'audio_encoder': 0, 'vision_encoder': 0, 'other': 0}
    for name, parameter in model.named_parameters():
        key = ('language' if name.startswith(('model.', 'lm_head.')) else
               'audio_encoder' if name.startswith('audio_tower.') else
               'vision_encoder' if name.startswith('visual.') else 'other')
        components[key] += parameter.numel()
    return components | {'total': sum(components.values())}


def evenly_spaced_layers(total: int, keep: int) -> list[int]:
    """Keep endpoints and spread remaining layers uniformly; ties round upward."""
    if type(total) is not int or type(keep) is not int or not 2 <= keep <= total:
        raise ValueError('Layer count must satisfy 2 <= keep <= total')
    return [(2 * i * (total - 1) + keep - 1) // (2 * (keep - 1)) for i in range(keep)]


def _indices(values: Sequence[int], size: int) -> list[int]:
    result = list(values)
    if (not result or any(type(v) is not int or not 0 <= v < size for v in result)
            or result != sorted(set(result))):
        raise ValueError('Retained indices must be nonempty, unique, increasing, and in range')
    return result


@torch.no_grad()
def prune_layers(model, keep: Sequence[int]) -> dict:
    language = _language(model)
    layers = language.layers
    keep = _indices(keep, len(layers))
    # Renumber caches as well as ModuleList names. Preserve original order.
    language.layers = nn.ModuleList([layers[i] for i in keep])
    for index, layer in enumerate(language.layers):
        layer.self_attn.layer_idx = index
    language.config.num_hidden_layers = len(keep)
    model.config.text_config.num_hidden_layers = len(keep)
    return {'kept_original_layer_indices': keep}


@torch.no_grad()
def magnitude_scores(mlp) -> torch.Tensor:
    """Squared L2 norm of all weights connected to each SwiGLU channel.

    score[j] = ||gate[j,:]||² + ||up[j,:]||² + ||down[:,j]||².
    Compute in FP32; sqrt is unnecessary for ranking. No calibration data.
    """
    gate, up, down = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError('Unsupported FFN: expected coupled gate/up/down projections')
    scores = gate.float().square().sum(1)
    scores += up.float().square().sum(1)
    scores += down.float().square().sum(0)
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite channel importance scores')
    return scores


def select_top_channels(scores: torch.Tensor, keep: int) -> list[int]:
    if scores.ndim != 1 or type(keep) is not int or not 1 <= keep <= scores.numel():
        raise ValueError('Invalid channel count or importance score shape')
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite channel importance scores')
    # Stable ties favor lower original indices; retained channels keep their order.
    return scores.argsort(descending=True, stable=True)[:keep].sort().values.cpu().tolist()


def _slice_linear(module: nn.Linear, keep: Sequence[int], dimension: int) -> nn.Linear:
    indices = torch.tensor(keep, device=module.weight.device, dtype=torch.long)
    weight = module.weight.index_select(dimension, indices).contiguous()
    # Meta construction avoids random initialization and does not consume RNG.
    result = nn.Linear(weight.shape[1], weight.shape[0], bias=module.bias is not None,
                       device='meta', dtype=weight.dtype)
    result.weight = nn.Parameter(weight, requires_grad=module.weight.requires_grad)
    if module.bias is not None:
        bias = module.bias.index_select(0, indices) if dimension == 0 else module.bias.clone()
        result.bias = nn.Parameter(bias, requires_grad=module.bias.requires_grad)
    result.train(module.training)
    return result


@torch.no_grad()
def prune_ffn_channels(model, kept_by_layer: Mapping[int, Sequence[int]]) -> dict:
    """Apply a selector's channel lists; uniform width keeps native config loading."""
    language = _language(model)
    if set(kept_by_layer) != set(range(len(language.layers))):
        raise ValueError('Provide retained channels for every decoder layer')
    validated = {i: _indices(kept_by_layer[i], layer.mlp.gate_proj.out_features)
                 for i, layer in enumerate(language.layers)}
    widths = {len(indices) for indices in validated.values()}
    if len(widths) != 1:
        raise ValueError('This exporter currently requires uniform FFN width across layers')
    # Validate every layer before changing any modules.
    for layer in language.layers:
        mlp = layer.mlp
        shape = mlp.gate_proj.weight.shape
        if mlp.up_proj.weight.shape != shape or mlp.down_proj.weight.shape != (shape[1], shape[0]):
            raise ValueError('Unsupported FFN projection shapes')
    width = widths.pop()
    for index, layer in enumerate(language.layers):
        mlp, keep = layer.mlp, validated[index]
        mlp.gate_proj = _slice_linear(mlp.gate_proj, keep, 0)
        mlp.up_proj = _slice_linear(mlp.up_proj, keep, 0)
        mlp.down_proj = _slice_linear(mlp.down_proj, keep, 1)
        mlp.intermediate_size = width
    language.config.intermediate_size = width
    model.config.text_config.intermediate_size = width
    return {'kept_original_ffn_channel_indices': {str(i): indices for i, indices in validated.items()}}


def apply_pruning(model, method: str, *, keep_layers: int | None = None,
                  ffn_width: int | None = None, importance: dict | None = None,
                  audio_ffn_width: int | None = None, vision_ffn_width: int | None = None,
                  audio_layers: int | None = None, vision_layers: int | None = None) -> dict:
    """Dispatch selection and surgery without depending on training or data code."""
    language = _language(model)
    if method == 'multimodal_depth_ffn_activation':
        from .hybrid_pruning import prune_hybrid
        if keep_layers is not None or importance is None:
            raise ValueError('Hybrid pruning preserves LM depth and requires importance scores')
        return prune_hybrid(model, {'language':ffn_width,'audio_encoder':audio_ffn_width,'vision_encoder':vision_ffn_width},
                            importance, audio_layers, vision_layers)
    if audio_layers is not None or vision_layers is not None:
        raise ValueError('Encoder layer budgets require multimodal_depth_ffn_activation')
    if method == 'encoder_ffn_activation':
        from .encoder_pruning import prune_encoder_ffns
        if keep_layers is not None or importance is None:
            raise ValueError('Encoder-only pruning requires scores and preserves every language layer')
        return prune_encoder_ffns(model, audio_ffn_width, vision_ffn_width, importance, ffn_width)
    if method == 'multimodal_ffn_activation':
        from .multimodal_pruning import prune_multimodal_ffns
        if keep_layers is not None or importance is None:
            raise ValueError('Multimodal FFN pruning requires scores and preserves every layer')
        return prune_multimodal_ffns(model, {'language': ffn_width, 'audio_encoder': audio_ffn_width,
                                           'vision_encoder': vision_ffn_width}, importance)
    if audio_ffn_width is not None or vision_ffn_width is not None:
        raise ValueError('Encoder widths are only valid for multimodal FFN pruning')
    if method == 'depth_uniform':
        if ffn_width is not None or keep_layers is None:
            raise ValueError('depth_uniform requires keep_layers only')
        selected = evenly_spaced_layers(len(language.layers), keep_layers)
        return {'selection_rule': 'evenly spaced, first/last retained, nearest integer ties upward',
                **prune_layers(model, selected)}
    if method == 'ffn_magnitude':
        if keep_layers is not None or ffn_width is None:
            raise ValueError('ffn_magnitude requires ffn_width only')
        selected = {i: select_top_channels(magnitude_scores(layer.mlp), ffn_width)
                    for i, layer in enumerate(language.layers)}
        return {'selection_rule': 'sum of squared gate row, up row, and down column L2 norms; per layer',
                'tie_break': 'lower original channel index', **prune_ffn_channels(model, selected)}
    if method in ('depth_bi', 'ffn_activation', 'depth_ffn_activation'):
        depth = method in ('depth_bi', 'depth_ffn_activation')
        width = method in ('ffn_activation', 'depth_ffn_activation')
        if ((depth and (type(keep_layers) is not int or not 2 <= keep_layers <= len(language.layers)))
                or (not depth and keep_layers is not None) or (width and ffn_width is None)
                or (not width and ffn_width is not None) or importance is None):
            raise ValueError('Calibrated pruning requires matching dimensions and importance scores')
        total = len(language.layers)
        bi = torch.tensor(importance['layer_bi'], dtype=torch.float64)
        if bi.shape != (total,) or not torch.isfinite(bi).all():
            raise ValueError('Invalid layer BI scores')
        layers = select_top_channels(bi, keep_layers) if depth else list(range(total))
        # Validate all selections before modifying the model. For a hybrid, use
        # scores from the same intact source, then retain original channel IDs.
        channels = {i: select_top_channels(torch.tensor(importance['ffn_activation'][str(i)], dtype=torch.float64), ffn_width)
                    for i in layers} if width else {}
        if width and any(len(importance['ffn_activation'][str(i)]) != language.layers[i].mlp.down_proj.in_features for i in layers):
            raise ValueError('FFN activation score width mismatch')
        result = {'selection_rule': 'ShortGPT token-mean BI layers; Minitron L2-over-examples mean-absolute-sequence gated FFN activations',
                  'tie_break': 'lower original index', 'importance_source': 'same intact merged task teacher, prompt-only training clips'}
        if depth:
            result.update(prune_layers(model, layers))
        if width:
            selected = {new: channels[original] for new, original in enumerate(layers)}
            surgery = prune_ffn_channels(model, selected)
            result['kept_ffn_channel_indices_by_compact_layer'] = surgery['kept_original_ffn_channel_indices']
            result['kept_original_ffn_channel_indices'] = {str(i): channels[i] for i in layers}
        return result
    raise ValueError(f'Unknown pruning method: {method}')
