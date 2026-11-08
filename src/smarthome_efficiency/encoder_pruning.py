"""Prune encoder FFNs only, preserving the recovered language model exactly.

Uses the existing activation collector unchanged, so previously pinned training-
only calibration caches remain valid. Output dimensions and all attention/layer
structures are preserved. Limits apply after merging recovery LoRA adapters.
"""
from __future__ import annotations
import torch
from .pruning import _language, _slice_linear, select_top_channels
from .multimodal_pruning import component_ffns, output_projection, slice_output, is_prunable_ffn_tensor

ENCODERS = ('audio_encoder', 'vision_encoder')
LANGUAGE_LIMIT = 2_000_000_000
ENCODER_LIMIT = 1_000_000_000


def encoder_parameters(counts):
    return counts['audio_encoder'] + counts['vision_encoder']


def validate_separate_budget(before, after):
    """Inclusive agreement limits, plus unchanged LM and actual encoder reduction."""
    if (not 0 < after['language'] <= LANGUAGE_LIMIT
            or not 0 < encoder_parameters(after) <= ENCODER_LIMIT
            or after['language'] != before['language']
            or encoder_parameters(after) >= encoder_parameters(before)
            or after['other'] != before['other']):
        raise ValueError('Encoder-only export must preserve LM <=2B and reduce combined encoders to <=1B')


def is_prunable_encoder_tensor(name):
    return name.startswith(('audio_tower.', 'visual.')) and is_prunable_ffn_tensor(name)


@torch.no_grad()
def prune_encoder_ffns(model, audio_width, vision_width, scores, expected_lm_width=None):
    language = _language(model)
    if expected_lm_width is not None and (
            language.config.intermediate_size != expected_lm_width
            or any(l.mlp.down_proj.in_features != expected_lm_width for l in language.layers)):
        raise ValueError('Encoder-only pruning cannot change the language FFN width')
    widths = dict(zip(ENCODERS, (audio_width, vision_width)))
    components = component_ffns(model)
    selections = {}
    # Validate both encoder selections before mutating either encoder.
    for key in ENCODERS:
        ffns = components[key]
        values = scores.get('components', {}).get(key, {}).get('ffn_activation', {})
        if set(values) != {str(i) for i in range(len(ffns))}:
            raise ValueError('Incomplete encoder FFN scores: '+key)
        selections[key] = {}
        for i, ffn in enumerate(ffns):
            output = output_projection(key, ffn)
            vector = torch.tensor(values[str(i)], dtype=torch.float64)
            if vector.shape != (output.in_features,) or (vector < 0).any():
                raise ValueError('Encoder activation score shape/range mismatch')
            selections[key][str(i)] = select_top_channels(vector, widths[key])
            if key == 'audio_encoder':
                if (ffn.fc1.weight.shape != (output.in_features, output.out_features)
                        or ffn.fc1.bias is None or output.bias is None):
                    raise ValueError('Unsupported audio FFN structure')
            elif (ffn.gate_proj.weight.shape != ffn.up_proj.weight.shape
                    or ffn.gate_proj.weight.shape != (output.in_features, output.out_features)):
                raise ValueError('Unsupported vision gated FFN structure')
    protected = [(n, id(p), p.data_ptr(), p._version) for n,p in model.named_parameters()
                 if not is_prunable_encoder_tensor(n)]
    for key in ENCODERS:
        for i,ffn in enumerate(components[key]):
            selected = selections[key][str(i)]
            if key == 'audio_encoder':
                ffn.fc1 = _slice_linear(ffn.fc1, selected, 0)
                ffn.fc2 = slice_output(ffn.fc2, selected)
            else:
                ffn.gate_proj = _slice_linear(ffn.gate_proj, selected, 0)
                ffn.up_proj = _slice_linear(ffn.up_proj, selected, 0)
                ffn.down_proj = slice_output(ffn.down_proj, selected)
                ffn.intermediate_size = vision_width
    model.audio_tower.config.encoder_ffn_dim = audio_width
    model.config.audio_config.encoder_ffn_dim = audio_width
    model.visual.config.intermediate_size = vision_width
    model.config.vision_config.intermediate_size = vision_width
    if protected != [(n,id(p),p.data_ptr(),p._version) for n,p in model.named_parameters()
                     if not is_prunable_encoder_tensor(n)]:
        raise RuntimeError('Encoder pruning modified the language model or a protected interface')
    return {'kept_ffn_channel_indices_by_component': selections,
            'selection_rule': 'encoder-only per-layer sqrt(sum_clips(mean_valid_tokens(abs(post_activation_FFN))**2))',
            'tie_break': 'lower original channel index',
            'importance_source': 'merged task-specific KD075 source; prompt-only training clips',
            'language_tensors_unchanged': True, 'av_interfaces_unchanged': True,
            'non_ffn_tensors_unchanged': True, 'all_layer_counts_unchanged': True}
