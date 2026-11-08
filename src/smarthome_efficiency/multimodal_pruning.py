"""Activation FFN pruning of all Thinker components with fixed AV interfaces.

The language and vision FFNs are gated; audio uses GELU and two biased linears.
Scores follow the existing Minitron-style per-clip mean-absolute / example-L2
reduction, extended to native packed (already unpadded) encoder token sequences.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

import torch

from .pruning import _language, _slice_linear, select_top_channels, sha256_file
from .pruning_calibration import calibration_spec

COMPONENTS = ('language', 'audio_encoder', 'vision_encoder')
BUDGETS = {
    'A_LM1536_A1024_V1024': (1536, 1024, 1024),
    'B_LM1280_A1536_V1024': (1280, 1536, 1024),
    'C_LM1024_A1536_V1536': (1024, 1536, 1536),
}


def expected_counts(lm_width, audio_width, vision_width):
    """Exact Qwen2.5-Omni-3B Thinker counts, including all FFN biases."""
    widths = (lm_width, audio_width, vision_width)
    if any(type(v) is not int or v < 1 for v in widths):
        raise ValueError('FFN widths must be positive integers')
    counts = {'language': 962_310_144 + 221_184 * lm_width,
              'audio_encoder': 218_082_304 + 81_952 * audio_width,
              'vision_encoder': 248_215_808 + 122_944 * vision_width, 'other': 0}
    return counts | {'total': sum(counts.values())}


def component_ffns(model):
    language = _language(model)
    return {'language': [layer.mlp for layer in language.layers],
            'audio_encoder': list(model.audio_tower.layers),
            'vision_encoder': [block.mlp for block in model.visual.blocks]}


def output_projection(component, ffn):
    return ffn.fc2 if component == 'audio_encoder' else ffn.down_proj


def slice_output(module, selected):
    result = _slice_linear(module, selected, 1)
    if module.bias is not None:
        result.bias = module.bias  # Output channels are unchanged; preserve bias identity.
    return result


def is_prunable_ffn_tensor(name):
    """Only these tensors may change; output biases and all AV interfaces stay fixed."""
    import re
    return bool(re.fullmatch(
        r'(model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight'
        r'|audio_tower\.layers\.\d+\.(fc1\.(weight|bias)|fc2\.weight)'
        r'|visual\.blocks\.\d+\.mlp\.((gate_proj|up_proj)\.(weight|bias)|down_proj\.weight))', name))


@torch.no_grad()
def prune_multimodal_ffns(model, widths, scores):
    """Validate all three components before any physical surgery takes place."""
    if set(widths) != set(COMPONENTS) or set(scores['components']) != set(COMPONENTS):
        raise ValueError('Provide widths and activation scores for all three components')
    components = component_ffns(model)
    selections = {}
    for component, ffns in components.items():
        values = scores['components'][component]['ffn_activation']
        if set(values) != {str(i) for i in range(len(ffns))}:
            raise ValueError('Incomplete FFN score coverage: ' + component)
        selections[component] = {}
        for i, ffn in enumerate(ffns):
            output = output_projection(component, ffn)
            vector = torch.tensor(values[str(i)], dtype=torch.float64)
            if vector.shape != (output.in_features,) or (vector < 0).any():
                raise ValueError('FFN score shape/range mismatch: ' + component)
            selected = select_top_channels(vector, widths[component])
            if component == 'audio_encoder':
                if (ffn.fc1.weight.shape != (output.in_features, output.out_features)
                        or ffn.fc1.bias is None or output.bias is None):
                    raise ValueError('Unsupported audio FFN structure')
            elif (ffn.gate_proj.weight.shape != ffn.up_proj.weight.shape
                  or ffn.gate_proj.weight.shape != (output.in_features, output.out_features)):
                raise ValueError('Unsupported gated FFN structure')
            selections[component][str(i)] = selected
    for component, ffns in components.items():
        for i, ffn in enumerate(ffns):
            selected = selections[component][str(i)]
            if component == 'audio_encoder':
                ffn.fc1 = _slice_linear(ffn.fc1, selected, 0)
                ffn.fc2 = slice_output(ffn.fc2, selected)
            else:
                ffn.gate_proj = _slice_linear(ffn.gate_proj, selected, 0)
                ffn.up_proj = _slice_linear(ffn.up_proj, selected, 0)
                ffn.down_proj = slice_output(ffn.down_proj, selected)
                ffn.intermediate_size = widths[component]
    model.model.config.intermediate_size = widths['language']
    model.config.text_config.intermediate_size = widths['language']
    model.audio_tower.config.encoder_ffn_dim = widths['audio_encoder']
    model.config.audio_config.encoder_ffn_dim = widths['audio_encoder']
    model.visual.config.intermediate_size = widths['vision_encoder']
    model.config.vision_config.intermediate_size = widths['vision_encoder']
    return {'kept_ffn_channel_indices_by_component': selections,
            'selection_rule': 'per component/layer sqrt(sum_clips(mean_valid_tokens(abs(post_activation_FFN))**2))',
            'tie_break': 'lower original channel index',
            'importance_source': 'merged task-specific KD075 source; prompt-only training clips',
            'av_interfaces_unchanged': True, 'all_layer_counts_unchanged': True}


class MultimodalImportanceCollector:
    """One prompt-only clip per forward; no padded encoder tokens or retained activations."""
    def __init__(self, model):
        self.ffns = component_ffns(model)
        self.handles = []
        self.squares = {key: [torch.zeros(output_projection(key, ffn).in_features, dtype=torch.float64)
                              for ffn in ffns] for key, ffns in self.ffns.items()}
        self.examples = {key: [0]*len(ffns) for key, ffns in self.ffns.items()}
        self.tokens = {key: [0]*len(ffns) for key, ffns in self.ffns.items()}

    def __enter__(self):
        for key, ffns in self.ffns.items():
            for i, ffn in enumerate(ffns):
                self.handles.append(output_projection(key, ffn).register_forward_pre_hook(self._hook(key, i)))
        return self

    def _hook(self, key, i):
        def hook(module, args):
            value = args[0].detach().float()
            if key == 'language':
                if value.ndim != 3 or value.shape[0] != 1:
                    raise ValueError('Expected one unpadded language clip')
                value = value[0]
            # Native audio removes CNN padding before encoder layers. Native
            # vision packs real patches only. Never flatten a padded batch here.
            if value.ndim != 2 or value.shape[0] < 1 or not torch.isfinite(value).all():
                raise ValueError('Expected finite packed valid encoder/LM tokens: ' + key)
            self.squares[key][i] += value.abs().mean(0).square().double().cpu()
            self.examples[key][i] += 1
            self.tokens[key][i] += value.shape[0]
        return hook

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def scores(self, expected_examples):
        components = {}
        for key in COMPONENTS:
            if (self.examples[key] != [expected_examples]*len(self.ffns[key])
                    or not all(self.tokens[key]) or len(set(self.tokens[key])) != 1):
                raise ValueError('Incomplete/unequal multimodal calibration coverage: ' + key)
            components[key] = {
                'ffn_activation': {str(i): value.sqrt().tolist() for i, value in enumerate(self.squares[key])},
                'examples_per_layer': self.examples[key], 'tokens_per_layer': self.tokens[key]}
        return {'components': components, 'encoder_tokens': 'native packed valid tokens, padding removed before layers',
                'rule': 'sqrt(sum_clips(mean_valid_tokens(abs(post_activation_FFN))**2))'}


def validate_multimodal_calibration(payload, spec, model):
    if payload.get('status') != 'completed' or payload.get('spec') != spec:
        raise ValueError('Stale/mismatched multimodal calibration provenance')
    scores = payload['scores']
    if set(scores['components']) != set(COMPONENTS):
        raise ValueError('Missing multimodal calibration components')
    for key, ffns in component_ffns(model).items():
        values = scores['components'][key]
        if (set(values['ffn_activation']) != {str(i) for i in range(len(ffns))}
                or values['examples_per_layer'] != [spec['examples']]*len(ffns)):
            raise ValueError('Multimodal calibration layer/example coverage mismatch')
        counts = values['tokens_per_layer']
        if len(counts) != len(ffns) or not all(type(v) is int and v > 0 for v in counts) or len(set(counts)) != 1:
            raise ValueError('Multimodal calibration token coverage mismatch')
        for i, ffn in enumerate(ffns):
            vector = torch.tensor(values['ffn_activation'][str(i)], dtype=torch.float64)
            if (vector.shape != (output_projection(key, ffn).in_features,)
                    or not torch.isfinite(vector).all() or (vector < 0).any()):
                raise ValueError('Invalid multimodal channel scores')
    return scores


def obtain_multimodal_calibration(runner, cfg, source, records, seed, cache_path, progress=None):
    from .checkpointing import write_json
    spec = calibration_spec(cfg, source, runner, records, seed)
    spec.update(scope='all_thinker_components', multimodal_collector_sha256=sha256_file(Path(__file__)))
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            scores = validate_multimodal_calibration(payload, spec, runner.model)
            return scores, {'path': str(cache_path.resolve()), 'sha256': sha256_file(cache_path), 'spec': spec, 'reused': True}
        with MultimodalImportanceCollector(runner.model) as collector:
            for index, record in enumerate(records):
                batch = runner.prepare(record)
                if batch['input_ids'].shape[0] != 1 or not torch.all(batch['attention_mask'] == 1):
                    raise ValueError('One unpadded training clip required for calibration')
                runner.score(batch)
                if progress and ((index+1) % 16 == 0 or index+1 == len(records)):
                    progress(index+1, len(records))
        payload = {'status': 'completed', 'spec': spec, 'scores': collector.scores(len(records))}
        scores = validate_multimodal_calibration(payload, spec, runner.model)
        write_json(cache_path, payload)
        return scores, {'path': str(cache_path.resolve()), 'sha256': sha256_file(cache_path), 'spec': spec, 'reused': False}


def adapter_gradient_coverage(model):
    """Reject silently frozen/disconnected newly pruned encoders on the first update."""
    totals = {key: 0. for key in COMPONENTS}
    tensors = {key: 0 for key in COMPONENTS}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'lora_' not in name:
            raise ValueError('Only LoRA parameters may be trained')
        key = ('audio_encoder' if 'audio_tower.' in name else
               'vision_encoder' if 'visual.' in name else 'language')
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise ValueError('Missing or nonfinite LoRA gradient: ' + name)
        totals[key] += float(gradient.detach().float().square().sum())
        tensors[key] += 1
    if any(not tensors[key] or totals[key] <= 0 for key in COMPONENTS):
        raise ValueError('Disconnected adapter component: ' + str(totals))
    return {key: {'gradient_l2': totals[key]**.5, 'tensors_with_gradient': tensors[key]} for key in COMPONENTS}
