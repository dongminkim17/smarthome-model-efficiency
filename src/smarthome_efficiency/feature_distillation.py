"""Training-only frozen encoder feature caches and token-aligned feature KD.

Final encoder outputs keep the same token layout and 2048-D interface after
FFN/depth pruning. Average cosine distance over valid tokens per clip, then clips,
so a long recording cannot dominate the feature objective.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from .distillation import teacher_spec, sample_fingerprint
from .pruning import sha256_file
from .queueing import read_json

FEATURE_VERSION = 1
FEATURE_KEYS = ('audio', 'vision')


def feature_spec(config, checkpoint, labels, token_ids, versions):
    return {'version': FEATURE_VERSION,
            'teacher': teacher_spec(config, checkpoint, labels, token_ids, versions),
            'feature_kind': 'native audio_tower.last_hidden_state and visual output before LM insertion',
            'padding': 'native packed valid tokens only', 'teacher_encoder_adapters': False,
            'storage_dtype': 'bfloat16'}


def feature_filename(identifier):
    return hashlib.sha256(identifier.encode()).hexdigest() + '.pt'


def validate_features(features, width=None):
    if not isinstance(features, dict) or set(features) != set(FEATURE_KEYS):
        raise ValueError('Require both native audio and vision features')
    for key, value in features.items():
        if (not torch.is_tensor(value) or value.ndim != 2 or min(value.shape) < 1
                or not value.is_floating_point() or not torch.isfinite(value).all()
                or (width is not None and value.shape[1] != width)):
            raise ValueError('Invalid packed encoder feature tensor: '+key)


class EncoderFeatureCapture:
    """Scoped forward hooks, removed before checkpoint backward recomputation."""
    def __init__(self, model):
        self.base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        self.handles = []; self.features = {}

    def __enter__(self):
        def save(key):
            def hook(module, args, output):
                if key in self.features:
                    raise ValueError('Encoder executed more than once in one feature-KD forward')
                self.features[key] = output.last_hidden_state if key == 'audio' else output
            return hook
        self.handles = [self.base.audio_tower.register_forward_hook(save('audio')),
                        self.base.visual.register_forward_hook(save('vision'))]
        return self

    def __exit__(self, *exc):
        for handle in self.handles: handle.remove()
        self.handles = []

    def result(self):
        validate_features(self.features)
        return self.features


def aligned_feature_losses(student, teachers):
    """Per-clip token-mean cosine distance with detached frozen targets."""
    validate_features(student)
    if not teachers: raise ValueError('Feature KD needs at least one target clip')
    for target in teachers: validate_features(target)
    losses = {}
    for key in FEATURE_KEYS:
        targets = [t[key] for t in teachers]
        lengths = [value.shape[0] for value in targets]
        prediction = student[key]
        if prediction.shape[0] != sum(lengths) or any(t.shape[1] != prediction.shape[1] for t in targets):
            raise ValueError('Teacher/student encoder token layout differs: '+key)
        clips = prediction.split(lengths, dim=0)
        values = [1-F.cosine_similarity(x.float(), y.detach().to(x.device).float(), dim=-1, eps=1e-8)
                  for x,y in zip(clips,targets)]
        losses[key] = torch.stack([v.mean() for v in values]).mean()
        if not torch.isfinite(losses[key]): raise ValueError('Nonfinite encoder feature KD')
    return losses


class TeacherFeatureCache:
    def __init__(self, path, expected_spec, records, max_cached_clips=32):
        self.path=Path(path).resolve();self.payload=read_json(self.path/'cache.json')
        if (self.payload.get('status')!='completed' or self.payload.get('spec')!=expected_spec
                or expected_spec['teacher']['split']!='train' or expected_spec['teacher']['test_set_used'] is not False):
            raise ValueError('Encoder feature cache has stale/non-training provenance')
        self.rows=self.payload['rows'];self.lru=OrderedDict();self.max_cached=max_cached_clips
        for record in records:
            row=self.rows.get(record['id'])
            if (row is None or row['sample_sha256']!=sample_fingerprint(record)
                    or row['file']!=feature_filename(record['id']) or not (self.path/'features'/row['file']).is_file()):
                raise ValueError('Missing/stale encoder features: '+record['id'])
        self.sha256=sha256_file(self.path/'cache.json')

    def load(self, record):
        key=record['id']
        if key not in self.rows: raise ValueError('Unknown feature-cache clip: '+key)
        if key in self.lru:
            self.lru.move_to_end(key);return self.lru[key]
        row=self.rows[key];path=self.path/'features'/row['file']
        if sha256_file(path)!=row['sha256']:raise ValueError('Encoder feature file changed: '+key)
        value=torch.load(path,map_location='cpu',weights_only=True);validate_features(value)
        if {k:list(v.shape) for k,v in value.items()}!=row['shapes']:
            raise ValueError('Encoder feature shapes differ from manifest')
        self.lru[key]=value
        if len(self.lru)>self.max_cached:self.lru.popitem(last=False)
        return value

    def losses(self, student, records):
        return aligned_feature_losses(student,[self.load(record) for record in records])
