"""Class-distribution distillation and provenance-checked training-only caches."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .pruning import processor_cache_identity, sha256_file
from .training import config_fingerprint

FORMAT_VERSION = 1
PROTOCOL_KEYS = ('task', 'split_seed', 'max_frames', 'max_pixels', 'use_transcript')


def sample_fingerprint(record):
    paths = [record.get(key) for key in ('video','media_path','media','path') if record.get(key)]
    if not paths:
        raise ValueError('Training record has no local media')
    path = Path(paths[0]).resolve()
    stat = path.stat()
    return config_fingerprint({'record':record,'media':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})


def teacher_spec(config, checkpoint, labels, token_ids, versions):
    """Bind task/input protocol, immutable teacher, label order, and tokenizer."""
    from .model import SYSTEM_PROMPT, build_prompt, build_input_prompt
    checkpoint = Path(checkpoint).resolve()
    metadata = json.loads((checkpoint/'metadata.json').read_text())
    teacher = metadata['config']
    for key in PROTOCOL_KEYS:
        if config[key] != teacher[key]:
            raise ValueError(f'Teacher/student input protocol differs: {key}')
    identity = processor_cache_identity(Path(config['model']))
    if identity != processor_cache_identity(Path(teacher['model'])):
        raise ValueError('Teacher/student processor cache identities differ')
    prompt_source = SYSTEM_PROMPT+inspect.getsource(build_prompt)
    if config['task']=='mer6' and config['use_transcript']:
        prompt_source += inspect.getsource(build_input_prompt)
    partition = ({'training_partition':config['mer_partition']} if config.get('mer_partition') else {})
    return {**partition, 'format_version':FORMAT_VERSION, 'split':'train', 'test_set_used':False,
            'teacher_checkpoint':str(checkpoint),
            'teacher_metadata_sha256':sha256_file(checkpoint/'metadata.json'),
            'teacher_adapter_sha256':sha256_file(checkpoint/'adapter/adapter_model.safetensors'),
            'teacher_base_config_sha256':sha256_file(Path(teacher['model'])/'config.json'),
            'input_protocol':{key:config[key] for key in PROTOCOL_KEYS},
            'processor_cache_identity':identity, 'labels':list(labels), 'label_token_ids':list(token_ids),
            'prompt_sha256':hashlib.sha256(prompt_source.encode()).hexdigest(),
            'versions':versions, 'logit_scope':'raw next-token logits for allowed class letters, prompt only'}


class TeacherLogitCache:
    def __init__(self, path, expected_spec, records):
        self.path = Path(path).resolve()
        payload = json.loads(self.path.read_text())
        if payload.get('status') != 'completed' or payload.get('spec') != expected_spec:
            raise ValueError('Teacher logit cache is incomplete or has mismatched teacher/input provenance')
        if expected_spec.get('split') != 'train' or expected_spec.get('test_set_used') is not False:
            raise ValueError('Distillation caches must contain training data only')
        self.rows = payload['rows']
        self.class_count = len(expected_spec['labels'])
        for record in records:
            row = self.rows.get(record['id'])
            if row is None or row['sample_sha256'] != sample_fingerprint(record):
                raise ValueError(f'Teacher logits missing or stale for training record {record["id"]}')
            values = torch.tensor(row['logits'],dtype=torch.float32)
            if values.shape != (self.class_count,) or not torch.isfinite(values).all():
                raise ValueError('Invalid teacher class logits')
        self.sha256 = sha256_file(self.path)

    def batch(self, records, device):
        return torch.tensor([self.rows[r['id']]['logits'] for r in records],dtype=torch.float32,device=device)


def distillation_kl(student_logits, teacher_logits, temperature):
    """T² KL(teacher || student), averaged over examples, with frozen targets."""
    if (student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape
            or student_logits.shape[1] < 2):
        raise ValueError('Teacher/student logits must have matching [batch, classes] shapes')
    if not isinstance(temperature,(int,float)) or isinstance(temperature,bool) or not 0 < temperature < float('inf'):
        raise ValueError('Distillation temperature must be finite and positive')
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError('Nonfinite distillation logits')
    teacher_prob = F.softmax(teacher_logits.detach().float()/temperature,dim=-1)
    return F.kl_div(F.log_softmax(student_logits.float()/temperature,dim=-1),teacher_prob,
                    reduction='batchmean') * temperature**2


def disagreement_distillation_loss(sft_per_example, student_logits, teacher_logits,
                                    gold_indices, temperature, alpha, disagreement_alpha):
    """Mix SFT and class KD per training example, reducing KD on teacher errors.

    The gate uses frozen cached-teacher argmax versus the training gold label;
    student predictions never control it. SFT already averages letter and EOS.
    Feature losses are added separately and are not gated. Returned diagnostics
    are detached so they can be summed over actual examples across DDP ranks.
    """
    if (student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape
            or student_logits.shape[0] < 1 or student_logits.shape[1] < 2):
        raise ValueError('Teacher/student logits must have matching nonempty [batch, classes] shapes')
    if sft_per_example.shape != (student_logits.shape[0],):
        raise ValueError('One differentiable SFT loss is required per example')
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 < temperature < float('inf'):
        raise ValueError('Distillation temperature must be finite and positive')
    for name, value in [('alpha', alpha), ('disagreement_alpha', disagreement_alpha)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(name + ' must be finite and in [0,1]')
    if not all(torch.isfinite(value).all() for value in (sft_per_example, student_logits, teacher_logits)):
        raise ValueError('Nonfinite distillation logits or SFT losses')
    gold = torch.as_tensor(gold_indices, device=student_logits.device)
    if (gold.shape != (student_logits.shape[0],) or gold.dtype not in (torch.int32, torch.int64)
            or (gold < 0).any() or (gold >= student_logits.shape[1]).any()):
        raise ValueError('Gold indices must be one valid integer class index per example')
    teacher = teacher_logits.detach().float()
    disagreement = teacher.argmax(dim=-1) != gold
    effective_alpha = torch.where(disagreement, disagreement_alpha, alpha).float()
    teacher_prob = F.softmax(teacher / temperature, dim=-1)
    per_example_kl = F.kl_div(F.log_softmax(student_logits.float() / temperature, dim=-1),
                            teacher_prob, reduction='none').sum(dim=-1) * temperature**2
    mixed = ((1 - effective_alpha) * sft_per_example + effective_alpha * per_example_kl).mean()
    return {'loss': mixed, 'distillation_loss': per_example_kl.mean(),
            'disagreement': disagreement.detach(), 'effective_alpha': effective_alpha.detach()}
