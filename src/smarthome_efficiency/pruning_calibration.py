"""Streaming, prompt-only audiovisual importance statistics for structured pruning.

Layer BI follows ShortGPT's 1-cosine definition. FFN scores follow Model Optimizer's
Minitron _register_mlp_importance: abs mean over sequence, L2 over examples, at the
input to the down projection (after the SwiGLU product). No modelopt dependency.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .pruning import _language, processor_cache_identity, sha256_file
from .training import config_fingerprint

CALIBRATION_VERSION = 1
REFERENCES = {
    'ShortGPT': {'url':'https://github.com/icip-cas/ShortGPT',
                 'commit':'9a2ee8dfe9e28d0bee430e83c140b787c02ddd6c',
                 'files':['shortgpt/metrics.py','shortgpt/shortgpt.py']},
    'Minitron': {'url':'https://github.com/NVlabs/Minitron',
                 'commit':'fed740c91886155e1f272159ef662536bb673593'},
    'Model-Optimizer': {'url':'https://github.com/NVIDIA/Model-Optimizer',
                        'commit':'d7cbff3486a3ffb1a357e32d9810da0bfc649709',
                        'file':'modelopt/torch/prune/plugins/mcore_minitron.py',
                        'functions':['_register_mlp_importance','_register_depth_cosine_importance']},
}


def block_influence(input_hidden, output_hidden):
    """Per-token BI without the reference implementation's token-square matrix."""
    if input_hidden.shape != output_hidden.shape or input_hidden.ndim != 3:
        raise ValueError('Block input/output must have matching [batch, sequence, hidden] shapes')
    x,y=input_hidden.detach().float(),output_hidden.detach().float()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite block activations')
    cosine=F.cosine_similarity(x,y,dim=-1,eps=1e-8).clamp(-1,1)
    # Match ShortGPT's nan_to_num(nan=.5) for zero-norm token pairs.
    zero=(x.norm(dim=-1)==0)|(y.norm(dim=-1)==0)
    return 1-torch.where(zero,torch.full_like(cosine,.5),cosine)


def ffn_example_squares(activation):
    """Sum_b mean_seq(abs(gated_activation))², HF layout [B,S,I]."""
    if activation.ndim != 3 or activation.shape[0]!=1 or activation.shape[1]<1:
        raise ValueError('Calibration requires one unpadded nonempty clip per forward')
    value=activation.detach().float()
    if not torch.isfinite(value).all():raise ValueError('Nonfinite FFN activations')
    return value.abs().mean(dim=1).square().sum(dim=0)


class ImportanceCollector:
    """No retained hidden states; O(layers*FFN_width) streaming accumulators."""
    def __init__(self,model):
        self.language=_language(model)
        self.handles=[]
        self.bi_sums=torch.zeros(len(self.language.layers),dtype=torch.float64)
        self.token_counts=[0]*len(self.language.layers)
        self.example_counts=[0]*len(self.language.layers)
        self.ffn_squares=[torch.zeros(l.mlp.down_proj.in_features,dtype=torch.float64)
                          for l in self.language.layers]

    def __enter__(self):
        for i,layer in enumerate(self.language.layers):
            self.handles.append(layer.register_forward_hook(self._block_hook(i),with_kwargs=True))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._ffn_hook(i)))
        return self

    def _block_hook(self,i):
        def hook(module,args,kwargs,output):
            x=kwargs.get('hidden_states',args[0] if args else None)
            y=output[0] if isinstance(output,(tuple,list)) else output
            if x is None or x.shape[0]!=1:raise ValueError('One unpadded calibration clip required')
            bi=block_influence(x,y)
            self.bi_sums[i]+=bi.double().sum().cpu()
            self.token_counts[i]+=bi.numel()
        return hook

    def _ffn_hook(self,i):
        def hook(module,args):
            self.ffn_squares[i]+=ffn_example_squares(args[0]).double().cpu()
            self.example_counts[i]+=1
        return hook

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def scores(self):
        if (not all(self.token_counts) or len(set(self.token_counts))!=1
                or not all(self.example_counts) or len(set(self.example_counts))!=1):
            raise ValueError('Incomplete or unequal calibration coverage across layers')
        return {'layer_bi':(self.bi_sums/torch.tensor(self.token_counts)).tolist(),
                'ffn_activation':{str(i):v.sqrt().tolist() for i,v in enumerate(self.ffn_squares)},
                'tokens_per_layer':self.token_counts,'examples_per_layer':self.example_counts,
                'layer_rule':'token mean of 1-cosine(raw block input, raw block output), FP32; lower BI removed',
                'ffn_rule':'sqrt(sum_examples(mean_tokens(abs(SwiGLU_output)) ** 2)), FP32 forward statistics',
                'references':REFERENCES}


def calibration_spec(cfg,source,runner,records,seed):
    from .distillation import sample_fingerprint
    from .model import SYSTEM_PROMPT,build_prompt
    source=Path(source).resolve()
    return {'version':CALIBRATION_VERSION,'split':'train','test_set_used':False,
            'source_checkpoint':str(source),'source_metadata_sha256':sha256_file(source/'metadata.json'),
            'source_adapter_sha256':sha256_file(source/'adapter/adapter_model.safetensors'),
            'processor_cache_identity':processor_cache_identity(Path(cfg['model'])),
            'input_protocol':{k:cfg[k] for k in ['task','split_seed','max_frames','max_pixels','use_transcript']},
            'calibration_seed':seed,'examples':len(records),'record_order':[r['id'] for r in records],
            'record_fingerprints':{r['id']:sample_fingerprint(r) for r in records},
            'prompt_sha256':hashlib.sha256((SYSTEM_PROMPT+inspect.getsource(build_prompt)).encode()).hexdigest(),
            'collector_source_sha256':sha256_file(Path(__file__)),
            'versions':runner.versions,'merged_source_adapter':True,'prompt_only':True,
            'references':REFERENCES}


def validate_calibration(payload,spec,model):
    layers=_language(model).layers
    if payload.get('status')!='completed' or payload.get('spec')!=spec:
        raise ValueError('Calibration cache has stale or mismatched source/input provenance')
    scores=payload['scores'];bi=torch.tensor(scores['layer_bi'],dtype=torch.float64)
    if bi.shape!=(len(layers),) or not torch.isfinite(bi).all() or (bi<0).any() or (bi>2).any():
        raise ValueError('Invalid layer importance scores')
    if set(scores['ffn_activation'])!={str(i) for i in range(len(layers))}:
        raise ValueError('Missing FFN importance scores')
    for i,layer in enumerate(layers):
        values=torch.tensor(scores['ffn_activation'][str(i)],dtype=torch.float64)
        if values.shape!=(layer.mlp.down_proj.in_features,) or not torch.isfinite(values).all() or (values<0).any():
            raise ValueError('Invalid FFN importance scores')
    if scores['examples_per_layer']!=[spec['examples']]*len(layers):
        raise ValueError('Calibration example coverage mismatch')
    counts=scores['tokens_per_layer']
    if len(counts)!=len(layers) or not all(type(v) is int and v>0 for v in counts) or len(set(counts))!=1:
        raise ValueError('Calibration token coverage mismatch')
    return scores


def obtain_calibration(runner,cfg,source,records,seed,cache_path,progress=None):
    """Reuse identical scores across all three methods; never read dev/test data."""
    import fcntl
    from .checkpointing import write_json
    spec=calibration_spec(cfg,source,runner,records,seed)
    cache_path=Path(cache_path);cache_path.parent.mkdir(parents=True,exist_ok=True)
    with cache_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if cache_path.exists():
            payload=json.loads(cache_path.read_text());scores=validate_calibration(payload,spec,runner.model)
            return scores,{'path':str(cache_path.resolve()),'sha256':sha256_file(cache_path),'spec':spec,'reused':True}
        with ImportanceCollector(runner.model) as collector:
            for index,record in enumerate(records):
                # prepare/score receive no target token. Hooks see prompt/media only.
                batch=runner.prepare(record)
                if batch['input_ids'].shape[0]!=1 or not torch.all(batch['attention_mask']==1):
                    raise ValueError('Calibration inputs must be single unpadded clips')
                runner.score(batch)
                if progress and ((index+1)%16==0 or index+1==len(records)):progress(index+1,len(records))
        payload={'status':'completed','spec':spec,'scores':collector.scores()}
        scores=validate_calibration(payload,spec,runner.model);write_json(cache_path,payload)
        return scores,{'path':str(cache_path.resolve()),'sha256':sha256_file(cache_path),'spec':spec,'reused':False}
