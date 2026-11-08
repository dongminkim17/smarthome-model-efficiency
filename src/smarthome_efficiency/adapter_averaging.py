"""Exact weighted averaging of LoRA updates by block concatenation.

sum(w_i * s_i * B_i @ A_i) is represented without factor cross terms.
This temporary expanded adapter merges into the SAME dense architecture.
"""
from __future__ import annotations
import copy
import math
import torch


def average_factors(states, configs, weights=None):
    if len(states)<2 or len(states)!=len(configs):raise ValueError('Need at least two compatible adapters')
    weights=list(weights) if weights is not None else [1/len(states)]*len(states)
    if len(weights)!=len(states) or any(not math.isfinite(w) or w<0 for w in weights) or abs(sum(weights)-1)>1e-7:
        raise ValueError('Averaging weights must be finite, nonnegative and sum to one')
    reference=configs[0]
    for cfg,state in zip(configs,states):
        if (cfg!=reference or set(state)!=set(states[0]) or cfg.get('use_rslora',False)
                or cfg.get('use_dora',False) or cfg.get('rank_pattern') or cfg.get('alpha_pattern')
                or cfg.get('bias','none')!='none' or cfg.get('modules_to_save')):
            raise ValueError('Require identical plain uniform-rank LoRA adapters without saved biases/modules')
    rank=reference['r'];combined=copy.deepcopy(reference)
    combined.update(r=rank*len(states),lora_alpha=reference['lora_alpha']*len(states),inference_mode=True)
    result={}
    for key in sorted(states[0]):
        if '.lora_A.' not in key and '.lora_B.' not in key:raise ValueError('Unsupported adapter tensor: '+key)
        tensors=[s[key].float() for s in states]
        if any(t.shape!=tensors[0].shape or not torch.isfinite(t).all() for t in tensors):raise ValueError('Invalid source factors')
        if '.lora_A.' in key:
            if tensors[0].shape[0]!=rank:raise ValueError('LoRA A rank differs from configuration')
            partner=key.replace('.lora_A.','.lora_B.')
            if partner not in states[0]:raise ValueError('Missing matching B factor')
            value=torch.cat(tensors,dim=0)
        else:
            if tensors[0].shape[1]!=rank:raise ValueError('LoRA B rank differs from configuration')
            if key.replace('.lora_B.','.lora_A.') not in states[0]:raise ValueError('Missing matching A factor')
            value=torch.cat([w*t for w,t in zip(weights,tensors)],dim=1)
        result[key]=value.contiguous()
    return result,combined


def averaging_windows(steps, best_step):
    """Two predeclared windows: three trained snapshots near best and last three."""
    trained=sorted(set(s for s in steps if s>0))
    if len(trained)<3:raise ValueError('Averaging needs three trained validation snapshots')
    index=min(range(len(trained)),key=lambda i:(abs(trained[i]-best_step),trained[i]))
    start=max(0,min(index-1,len(trained)-3))
    windows={'near_best':trained[start:start+3],'last_three':trained[-3:]}
    if windows['near_best']==windows['last_three']:windows.pop('last_three')
    return windows
