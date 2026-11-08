"""Encoder-depth + activation-FFN pruning, retaining native AV interfaces."""
from __future__ import annotations
import fcntl,json,re
from pathlib import Path
import torch
from torch import nn
from .multimodal_pruning import (MultimodalImportanceCollector, validate_multimodal_calibration,
                                prune_multimodal_ffns, is_prunable_ffn_tensor)
from .pruning_calibration import calibration_spec,block_influence
from .pruning import sha256_file,select_top_channels

D_COUNTS={'language':1_415_294_976,'audio_encoder':259_980_288,'vision_encoder':313_741_568,'other':0,'total':1_989_016_832}


def select_encoder_layers(scores, keep, mandatory):
    values=torch.tensor(scores,dtype=torch.float64);mandatory=sorted(set(mandatory))
    if (values.ndim!=1 or not torch.isfinite(values).all() or (values<0).any() or (values>2).any()
            or type(keep) is not int or not len(mandatory)<=keep<=len(values)
            or any(type(i) is not int or not 0<=i<len(values) for i in mandatory)):
        raise ValueError('Invalid encoder layer importance/budget/protected indices')
    candidates=[i for i in range(len(values)) if i not in mandatory]
    count=keep-len(mandatory)
    extras=select_top_channels(values[candidates],count) if count else []
    return sorted(mandatory+[candidates[i] for i in extras])


def protected_identities(model,audio_keep,vision_keep,compact=False):
    result=[]
    for name,p in model.named_parameters():
        if is_prunable_ffn_tensor(name):continue
        matched=re.match(r'(audio_tower\.layers|visual\.blocks)\.(\d+)\.(.*)',name)
        if matched:
            prefix,index,suffix=matched.groups();index=int(index)
            keep=audio_keep if prefix=='audio_tower.layers' else vision_keep
            if compact:index=keep[index]
            elif index not in keep:continue
            name=f'{prefix}.{index}.{suffix}'
        result.append((name,p.data_ptr(),p._version))
    return result


@torch.no_grad()
def prune_hybrid(model,widths,scores,audio_layers,vision_layers):
    audio=list(model.audio_tower.layers);vision=list(model.visual.blocks)
    original_full=list(model.visual.fullatt_block_indexes)
    a_keep=select_encoder_layers(scores['audio_layer_bi'],audio_layers,[0,len(audio)-1])
    v_keep=select_encoder_layers(scores['vision_layer_bi'],vision_layers,[0,len(vision)-1,*original_full])
    before=protected_identities(model,a_keep,v_keep)
    # The common physical surgery validates every FFN before any modifications.
    selection=prune_multimodal_ffns(model,widths,scores)
    model.audio_tower.layers=nn.ModuleList([audio[i] for i in a_keep])
    model.audio_tower.config.encoder_layers=audio_layers
    model.config.audio_config.encoder_layers=audio_layers
    model.visual.blocks=nn.ModuleList([vision[i] for i in v_keep])
    full=[new for new,old in enumerate(v_keep) if old in original_full]
    model.visual.fullatt_block_indexes=full
    model.visual.config.fullatt_block_indexes=full
    model.config.vision_config.fullatt_block_indexes=full
    model.visual.config.depth=vision_layers;model.config.vision_config.depth=vision_layers
    if protected_identities(model,a_keep,v_keep,compact=True)!=before:
        raise RuntimeError('Hybrid pruning changed a retained non-FFN tensor or interface')
    selection.update(kept_original_audio_layer_indices=a_keep,kept_original_vision_layer_indices=v_keep,
                     original_vision_full_attention_indices=original_full,compact_vision_full_attention_indices=full,
                     retained_non_ffn_tensors_unchanged=True,all_layer_counts_unchanged=False,
                     selection_rule='protected endpoints/global vision attention; highest token-mean BI for encoder layers; activation-ranked FFNs')
    return selection


class HybridCollector(MultimodalImportanceCollector):
    def __init__(self,model):
        super().__init__(model)
        self.blocks={'audio':list(model.audio_tower.layers),'vision':list(model.visual.blocks)}
        self.bi={key:torch.zeros(len(blocks),dtype=torch.float64) for key,blocks in self.blocks.items()}
        self.counts={key:[0]*len(blocks) for key,blocks in self.blocks.items()}

    def __enter__(self):
        super().__enter__()
        for key,blocks in self.blocks.items():
            for i,block in enumerate(blocks):
                def hook(module,args,kwargs,output,key=key,i=i):
                    x=kwargs.get('hidden_states',args[0] if args else None)
                    y=output[0] if isinstance(output,(tuple,list)) else output
                    if x is None or x.ndim!=2 or y.shape!=x.shape:raise ValueError('Expected packed valid encoder block tokens')
                    values=block_influence(x.unsqueeze(0),y.unsqueeze(0))
                    self.bi[key][i]+=values.double().sum().cpu();self.counts[key][i]+=values.numel()
                self.handles.append(block.register_forward_hook(hook,with_kwargs=True))
        return self

    def scores(self,expected_examples):
        result=super().scores(expected_examples)
        for key in self.blocks:
            if not all(self.counts[key]) or len(set(self.counts[key]))!=1:raise ValueError('Incomplete encoder BI coverage')
            result[key+'_layer_bi']=(self.bi[key]/torch.tensor(self.counts[key])).tolist()
        return result


def obtain_hybrid_calibration(runner,cfg,source,records,seed,cache_path,progress=None):
    from .checkpointing import write_json
    spec=calibration_spec(cfg,source,runner,records,seed)
    spec.update(scope='all_thinker_components',encoder_depth_scoring='packed_token_mean_BI',
                hybrid_collector_sha256=sha256_file(Path(__file__)))
    cache_path=Path(cache_path);cache_path.parent.mkdir(parents=True,exist_ok=True)
    def validate(payload):
        scores=validate_multimodal_calibration(payload,spec,runner.model)
        for key,blocks in [('audio',runner.model.audio_tower.layers),('vision',runner.model.visual.blocks)]:
            vector=torch.tensor(scores[key+'_layer_bi'],dtype=torch.float64)
            if vector.shape!=(len(blocks),) or not torch.isfinite(vector).all() or (vector<0).any() or (vector>2).any():
                raise ValueError('Invalid encoder block influence scores')
        return scores
    with cache_path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if cache_path.exists():
            scores=validate(json.loads(cache_path.read_text()))
            return scores,{'path':str(cache_path.resolve()),'sha256':sha256_file(cache_path),'spec':spec,'reused':True}
        with HybridCollector(runner.model) as collector:
            for i,record in enumerate(records):
                batch=runner.prepare(record)
                if batch['input_ids'].shape[0]!=1 or not torch.all(batch['attention_mask']==1):raise ValueError('Unpadded train clips required')
                runner.score(batch)
                if progress and ((i+1)%16==0 or i+1==len(records)):progress(i+1,len(records))
        payload={'status':'completed','spec':spec,'scores':collector.scores(len(records))}
        scores=validate(payload);write_json(cache_path,payload)
        return scores,{'path':str(cache_path.resolve()),'sha256':sha256_file(cache_path),'spec':spec,'reused':False}
