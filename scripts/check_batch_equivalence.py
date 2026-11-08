#!/usr/bin/env python3
"""FP32 check to separate batch-alignment errors from BF16 kernel rounding."""
import argparse,json,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from smarthome_efficiency.data import LABELS,load_records
from smarthome_efficiency.model import PilotModel
from smarthome_efficiency.training import load_config,configure_determinism

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
assert not args.output.exists()
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
result={'status':'running','dtype':'float32','tf32':False,'tasks':{},'test_set_used':False}
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
save();shared=None
for task in ['mintrec20','mer6']:
 cfg=load_config(ROOT/'configs'/f'{task}.yaml');configure_determinism(cfg)
 runner=PilotModel(cfg['model'],task,LABELS[task],cfg['cache_dir'],max_frames=cfg['max_frames'],max_pixels=cfg['max_pixels'],use_transcript=cfg['use_transcript'])
 if shared is None:
  runner.load_model();runner.enable_lora(rank=16,alpha=32,dropout=0,target_modules=r'model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))');runner.model.float();shared=runner.model
 else:runner.model=shared;runner.lora_enabled=True
 records=load_records(task,'train',data_root=cfg['data_root'])
 selected=random.Random(42).sample(range(len(records)),8)[:2]
 batches=[runner.prepare(records[i]) for i in selected]
 targets=[list(LABELS[task]).index(records[i]['label']) for i in selected]
 shared.zero_grad(set_to_none=True)
 singles=[]
 for b,t in zip(batches,targets):
  item=runner.loss_batch(b,t);(item['loss']/len(batches)).backward();singles.append(item)
 parameters=[p for p in shared.parameters() if p.requires_grad]
 reference_grad=torch.cat([p.grad.detach().cpu().flatten() for p in parameters if p.grad is not None])
 shared.zero_grad(set_to_none=True)
 merged=runner.loss_microbatch(batches,targets);merged['loss'].backward()
 batch_grad=torch.cat([p.grad.detach().cpu().flatten() for p in parameters if p.grad is not None])
 gradient_relative_l2=float((batch_grad-reference_grad).norm()/reference_grad.norm().clamp_min(1e-12))
 logits=torch.stack([s['class_logits'] for s in singles]); expected=torch.stack([s['loss'] for s in singles]).mean()
 logit_delta=float((merged['class_logits']-logits).abs().max());loss_delta=abs(float(merged['loss']-expected))
 passed=logit_delta<1e-3 and loss_delta<1e-4 and gradient_relative_l2<1e-3
 result['tasks'][task]={'logit_max_abs':logit_delta,'loss_abs_difference':loss_delta,'gradient_relative_l2':gradient_relative_l2,'passed':passed,
                       'ids':[records[i]['id'] for i in selected],'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
 print(task,result['tasks'][task],flush=True);save()
 shared.zero_grad(set_to_none=True)
 del singles,merged,reference_grad,batch_grad
 torch.cuda.empty_cache()
result['status']='passed' if all(v['passed'] for v in result['tasks'].values()) else 'failed';save()
