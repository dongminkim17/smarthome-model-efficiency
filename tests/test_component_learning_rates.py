"""Verify component-specific updates, exact resume and legacy configuration identity."""
import copy,tempfile,unittest
from pathlib import Path
import torch
from torch import nn
import yaml
from smarthome_efficiency.training import (DEFAULTS,load_config,optimizer_groups,set_optimizer_learning_rate,
    make_schedule,learning_rate,validate_resume)


def toy_model():
    model=nn.Module();model.model=nn.Linear(2,2,bias=False)
    model.audio_tower=nn.Linear(2,2,bias=False);model.visual=nn.Linear(2,2,bias=False)
    return model


class ComponentLearningRateTests(unittest.TestCase):
    def test_legacy_config_and_optimizer_membership_are_unchanged(self):
        cfg=dict(DEFAULTS,model='/model',lora_target='decoder_encoder_ffn');model=toy_model()
        groups=optimizer_groups(model,cfg)
        self.assertEqual(len(groups),1);self.assertEqual(set(groups[0]),{'params'})
        self.assertEqual([id(p) for p in groups[0]['params']],[id(p) for p in model.parameters()])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'c.yaml';p.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(p),cfg)
            extra=cfg|{'encoder_learning_rate':1e-5};p.write_text(yaml.safe_dump(extra));self.assertEqual(load_config(p),extra)
            for value in [0,-1,float('inf'),True,None]:
                p.write_text(yaml.safe_dump(cfg|{'encoder_learning_rate':value}))
                with self.assertRaises(ValueError):load_config(p)
            for change in [{'encoder_learning_rate':2e-5},{'encoder_learning_rate':None}]:
                with self.assertRaises(ValueError):validate_resume(extra,extra|change,1,1)

    def test_real_optimizer_updates_have_correct_ratio_and_resume_exactly(self):
        cfg=dict(DEFAULTS,learning_rate=1e-3,encoder_learning_rate=1e-4,warmup_ratio=0.)
        model=toy_model();original=copy.deepcopy(model.state_dict());schedule=make_schedule(cfg,10)
        opt=torch.optim.AdamW(optimizer_groups(model,cfg),lr=cfg['learning_rate'],weight_decay=0.)
        self.assertEqual([g['component'] for g in opt.param_groups],['language','audio_encoder','vision_encoder'])
        for group in opt.param_groups:
            for p in group['params']:p.grad=torch.ones_like(p)
        set_optimizer_learning_rate(opt,learning_rate(schedule,0));opt.step();opt.zero_grad()
        for name,p in model.named_parameters():
            expected=1e-3 if name.startswith('model.') else 1e-4
            torch.testing.assert_close(original[name]-p,torch.full_like(p,expected),rtol=1e-3,atol=1e-7)
        resumed=copy.deepcopy(model);saved=copy.deepcopy(opt.state_dict())
        other=torch.optim.AdamW(optimizer_groups(resumed,cfg),lr=cfg['learning_rate'],weight_decay=0.)
        other.load_state_dict(saved)
        for actual,optm in [(model,opt),(resumed,other)]:
            set_optimizer_learning_rate(optm,learning_rate(schedule,1))
            self.assertAlmostEqual(optm.param_groups[1]['lr']/optm.param_groups[0]['lr'],.1)
            for p in actual.parameters():p.grad=torch.full_like(p,.3)
            optm.step()
        for a,b in zip(model.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
