"""Feature alignment, per-clip weighting, frozen cache integrity and native gradients."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from peft import LoraConfig,TaskType,get_peft_model,PeftModel
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_pruning import tiny_model
from smarthome_efficiency.feature_distillation import (EncoderFeatureCapture,aligned_feature_losses,
    TeacherFeatureCache,feature_filename,validate_features)
from smarthome_efficiency.training import lora_targets,load_config,validate_resume,DEFAULTS
from smarthome_efficiency.pruning import sha256_file,parameter_counts
from smarthome_efficiency.queueing import atomic_json


def media_batch():
    return {'input_features':torch.randn(1,128,8),'feature_attention_mask':torch.ones(1,8,dtype=torch.long),
            'pixel_values_videos':torch.randn(16,24),'video_grid_thw':torch.tensor([[1,4,4]])}


def encode(model,batch):
    audio=model.get_audio_features(batch['input_features'],feature_attention_mask=batch['feature_attention_mask'])
    vision=model.get_video_features(batch['pixel_values_videos'],batch['video_grid_thw'])
    return {'audio':audio,'vision':vision}


class FeatureDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_cosine_loss_averages_clips_equally_and_detaches_teacher(self):
        a=torch.tensor([[1.,0.]],requires_grad=True);b=torch.tensor([[0.,1.]]*3,requires_grad=True)
        student={'audio':torch.cat([a,b]),'vision':torch.cat([a,b])}
        targets=[{k:torch.tensor([[1.,0.]],requires_grad=True) for k in ('audio','vision')},
                 {k:torch.tensor([[1.,0.]]*3,requires_grad=True) for k in ('audio','vision')}]
        losses=aligned_feature_losses(student,targets)
        for value in losses.values():self.assertAlmostEqual(value.item(),.5)
        sum(losses.values()).backward();self.assertIsNotNone(a.grad);self.assertIsNotNone(b.grad)
        self.assertTrue(all(t.grad is None for d in targets for t in d.values()))
        with self.assertRaises(ValueError):aligned_feature_losses(student,targets[:1])
        with self.assertRaises(ValueError):validate_features({'audio':torch.ones(1,2)})

    def test_capture_removes_hooks_on_exception_and_preserves_autograd(self):
        class Audio(nn.Module):
            def forward(self,x):return SimpleNamespace(last_hidden_state=x*2)
        class Model(nn.Module):
            def __init__(self):super().__init__();self.audio_tower=Audio();self.visual=nn.Identity()
        model=Model();x=torch.ones(3,4,requires_grad=True)
        with EncoderFeatureCapture(model) as capture:model.audio_tower(x);model.visual(x)
        features=capture.result();features['audio'].sum().backward();self.assertTrue(torch.equal(x.grad,torch.full_like(x,2)))
        with self.assertRaises(RuntimeError):
            with EncoderFeatureCapture(model):raise RuntimeError('interrupted')
        self.assertFalse(model.audio_tower._forward_hooks);self.assertFalse(model.visual._forward_hooks)

    def test_cache_checks_media_fingerprint_hash_shape_and_training_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record={'id':'clip'};filename=feature_filename('clip');(root/'features').mkdir()
            value={k:torch.randn(3,4).to(torch.bfloat16) for k in ('audio','vision')};path=root/'features'/filename;torch.save(value,path)
            spec={'teacher':{'split':'train','test_set_used':False}}
            row={'file':filename,'sha256':sha256_file(path),'shapes':{k:list(v.shape) for k,v in value.items()},'sample_sha256':'sample'}
            payload={'status':'completed','spec':spec,'rows':{'clip':row}}
            atomic_json(root/'cache.json',payload)
            with patch('smarthome_efficiency.feature_distillation.sample_fingerprint',return_value='sample'):
                cache=TeacherFeatureCache(root,spec,[record]);actual=cache.load(record)
                for k in value:torch.testing.assert_close(actual[k],value[k])
                badspec={'teacher':{'split':'test','test_set_used':True}}
                with self.assertRaises(ValueError):TeacherFeatureCache(root,badspec,[record])
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):TeacherFeatureCache(root,spec,[record]).load(record)
            with patch('smarthome_efficiency.feature_distillation.sample_fingerprint',return_value='changed'):
                with self.assertRaises(ValueError):TeacherFeatureCache(root,spec,[record])

    def test_old_configs_remain_identical_and_feature_resume_is_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'config.yaml';import yaml
            cfg=dict(DEFAULTS,model='/model',data_root='/data',cache_dir='/cache',lora_target='decoder_encoder_linear')
            p.write_text(yaml.safe_dump(cfg));self.assertEqual(load_config(p),cfg)
            extra=dict(cfg,feature_teacher_cache='/features',feature_audio_weight=.1,feature_vision_weight=.1)
            p.write_text(yaml.safe_dump(extra));self.assertEqual(load_config(p),extra)
            for change in [dict(feature_audio_weight=.2),dict(feature_teacher_cache='/different')]:
                with self.assertRaises(ValueError):validate_resume(extra,extra|change,1,1)
            for change in [dict(feature_audio_weight=-1),dict(feature_teacher_cache=None),dict(lora_target='qv')]:
                p.write_text(yaml.safe_dump(extra|change))
                with self.assertRaises(ValueError):load_config(p)

    def test_native_packed_features_keep_clip_order_with_variable_audio_padding(self):
        model=tiny_model();first=media_batch();second=media_batch()
        second['input_features']=torch.randn(1,128,12)
        second['feature_attention_mask']=torch.ones(1,12,dtype=torch.long)
        padded=torch.nn.functional.pad(first['input_features'],(0,4))
        batch={'input_features':torch.cat([padded,second['input_features']]),
               'feature_attention_mask':torch.cat([torch.nn.functional.pad(first['feature_attention_mask'],(0,4)),second['feature_attention_mask']]),
               'pixel_values_videos':torch.cat([first['pixel_values_videos'],second['pixel_values_videos']]),
               'video_grid_thw':torch.cat([first['video_grid_thw'],second['video_grid_thw']])}
        with torch.no_grad():
            targets=[encode(model,clip) for clip in (first,second)];actual=encode(model,batch)
        for key in actual:
            torch.testing.assert_close(actual[key],torch.cat([t[key] for t in targets]),rtol=2e-5,atol=2e-6)
        for value in aligned_feature_losses(actual,targets).values():
            self.assertLess(abs(value.item()),1e-6)

    def test_native_encoder_features_match_capture_and_broader_lora_can_train_reload_merge(self):
        model=tiny_model();model.requires_grad_(False);batch=media_batch()
        with torch.no_grad(),EncoderFeatureCapture(model) as capture:expected=encode(model,batch)
        for key in expected:torch.testing.assert_close(capture.result()[key],expected[key])
        wrapped=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=lora_targets('decoder_encoder_linear'),task_type=TaskType.CAUSAL_LM,bias='none'))
        base=wrapped.get_base_model();counts={k:0 for k in ('audio','vision','decoder')}
        for name,p in wrapped.named_parameters():
            if p.requires_grad:
                if 'audio_tower' in name:counts['audio']+=1
                elif 'visual.' in name:counts['vision']+=1
                else:counts['decoder']+=1
        self.assertTrue(all(counts.values()))
        teacher={k:v.detach()+.1*torch.randn_like(v) for k,v in expected.items()}
        with EncoderFeatureCapture(wrapped) as capture:encode(base,batch)
        losses=aligned_feature_losses(capture.result(),[teacher]);sum(losses.values()).backward()
        for name,p in wrapped.named_parameters():
            if p.requires_grad and ('audio_tower' in name or 'visual.' in name):
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        optimizer=torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],lr=.01);optimizer.step();wrapped.eval()
        with torch.no_grad():trained=encode(base,batch)
        with tempfile.TemporaryDirectory() as folder:
            wrapped.save_pretrained(folder,safe_serialization=True)
            fresh=tiny_model();loaded=PeftModel.from_pretrained(fresh,folder,is_trainable=False).eval()
            with torch.no_grad():actual=encode(loaded.get_base_model(),batch)
            for key in actual:torch.testing.assert_close(actual[key],trained[key],rtol=0,atol=0)
            merged=loaded.merge_and_unload(safe_merge=True)
            self.assertEqual(parameter_counts(merged),parameter_counts(tiny_model()))

if __name__=='__main__':unittest.main()
