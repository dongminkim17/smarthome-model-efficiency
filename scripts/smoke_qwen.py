#!/usr/bin/env python3
"""Bounded Qwen Omni wiring checks; outputs are not benchmark quality scores."""
import argparse
import importlib.metadata
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local snapshot path or Hugging Face repo ID')
    parser.add_argument('--media', type=Path, help='Optional local video containing an audio track')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-step', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--clip-seconds', type=float, default=4.0)
    parser.add_argument('--max-frames', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64 or not 0 < args.clip_seconds <= 10:
        parser.error('Smoke limits: 1–64 output tokens and >0–10 seconds of video')
    if not 2 <= args.max_frames <= 16 or args.max_frames % 2:
        parser.error('--max-frames must be even and between 2 and 16')
    if args.media and not args.media.is_file():
        parser.error('--media must be an existing local video')
    torch.manual_seed(42)
    report = {'purpose': 'Environment/inference/gradient wiring only; not a quality evaluation',
              'model': args.model, 'media': str(args.media) if args.media else None,
              'limits': vars(args) | {'media': str(args.media), 'output': str(args.output)},
              'versions': {p: importlib.metadata.version(p) for p in
                           ['torch', 'transformers', 'peft', 'accelerate', 'qwen-omni-utils']}}

    def memory():
        torch.cuda.synchronize()
        return {'allocated_gib': torch.cuda.memory_allocated() / 2**30,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}

    try:
        if args.media and not shutil.which('ffmpeg'):
            raise RuntimeError('ffmpeg is missing from PATH; activate the project conda environment or use conda run -p ENV')
        if not torch.cuda.is_available():
            raise RuntimeError('A visible CUDA GPU is required')
        report['gpu'] = torch.cuda.get_device_name(0)
        start = time.perf_counter()
        processor = Qwen2_5OmniProcessor.from_pretrained(args.model)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0})
        counts = defaultdict(int)
        for name, parameter in model.named_parameters():
            counts[name.split('.')[0]] += parameter.numel()
        report['parameters'] = {'components': dict(counts), 'total': sum(counts.values()),
                                'scope': 'Thinker: language model + audio/vision encoders; no speech generator'}
        report['load_seconds'] = time.perf_counter() - start
        system = {'role': 'system', 'content': [{'type': 'text', 'text':
            'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'}]}
        conversations = [('text', [system, {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Reply with the single word ready.'}]}])]
        if args.media:
            conversations.append(('audio_video', [system, {'role': 'user', 'content': [
                {'type': 'video', 'video': str(args.media.resolve()), 'video_start': 0,
                 'video_end': args.clip_seconds, 'fps': 2.0, 'min_frames': 2,
                 'max_frames': args.max_frames, 'min_pixels': 56**2, 'max_pixels': 224**2},
                {'type': 'text', 'text': 'Briefly describe what you see and hear.'}]}]))
        report['inference'] = []
        for kind, conversation in conversations:
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            use_audio = kind == 'audio_video'
            audios, images, videos, video_kwargs = process_mm_info(
                conversation, use_audio_in_video=use_audio, return_video_kwargs=True)
            video_kwargs.pop('do_sample_frames', None)  # Not accepted by Transformers 4.52.3.
            if isinstance(video_kwargs.get('fps'), list):  # This processor broadcasts a scalar itself.
                video_kwargs['fps'] = video_kwargs['fps'][0] if video_kwargs['fps'] else 2.0
            if use_audio and (not audios or not videos):
                raise RuntimeError('Both native audio and video inputs must be present')
            prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            encoded = processor(text=prompt, audio=audios, images=images, videos=videos,
                                use_audio_in_video=use_audio, min_pixels=56**2, max_pixels=224**2,
                                return_tensors='pt', padding=True, **video_kwargs)
            batch = encoded.to(model.device).to(model.dtype)
            model.eval()
            with torch.inference_mode():
                generated = model.generate(**batch, use_audio_in_video=use_audio,
                                           max_new_tokens=args.max_new_tokens, do_sample=False,
                                           eos_token_id=processor.tokenizer.eos_token_id,
                                           pad_token_id=processor.tokenizer.pad_token_id)
            usage = memory()
            report['inference'].append({'kind': kind, 'seconds': time.perf_counter() - start,
                'input_tokens': batch['input_ids'].shape[-1],
                'input_shapes': {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
                'video_kwargs': video_kwargs, 'memory': usage,
                'generated_token_ids': generated[0, batch['input_ids'].shape[-1]:].tolist(),
                'output': processor.batch_decode(generated[:, batch['input_ids'].shape[-1]:],
                                                 skip_special_tokens=True)[0]})
            del generated
        if args.train_step:
            # Last case is audiovisual when provided. Synthetic target tests wiring, not accuracy.
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            target = 'This is a short audiovisual clip.' if args.media else 'ready'
            target_ids = processor.tokenizer(target + processor.tokenizer.eos_token,
                                             add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
            prefix_len = batch['input_ids'].shape[-1]
            batch['input_ids'] = torch.cat([batch['input_ids'], target_ids], dim=1)
            batch['attention_mask'] = torch.cat([batch['attention_mask'], torch.ones_like(target_ids)], dim=1)
            labels = batch['input_ids'].clone()
            labels[:, :prefix_len] = -100
            # Recompute multimodal positions for the new full sequence after generation.
            model.rope_deltas = None
            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                target_modules=r'model\.layers\.\d+\.self_attn\.(q_proj|v_proj)',
                task_type=TaskType.CAUSAL_LM, bias='none'))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
            model.train()
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=1e-4)
            loss = model(**batch, labels=labels, use_audio_in_video=use_audio, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            grads = [p for p in trainable if p.grad is not None]
            if not grads or not all(torch.isfinite(p.grad).all().item() for p in grads):
                raise RuntimeError('Missing or nonfinite LoRA gradients')
            changed = next((p for p in grads if p.grad.abs().max().item() > 0), None)
            if changed is None:
                raise RuntimeError('All LoRA gradients are zero')
            before = changed.detach().clone()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0).item()
            optimizer.step()
            updated = not torch.equal(before, changed.detach())
            report['train_step'] = {'loss': loss.item(), 'grad_norm': grad_norm,
                'trainable_parameters': sum(p.numel() for p in trainable), 'parameter_updated': updated,
                'target': target, 'memory': memory(), 'seconds': time.perf_counter() - start}
            if not updated:
                raise RuntimeError('Optimizer step did not change the sampled LoRA parameter')
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
