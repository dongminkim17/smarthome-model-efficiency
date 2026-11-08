#!/usr/bin/env python3
"""Train one task with torchrun DDP, dual evaluation, and complete checkpoints.

This entry point only opens training/development splits. Run both tasks with
scripts/launch_experiments.py; official test evaluation is deliberately separate.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from smarthome_efficiency.checkpointing import (
    atomic_link, capture_rng, inherit_best_checkpoint, resolve_checkpoint, restore_rng,
    save_checkpoint, trim_metrics, write_json,
)
from smarthome_efficiency.data import LABELS, load_records, select_records
from smarthome_efficiency.evaluation import classification_metrics, prediction_agreement
from smarthome_efficiency.model import PilotModel
from smarthome_efficiency.distillation import TeacherLogitCache, teacher_spec, distillation_kl, disagreement_distillation_loss
from smarthome_efficiency.tracking import Tracker, plot_metrics
from smarthome_efficiency.training import (
    configure_determinism, config_fingerprint, epoch_order, learning_rate, load_config, make_schedule,
    rank_update, validate_resume, lora_targets, optimizer_groups, set_optimizer_learning_rate, distillation_weight,
)


def gather_objects(value, world_size):
    if world_size == 1:
        return [value]
    result = [None] * world_size
    dist.all_gather_object(result, value)
    return result


def barrier(world_size):
    if world_size > 1:
        dist.barrier(device_ids=[torch.cuda.current_device()])


def all_reduce(value, world_size, operation=dist.ReduceOp.SUM):
    if world_size > 1:
        dist.all_reduce(value, op=operation)
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', type=Path, help='Full checkpoint/run directory; preserve optimizer and RNG')
    group.add_argument('--init-adapter', type=Path, help='LoRA adapter directory; start a new optimizer/schedule')
    parser.add_argument('--max-steps', type=int, help='Absolute global-step limit for bounded smoke tests')
    parser.add_argument('--train-ids', type=Path, help='Training-only memory-smoke IDs; requires a bounded subset run')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    return args


def memory_probe_records(records, path):
    probe = json.loads(Path(path).read_text())
    ids = probe['ids']
    by_id = {record['id']: record for record in records}
    if (probe.get('split') != 'train' or probe.get('test_set_used') is not False
            or len(ids) != 8 or len(ids) != len(set(ids)) or any(i not in by_id for i in ids)):
        raise ValueError('Memory probe contains invalid or non-training IDs')
    return [by_id[i] for i in ids]


def run(args):
    cfg = load_config(args.config)
    configure_determinism(cfg)
    output = args.output.resolve()
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError('A separate visible CUDA device is required for every local rank')
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(cfg['num_threads'])
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    device = torch.device('cuda', local_rank)
    started = time.perf_counter()
    elapsed_before = 0.0
    tracker = None
    global_step = 0
    report = {'status': 'initializing', 'config': cfg, 'world_size': world,
              'test_set_used': False, 'started_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        resume_path = resolve_checkpoint(args.resume) if args.resume else None
        # training_state.pt is a trusted local artifact produced by this project.
        saved = torch.load(resume_path / 'training_state.pt', map_location='cpu', weights_only=False) if resume_path else None
        if saved is not None:
            validate_resume(saved['config'], cfg, saved['world_size'], world)
            global_step = saved['global_step']
            elapsed_before = saved['elapsed_seconds']
        if rank == 0:
            if output.exists() and any(output.iterdir()) and saved is None:
                raise ValueError(f'Output is not empty: {output}; choose a new run or --resume')
            output.mkdir(parents=True, exist_ok=True)
            if saved is not None:
                source_output = resume_path.parent.parent
                if output == source_output:
                    later = [path for path in (output/'checkpoints').glob('step-*')
                             if path.name[5:].isdigit() and int(path.name[5:]) > global_step]
                    if later:
                        raise ValueError('Resuming an older checkpoint requires a new --output directory; later checkpoints already exist')
                if output != source_output:
                    if any(output.iterdir()):
                        raise ValueError('A different continuation output must be empty')
                    (output / 'checkpoints').mkdir(exist_ok=True)
                    atomic_link(output / 'checkpoints/latest', resume_path)
                    source_metrics = source_output / 'metrics.jsonl'
                    if source_metrics.exists():
                        (output / 'metrics.jsonl').write_text(source_metrics.read_text())
                    inherit_best_checkpoint(source_output, output, saved.get('best_step'))
                trim_metrics(output / 'metrics.jsonl', global_step)
            (output / 'predictions').mkdir(exist_ok=True)
            write_json(output / 'resolved_config.json', cfg)
        barrier(world)
        tracker = Tracker(output, enabled=rank == 0, purge_step=global_step + 1 if saved else None)

        def emit(event, **values):
            record = {'event': event, 'global_step': global_step,
                      'elapsed_seconds': elapsed_before + time.perf_counter()-started, **values}
            if rank == 0:
                tracker.log(record)
                print(json.dumps(record, allow_nan=False), flush=True)
            return record

        full_train = load_records(cfg['task'], 'train', seed=cfg['split_seed'], data_root=cfg['data_root'], mer_partition=cfg.get('mer_partition'))
        full_dev = load_records(cfg['task'], 'dev', seed=cfg['split_seed'], data_root=cfg['data_root'], mer_partition=cfg.get('mer_partition'))
        train = select_records(full_train, cfg['train_limit'], seed=cfg['split_seed'])
        if getattr(args, 'train_ids', None) is not None:
            if args.max_steps is None or args.max_steps > 2 or cfg['epochs'] > 2 or cfg['train_limit'] is None or cfg['dev_limit'] is None:
                raise ValueError('Explicit training IDs are restricted to bounded subset smoke runs')
            train = memory_probe_records(full_train, args.train_ids)
        dev = select_records(full_dev, cfg['dev_limit'], seed=cfg['split_seed'])
        if set(x['id'] for x in train) & set(x['id'] for x in dev):
            raise RuntimeError('Training/development overlap')
        manifest = {'train': train, 'dev': dev, 'split_seed': cfg['split_seed'],
                    'full_train_count': len(full_train), 'full_dev_count': len(full_dev)}
        data_fingerprint = config_fingerprint(manifest)
        if saved is not None and saved['data_fingerprint'] != data_fingerprint:
            raise ValueError('Dataset records differ from the checkpoint; full resume would not be reproducible')
        compact_sidecar = Path(cfg['model']) / 'compression.json'
        base_manifest_sha256 = hashlib.sha256(compact_sidecar.read_bytes()).hexdigest() if compact_sidecar.is_file() else None
        if saved is not None and saved.get('base_model_manifest_sha256') != base_manifest_sha256:
            raise ValueError('Compact base checkpoint changed since the saved training state')
        labels = list(LABELS[cfg['task']])
        label_to_index = {label: i for i, label in enumerate(labels)}
        global_batch = world * cfg['micro_batch_size'] * cfg['gradient_accumulation_steps']
        steps_per_epoch = math.ceil(len(train) / global_batch)
        total_steps = steps_per_epoch * cfg['epochs']
        kd_schedule_total_steps = saved.get('kd_schedule_total_steps',total_steps) if saved is not None else total_steps
        eval_interval = max(1, round(steps_per_epoch * cfg['eval_every_fraction']))
        if global_step > total_steps:
            raise ValueError('Checkpoint is beyond the requested epoch budget')
        source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [Path(__file__).resolve(), *sorted((ROOT/'src/smarthome_efficiency').glob('*.py'))]}
        if saved is not None and saved.get('source_sha256') != source_hashes:
            emit('source_changed', warning='Source hashes changed since checkpoint; numerical identity is not guaranteed')
        if rank == 0:
            write_json(output / 'data_manifest.json', manifest)
            write_json(output / 'provenance.json', {'source_sha256': source_hashes,
                       'python': sys.version, 'torch': torch.__version__, 'labels': labels,
                       'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                       'world_size': world, 'data_fingerprint': data_fingerprint,
                       'resume_from': str(resume_path) if resume_path else None,
                       'init_adapter': str(args.init_adapter.resolve()) if args.init_adapter else None,
                       'global_batch_size': global_batch, 'base_model_manifest_sha256': base_manifest_sha256})
        random.seed(cfg['seed'] + rank)
        np.random.seed(cfg['seed'] + rank)
        torch.manual_seed(cfg['seed'] + rank)
        runner = PilotModel(cfg['model'], cfg['task'], labels, cfg['cache_dir'],
                            max_frames=cfg['max_frames'], max_pixels=cfg['max_pixels'],
                            use_transcript=cfg['use_transcript'], device_index=local_rank,
                            gradient_checkpointing=cfg.get('gradient_checkpointing', True))
        teacher_cache = None
        if cfg['objective'] == 'sft_kd':
            spec=teacher_spec(cfg,cfg['teacher_checkpoint'],labels,runner.label_token_ids,runner.versions)
            teacher_cache=TeacherLogitCache(cfg['teacher_cache'],spec,train)
        feature_cache = None
        if cfg.get('feature_teacher_cache'):
            from smarthome_efficiency.feature_distillation import TeacherFeatureCache, feature_spec
            spec = feature_spec(cfg, cfg['teacher_checkpoint'], labels, runner.label_token_ids, runner.versions)
            feature_cache = TeacherFeatureCache(cfg['feature_teacher_cache'], spec, train)
        feature_cache_sha256 = feature_cache.sha256 if feature_cache is not None else None
        if saved is not None and saved.get('feature_cache_sha256') != feature_cache_sha256:
            raise ValueError('Encoder feature cache changed since checkpoint')
        teacher_cache_sha256=teacher_cache.sha256 if teacher_cache is not None else None
        if saved is not None and saved.get('teacher_cache_sha256') != teacher_cache_sha256:
            raise ValueError('Teacher logit cache changed since the saved training state')
        if rank==0:
            provenance=json.loads((output/'provenance.json').read_text())
            provenance.update(objective=cfg['objective'],teacher_cache_sha256=teacher_cache_sha256, feature_cache_sha256=feature_cache_sha256)
            write_json(output/'provenance.json',provenance)
        runner.load_model()
        base_parameters = dict(runner.parameters_info)
        if resume_path or args.init_adapter:
            adapter_path = resume_path / 'adapter' if resume_path else args.init_adapter
            runner.load_adapter(adapter_path, trainable=True)
            adapter_cfg = runner.model.peft_config['default']
            expected_target = lora_targets(cfg['lora_target'])
            if (adapter_cfg.r, adapter_cfg.lora_alpha, adapter_cfg.lora_dropout, adapter_cfg.target_modules) != (
                    cfg['lora_rank'], cfg['lora_alpha'], cfg['lora_dropout'], expected_target):
                raise ValueError('Adapter scope/rank/alpha/dropout does not match this configuration')
        else:
            target = lora_targets(cfg['lora_target'])
            runner.enable_lora(rank=cfg['lora_rank'], alpha=cfg['lora_alpha'],
                               dropout=cfg['lora_dropout'], target_modules=target)
        if runner.model.is_gradient_checkpointing != cfg.get('gradient_checkpointing', True):
            raise RuntimeError('Gradient checkpointing state does not match the configuration')
        model = (DistributedDataParallel(runner.model, device_ids=[local_rank], output_device=local_rank,
                 broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
                 if world > 1 else runner.model)
        parameters = [p for p in runner.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(optimizer_groups(runner.model, cfg), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
                                      betas=(0.9, 0.999), eps=1e-8)
        next_epoch, next_update = 0, 0
        best_score, best_step = None, None
        last_validation = None
        schedule = make_schedule(cfg, total_steps)
        if saved is not None:
            optimizer.load_state_dict(saved['optimizer'])
            next_epoch, next_update = saved['next_epoch'], saved['next_update']
            best_score, best_step = saved['best_score'], saved['best_step']
            last_validation = saved['last_validation']
            schedule = saved['schedule']
            if cfg['epochs'] > saved['config']['epochs']:
                schedule = make_schedule(cfg, total_steps, start_step=global_step)
                emit('schedule_restarted', reason='epoch_budget_extended', schedule=schedule)
            restore_rng(saved['rng_states'][rank])
        del saved
        report.update({'status': 'running', 'labels': labels, 'base_parameters': base_parameters,
                       'trainable_parameters': sum(p.numel() for p in parameters),
                       'objective': cfg['objective'], 'teacher_cache_sha256': teacher_cache_sha256, 'feature_cache_sha256': feature_cache_sha256,
                       'train_count': len(train), 'dev_count': len(dev), 'global_batch_size': global_batch,
                       'steps_per_epoch': steps_per_epoch, 'requested_steps': total_steps,
                       'resume_from': str(resume_path) if resume_path else None})
        emit('run_start' if not resume_path else 'run_resume', world_size=world, local_device_count=torch.cuda.device_count(),
             train_count=len(train), dev_count=len(dev), global_batch_size=global_batch,
             steps_per_epoch=steps_per_epoch, total_steps=total_steps, eval_modes=cfg['eval_modes'],
             trainable_parameters=report['trainable_parameters'], micro_batch_size=cfg['micro_batch_size'],
             gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
             gradient_checkpointing=cfg.get('gradient_checkpointing', True),
             epoch=next_epoch + next_update/steps_per_epoch, objective=cfg['objective'],
             distillation_alpha=cfg['distillation_alpha'], distillation_temperature=cfg['distillation_temperature'],
             teacher_cache_sha256=teacher_cache_sha256)

        def evaluate(epoch_fraction, phase):
            emit('phase', phase='validating', epoch=epoch_fraction)
            barrier(world)
            begin = time.perf_counter()
            records = []
            runner.model.eval()
            for index in range(rank, len(dev), world):
                record = dev[index]
                try:
                    result = runner.evaluate_batch(runner.prepare(record), label_to_index[record['label']],
                                                   modes=cfg['eval_modes'], max_new_tokens=cfg['max_new_tokens'])
                except Exception as error:
                    raise RuntimeError(f'Validation failed on {record["id"]} on rank {rank}') from error
                records.append({'id': record['id'], 'true_label': record['label'], **result})
            gathered = gather_objects(records, world)
            rows = sorted([row for shard in gathered for row in shard], key=lambda row: row['id'])
            if len(rows) != len(dev) or {row['id'] for row in rows} != {record['id'] for record in dev}:
                raise RuntimeError('Distributed validation lost or duplicated examples')
            result = {'epoch': epoch_fraction, 'phase': phase, 'n': len(rows),
                      **{key: float(np.mean([row[key] for row in rows])) for key in ('sft_loss', 'class_loss', 'eos_loss')}}
            truth = [row['true_label'] for row in rows]
            for mode in cfg['eval_modes']:
                result[mode] = classification_metrics(truth, [row[f'{mode}_predicted_label'] for row in rows], labels)
            if len(cfg['eval_modes']) == 2:
                result['agreement'] = prediction_agreement([row['logit_predicted_label'] for row in rows],
                                                           [row['greedy_predicted_label'] for row in rows])
            times = torch.tensor([time.perf_counter()-begin], dtype=torch.float64, device=device)
            all_reduce(times, world, dist.ReduceOp.MAX)
            result['seconds'] = float(times.item())
            if rank == 0:
                path = output / 'predictions' / f'dev-step-{global_step:08d}.jsonl'
                path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows))
            return emit('validation', **result)

        def checkpoint(validation, epoch_index, update_index):
            nonlocal best_score, best_step, last_validation
            metric = 'accuracy' if cfg['task'].startswith('mintrec') else 'weighted_f1'
            score = validation[cfg['selection_mode']][metric]
            is_best = best_score is None or score > best_score
            if is_best:
                best_score, best_step = score, global_step
            last_validation = validation
            emit('phase', phase='saving', epoch=validation['epoch'])
            rng_states = gather_objects(capture_rng(), world)
            if rank == 0:
                state = {'format_version': 1, 'config': cfg, 'world_size': world,
                         'base_model_manifest_sha256': base_manifest_sha256,
                         'teacher_cache_sha256': teacher_cache_sha256, 'feature_cache_sha256': feature_cache_sha256,
                         'data_fingerprint': data_fingerprint, 'global_step': global_step,
                         'next_epoch': epoch_index, 'next_update': update_index,
                         'optimizer': optimizer.state_dict(), 'rng_states': rng_states,
                         'schedule': schedule, 'kd_schedule_total_steps':kd_schedule_total_steps, 'best_score': best_score, 'best_step': best_step,
                         'last_validation': validation, 'source_sha256': source_hashes,
                         'elapsed_seconds': elapsed_before + time.perf_counter()-started}
                path = save_checkpoint(output, runner.model, state, is_best=is_best)
                emit('checkpoint', path=str(path), best_score=best_score, best_step=best_step,
                     epoch=validation['epoch'])
                try:
                    plot_metrics(output)
                except Exception as error:
                    emit('plot_warning', error=str(error))
                report.update({'global_step': global_step, 'best_score': best_score, 'best_step': best_step,
                               'last_validation': validation, 'latest_checkpoint': str(path)})
                write_json(output / 'report.json', report)
            barrier(world)
            # Logging/plotting must not change the random stream captured for resume.
            restore_rng(rng_states[rank])

        if resume_path is None:
            if compact_sidecar.is_file() and args.init_adapter is None:
                if any(torch.count_nonzero(p).item() for n,p in runner.model.named_parameters() if 'lora_B' in n):
                    raise RuntimeError('Pre-recovery validation requires zero-initialized LoRA updates')
            validation = evaluate(0.0, 'baseline')
            if rank == 0 and compact_sidecar.is_file() and args.init_adapter is None:
                write_json(output / 'pruned_dev_before_recovery.json', {
                    **validation, 'task': cfg['task'], 'split': 'dev', 'test_set_used': False,
                    'recovery_updates': 0, 'fresh_lora_update_is_zero': True,
                    'model': cfg['model'], 'base_model_manifest_sha256': base_manifest_sha256,
                    'parameters': base_parameters,
                })
            checkpoint(validation, 0, 0)
        last_eval_step = global_step
        torch.cuda.reset_peak_memory_stats()
        stopped = False
        for epoch in range(next_epoch, cfg['epochs']):
            order = epoch_order(len(train), cfg['seed'], epoch)
            start_update = next_update if epoch == next_epoch else 0
            for update in range(start_update, steps_per_epoch):
                if args.max_steps is not None and global_step >= args.max_steps:
                    stopped = True
                    break
                begin = time.perf_counter()
                micro_size = cfg['micro_batch_size']
                indices, valid_count = rank_update(order, update, world, rank,
                                                   cfg['gradient_accumulation_steps'] * micro_size)
                lr = learning_rate(schedule, global_step)
                alpha = distillation_weight(cfg, global_step, kd_schedule_total_steps)
                set_optimizer_learning_rate(optimizer, lr)
                optimizer.zero_grad(set_to_none=True)
                totals = torch.zeros(7, dtype=torch.float64, device=device)
                feature_totals = torch.zeros(2, dtype=torch.float64, device=device) if feature_cache is not None else None
                disagreement_totals = (torch.zeros(2, dtype=torch.float64, device=device)
                                       if 'distillation_disagreement_alpha' in cfg else None)
                for micro in range(cfg['gradient_accumulation_steps']):
                    chunk = indices[micro * micro_size:(micro + 1) * micro_size]
                    real_indices = [index for index in chunk if index is not None]
                    records = [train[index] for index in real_indices] or [train[0]]
                    sync_context = model.no_sync() if world > 1 and micro < cfg['gradient_accumulation_steps']-1 else nullcontext()
                    with sync_context:
                        try:
                            batches = [runner.prepare(record) for record in records]
                            targets = [label_to_index[record['label']] for record in records]
                            kd_args={'return_class_gradients':True} if teacher_cache is not None else {}
                            if feature_cache is not None: kd_args['return_encoder_features'] = True
                            if disagreement_totals is not None: kd_args['return_per_example_sft'] = True
                            losses = (runner.loss_batch(batches[0], targets[0], forward_model=model, train=True, **kd_args)
                                      if len(batches) == 1 else
                                      runner.loss_microbatch(batches, targets, forward_model=model, train=True, **kd_args))
                        except Exception as error:
                            raise RuntimeError(f"Training failed on {[r['id'] for r in records]} on rank {rank}") from error
                        kd_loss=torch.zeros((),device=device)
                        gated_kd = None
                        if teacher_cache is not None:
                            teacher_logits=teacher_cache.batch(records,device)
                            if disagreement_totals is None:
                                # Preserve ordinary KD arithmetic for all historical configs.
                                kd_loss=distillation_kl(losses.pop('class_logits_for_kd'),teacher_logits,cfg['distillation_temperature'])
                                losses['loss']=(1-alpha)*losses['sft_loss']+alpha*kd_loss
                            else:
                                gated_kd = disagreement_distillation_loss(
                                    losses.pop('sft_per_example_for_kd'), losses.pop('class_logits_for_kd'),
                                    teacher_logits, targets, cfg['distillation_temperature'], alpha,
                                    cfg['distillation_disagreement_alpha'],
                                )
                                kd_loss = gated_kd['distillation_loss']
                                losses['loss'] = gated_kd['loss']
                        feature_losses = None
                        if feature_cache is not None:
                            feature_losses = feature_cache.losses(losses.pop('encoder_features_for_kd'), records)
                            losses['loss'] = (losses['loss'] + cfg['feature_audio_weight']*feature_losses['audio']
                                              + cfg['feature_vision_weight']*feature_losses['vision'])
                        if not torch.isfinite(losses['loss']).item():
                            raise RuntimeError('Nonfinite training loss')
                        # Each forward averages its real examples; DDP averages ranks.
                        scale = world * len(real_indices) / valid_count
                        (losses['loss'] * scale).backward()
                    if real_indices:
                        totals[:3] += len(real_indices) * torch.stack([losses[key].detach().double() for key in ('sft_loss','class_loss','eos_loss')])
                        totals[3] += len(real_indices)
                        totals[4] += sum(int(b['attention_mask'].sum()) for b in batches)
                        totals[5] += len(real_indices)*losses['loss'].detach().double()
                        totals[6] += len(real_indices)*kd_loss.detach().double()
                        if gated_kd is not None:
                            disagreement_totals[0] += gated_kd['disagreement'].sum().double()
                            disagreement_totals[1] += gated_kd['effective_alpha'].sum().double()
                        if feature_losses is not None:
                            feature_totals += len(real_indices)*torch.stack([feature_losses[k].detach().double() for k in ('audio','vision')])
                    del losses, batches
                if global_step == 0 and cfg['lora_target'] in ('decoder_encoder_ffn','decoder_encoder_linear'):
                    from smarthome_efficiency.multimodal_pruning import adapter_gradient_coverage
                    coverage = adapter_gradient_coverage(runner.model)
                    emit('adapter_gradient_coverage', components=coverage)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm']).item())
                if not math.isfinite(grad_norm):
                    raise RuntimeError('Nonfinite gradient norm')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                all_reduce(totals, world)
                feature_metrics = {}
                if feature_totals is not None:
                    all_reduce(feature_totals, world)
                    feature_metrics = {'audio_feature_loss': float(feature_totals[0]/totals[3]),
                                       'vision_feature_loss': float(feature_totals[1]/totals[3])}
                disagreement_metrics = {}
                if disagreement_totals is not None:
                    all_reduce(disagreement_totals, world)
                    disagreement_metrics = {
                        'teacher_disagreement_count': int(disagreement_totals[0].item()),
                        'teacher_disagreement_rate': float(disagreement_totals[0] / totals[3]),
                        'distillation_alpha_effective_mean': float(disagreement_totals[1] / totals[3]),
                    }
                torch.cuda.synchronize()
                info = torch.tensor([time.perf_counter()-begin, torch.cuda.max_memory_allocated()/2**30,
                                     torch.cuda.max_memory_reserved()/2**30], dtype=torch.float64, device=device)
                all_reduce(info, world, dist.ReduceOp.MAX)
                seen = min((update+1)*global_batch, len(train))
                epoch_fraction = epoch + seen/len(train)
                component_metrics = ({'encoder_learning_rate': lr * cfg['encoder_learning_rate']/cfg['learning_rate']}
                                     if cfg.get('encoder_learning_rate') is not None else {})
                emit('train', epoch=epoch_fraction, loss=float(totals[5]/totals[3]),
                     sft_loss=float(totals[0]/totals[3]), distillation_loss=float(totals[6]/totals[3]),
                     class_loss=float(totals[1]/totals[3]), eos_loss=float(totals[2]/totals[3]),
                     learning_rate=lr, distillation_alpha=alpha, grad_norm=grad_norm, examples=int(totals[3].item()),
                     step_seconds=float(info[0]), examples_per_second=float(totals[3]/info[0]),
                     input_tokens=int(totals[4].item()), input_tokens_per_second=float(totals[4]/info[0]),
                     peak_allocated_gib=float(info[1]), peak_reserved_gib=float(info[2]), **feature_metrics, **component_metrics, **disagreement_metrics)
                next_epoch_state = epoch+1 if update+1 == steps_per_epoch else epoch
                next_update_state = 0 if update+1 == steps_per_epoch else update+1
                should_stop = args.max_steps is not None and global_step >= args.max_steps
                if (update+1) % eval_interval == 0 or update+1 == steps_per_epoch or should_stop:
                    validation = evaluate(epoch_fraction, 'training')
                    checkpoint(validation, next_epoch_state, next_update_state)
                    last_eval_step = global_step
                if should_stop:
                    stopped = True
                    break
            if stopped:
                break
        if global_step != last_eval_step:
            raise RuntimeError('Last optimizer update was not checkpointed')
        status = 'paused' if global_step < total_steps else 'completed'
        emit('run_complete', status=status, epoch=global_step/steps_per_epoch, best_score=best_score, best_step=best_step)
        if rank == 0:
            report.update({'status': status, 'global_step': global_step, 'best_score': best_score, 'best_step': best_step,
                           'last_validation': last_validation,
                           'total_elapsed_seconds': elapsed_before+time.perf_counter()-started,
                           'session_seconds': time.perf_counter()-started,
                           'latest_checkpoint': str((output/'checkpoints/latest').resolve())})
            write_json(output / 'report.json', report)
            try:
                plot_metrics(output)
            except Exception as error:
                emit('plot_warning', error=str(error))
        barrier(world)
    except Exception as error:
        if rank == 0 and output.exists():
            failure = {'status': 'failed', 'global_step': global_step, 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc()}
            write_json(output / 'failure.json', failure)
        raise
    finally:
        if tracker is not None:
            tracker.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    run(parse_args())
