"""Answer-only LoRA training, one example per GPU with gradient accumulation.

Supports torchrun DDP (one full decoder replica per GPU), adapter-only saves,
optimizer resume, optional validation loss, and identical seeded ablations.
Scene tokens are never sampled or truncated to fit a budget.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from ..geometry.tsdf import FREE, UNKNOWN
from .common import VARIANTS, check_rows, load_index, load_scene, read_jsonl, sha256, write_json


class SceneCache:
    def __init__(self, index, size=2):
        self.index, self.size, self.cache = index, size, OrderedDict()

    def __getitem__(self, key):
        if key not in self.cache:
            self.cache[key] = load_scene(self.index[key])
            while len(self.cache) > self.size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--train-qa', type=Path, required=True)
    p.add_argument('--val-qa', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path, help='Checkpoint directory; model settings come from this checkpoint')
    p.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--model-revision', default='main')
    p.add_argument('--variant', choices=VARIANTS, default='surface_both')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=0, help='Total scheduled update limit; 0 = all epochs')
    p.add_argument('--gradient-accumulation', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup-ratio', type=float, default=.03)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--lora-alpha', type=float, default=32.)
    p.add_argument('--lora-dropout', type=float, default=.05)
    p.add_argument('--rope-metres-per-unit', type=float, default=.02)
    p.add_argument('--max-sequence-tokens', type=int, default=262144)
    p.add_argument('--max-answer-tokens', type=int, default=2048)
    p.add_argument('--attention', choices=['sdpa', 'flash_attention_2'], default='sdpa')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    p.add_argument('--no-gradient-checkpointing', action='store_true')
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Validate every N steps in addition to epoch ends and the final step')
    p.add_argument('--eval-max-examples', type=int, default=128, help='0 = entire validation set')
    p.add_argument('--distributed-timeout-minutes', type=float, default=60.)
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--scene-cache-size', type=int, default=2)
    p.add_argument('--preflight-only', action='store_true',
                   help='Count scene tokens for the selected QA and variant without loading features or the model')
    p.add_argument('--seed', type=int, default=42)
    return p


def count_scene_tokens(index, keys, variant):
    counts = {}
    for key in keys:
        with np.load(index[key]['regions'], allow_pickle=False) as geometry:
            count = len(geometry['surface_index'])
            if variant != 'surface':
                labels = geometry['region_state']
                if variant in ('surface_free', 'surface_both'):
                    count += int(np.count_nonzero(labels == FREE))
                if variant in ('surface_unknown', 'surface_both'):
                    count += int(np.count_nonzero(labels == UNKNOWN))
            counts[key] = count
    return counts


def check_scene_context(counts, limit):
    oversized = [(key, count) for key, count in counts.items() if count >= limit]
    if oversized:
        details = ', '.join(f'{key}: {count:,}' for key, count in oversized[:20])
        suffix = f' (+{len(oversized) - 20} more)' if len(oversized) > 20 else ''
        raise ValueError(f'{len(oversized)} region(s) reach or exceed the {limit:,}-token context '
                         f'with scene tokens alone ({details}{suffix}). Prompt and answer tokens add to these counts. '
                         'Rebuild at a coarser resolution or use a supported larger context.')


def lock_output_directory(path, resume):
    import fcntl

    path.mkdir(parents=True, exist_ok=True)
    lock_path = path / '.training.lock'
    lock = lock_path.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        same_last = resume and resume.resolve() == (path / 'last').resolve()
        if not same_last and any(item != lock_path for item in path.iterdir()):
            raise ValueError('Training output must be empty unless resuming its last checkpoint')
    except BlockingIOError as exc:
        lock.close()
        raise ValueError(f'Another training process is using {path}') from exc
    except Exception:
        lock.close()
        raise
    return lock


def main():
    args = parser().parse_args()
    for key in ('epochs', 'gradient_accumulation', 'scene_cache_size', 'lr', 'log_every',
                'max_grad_norm', 'distributed_timeout_minutes', 'max_sequence_tokens'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('max_steps', 'save_every', 'eval_every', 'eval_max_examples', 'weight_decay'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError(f'{key} must be nonnegative and finite')
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError('Warmup ratio must be between zero and one')
    index = load_index(args.index)
    rows = read_jsonl(args.train_qa)
    check_rows(rows, index)
    val = read_jsonl(args.val_qa) if args.val_qa else []
    if args.val_qa:
        check_rows(val, index)
        if {r['scene_id'] for r in rows} & {r['scene_id'] for r in val}:
            raise ValueError('Training and validation scenes overlap')
    used_regions = sorted({r['region_id'] for r in rows + val})
    # Read only geometry before feature checksums, CUDA initialization, or model downloads.
    preflight_settings = (json.loads((args.resume / 'model_config.json').read_text()) if args.resume
                          else {'variant': args.variant, 'max_sequence_tokens': args.max_sequence_tokens})
    scene_counts = count_scene_tokens(index, used_regions, preflight_settings['variant'])
    check_scene_context(scene_counts, preflight_settings['max_sequence_tokens'])
    if args.preflight_only:
        print(json.dumps({'variant': preflight_settings['variant'], 'scene_tokens': scene_counts,
                          'limit': preflight_settings['max_sequence_tokens'],
                          'scope': 'geometry only; prompt and answer tokens are additional'}, indent=2))
        return
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler

    from .model import ModelConfig, SpatialVLM

    rank, local_rank, world = (int(os.getenv(k, d)) for k, d in [('RANK', '0'), ('LOCAL_RANK', '0'), ('WORLD_SIZE', '1')])
    if not torch.cuda.is_available():
        raise RuntimeError('Training CLI requires a CUDA GPU')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=args.distributed_timeout_minutes))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_lock = lock_output_directory(args.output, args.resume) if rank == 0 else None
    if world > 1:
        dist.barrier()
    cache = SceneCache(index, args.scene_cache_size)
    feature_hashes = {key: sha256(index[key]['features']) for key in used_regions}
    source_identity = {'train_qa': sha256(args.train_qa),
                       'val_qa': sha256(args.val_qa) if args.val_qa else None,
                       'index': sha256(args.index), 'feature_caches': feature_hashes}
    training_options = {key: getattr(args, key) for key in (
        'epochs', 'max_steps', 'gradient_accumulation', 'lr', 'weight_decay', 'warmup_ratio',
        'max_grad_norm', 'no_gradient_checkpointing', 'eval_every', 'eval_max_examples', 'seed')}
    restored = None
    if args.resume:
        restored = torch.load(args.resume / 'training.pt', map_location='cpu', weights_only=True)
        old_identity = restored['source_identity']
        if (any(source_identity.get(key) != value for key, value in old_identity.items())
                or restored['world_size'] != world):
            raise ValueError('Resume requires the same QA, index, feature caches, and number of GPU processes')
        old_options = restored.get('training_options')
        if old_options is None:
            # Legacy checkpoints did not record the training options in the checkpoint itself.
            run_manifest = args.resume.parent / 'run.json'
            if run_manifest.is_file():
                old_options = json.loads(run_manifest.read_text())['arguments']
        changed = [key for key, value in (old_options or {}).items()
                   if key in training_options and value != training_options[key]]
        if changed:
            raise ValueError(f'Resume requires unchanged training options: {", ".join(changed)}')
        settings = restored.get('model_config')
        if settings is None:
            settings = json.loads((args.resume / 'model_config.json').read_text())
        config = ModelConfig(**settings)
    else:
        first = cache[rows[0]['region_id']]
        config = ModelConfig(base_model=args.model, base_revision=args.model_revision,
            encoder=first['encoder'], encoder_revision=first['encoder_revision'], feature_dim=first['feature_dim'],
            variant=args.variant, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, rope_metres_per_unit=args.rope_metres_per_unit,
            max_sequence_tokens=args.max_sequence_tokens, max_answer_tokens=args.max_answer_tokens,
            attention=args.attention, dtype=args.dtype)
    core = SpatialVLM(config, device, gradient_checkpointing=not args.no_gradient_checkpointing)
    check_scene_context(scene_counts, core.max_length)
    if restored is not None:
        if 'adapter' in restored:
            expected = {name for name, value in core.named_parameters() if value.requires_grad}
            if set(restored['adapter']) != expected:
                raise ValueError('Checkpoint adapter parameters do not match the model configuration')
            core.load_state_dict(restored['adapter'], strict=False)
        else:
            core.load_adapter(args.resume)
    trainable = [p for p in core.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=config.dtype == 'float16')
    global_step, first_epoch, first_batch, best_val = 0, 0, 0, float('inf')
    skipped_steps = 0
    last_eval_step = None
    if restored is not None:
        optimizer.load_state_dict(restored['optimizer'])
        scaler.load_state_dict(restored['scaler'])
        global_step, first_epoch, first_batch = restored['global_step'], restored['epoch'], restored['next_batch']
        best_val = restored['best_val']
        skipped_steps = restored.get('skipped_steps', 0)
        last_eval_step = restored.get('last_eval_step')
        if 'training_options' not in restored and val:
            # A legacy best loss was measured on the first N rows rather than a seeded sample.
            best_val = float('inf')
        del restored
    model = DistributedDataParallel(core, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else core
    torch.manual_seed(args.seed + rank + global_step)
    sampler = DistributedSampler(rows, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(rows, batch_size=1, sampler=sampler, collate_fn=lambda batch: batch[0], num_workers=0)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup = round(total_steps * args.warmup_ratio)
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[config.dtype]
    amp = lambda: torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32)
    train_scenes = sorted({r['scene_id'] for r in rows})
    val_scenes = sorted({r['scene_id'] for r in val})
    chosen_val = val
    if args.eval_max_examples and len(val) > args.eval_max_examples:
        chosen_val = random.Random(args.seed).sample(val, args.eval_max_examples)
    if rank == 0:
        write_json(args.output / 'run.json', {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   'model': asdict(config), 'source_identity': source_identity, 'train_scenes': train_scenes,
                   'val_scenes': val_scenes, 'world_size': world,
                   'effective_examples_per_step': world * args.gradient_accumulation,
                   'scene_tokens': scene_counts,
                   'trainable_parameters': sum(p.numel() for p in trainable), 'status': 'running',
                   'feature_caches': feature_hashes})

    def save(directory, epoch, next_batch):
        if rank == 0:
            core.save(directory, {'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                       'global_step': global_step, 'epoch': epoch, 'next_batch': next_batch,
                       'best_val': best_val, 'source_identity': source_identity, 'world_size': world,
                       'last_eval_step': last_eval_step, 'skipped_steps': skipped_steps,
                       'training_options': training_options,
                       # Keep resume weights and optimizer state in the same atomic file.
                       'adapter': core.adapter_state(), 'model_config': asdict(config)})
            write_json(Path(directory) / 'data_manifest.json', {'train_scenes': train_scenes,
                       'val_scenes': val_scenes, 'source_identity': source_identity,
                       'seed': args.seed, 'variant': config.variant, 'global_step': global_step,
                       'training_options': training_options, 'world_size': world,
                       'adapter_sha256': sha256(Path(directory) / 'adapter.pt')})

    def evaluate(epoch, next_batch):
        nonlocal best_val, last_eval_step
        if not val or last_eval_step == global_step:
            return
        core.eval()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.no_grad(), amp():
            for row in chosen_val[rank::world]:
                totals[0] += core(cache[row['region_id']], row['question'], row['answer'],
                                  history=row.get('history', ())).detach()
                totals[1] += 1
        if world > 1:
            dist.all_reduce(totals)
        loss_sum, count = totals.tolist()
        value = loss_sum / count
        if not math.isfinite(value):
            raise FloatingPointError('Non-finite validation loss')
        last_eval_step = global_step
        if rank == 0:
            record = {'step': global_step, 'validation_answer_loss': value, 'examples': int(count)}
            print(json.dumps(record), flush=True)
            with (args.output / 'validation.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            if value < best_val:
                best_val = value
                save(args.output / 'best', epoch, next_batch)
        core.train()
        if world > 1:
            dist.barrier()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_epoch, next_batch = first_epoch, first_batch
    begin = perf_counter()
    for epoch in range(first_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for batch_idx, row in enumerate(loader):
            if epoch == first_epoch and batch_idx < first_batch:
                continue
            if global_step >= total_steps:
                break
            group_start = (batch_idx // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(loader) - group_start)
            update = (batch_idx + 1) % args.gradient_accumulation == 0 or batch_idx + 1 == len(loader)
            synchronization = model.no_sync() if world > 1 and not update else nullcontext()
            with synchronization:
                with amp():
                    loss = model(cache[row['region_id']], row['question'], row['answer'],
                                 history=row.get('history', ()))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite loss for {row["id"]}')
                    scaled_loss = loss / group_size
                scaler.scale(scaled_loss).backward()
            last_epoch, next_batch = epoch, batch_idx + 1
            if not update:
                continue
            step = global_step + 1
            factor = step / max(1, warmup) if step <= warmup else .5 * (1 + math.cos(math.pi * max(0, step - 1 - warmup) / max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group['lr'] = args.lr * factor
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm,
                                           error_if_nonfinite=not scaler.is_enabled())
            old_scale = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            skipped = old_scale is not None and scaler.get_scale() < old_scale
            skipped_steps += int(skipped)
            optimizer.zero_grad(set_to_none=True)
            global_step = step
            if global_step % args.log_every == 0:
                scalar = loss.detach().float()
                if world > 1:
                    dist.all_reduce(scalar)
                    scalar /= world
                if rank == 0:
                    record = {'step': global_step, 'epoch': epoch, 'last_microbatch_loss': float(scalar),
                              'lr': args.lr * factor, 'sequence_tokens_rank0': core.last_sequence_length,
                              'skipped_optimizer_update': skipped,
                              'peak_gpu_gb_rank0': torch.cuda.max_memory_allocated(device) / 1e9,
                              'elapsed_seconds': perf_counter() - begin}
                    print(json.dumps(record), flush=True)
                    with (args.output / 'train.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
            if args.eval_every > 0 and global_step % args.eval_every == 0:
                evaluate(epoch, next_batch)
            if args.save_every > 0 and global_step % args.save_every == 0:
                save(args.output / 'last', epoch, next_batch)
                if world > 1:
                    dist.barrier()
        if next_batch == len(loader):
            last_epoch, next_batch = epoch + 1, 0
            evaluate(last_epoch, 0)
        if global_step >= total_steps:
            break
    evaluate(last_epoch, next_batch)
    save(args.output / 'last', last_epoch, next_batch)
    if rank == 0:
        path = args.output / 'run.json'
        manifest = json.loads(path.read_text())
        manifest.update(status='complete', scheduled_steps=global_step,
                        optimizer_steps=global_step - skipped_steps, skipped_optimizer_steps=skipped_steps)
        write_json(path, manifest)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if output_lock is not None:
        output_lock.close()


if __name__ == '__main__':
    main()
