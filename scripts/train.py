"""Fine-tune one variant: scene-token embeddings + LoRA on the frozen Qwen3-VL decoder, answer-only loss.

Variants: surface_only, surface_free, surface_unknown, surface_both. Every variant uses the same seed, so the
same initialization, training examples and example order; they differ only in which box tokens they see.
One example per forward pass (scenes have 1.5k-19k tokens), gradients accumulated over --grad-accum examples.
After every epoch: validation (generation + VSI-Bench scores) and a checkpoint of the trainable weights;
best.pt is the epoch with the highest validation mean over tasks.

Outputs (--output): args.json, log.jsonl (training loss per optimizer step, validation per epoch),
epoch<k>.pt, best.pt, val_predictions_epoch<k>.jsonl

Usage (from the repo root, after build_scenes.py and extract_features.py):
    venv/bin/python scripts/train.py --variant surface_both --output runs/surface_both
    venv/bin/python scripts/train.py --variant surface_only --train-scenes 10 --output runs/pilot_surface_only
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volumetric_reasoning.evaluation import evaluate  # noqa: E402
from volumetric_reasoning.model import FreeSpaceVLM  # noqa: E402
from volumetric_reasoning.qa_dataset import VARIANTS, SceneQADataset  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--qa-dir", type=Path, default=Path("data/qa"))
    p.add_argument("--scene-root", type=Path, default=Path("outputs/scenes"))
    p.add_argument("--train-scenes", type=int, default=0, help="Use N random training scenes (0 = all)")
    p.add_argument("--train-limit", type=int, default=0, help="Use N random training examples (0 = all; for tests)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=float, default=0.03, help="Fraction of optimizer steps")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--val-limit", type=int, default=0, help="Validate on N evenly spaced examples (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                       for k, v in vars(args).items()}, indent=2) + "\n")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    scenes = None
    if args.train_scenes:
        train_ids = sorted(json.loads((args.qa_dir / "splits.json").read_text())["train"])
        scenes = set(rng.choice(train_ids, args.train_scenes, replace=False).tolist())
    train = SceneQADataset(args.qa_dir / "train.jsonl", args.variant, args.scene_root, args.model, scenes=scenes)
    val = SceneQADataset(args.qa_dir / "val.jsonl", args.variant, args.scene_root, args.model)
    pool = np.arange(len(train))
    if args.train_limit:
        pool = np.sort(rng.permutation(len(train))[:args.train_limit])
    order = [rng.permutation(pool) for _ in range(args.epochs)]  # fixed by the seed, same for every variant
    n_examples = sum(len(o) for o in order)
    total_steps = math.ceil(n_examples / args.grad_accum)
    print(f"{args.variant}: {len(train)} training examples from {len({r['scene_id'] for r in train.rows})} scenes, "
          f"{args.epochs} epochs, {total_steps} optimizer steps; {len(val)} validation examples; device {device}",
          flush=True)

    model = FreeSpaceVLM(args.model, lora_rank=args.lora_rank, lora_alpha=2 * args.lora_rank).to(device)
    if not args.no_gradient_checkpointing:
        model.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, round(args.warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    log = open(args.output / "log.jsonl", "a")
    step, seen, best = 0, 0, -1.0
    t0, window, window_tokens = time.time(), [], 0
    for epoch in range(1, args.epochs + 1):
        for i in order[epoch - 1]:
            ex = train[int(i)]
            loss = model.loss(ex)
            (loss / args.grad_accum).backward()
            window.append(loss.item())
            window_tokens += len(ex["surface_feature"]) + len(ex["box_type"])
            seen += 1
            if seen % args.grad_accum == 0 or seen == n_examples:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "epoch": epoch, "loss": float(np.mean(window)), "grad_norm": grad_norm,
                       "lr": sched.get_last_lr()[0], "scene_tokens_per_s": window_tokens / (time.time() - t0)}
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step % 10 == 0 or step == total_steps:
                    print(f"step {step}/{total_steps} epoch {epoch} loss {rec['loss']:.4f} grad {grad_norm:.2f} "
                          f"lr {rec['lr']:.2e} ({rec['scene_tokens_per_s']:.0f} scene tokens/s)", flush=True)
                t0, window, window_tokens = time.time(), [], 0

        preds, summary = evaluate(model, val, args.val_limit)
        torch.save(model.trainable_state_dict(), args.output / f"epoch{epoch}.pt")
        with open(args.output / f"val_predictions_epoch{epoch}.jsonl", "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in preds)
        if summary["mean_over_tasks"] > best:
            best = summary["mean_over_tasks"]
            shutil.copy(args.output / f"epoch{epoch}.pt", args.output / "best.pt")
        log.write(json.dumps({"epoch": epoch, "val": summary}) + "\n"); log.flush()
        print(f"epoch {epoch} validation: mean over tasks {summary['mean_over_tasks']:.4f} "
              f"{ {k: round(v, 3) for k, v in summary['per_task'].items()} }", flush=True)
        t0 = time.time()


if __name__ == "__main__":
    main()
