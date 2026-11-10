"""Evaluate a trained variant on a QA split: ViCA and navigation questions, every view (greedy generation).

Reads the run's args.json for the variant, model and LoRA rank, loads the checkpoint's trainable weights and
writes <run>/<split>_predictions.jsonl and <run>/<split>_summary.json.

Usage (from the repo root):
    venv/bin/python scripts/evaluate.py --run runs/surface_both --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volumetric_reasoning.evaluation import evaluate  # noqa: E402
from volumetric_reasoning.model import FreeSpaceVLM  # noqa: E402
from volumetric_reasoning.qa_dataset import ALL_TASKS, VIEWS, SceneQADataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--limit", type=int, default=0, help="Evaluate N evenly spaced examples (0 = all)")
    args = p.parse_args()
    run = json.loads((args.run / "args.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = FreeSpaceVLM(run["model"], lora_rank=run["lora_rank"], lora_alpha=2 * run["lora_rank"]).to(device)
    model.load_trainable_state_dict(torch.load(args.run / args.checkpoint, map_location=device))
    qa = Path(run["qa_dir"])
    data = SceneQADataset([qa / f"{args.split}.jsonl", qa / f"nav_{args.split}.jsonl"], run["variant"], VIEWS,
                          run["model"], tasks=ALL_TASKS)
    preds, summary = evaluate(model, data, args.limit)
    summary = {"run": str(args.run), "checkpoint": args.checkpoint, "split": args.split, "variant": run["variant"],
               **summary}
    with open(args.run / f"{args.split}_predictions.jsonl", "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in preds)
    (args.run / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
