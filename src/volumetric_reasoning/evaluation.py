"""Generate answers for a QA dataset and score them with the VSI-Bench metrics."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .qa_dataset import TASKS, score


@torch.no_grad()
def evaluate(model, dataset, limit: int = 0) -> tuple[list[dict], dict]:
    """Predictions (one dict per example) and a summary: per-task mean score, mean over tasks, mean over examples.

    limit > 0 evaluates an evenly spaced subset of the dataset (same subset for every model).
    """
    was_training = model.training
    model.eval()
    idx = np.arange(len(dataset)) if limit <= 0 else np.linspace(0, len(dataset) - 1, limit).round().astype(int)
    preds = []
    for i in idx:
        ex = dataset[int(i)]
        out = model.generate(ex)
        preds.append({"id": ex["id"], "scene_id": ex["scene_id"], "task": ex["task"], "answer": ex["answer"],
                      "prediction": out, "score": score(ex["task"], out, ex["answer"])})
    model.train(was_training)
    return preds, summarize(preds)


def summarize(preds: list[dict]) -> dict:
    by_task = defaultdict(list)
    for p in preds:
        by_task[p["task"]].append(p["score"])
    per_task = {t: float(np.mean(by_task[t])) for t in TASKS if by_task[t]}
    return {"mean_over_tasks": float(np.mean(list(per_task.values()))),
            "mean_over_examples": float(np.mean([p["score"] for p in preds])),
            "per_task": per_task, "counts": {t: len(by_task[t]) for t in per_task}}
