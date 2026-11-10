"""Generate answers for a QA dataset and score them.

Groups: every ViCA task; every navigation task per view ("nav_collision/first8", ...).
Group score: mean example score, except nav_collision: balanced accuracy (mean of the accuracy on the "yes" and on
the "no" questions, so the share of "yes" answers does not matter). For nav_collision also the collision rate:
the share of truly blocked ("no") questions answered "yes".
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .qa_dataset import ALL_TASKS, NAV_TASKS, VIEWS, parse_yes_no, score


@torch.no_grad()
def evaluate(model, dataset, limit: int = 0) -> tuple[list[dict], dict]:
    """Predictions (one dict per example) and a summary (see summarize).

    limit > 0 evaluates an evenly spaced subset of the dataset (same subset for every model).
    """
    was_training = model.training
    model.eval()
    idx = np.arange(len(dataset)) if limit <= 0 else np.linspace(0, len(dataset) - 1, limit).round().astype(int)
    preds = []
    for i in idx:
        ex = dataset[int(i)]
        out = model.generate(ex)
        preds.append({"id": ex["id"], "scene_id": ex["scene_id"], "task": ex["task"], "view": ex["view"],
                      "answer": ex["answer"], "prediction": out, "score": score(ex["task"], out, ex["answer"])})
    model.train(was_training)
    return preds, summarize(preds)


def group_of(p: dict) -> str:
    return f"{p['task']}/{p['view']}" if p["task"] in NAV_TASKS else p["task"]


def summarize(preds: list[dict]) -> dict:
    by_group = defaultdict(list)
    for p in preds:
        by_group[group_of(p)].append(p)
    order = [t for t in ALL_TASKS if t not in NAV_TASKS] + [f"{t}/{v}" for t in NAV_TASKS for v in VIEWS]
    per_task, collision_rate = {}, {}
    for g in [g for g in order if g in by_group]:
        ps = by_group[g]
        if g.startswith("nav_collision/"):
            per_class = [np.mean([p["score"] for p in ps if p["answer"] == a]) for a in ("yes", "no")
                         if any(p["answer"] == a for p in ps)]
            per_task[g] = float(np.mean(per_class))
            blocked = [p for p in ps if p["answer"] == "no"]
            if blocked:
                collision_rate[g] = float(np.mean([parse_yes_no(p["prediction"]) == "yes" for p in blocked]))
        else:
            per_task[g] = float(np.mean([p["score"] for p in ps]))
    return {"mean_over_tasks": float(np.mean(list(per_task.values()))),
            "mean_over_examples": float(np.mean([p["score"] for p in preds])),
            "per_task": per_task, "collision_rate": collision_rate,
            "counts": {g: len(by_group[g]) for g in per_task}}
