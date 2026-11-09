"""Question-answer examples with scene tokens, and VSI-Bench scoring.

One example = the scene tokens of one QA scene (for one variant) + the question + the answer.

Scene tokens (from build_scenes.py and extract_features.py, see outputs/scenes/<scene>/):
  * surface: one per surface block with a feature (blocks no depth pixel landed in are dropped, ~2%),
    position = mean of its occupied voxels, feature = pooled frozen Qwen3-VL image feature;
  * free / unknown boxes: center, size (meters) and type; which types are included depends on the variant.
Coordinates are per-scene normalized: x, y centered on the surface tokens' bounding box, z = 0 at the
floor (1st percentile of surface heights; ScanNet++ scenes are z-up). Metric scale is unchanged.

Tasks: only those with an objectively scorable answer (VSI-Bench metrics):
  * multiple choice (object_relative_distance): accuracy of the option letter;
  * numeric (distance, size, count, room size): mean relative accuracy (MRA).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .geometry.labels import FREE, UNKNOWN

NUMERIC_TASKS = ("object_abs_distance", "object_size_estimation", "object_count", "room_size")
CHOICE_TASKS = ("object_relative_distance",)
TASKS = NUMERIC_TASKS + CHOICE_TASKS
VARIANTS = {"surface_only": (), "surface_free": (FREE,), "surface_unknown": (UNKNOWN,),
            "surface_both": (FREE, UNKNOWN)}
# VSI-Bench post-prompt for numeric answers; multiple-choice questions carry their own instruction.
NUMERIC_PROMPT = "Please answer the question using a single word or phrase."


# --- scenes -------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_scene(scene_dir: str, feature_file: str) -> dict:
    """All scene tokens of one scene (every box type), with normalized coordinates."""
    d = Path(scene_dir)
    with np.load(d / "tokens.npz") as t, np.load(d / feature_file) as f:
        assert len(f["features"]) == len(t["surface_block"])
        has_feature = f["n_pixels"] > 0
        position = t["surface_position"][has_feature]
        lo, hi = position.min(0), position.max(0)
        shift = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, np.percentile(position[:, 2], 1)])
        return {"surface_feature": torch.from_numpy(f["features"][has_feature]),
                "surface_position": torch.from_numpy((position - shift).astype(np.float32)),
                "box_center": torch.from_numpy((t["box_center"] - shift).astype(np.float32)),
                "box_size": torch.from_numpy(t["box_size"].astype(np.float32)),
                "box_type": torch.from_numpy(t["box_type"].astype(np.int64))}


def scene_tokens(scene: dict, variant: str) -> dict:
    """The scene's tokens for one variant: all surface tokens + the boxes of the variant's types."""
    keep = torch.isin(scene["box_type"], torch.tensor(VARIANTS[variant], dtype=torch.int64))
    return {k: v[keep] if k.startswith("box_") else v for k, v in scene.items()}


# --- questions ----------------------------------------------------------------------
def load_qa(path, tasks=TASKS, scenes=None) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if r["task"] in tasks and (scenes is None or r["scene_id"] in scenes)]


def prompt(row: dict) -> str:
    return row["question"] if row["task"] in CHOICE_TASKS else f"{row['question']}\n{NUMERIC_PROMPT}"


class SceneQADataset(torch.utils.data.Dataset):
    """Examples of one QA split for one variant. Scenes are loaded once and kept in memory."""

    def __init__(self, qa_path, variant: str, scene_root="outputs/scenes", model="Qwen/Qwen3-VL-4B-Instruct",
                 tasks=TASKS, scenes=None):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {list(VARIANTS)}")
        self.rows = load_qa(qa_path, tasks, scenes)
        self.variant = variant
        self.scene_root = Path(scene_root)
        self.feature_file = f"features_{model.split('/')[-1]}.npz"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i) -> dict:
        r = self.rows[i]
        scene = load_scene(str(self.scene_root / r["scene_id"]), self.feature_file)
        return {"id": r["id"], "scene_id": r["scene_id"], "task": r["task"], "prompt": prompt(r),
                "answer": r["answer"], **scene_tokens(scene, self.variant)}


# --- scoring (VSI-Bench) ------------------------------------------------------------
MRA_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def parse_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def parse_choice(text: str) -> str | None:
    """Option letter: a leading 'B', 'B.', 'B)' ..., else the first 'B.' in the text."""
    text = text.strip()
    m = re.match(r"\(?([A-D])(?![A-Za-z])", text) or re.search(r"\b([A-D])\.", text)
    return m.group(1) if m else None


def mean_relative_accuracy(pred: float, gt: float) -> float:
    """Fraction of thresholds theta in {0.5, 0.55, ..., 0.95} with |pred - gt| / gt < 1 - theta."""
    return float(np.mean(abs(pred - gt) / abs(gt) < 1 - MRA_THRESHOLDS))


def score(task: str, prediction: str, answer: str) -> float:
    if task in CHOICE_TASKS:
        return float(parse_choice(prediction) == answer[0])
    pred = parse_number(prediction)
    return 0.0 if pred is None else mean_relative_accuracy(pred, float(answer))
