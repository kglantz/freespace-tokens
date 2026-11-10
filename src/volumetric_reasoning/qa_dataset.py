"""Question-answer examples with scene tokens, and their scoring.

One example = the scene tokens of one QA scene in one view (for one variant) + the question + the answer.

Scene tokens (from build_scenes.py and extract_features.py, see outputs/scenes/<scene>/):
  * surface: one per surface block with a feature (blocks no depth pixel landed in are dropped, ~2%),
    position = mean of its occupied voxels, feature = pooled frozen Qwen3-VL image feature;
  * free / unknown boxes: center, size (meters) and type; which types are included depends on the variant.
Coordinates are per-scene normalized: x, y centered on the surface tokens' bounding box, z = 0 at the
floor (1st percentile of surface heights; ScanNet++ scenes are z-up). Metric scale is unchanged.

Views (VIEWS): the scene built from all 32 frames, or from the first 16 / 8 of them (what a robot has seen after
half / a quarter of the walk). ViCA questions are about the whole scene and use the full view only; a navigation
question is used in every view in which both of its endpoints are seen.

Navigation questions (nav_qa.py) name two endpoint objects, (A) and (B); every example carries `mark`, one entry
per surface token: 1 = token of A, 2 = token of B, 0 = other (all 0 for ViCA questions).

Tasks and scores:
  * ViCA, VSI-Bench metrics: multiple choice (object_relative_distance): accuracy of the option letter;
    numeric (distance, size, count, room size): mean relative accuracy (MRA);
  * nav_collision: yes / no; per example 1 if correct (summaries: balanced accuracy, see evaluation.py);
  * nav_distance: meters, MRA.
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
TASKS = NUMERIC_TASKS + CHOICE_TASKS                     # ViCA
NAV_TASKS = ("nav_collision", "nav_distance")
ALL_TASKS = TASKS + NAV_TASKS
VARIANTS = {"surface_only": (), "surface_free": (FREE,), "surface_unknown": (UNKNOWN,),
            "surface_both": (FREE, UNKNOWN)}
VIEWS = {"full": "outputs/scenes", "first16": "outputs/scenes_first16", "first8": "outputs/scenes_first8"}
# VSI-Bench post-prompt for numeric answers; multiple-choice and yes / no questions carry their own instruction.
NUMERIC_PROMPT = "Please answer the question using a single word or phrase."


# --- scenes -------------------------------------------------------------------------
@lru_cache(maxsize=None)
def surface_positions(scene_dir: str, feature_file: str) -> tuple[np.ndarray, np.ndarray]:
    """Normalized positions (Ns, 3) of the surface tokens with a feature, and the shift (world = position + shift)."""
    d = Path(scene_dir)
    with np.load(d / "tokens.npz") as t, np.load(d / feature_file) as f:
        has_feature = f["n_pixels"] > 0
        assert len(has_feature) == len(t["surface_block"])
        position = t["surface_position"][has_feature]
    lo, hi = position.min(0), position.max(0)
    shift = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, np.percentile(position[:, 2], 1)])
    return (position - shift).astype(np.float32), shift


@lru_cache(maxsize=None)
def load_scene(scene_dir: str, feature_file: str) -> dict:
    """All scene tokens of one scene (every box type), with normalized coordinates, and the shift."""
    d = Path(scene_dir)
    position, shift = surface_positions(scene_dir, feature_file)
    with np.load(d / "tokens.npz") as t, np.load(d / feature_file) as f:
        return {"surface_feature": torch.from_numpy(f["features"][f["n_pixels"] > 0]),
                "surface_position": torch.from_numpy(position),
                "box_center": torch.from_numpy((t["box_center"] - shift).astype(np.float32)),
                "box_size": torch.from_numpy(t["box_size"].astype(np.float32)),
                "box_type": torch.from_numpy(t["box_type"].astype(np.int64)),
                "shift": shift}


def scene_tokens(scene: dict, variant: str) -> dict:
    """The scene's tokens for one variant: all surface tokens + the boxes of the variant's types."""
    keep = torch.isin(scene["box_type"], torch.tensor(VARIANTS[variant], dtype=torch.int64))
    return {"surface_feature": scene["surface_feature"], "surface_position": scene["surface_position"],
            **{k: scene[k][keep] for k in ("box_center", "box_size", "box_type")}}


# --- navigation endpoints -------------------------------------------------------------
def in_obb(points: np.ndarray, obb: dict, pad: float = 0.0) -> np.ndarray:
    """Which of the (N, 3) points lie inside an annotated ScanNet++ oriented box grown by `pad` meters.
    The rows of obb["normalizedAxes"] (3 x 3, flattened) are the box axes, in the order of obb["axesLengths"]."""
    axes = np.asarray(obb["normalizedAxes"], dtype=np.float64).reshape(3, 3)
    local = (np.asarray(points, dtype=np.float64) - np.asarray(obb["centroid"])) @ axes.T
    return (np.abs(local) <= np.asarray(obb["axesLengths"]) / 2 + pad).all(axis=1)


# An endpoint is seen in a view if at least MIN_ENDPOINT_TOKENS of the view's surface tokens lie inside its box,
# grown by half a block (token positions are means over 16 cm blocks). With all 32 frames 97% of endpoints are seen.
ENDPOINT_PAD = 0.08
MIN_ENDPOINT_TOKENS = 3


def endpoint_tokens(surface_position: np.ndarray, obb: dict, shift=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Which surface tokens belong to the endpoint. obb is in world coordinates; the positions are world
    coordinates minus `shift` (as returned by surface_positions)."""
    return in_obb(np.asarray(surface_position, dtype=np.float64) + np.asarray(shift), obb, ENDPOINT_PAD)


def endpoint_marks(surface_position: np.ndarray, shift, row: dict) -> np.ndarray:
    """(Ns,) int64: 1 for tokens of endpoint A, 2 for tokens of B, 0 otherwise (a token in both boxes counts for A)."""
    mark = np.zeros(len(surface_position), np.int64)
    mark[endpoint_tokens(surface_position, row["b"]["obb"], shift)] = 2
    mark[endpoint_tokens(surface_position, row["a"]["obb"], shift)] = 1
    return mark


# --- questions ----------------------------------------------------------------------
def load_qa(path, tasks=TASKS, scenes=None) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    return [r for r in rows if r["task"] in tasks and (scenes is None or r["scene_id"] in scenes)]


def prompt(row: dict) -> str:
    own_instruction = row["task"] in CHOICE_TASKS or row["task"] == "nav_collision"
    return row["question"] if own_instruction else f"{row['question']}\n{NUMERIC_PROMPT}"


class SceneQADataset(torch.utils.data.Dataset):
    """Examples of one or more QA files for one variant. Scenes are loaded once and kept in memory.

    scene_root: the full-view scene directory, or a dict view name -> directory (must include "full").
    Example i = (self.rows[i], self.views[i]): ViCA questions in the full view; navigation questions in every
    view of scene_root in which both endpoints are seen.
    """

    def __init__(self, qa_path, variant: str, scene_root="outputs/scenes", model="Qwen/Qwen3-VL-4B-Instruct",
                 tasks=TASKS, scenes=None):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {list(VARIANTS)}")
        self.variant = variant
        self.view_roots = {k: Path(v) for k, v in (scene_root if isinstance(scene_root, dict)
                                                   else {"full": scene_root}).items()}
        self.feature_file = f"features_{model.split('/')[-1]}.npz"
        self.rows, self.views = [], []
        for path in [qa_path] if isinstance(qa_path, (str, Path)) else qa_path:
            for r in load_qa(path, tasks, scenes):
                for view in (self.view_roots if r["task"] in NAV_TASKS else ["full"]):
                    if r["task"] not in NAV_TASKS or self._seen(r, view):
                        self.rows.append(r)
                        self.views.append(view)

    def _scene_dir(self, row: dict, view: str) -> str:
        return str(self.view_roots[view] / row["scene_id"])

    def _seen(self, row: dict, view: str) -> bool:
        position, shift = surface_positions(self._scene_dir(row, view), self.feature_file)
        return all(endpoint_tokens(position, row[k]["obb"], shift).sum() >= MIN_ENDPOINT_TOKENS for k in "ab")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i) -> dict:
        r, view = self.rows[i], self.views[i]
        scene = load_scene(self._scene_dir(r, view), self.feature_file)
        tokens = scene_tokens(scene, self.variant)
        if r["task"] in NAV_TASKS:
            mark = torch.from_numpy(endpoint_marks(scene["surface_position"].numpy(), scene["shift"], r))
        else:
            mark = torch.zeros(len(tokens["surface_position"]), dtype=torch.int64)
        return {"id": r["id"], "scene_id": r["scene_id"], "task": r["task"], "view": view, "prompt": prompt(r),
                "answer": r["answer"], "mark": mark, **tokens}


# --- scoring ------------------------------------------------------------------------
MRA_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def parse_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def parse_choice(text: str) -> str | None:
    """Option letter: a leading 'B', 'B.', 'B)' ..., else the first 'B.' in the text."""
    text = text.strip()
    m = re.match(r"\(?([A-D])(?![A-Za-z])", text) or re.search(r"\b([A-D])\.", text)
    return m.group(1) if m else None


def parse_yes_no(text: str) -> str | None:
    """'yes' / 'no' if the answer starts with that word (any case, trailing punctuation allowed)."""
    m = re.match(r"(yes|no)\b", text.strip().lower())
    return m.group(1) if m else None


def mean_relative_accuracy(pred: float, gt: float) -> float:
    """Fraction of thresholds theta in {0.5, 0.55, ..., 0.95} with |pred - gt| / gt < 1 - theta."""
    return float(np.mean(abs(pred - gt) / abs(gt) < 1 - MRA_THRESHOLDS))


def score(task: str, prediction: str, answer: str) -> float:
    if task in CHOICE_TASKS:
        return float(parse_choice(prediction) == answer[0])
    if task == "nav_collision":
        return float(parse_yes_no(prediction) == answer)
    pred = parse_number(prediction)
    return 0.0 if pred is None else mean_relative_accuracy(pred, float(answer))
