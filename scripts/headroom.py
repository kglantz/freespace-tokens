"""Check A: how much can free / unknown tokens tell a robot beyond surface tokens, on the navigation questions?

No learning. Hand-made rules read the tokens of one view (32, 16 or 8 frames) and answer the navigation
questions; the ground truth comes from the laser scan (nav_qa.py). All rules share everything except which
columns of the 16 cm block grid they consider walkable:

  obstacle column: a surface token 0.15-1.5 m above the floor (floor height = 1st percentile of the surface
                   tokens' heights, as in qa_dataset.py); every rule avoids these
  naive      : every other column is walkable (surface tokens read as "nothing there = free")
  floor      : only columns where a floor-level surface token was seen (seeing the floor there means the camera
               probably saw the space above it): the best we found that uses surface tokens alone
  free       : only columns whose body band is observed free (free boxes cover at least half its block layers)
  floor|free : either
Questions: those whose two endpoints are seen in the view (qa_dataset.endpoint_tokens); endpoint footprint = the
columns of its tokens. As in nav_qa.py, the robot centre must be more than r from every obstacle column:
  collision: "yes" if a straight walkable segment joins the zones (robot centre within 0.8 m of the footprints)
  distance : shortest walkable path between the zones (robot edge within 0.3 m); if there is none, the
             straight-line distance between the zones ("straight" also reports that for every question)
Scores per view and rule: collision balanced accuracy (mean of the accuracy on yes and on no questions) and the
collision rate ("yes" on truly blocked questions); distance mean relative accuracy, also on detours
(true path > 1.2 x straight line). Only train and val questions are used (the test split stays untouched).

Usage (from the repo root, after make_nav_qa.py and the three views):
    venv/bin/python scripts/headroom.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volumetric_reasoning.geometry.labels import FREE  # noqa: E402
from volumetric_reasoning.nav_qa import (BAND, COLLISION_REACH, REACH, TRAVEL_WIDTH, GridPaths,  # noqa: E402
                                         straight_distance, straight_line)
from volumetric_reasoning.qa_dataset import MIN_ENDPOINT_TOKENS, endpoint_tokens, mean_relative_accuracy  # noqa: E402

VIEWS = {"full": "outputs/scenes", "first16": "outputs/scenes_first16", "first8": "outputs/scenes_first8"}
RULES = ("naive", "floor", "free", "floor|free")
FEATURES = "features_Qwen3-VL-4B-Instruct.npz"


class TokenGrid:
    """Columns of one view of one scene: obstacles, seen floor, observed-free body band."""

    def __init__(self, scene_dir: Path):
        with np.load(scene_dir / "tokens.npz") as t, np.load(scene_dir / FEATURES) as f, \
                np.load(scene_dir / "block_map.npz") as m:
            seen = f["n_pixels"] > 0                                   # the surface tokens the model receives
            blocks, self.position = t["surface_block"][seen], t["surface_position"][seen]
            self.cell, origin, shape = float(t["block_size_m"]), t["origin"], m["shape_blocks"]
            assert np.allclose(origin, m["origin"])
            box_start, box_stop, box_type = t["box_start"], t["box_stop"], t["box_type"]
        self.columns = blocks[:, :2]
        floor_z = np.percentile(self.position[:, 2], 1)
        h = self.position[:, 2] - floor_z
        band = (h > BAND[0]) & (h < BAND[1])
        self.obstacle = np.zeros(shape[:2], bool)
        self.obstacle[blocks[band, 0], blocks[band, 1]] = True
        floor = np.zeros(shape[:2], bool)
        floor[blocks[h <= BAND[0], 0], blocks[h <= BAND[0], 1]] = True
        layer_h = origin[2] + (np.arange(shape[2]) + 0.5) * self.cell - floor_z
        layers = np.flatnonzero((layer_h > BAND[0]) & (layer_h < BAND[1]))   # block layers of the body band
        free = np.zeros(shape[:2], int)                                        # free band layers per column
        for a, b, kind in zip(box_start, box_stop, box_type):                  # free boxes do not overlap
            if kind == FREE:
                free[a[0]:b[0], a[1]:b[1]] += ((layers >= a[2]) & (layers < b[2])).sum()
        band_free = free >= 0.5 * len(layers)
        self.candidates = {"naive": np.ones(shape[:2], bool), "floor": floor, "free": band_free,
                           "floor|free": floor | band_free}
        self.clearance = ndimage.distance_transform_edt(~self.obstacle) * self.cell
        self._paths = {}

    def walkable(self, rule: str, radius: float) -> np.ndarray:
        return self.candidates[rule] & (self.clearance > radius)

    def footprint(self, obb: dict) -> np.ndarray | None:
        """Columns of the endpoint's tokens, or None if it is not seen in this view."""
        mine = endpoint_tokens(self.position, obb)
        if mine.sum() < MIN_ENDPOINT_TOKENS:
            return None
        fp = np.zeros(self.obstacle.shape, bool)
        fp[self.columns[mine, 0], self.columns[mine, 1]] = True
        return fp

    def zone(self, walk: np.ndarray, fp: np.ndarray, reach: float) -> np.ndarray:
        return walk & (ndimage.distance_transform_edt(~fp) * self.cell <= reach)

    def collision(self, rule: str, fa: np.ndarray, fb: np.ndarray, width: float) -> str:
        walk = self.walkable(rule, width / 2)
        za, zb = self.zone(walk, fa, COLLISION_REACH), self.zone(walk, fb, COLLISION_REACH)
        return "yes" if straight_line(walk, za, zb) is not None else "no"

    def distance(self, rule: str, fa: np.ndarray, fb: np.ndarray) -> tuple[float, float]:
        """(predicted travel distance, straight-line distance between the zones of the naive rule)."""
        r = TRAVEL_WIDTH / 2
        if rule not in self._paths:
            self._paths[rule] = GridPaths(self.walkable(rule, r), self.cell)
        paths = self._paths[rule]
        za, zb = self.zone(paths.walk, fa, r + REACH), self.zone(paths.walk, fb, r + REACH)
        naive = self.walkable("naive", r)
        na, nb = self.zone(naive, fa, r + REACH), self.zone(naive, fb, r + REACH)
        straight = straight_distance(na, nb, self.cell) if na.any() and nb.any() else np.nan
        travel = float(paths.distances(za)[zb].min()) if za.any() and zb.any() else np.inf
        return (travel if np.isfinite(travel) else straight), straight


def main():
    questions = [json.loads(line) for s in ("train", "val") for line in open(f"data/qa/nav_{s}.jsonl")]
    results = {}
    for view, root in VIEWS.items():
        grids = {}
        col = defaultdict(list)    # rule -> [(truth, prediction)]
        dist = defaultdict(list)   # rule -> [(truth, prediction, detour)]
        for q in questions:
            if q["scene_id"] not in grids:
                grids[q["scene_id"]] = TokenGrid(Path(root) / q["scene_id"])
            g = grids[q["scene_id"]]
            fa, fb = g.footprint(q["a"]["obb"]), g.footprint(q["b"]["obb"])
            if fa is None or fb is None:
                continue
            if q["task"] == "nav_collision":
                for rule in RULES:
                    col[rule].append((q["answer"], g.collision(rule, fa, fb, q["robot_width_m"])))
            else:
                truth, detour = float(q["answer"]), q["travel_m"] > 1.2 * q["straight_m"]
                for rule in RULES:
                    travel, straight = g.distance(rule, fa, fb)
                    dist[rule].append((truth, travel, detour))
                dist["straight"].append((truth, straight, detour))
        results[view] = {"collision": {}, "distance": {}}
        print(f"\n== {view}: {len(col['naive'])} collision, {len(dist['naive'])} distance questions "
              f"({sum(d for _, _, d in dist['naive'])} with a detour)")
        for rule in RULES:
            t = np.array([a == "yes" for a, _ in col[rule]]); p = np.array([b == "yes" for _, b in col[rule]])
            balanced = (np.mean(p[t]) + np.mean(~p[~t])) / 2
            crash = np.mean(p[~t])
            results[view]["collision"][rule] = {"balanced_accuracy": balanced, "collision_rate": crash}
            print(f"  collision {rule:10s}: balanced accuracy {balanced:.3f}, says yes when blocked {crash:.3f}")
        for rule in ("straight",) + RULES:
            mra = [mean_relative_accuracy(p, t) if np.isfinite(p) else 0.0 for t, p, _ in dist[rule]]
            det = [m for m, (_, _, d) in zip(mra, dist[rule]) if d]
            results[view]["distance"][rule] = {"mra": float(np.mean(mra)), "mra_detour": float(np.mean(det))}
            print(f"  distance  {rule:10s}: MRA {np.mean(mra):.3f}, on detours {np.mean(det):.3f}")
    Path("outputs/headroom.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
