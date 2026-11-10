"""Generate the navigation questions (nav_qa.py) for every QA scene, with ground truth from the laser scan.

Writes data/qa/nav_{train,val,test}.jsonl (scene split of data/qa/splits.json) and, with --render, one picture per
scene (outputs/nav_qa/<scene>.png): floor, obstacles, walkable cells, endpoints, and a few questions drawn in.

Usage (from the repo root, after downloading mesh_aligned_0.05.ply, segments.json, segments_anno.json):
    venv/bin/python scripts/make_nav_qa.py --render outputs/nav_qa
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volumetric_reasoning.nav_qa import (CELL, MARGIN, TRAVEL_WIDTH, FloorGraph, Scan,  # noqa: E402
                                         collision_segment, make_questions)


def render(scan: Scan, rows: list[dict], path: Path, scene_id: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    walk = scan.walkable(TRAVEL_WIDTH / 2)
    img = np.ones(scan.shape + (3,))
    img[scan.floor] = (0.85, 0.85, 0.85)
    img[walk] = (0.7, 0.85, 1.0)
    img[scan.obstacle] = (0.2, 0.2, 0.2)
    ext = [scan.lo[0], scan.lo[0] + scan.shape[0] * CELL, scan.lo[1], scan.lo[1] + scan.shape[1] * CELL]
    fig, ax = plt.subplots(figsize=(10, 10 * scan.shape[1] / scan.shape[0] + 1))
    ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=ext)
    for i in scan.endpoints():
        c = scan.objects[i]["obb"]["centroid"][:2]
        ax.plot(*c, "o", color="orange", ms=4)
        ax.text(*c, scan.objects[i]["label"], fontsize=7)
    to_m = lambda cell: np.asarray(cell) * CELL + scan.lo + CELL / 2   # noqa: E731
    for r in [r for r in rows if r["task"] == "nav_collision" and r["answer"] == "yes"][:3]:  # the segment found
        seg = collision_segment(scan, r["a"]["id"], r["b"]["id"], r["robot_width_m"] / 2 + MARGIN)
        p, q = map(to_m, seg)
        ax.plot([p[0], q[0]], [p[1], q[1]], "-", color="green", lw=1.5)
        ax.text(*q, f"yes {round(r['robot_width_m'] * 100)} cm", color="green", fontsize=7)
    for r in [r for r in rows if r["task"] == "nav_collision" and r["answer"] == "no"][:3]:   # centre to centre
        a, b = scan.objects[r["a"]["id"]]["obb"]["centroid"][:2], scan.objects[r["b"]["id"]]["obb"]["centroid"][:2]
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="red", lw=1)
        ax.text(*b, f"no {round(r['robot_width_m'] * 100)} cm", color="red", fontsize=7)
    graph = FloorGraph(scan, TRAVEL_WIDTH / 2)
    for r in [r for r in rows if r["task"] == "nav_distance"][:3]:   # path: walk back down the distance field
        d = graph.distances_from(r["a"]["id"])
        zb = graph.zone(r["b"]["id"])
        cur = np.unravel_index(np.where(zb, d, np.inf).argmin(), d.shape)
        pts = [cur]
        while d[cur] > 1e-9:
            nb = [(cur[0] + i, cur[1] + j) for i in range(-2, 3) for j in range(-2, 3) if (i, j) != (0, 0)]
            cur = min(nb, key=lambda p: d[p])
            pts.append(cur)
        p = np.array(pts) * CELL + scan.lo + CELL / 2
        ax.plot(p[:, 0], p[:, 1], "-", color="purple", lw=1.5)
        ax.text(*p[0], f"{r['answer']} m", color="purple", fontsize=8)
    n = Counter((r["task"], r["answer"] if r["task"] == "nav_collision" else "") for r in rows)
    ax.set_title(f"{scene_id}: {dict(n)}  (green: straight segment found (yes); red dashed: no, centre to centre; purple: shortest paths)", fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/scannetpp"))
    p.add_argument("--splits", type=Path, default=Path("data/qa/splits.json"))
    p.add_argument("--output", type=Path, default=Path("data/qa"))
    p.add_argument("--max-pairs", type=int, default=150, help="Object pairs per scene (0 = all)")
    p.add_argument("--render", type=Path, help="Directory for one picture per scene")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.render:
        args.render.mkdir(parents=True, exist_ok=True)
    for split, scenes in json.loads(args.splits.read_text()).items():
        rows = []
        for s in sorted(scenes):
            scan = Scan.load(args.data_root / "data" / s / "scans")
            rng = np.random.default_rng([args.seed, zlib.crc32(s.encode())])
            r = make_questions(scan, s, rng, args.max_pairs)
            rows += r
            n = Counter(x["task"] for x in r)
            print(f"{split} {s}: {len(scan.endpoints())} endpoints, {n['nav_collision']} collision, "
                  f"{n['nav_distance']} distance", flush=True)
            if args.render:
                render(scan, r, args.render / f"{s}.png", s)
        with open(args.output / f"nav_{split}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        n = Counter(r["task"] for r in rows)
        print(f"== {split}: {len(rows)} questions {dict(n)} from {len(scenes)} scenes", flush=True)


if __name__ == "__main__":
    main()
