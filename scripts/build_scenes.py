"""Build the scene representation of every QA scene: 2 cm map (as step 1) + block tokens (as step 3).

For each scene in data/qa/splits.json:
  * 32 iPhone RGB-D frames spread uniformly over the capture, ARKit poses,
  * fused into a BlockMap (2 cm voxels, 8^3 blocks) with occupied / free / unknown labels,
  * tokenized with the step-3 defaults (16 cm blocks, boxes capped at 1.5 m, unknown inside the room).
No laser data is used. A scene whose tokens.npz exists is skipped, so the script can be re-run.

Partial views (what a robot has seen after walking part of the capture): --first N keeps only the first N of
the 32 frames, e.g. --first 8 --output outputs/scenes_first8 (a quarter of the capture).

Outputs (<output>/<scene>/): block_map.npz (incl. frame_indices), tokens.npz, summary.json

Usage (from the repo root):
    venv/bin/python scripts/build_scenes.py --jobs 4
    venv/bin/python scripts/build_scenes.py --jobs 4 --first 8 --output outputs/scenes_first8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volumetric_reasoning.datasets.scannetpp import ScanNetPPIphoneScene  # noqa: E402
from volumetric_reasoning.geometry.block_map import BlockMap  # noqa: E402
from volumetric_reasoning.geometry.block_tokens import tokenize  # noqa: E402
from volumetric_reasoning.geometry.labels import LABEL_NAMES  # noqa: E402


def build(scene_id: str, data_root: Path, output: Path, num_frames: int, first: int = 0) -> str:
    out = output / scene_id
    if (out / "tokens.npz").exists():
        return f"{scene_id}: already built"
    t0 = time.time()
    scene = ScanNetPPIphoneScene(data_root, scene_id)
    frame_ids = scene.select_frames(num_frames)
    if first:
        frame_ids = frame_ids[:first]
    frames = scene.load_frames(frame_ids, with_rgb=False)
    bm = BlockMap.from_frames(frames)
    out.mkdir(parents=True, exist_ok=True)
    bm.save(out / "block_map.npz", frame_indices=np.array(frame_ids))

    state = bm.to_dense_states()
    tokens, _ = tokenize(state, bm.B, bm.origin, bm.h)
    summary = {"scene": scene_id, "frame_indices": frame_ids,
               "domain_size_m": (np.array(bm.shape_blocks) * bm.block_extent).tolist(),
               "voxels": {name: int((state == k).sum()) for k, name in LABEL_NAMES.items()},
               "tokens": tokens.counts(), "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # tokens.npz marks the scene as done, so write it last and atomically.
    tokens.save(out / "tokens.tmp.npz")
    os.replace(out / "tokens.tmp.npz", out / "tokens.npz")
    c = tokens.counts()
    return (f"{scene_id}: {c['surface']} surface + {c['free']} free + {c['unknown']} unknown = "
            f"{c['surface_both']} tokens ({summary['seconds']} s)")


def build_or_report(scene_id, **kw) -> str:
    try:
        return build(scene_id, **kw)
    except Exception as e:  # keep building the other scenes
        return f"{scene_id}: FAILED ({type(e).__name__}: {e})"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/scannetpp"))
    p.add_argument("--splits", type=Path, default=Path("data/qa/splits.json"))
    p.add_argument("--scenes", nargs="*", help="Scene ids (default: all scenes in --splits)")
    p.add_argument("--output", type=Path, default=Path("outputs/scenes"))
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--first", type=int, default=0, help="Use only the first N of the --frames frames (0 = all)")
    p.add_argument("--jobs", type=int, default=1, help="Scenes built in parallel")
    args = p.parse_args()
    scenes = args.scenes or sorted(s for ids in json.loads(args.splits.read_text()).values() for s in ids)
    print(f"Building {len(scenes)} scenes with {args.jobs} job(s) -> {args.output}", flush=True)

    work = partial(build_or_report, data_root=args.data_root, output=args.output, num_frames=args.frames,
                   first=args.first)
    with Pool(args.jobs) as pool:
        for n, line in enumerate(pool.imap_unordered(work, scenes), 1):
            print(f"[{n}/{len(scenes)}] {line}", flush=True)


if __name__ == "__main__":
    main()
