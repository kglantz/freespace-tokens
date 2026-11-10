"""Surface-token features: frozen Qwen3-VL image features, mean-pooled per 16 cm surface block.

For each built scene (outputs/scenes/<scene>/, from build_scenes.py): the same 32 frames that built the map
are encoded by the vision encoder of --model, and the patch features are pooled into the scene's surface
blocks via the depth (see volumetric_reasoning/features.py). Rows follow tokens.npz["surface_block"].
Features depend on the model (the 2B and 4B encoders differ), so the file is named after it.

Output: outputs/scenes/<scene>/features_<model>.npz with
    features  (Ns, dim) float16   mean feature per surface block (zeros where n_pixels == 0)
    n_pixels  (Ns,) int           depth pixels that contributed

Usage (from the repo root, after build_scenes.py):
    venv/bin/python scripts/extract_features.py --model Qwen/Qwen3-VL-4B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volumetric_reasoning.datasets.scannetpp import ScanNetPPIphoneScene  # noqa: E402
from volumetric_reasoning.features import IMAGE_HEIGHT, IMAGE_WIDTH, VisionEncoder, pool_block_features  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--data-root", type=Path, default=Path("data/scannetpp"))
    p.add_argument("--splits", type=Path, default=Path("data/qa/splits.json"))
    p.add_argument("--scenes", nargs="*", help="Scene ids (default: all scenes in --splits)")
    p.add_argument("--scene-root", type=Path, default=Path("outputs/scenes"))
    args = p.parse_args()
    scenes = args.scenes or sorted(s for ids in json.loads(args.splits.read_text()).values() for s in ids)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    name = f"features_{args.model.split('/')[-1]}.npz"
    print(f"Loading {args.model} vision encoder on {device} ...", flush=True)
    encoder = VisionEncoder(args.model, device)

    for n, scene_id in enumerate(scenes, 1):
        d = args.scene_root / scene_id
        if (d / name).exists():
            print(f"[{n}/{len(scenes)}] {scene_id}: already done", flush=True)
            continue
        t0 = time.time()
        with np.load(d / "block_map.npz") as m:
            frame_ids, shape_blocks, map_origin = m["frame_indices"].tolist(), m["shape_blocks"], m["origin"]
        with np.load(d / "tokens.npz") as t:
            surface_block, origin, block_size = t["surface_block"], t["origin"], float(t["block_size_m"])
        assert np.allclose(origin, map_origin)
        frames = ScanNetPPIphoneScene(args.data_root, scene_id).load_frames(frame_ids, with_rgb=True)
        feats, n_pixels = pool_block_features(frames, encoder, surface_block, origin, block_size, shape_blocks)
        np.savez_compressed(d / "features.tmp.npz", features=feats.astype(np.float16), n_pixels=n_pixels,
                            model=args.model, image_size=np.array([IMAGE_WIDTH, IMAGE_HEIGHT]))
        os.replace(d / "features.tmp.npz", d / name)
        print(f"[{n}/{len(scenes)}] {scene_id}: {len(feats)} surface blocks, {(n_pixels == 0).sum()} without "
              f"features, median {int(np.median(n_pixels))} pixels/block ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
