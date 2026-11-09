"""Checks of qa_dataset.py: VSI-Bench scoring and every example of every split and variant against the files.

Run from the repo root after build_scenes.py and extract_features.py:  python tests/test_dataset.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volumetric_reasoning.geometry.labels import FREE, UNKNOWN  # noqa: E402
from volumetric_reasoning.qa_dataset import (NUMERIC_PROMPT, TASKS, VARIANTS, SceneQADataset,  # noqa: E402
                                             mean_relative_accuracy, parse_choice, parse_number, score)

MODEL = "Qwen/Qwen3-VL-4B-Instruct"

# 1. scoring, hand-computed (1 - theta = .50 .45 .40 .35 .30 .25 .20 .15 .10 .05)
assert mean_relative_accuracy(1.0, 1.0) == 1.0
assert mean_relative_accuracy(1.04, 1.0) == 1.0     # rel .04 below all ten
assert mean_relative_accuracy(0.88, 1.0) == 0.8     # rel .12 below .50 ... .15 (8 of 10)
assert mean_relative_accuracy(13.2, 10) == 0.4      # rel .32 below .50 .45 .40 .35 (4 of 10)
assert mean_relative_accuracy(1.6, 1.0) == 0.0
assert parse_number("about 2.5 meters") == 2.5 and parse_number("1,200") == 1200 and parse_number("no") is None
for text, letter in [("D", "D"), ("D. mug", "D"), ("(B) chair", "B"), ("C)", "C"), ("The answer is A. sofa", "A"),
                     ("Dresser", None), ("answer: d", None)]:
    assert parse_choice(text) == letter, (text, parse_choice(text))
assert score("object_relative_distance", "D. mug", "D. mug") == 1 and score("object_relative_distance", "B", "D. mug") == 0
assert score("object_count", "three", "3") == 0 and score("object_count", "3", "3") == 1
print("1. scoring OK")

# 2. every example of every split and variant, cross-checked against the raw files
for split in ("train", "val", "test"):
    rows = [r for r in map(json.loads, open(ROOT / f"data/qa/{split}.jsonl")) if r["task"] in TASKS]
    datasets = {v: SceneQADataset(ROOT / f"data/qa/{split}.jsonl", v, ROOT / "outputs/scenes", MODEL) for v in VARIANTS}
    for ds in datasets.values():
        assert len(ds) == len(rows)
    for i, r in enumerate(rows):
        ex = {v: ds[i] for v, ds in datasets.items()}
        e = ex["surface_both"]
        assert e["id"] == r["id"] and e["answer"] == r["answer"] and e["task"] == r["task"]
        assert score(r["task"], r["answer"], r["answer"]) == 1.0          # ground truth scores perfectly
        assert e["prompt"].startswith(r["question"])
        assert e["prompt"].endswith(NUMERIC_PROMPT) == (r["task"] != "object_relative_distance")
        for v, x in ex.items():                                           # variants differ only in boxes
            assert x["surface_feature"] is e["surface_feature"] and x["surface_position"] is e["surface_position"]
            assert set(x["box_type"].tolist()) <= set(VARIANTS[v])
    for s in sorted({r["scene_id"] for r in rows}):                       # per scene: counts and coordinates
        i = next(i for i, r in enumerate(rows) if r["scene_id"] == s)
        t = np.load(ROOT / f"outputs/scenes/{s}/tokens.npz")
        f = np.load(ROOT / f"outputs/scenes/{s}/features_{MODEL.split('/')[-1]}.npz")
        ok = f["n_pixels"] > 0
        e = datasets["surface_both"][i]
        assert torch.equal(e["surface_feature"], torch.from_numpy(f["features"][ok]))
        assert len(datasets["surface_only"][i]["box_type"]) == 0
        assert len(datasets["surface_free"][i]["box_type"]) == (t["box_type"] == FREE).sum()
        assert len(datasets["surface_unknown"][i]["box_type"]) == (t["box_type"] == UNKNOWN).sum()
        assert len(e["box_type"]) == len(t["box_type"])
        shift = t["surface_position"][ok][0] - e["surface_position"][0].numpy()   # one shift for all tokens
        assert np.allclose(t["box_center"] - shift, e["box_center"].numpy(), atol=1e-4)
        assert np.allclose(t["box_size"], e["box_size"].numpy())
        p = e["surface_position"].numpy()
        assert abs((p[:, 0].min() + p[:, 0].max()) / 2) < 1e-4 and abs((p[:, 1].min() + p[:, 1].max()) / 2) < 1e-4
        assert abs(np.percentile(p[:, 2], 1)) < 1e-4
    print(f"2. {split}: {len(rows)} examples x {len(VARIANTS)} variants OK")
