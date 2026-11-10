"""Checks of qa_dataset.py / evaluation.py: scoring and summaries, and every example of every split, view and
variant against the files (ViCA questions; navigation questions with their visibility and endpoint marks).

Run from the repo root after build_scenes.py and extract_features.py (all views) and make_nav_qa.py:
    python tests/test_dataset.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volumetric_reasoning.geometry.labels import FREE, UNKNOWN  # noqa: E402
from volumetric_reasoning.evaluation import summarize  # noqa: E402
from volumetric_reasoning.qa_dataset import (ALL_TASKS, ENDPOINT_PAD, MIN_ENDPOINT_TOKENS, NAV_TASKS,  # noqa: E402
                                             NUMERIC_PROMPT, TASKS, VARIANTS, VIEWS, SceneQADataset, in_obb,
                                             mean_relative_accuracy, parse_choice, parse_number, parse_yes_no, score)

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
for text, word in [("yes", "yes"), ("No.", "no"), (" YES, it can", "yes"), ("no way", "no"), ("Nope", None),
                   ("Yesterday", None), ("I think yes", None), ("", None)]:
    assert parse_yes_no(text) == word, (text, parse_yes_no(text))
assert score("nav_collision", "Yes.", "yes") == 1 and score("nav_collision", "no", "yes") == 0
assert score("nav_collision", "maybe", "no") == 0 and score("nav_distance", "3.4", "3.4") == 1
print("1. scoring OK")

# 1a. summaries: collision = balanced accuracy; collision rate = "yes" on truly blocked questions
def p(task, view, answer, prediction):
    return {"task": task, "view": view, "answer": answer, "prediction": prediction,
            "score": score(task, prediction, answer)}
preds = ([p("nav_collision", "first8", "yes", "yes")] * 3 + [p("nav_collision", "first8", "yes", "no")]
         + [p("nav_collision", "first8", "no", "yes")] + [p("nav_collision", "first8", "no", "no")]
         + [p("nav_distance", "full", "2.0", "2.0"), p("nav_distance", "full", "2.0", "4.0"),
            p("object_count", "full", "3", "3")])
s = summarize(preds)
assert s["per_task"] == {"object_count": 1.0, "nav_collision/first8": (3 / 4 + 1 / 2) / 2, "nav_distance/full": 0.5}
assert s["collision_rate"] == {"nav_collision/first8": 0.5}
assert s["counts"] == {"object_count": 1, "nav_collision/first8": 6, "nav_distance/full": 2}
print("1a. summaries OK")

# 1b. oriented boxes: 2 m x 0.4 m x 1 m box at (1, 2, 0.5), rotated 30 degrees about z (rows = box axes)
c, a = np.array([1.0, 2.0, 0.5]), np.radians(30)
ax0, ax1 = np.array([np.cos(a), np.sin(a), 0]), np.array([-np.sin(a), np.cos(a), 0])
obb = {"centroid": c.tolist(), "axesLengths": [2.0, 0.4, 1.0], "normalizedAxes": np.stack([ax0, ax1, [0, 0, 1]]).ravel().tolist()}
pts = np.stack([c, c + 0.99 * ax0, c + 1.01 * ax0, c + 0.19 * ax1, c + 0.21 * ax1, c + 0.5 * ax1, c + [0, 0, 0.49]])
assert in_obb(pts, obb).tolist() == [True, True, False, True, False, False, True]
assert in_obb(pts, obb, pad=0.05).tolist() == [True, True, True, True, True, False, True]
print("1b. oriented boxes OK")

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

# 3. navigation questions: every split, view and variant, against an independent computation from the raw files
#    (world coordinates straight from tokens.npz, no normalization)
FEATURES = f"features_{MODEL.split('/')[-1]}.npz"
for split in ("train", "val", "test"):
    rows = [json.loads(line) for line in open(ROOT / f"data/qa/nav_{split}.jsonl")]
    views = {k: ROOT / v for k, v in VIEWS.items()}
    expected, world = [], {}
    for r in rows:
        for view, root in views.items():
            key = (view, r["scene_id"])
            if key not in world:
                t, f = np.load(root / r["scene_id"] / "tokens.npz"), np.load(root / r["scene_id"] / FEATURES)
                world[key] = t["surface_position"][f["n_pixels"] > 0]
            a, b = (in_obb(world[key], r[k]["obb"], ENDPOINT_PAD) for k in "ab")
            if a.sum() >= MIN_ENDPOINT_TOKENS and b.sum() >= MIN_ENDPOINT_TOKENS:
                expected.append((r["id"], view, a, b))
    datasets = {v: SceneQADataset(ROOT / f"data/qa/nav_{split}.jsonl", v, views, MODEL, tasks=NAV_TASKS) for v in VARIANTS}
    for ds in datasets.values():
        assert [(r["id"], v) for r, v in zip(ds.rows, ds.views)] == [(i, v) for i, v, _, _ in expected]
    for i, (qid, view, a, b) in enumerate(expected):
        ex = {v: ds[i] for v, ds in datasets.items()}
        e = ex["surface_both"]
        assert e["id"] == qid and e["view"] == view
        mark = np.where(a, 1, np.where(b, 2, 0))                     # a token in both boxes counts for A
        assert np.array_equal(e["mark"].numpy(), mark), (qid, view)
        assert len(e["mark"]) == len(e["surface_feature"]) == len(e["surface_position"])
        assert e["prompt"].endswith("Answer yes or no.") if e["task"] == "nav_collision" else e["prompt"].endswith(NUMERIC_PROMPT)
        assert score(e["task"], e["answer"], e["answer"]) == 1.0
        for v, x in ex.items():                                      # variants differ only in boxes
            assert x["mark"].equal(e["mark"]) and x["surface_feature"] is e["surface_feature"]
            assert set(x["box_type"].tolist()) <= set(VARIANTS[v])
    n = {v: sum(view == v for _, view, _, _ in expected) for v in VIEWS}
    print(f"3. nav {split}: {len(expected)} examples {n} x {len(VARIANTS)} variants OK")

# 4. ViCA + navigation in one dataset: ViCA in the full view with no marks, then the navigation examples
vica = [r for r in map(json.loads, open(ROOT / "data/qa/val.jsonl")) if r["task"] in TASKS]
both = SceneQADataset([ROOT / "data/qa/val.jsonl", ROOT / "data/qa/nav_val.jsonl"], "surface_both",
                      {k: ROOT / v for k, v in VIEWS.items()}, MODEL, tasks=ALL_TASKS)
assert [r["id"] for r in both.rows[:len(vica)]] == [r["id"] for r in vica] and set(both.views[:len(vica)]) == {"full"}
assert all(both.rows[i]["task"] in NAV_TASKS for i in range(len(vica), len(both)))
assert all(both[i]["mark"].sum() == 0 for i in range(0, len(vica), 50))
print(f"4. ViCA + navigation: {len(vica)} + {len(both) - len(vica)} examples OK")
