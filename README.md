# FreeSpace Tokens

Explicit free and unknown 3D space as input tokens for a vision-language model
([research proposal](documents/FreeSpace_Tokens_Research_Proposal.tex)).

## Data representation

1. **Reconstruction (2 cm).** 32 iPhone RGB-D frames from ScanNet++ (spread uniformly over the
   capture) with their ARKit poses, fused
   into a sparse VoxelHashing map (2 cm voxels, 8³ blocks). Every voxel is labelled occupied (TSDF
   zero surface), free (traversed by a depth ray) or unknown.
2. **Tokens (16 cm blocks).** One surface token per block containing surface (position = mean of
   its occupied voxels; feature = mean of the frozen Qwen3-VL patch features of its depth pixels). Free blocks and unknown blocks inside the room are merged separately into
   axis-aligned boxes (max 1.5 m per side); one token per box (center, size, type).

On scene `1ada7a0617`: 5,052 surface + 509 free + 213 unknown = 5,774 scene tokens. Over the 50 QA
scenes: 1,505–19,077 scene tokens (median 5,105).

## Scripts

Run from the repository root, in order:

| Script | Does | Output (`outputs/<scene>/`) |
|---|---|---|
| `scripts/step1_reconstruction.py` | Frames → 2 cm map; mesh vs. laser scan | `step1_reconstruction/` |
| `scripts/step2_labels_vs_gt.py` | Occupied/free/unknown labels vs. laser ground truth | `step2_labels/` |
| `scripts/step3_tokens.py` | Block tokens and token counts | `step3_tokens/` |

Steps 1–3 study the development scene `1ada7a0617` (the only one with laser ground truth). The QA
scenes (`data/qa/splits.json`) are built with the step-1 map and step-3 default tokens, without laser data:

```bash
venv/bin/python scripts/build_scenes.py --jobs 4   # -> outputs/scenes/<scene>/{block_map,tokens}.npz
venv/bin/python scripts/extract_features.py        # -> outputs/scenes/<scene>/features_<model>.npz
```

Partial views (what a robot has seen after a quarter / half of the walk: the first 8 / 16 of the 32 frames):

```bash
venv/bin/python scripts/build_scenes.py --jobs 4 --first 8 --output outputs/scenes_first8
venv/bin/python scripts/extract_features.py --scene-root outputs/scenes_first8
# same with --first 16 / outputs/scenes_first16
```

Navigation questions with ground truth from the laser scan and its annotations (`data/README.md`):

```bash
venv/bin/python scripts/make_nav_qa.py --render outputs/nav_qa   # -> data/qa/nav_{train,val,test}.jsonl
venv/bin/python tests/test_nav_qa.py
```

## Training and evaluation

Qwen3-VL-4B decoder (frozen, LoRA on attention) reading the scene tokens in place of image tokens; answer-only
loss; 3-D rotary positions in Qwen's image convention (see `src/volumetric_reasoning/model.py`). Training and
evaluation use the ViCA questions (full view) and the navigation questions (all three views, endpoints marked as
A / B in the input). One run per variant (`surface_only`, `surface_free`, `surface_unknown`, `surface_both`), all
with the same seed:

```bash
venv/bin/python scripts/train.py --variant surface_both --output runs/surface_both     # --train-scenes 10 for a pilot
venv/bin/python scripts/evaluate.py --run runs/surface_both --split test
```

Each run validates and checkpoints after every epoch (`best.pt` = best validation mean over tasks). Scores: the
VSI-Bench metrics for ViCA (option accuracy for relative distance, mean relative accuracy for distance, size, count,
room size); per view for navigation: balanced accuracy and collision rate ("yes" on blocked paths) for
`nav_collision`, mean relative accuracy for `nav_distance`. `scripts/headroom.py` scores hand-made rules on the
same navigation questions (how much free / unknown tokens can tell beyond surface tokens, without learning).

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python scripts/step1_reconstruction.py --scene 1ada7a0617
venv/bin/python scripts/step2_labels_vs_gt.py --scene 1ada7a0617
venv/bin/python scripts/step3_tokens.py --scene 1ada7a0617
```

## Code

| Module (`src/volumetric_reasoning/`) | Purpose |
|---|---|
| `datasets/scannetpp.py` | iPhone RGB-D frames, ARKit poses, frame selection |
| `datasets/scannetpp_laser.py` | Laser panoramas and scanner registration (ground truth only) |
| `geometry/backprojection.py` | Depth → 3D points |
| `geometry/block_map.py` | VoxelHashing TSDF fusion with free/unknown labels, mesh extraction |
| `geometry/visibility.py` | Ray traversal (free space) |
| `geometry/zero_surface.py` | Occupied voxels from the TSDF zero surface |
| `geometry/block_tokens.py` | Block labels, surface tokens, box merging |
| `geometry/labels.py` | Label constants |
| `features.py` | Frozen Qwen3-VL image features, mean-pooled per surface block |
| `qa_dataset.py` | QA examples (ViCA + navigation) with scene tokens per variant and view; endpoint marks; scoring |
| `nav_qa.py` | Navigation questions with ground truth from the laser scan |
| `model.py` | Scene-token embeddings, 3-D RoPE positions, LoRA Qwen3-VL decoder, loss, greedy decoding |
| `evaluation.py` | Generation + scoring of a dataset |
| `render3d.py` | Shared 3D rendering for figures |

Data layout and download: [data/README.md](data/README.md).
