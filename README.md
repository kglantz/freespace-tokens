# FreeSpace Tokens

Explicit free and unknown 3D space as input tokens for a vision-language model
([research proposal](documents/FreeSpace_Tokens_Research_Proposal.tex)).

## Data representation

1. **Reconstruction (2 cm).** 32 posed iPhone RGB-D frames from ScanNet++. ARKit poses are refined
   by multi-view ICP and fused into a sparse VoxelHashing map (2 cm voxels, 8³ blocks). Every voxel
   is labelled occupied (TSDF zero surface), free (traversed by a depth ray) or unknown.
2. **Tokens (16 cm blocks).** One surface token per block containing surface (position = mean of
   its occupied voxels). Free blocks and unknown blocks inside the room are merged separately into
   axis-aligned boxes (max 1.5 m per side); one token per box (center, size, type).

On scene `1ada7a0617`: 4,556 surface + 460 free + 195 unknown = 5,211 scene tokens.

## Scripts

Run from the repository root, in order:

| Script | Does | Output (`outputs/<scene>/`) |
|---|---|---|
| `scripts/step1_reconstruction.py` | Frames → ICP → 2 cm map; mesh vs. laser scan | `step1_reconstruction/` |
| `scripts/step2_labels_vs_gt.py` | Occupied/free/unknown labels vs. laser ground truth | `step2_labels/` |
| `scripts/step3_tokens.py` | Block tokens and token counts | `step3_tokens/` |

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
| `geometry/pose_refinement.py` | Multi-view ICP |
| `geometry/block_map.py` | VoxelHashing TSDF fusion with free/unknown labels, mesh extraction |
| `geometry/visibility.py` | Ray traversal (free space) |
| `geometry/zero_surface.py` | Occupied voxels from the TSDF zero surface |
| `geometry/block_tokens.py` | Block labels, surface tokens, box merging |
| `geometry/labels.py` | Label constants |
| `render3d.py` | Shared 3D rendering for figures |

Data layout and download: [data/README.md](data/README.md).
