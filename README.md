# freespace-tokens

Volumetric scene representations that encode **occupied, free, and unknown space** as a
basis for spatial visual tokens in vision-language models (VLMs).

Student research project, Visual Computing & AI group, TUM (supervisor: Prof. Matthias Nießner).

## Motivation

VLMs process images as sets of visual tokens. When several posed RGB-D views of a scene
are available, simply concatenating per-image tokens ignores that many of them observe the
same 3D region. We investigate organizing visual information in a **shared 3D volumetric
representation** instead:

1. Fuse posed RGB-D frames into a voxel grid (TSDF fusion; Curless & Levoy 1996, KinectFusion 2011).
2. Distinguish occupied, free, and unknown space. Unobserved voxels are *not* treated as free.
3. Assign image features to voxels and aggregate multi-view observations (mean / max / learned, to be compared).
4. Use voxel centers as positional information for downstream VLM reasoning.

The token representation, the aggregation strategy, and how to encode free space are
**open research questions**.

## Status

| Milestone | Topic | Status |
|---|---|---|
| 0 | Repository setup | done |
| 1 | ScanNet++ loading and inspection | planned |
| 2 | Depth back-projection, multi-view point clouds | planned |
| 3 | TSDF fusion | planned |
| 4 | Marching Cubes mesh extraction and visualization | planned |
| 5 | Occupied / free / unknown classification | planned |
| 6 | Visual feature integration and aggregation | planned |

No pipeline functionality is implemented yet.

## Repository structure

```text
configs/default.yaml        Dataset path, scene, frame subset, output dir
data/                       Local datasets (git-ignored, see data/README.md)
experiments/                Experiment notes and scripts
src/volumetric_reasoning/
    datasets/               Data loaders (M1)
    geometry/               Back-projection, TSDF, mesh extraction (M2–M5)
    visualization/          Image / point cloud / mesh / voxel visualization
    features/               Feature extraction and aggregation (M6)
tests/                      Synthetic tests for geometric transformations
pyproject.toml              Makes the package installable with `pip install -e .`
requirements.txt            Runtime dependencies
```

`scripts/` will be added with the first runnable entry point (Milestone 1).

## Environment setup

Requires Python ≥ 3.10 (tested with 3.13). Keep the repository out of cloud-synced folders
(iCloud, Dropbox, OneDrive). Virtual environments break there.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Activate it again in each new shell with `source .venv/bin/activate`.

## Dataset setup

We use [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/) (access must be requested
separately). The data is **never committed**. Symlink it to `data/scannetpp/` or point
`dataset.root` in a git-ignored `configs/local.yaml` to its location. See
[data/README.md](data/README.md).

## Running experiments

Nothing is runnable yet. From Milestone 1 on, scripts in `scripts/` will take a config file,
for example:

```bash
python scripts/visualize_scene.py --config configs/local.yaml   # planned
```

Tests (once added): `pytest`.

## References

- B. Curless, M. Levoy. *A Volumetric Method for Building Complex Models from Range Images.* SIGGRAPH 1996.
- S. Izadi et al. *KinectFusion: Real-time 3D Reconstruction and Interaction Using a Moving Depth Camera.* UIST 2011.
- C. Yeshwanth et al. *ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes.* ICCV 2023.
