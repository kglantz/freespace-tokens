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
| 1 | ScanNet++ loading and inspection | done |
| 2 | Depth back-projection, multi-view point clouds | done |
| 3 | TSDF fusion | in progress: Open3D reference done, own implementation next |
| 4 | Marching Cubes mesh extraction and visualization | in progress: meshes + metrics vs. laser scan for the reference |
| 5 | Occupied / free / unknown classification | planned |
| 6 | Visual feature integration and aggregation | planned |

Implemented so far: an iPhone RGB-D loader for ScanNet++ (`datasets/scannetpp.py`) with
verified pose / intrinsics / depth conventions, a mesh depth renderer used for validation
(`geometry/raycast.py`), depth back-projection to world-space point clouds
(`geometry/backprojection.py`), reference TSDF fusion with Open3D (`geometry/tsdf_open3d.py`),
mesh-vs-laser-scan metrics (`geometry/mesh_eval.py`), and notebooks 01–03 in `experiments/`.

## Repository structure

```text
configs/default.yaml        Dataset path, scene, frame subset, output dir
data/                       Local datasets (git-ignored, see data/README.md)
experiments/                Experiment notes and scripts
src/volumetric_reasoning/
    config.py               Loads configs/default.yaml + optional configs/local.yaml
    datasets/scannetpp.py   ScanNet++ iPhone RGB-D loader (M1)
    geometry/raycast.py     Render reference-mesh depth for validation (M1)
    geometry/backprojection.py   Pixel -> camera -> world, multi-frame point clouds (M2)
    geometry/tsdf_open3d.py      Reference TSDF fusion + Marching Cubes via Open3D (M3/M4)
    geometry/mesh_eval.py        Accuracy / completeness / F-score vs. reference mesh
    geometry/               Later: own TSDF with free/unknown space (M3, M5)
    visualization/          Image / point cloud / mesh / voxel visualization
    features/               Feature extraction and aggregation (M6)
tests/                      Synthetic tests for geometric transformations
pyproject.toml              Makes the package installable with `pip install -e .`
requirements.txt            Runtime dependencies
```

`scripts/` will be added when we need command-line entry points.

## Environment setup

Requires Python ≥ 3.10 (tested with 3.13). On macOS, Open3D additionally needs
`brew install libusb`. Keep the repository out of cloud-synced folders
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

Open the notebooks in `experiments/` (in order) with the `.venv` kernel (PyCharm, VS Code or Jupyter)
and run all cells. The scene and frame subset come from `configs/default.yaml` (override in
`configs/local.yaml`).

Synthetic tests for loader and geometry helpers (no dataset needed):

```bash
pytest
```

## References

- B. Curless, M. Levoy. *A Volumetric Method for Building Complex Models from Range Images.* SIGGRAPH 1996.
- S. Izadi et al. *KinectFusion: Real-time 3D Reconstruction and Interaction Using a Moving Depth Camera.* UIST 2011.
- C. Yeshwanth et al. *ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes.* ICCV 2023.
