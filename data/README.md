# Local data

This directory is **git-ignored** (except this file). Never commit dataset files.

## Setup

Download ScanNet++ with the official download script (outside this repo) and symlink it here:

```bash
ln -s ~/dev/scannetpp data/scannetpp
```

Alternatively, set `dataset.root` in `configs/local.yaml` (git-ignored).

## Layout (ScanNet++ v2, verified on scene `3e8bba0176`)

```text
data/scannetpp/
├── splits/          nvs_sem_train.txt, nvs_sem_val.txt, ...   (3e8bba0176 is in nvs_sem_val)
├── metadata/        semantic / instance class lists
└── data/<scene_id>/
    ├── iphone/                 ← our RGB-D input
    │   ├── rgb.mkv             H.264 video, 1920×1440, 60 fps
    │   ├── depth.bin           LiDAR depth 256×192, uint16 mm, LZ4-compressed per frame
    │   ├── pose_intrinsic_imu.json   per-frame ARKit intrinsics + poses ("aligned_pose")
    │   └── colmap/, nerfstudio/      refined poses for ~every 10th frame (unused for now)
    ├── scans/                  ← laser-scan reference, evaluation only
    │   ├── mesh_aligned_0.05.ply           reference mesh
    │   ├── mesh_aligned_0.05_semantic.ply  mesh with semantic labels
    │   └── segments.json, segments_anno.json
    ├── dslr/                   fisheye still images + COLMAP poses (no depth)
    └── panocam/                panoramas with depth from fixed positions
```

Frame counts match across `rgb.mkv`, `depth.bin` and `pose_intrinsic_imu.json` (7209 for 3e8bba0176).
Conventions (pose direction, camera axes, intrinsics scaling, depth units) are documented and verified in
`src/volumetric_reasoning/datasets/scannetpp.py` and `experiments/01_scene_inspection.ipynb`.

- Dataset website: https://scannetpp.mlsg.cit.tum.de/scannetpp/
- Official toolkit: https://github.com/scannetpp/scannetpp
