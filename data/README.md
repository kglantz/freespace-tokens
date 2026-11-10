# Local data

The scripts read ScanNet++ scenes from `--data-root`, laid out as:

```text
<data-root>/
  data/
    <scene-id>/
      iphone/
        rgb.mkv                   RGB video
        depth.bin                 LiDAR depth
        pose_intrinsic_imu.json   ARKit poses and intrinsics
      scans/
        mesh_aligned_0.05.ply     laser mesh (evaluation / question ground truth only)
        segments.json             mesh vertex -> segment
        segments_anno.json        annotated instances (label, segments, oriented box)
      panocam/                    laser panoramas: depth, azim, elev (evaluation only)
```

Two data roots are used:

* `data` (default of steps 1–3): the development scene `1ada7a0617`, with laser data.
* `data/scannetpp` (default of `build_scenes.py`): the 50 QA scenes: iPhone data, plus the laser
  mesh and its annotations (`scans/`, used only for the navigation questions' ground truth).

`qa/` holds the ScanNet++ question-answer pairs from ViCA-322K for these 50 scenes
(`train/val/test.jsonl`, scene split in `splits.json`: 30 / 5 / 15 scenes). Two changes to the
original 40 / 5 / 5 split, both recorded in `qa/manifest.json`; the original files are in `qa/original/`:

1. `469f112e38` (poorly reconstructed from 32 frames) moved val → train, `1b379f1114` train → val.
2. 10 train scenes, drawn at random (seed 0), moved to test for a larger test set.

`qa/nav_{train,val,test}.jsonl` hold our navigation questions for the same scenes and split, made by
`scripts/make_nav_qa.py` from the laser mesh and annotations (`src/volumetric_reasoning/nav_qa.py`):
`nav_collision` (yes / no, balanced per scene and robot width) and `nav_distance` (meters). Each row also stores both
endpoint objects (`a`, `b`: instance id, label, oriented box) and the measured values.

Download with `data_download/download_scannetpp.py`; access requires a token from the
[ScanNet++ portal](https://scannetpp.mlsg.cit.tum.de/scannetpp/).

Only this README is tracked under `data/`.
