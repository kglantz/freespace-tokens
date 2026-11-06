# Local data

The scripts read ScanNet++ scenes from `--data-root` (default `data`), laid out as:

```text
<data-root>/
  data/
    <scene-id>/
      iphone/
        rgb.mkv                   RGB video
        depth.bin                 LiDAR depth
        pose_intrinsic_imu.json   ARKit poses and intrinsics
        colmap/images.txt         COLMAP-registered frames (used for frame selection)
      scans/
        mesh_aligned_0.05.ply     laser mesh (evaluation only)
      panocam/                    laser panoramas: depth, azim, elev (evaluation only)
```

Download with `data_download/download_scannetpp.py`; access requires a token from the
[ScanNet++ portal](https://scannetpp.mlsg.cit.tum.de/scannetpp/).

Only this README is tracked under `data/`.
