# Local data

The pipeline takes a ScanNet++ download root containing this layout:

```text
<download-root>/
  data/
    <scene-id>/
      iphone/
        rgb.mkv
        depth.bin
        pose_intrinsic_imu.json
```

Pass the outer directory as `--data-root`. For example, `--data-root data/scannetpp` reads scenes
under `data/scannetpp/data/`. The [GPU runbook](../documents/GPU_RUNBOOK.md) shows how to select
ViCA-annotated scenes and download just these three assets. Access requires a token from the
[ScanNet++ portal](https://scannetpp.mlsg.cit.tum.de/scannetpp/).

Only this README is tracked under `data/`. Keep dataset files and generated downloader configs here.
