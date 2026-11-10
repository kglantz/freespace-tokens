# Old notebooks (Kai's dense TSDF pipeline)

These notebooks belong to the first pipeline: a dense TSDF with occupied / free / unknown / behind-surface
states (`geometry/tsdf.py`, `tsdf_open3d.py`, `raycast.py`, `mesh_eval.py`, `config.py` + `configs/`).
That code was replaced by Peter's block-map pipeline and is **no longer on this branch**, so the notebooks
here do not run against the current `src/`. They are kept for reference.

To run them, check out the tag that holds the complete old state:

```bash
git switch --detach kai-tsdf-v1     # back to the old code + notebooks (in experiments/)
git switch develop_Kai              # back to the current branch
```

Or, to keep both side by side, put the tag into its own folder:

```bash
git worktree add ../freespace-tokens-old kai-tsdf-v1
```

| Notebook | Purpose |
|---|---|
| `01_scene_inspection.ipynb` | Load one scene; RGB/depth alignment, depth statistics, trajectory, pose check against the laser mesh |
| `02_backprojection.ipynb` | Depth → world points; multi-view alignment and error vs. depth against the laser mesh |
| `03_tsdf_open3d.ipynb` | Open3D TSDF fusion + Marching Cubes vs. laser mesh; accuracy / completeness / F-score; voxel-size sweep |
| `04_tsdf_free_space.ipynb` | Own dense TSDF; occupied / free / unknown / behind slices; pooled voxel sizes 2–64 cm; frame count 8–276 |
| `05_feature_positions.ipynb` | Where back-projected image tokens land (patch center vs. 3×3 samples), on the mesh and as token voxels |

`README_original.md` is the original experiments README of that pipeline.
