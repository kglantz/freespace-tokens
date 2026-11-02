# Experiments

Notebooks for exploration and visual checks. Reusable code lives in `src/volumetric_reasoning/`
and is imported here; notebooks should not define core pipeline functions.

Please **clear outputs before committing** notebooks (embedded images bloat the repo).
Generated files (meshes, point clouds, plots) go to `outputs/`, which is git-ignored.

| Notebook | Milestone | Purpose |
|---|---|---|
| `01_scene_inspection.ipynb` | 1 | Load one scene; check RGB/depth alignment, depth statistics, trajectory, pose validation against the laser mesh |
| `02_backprojection.ipynb` | 2 | Back-project depth to world points; multi-view alignment and error vs. depth against the laser mesh; saves a point cloud to `outputs/` |
| `03_tsdf_open3d.ipynb` | 3–4 | Open3D TSDF fusion + Marching Cubes; renders vs. laser mesh; accuracy/completeness/F-score; voxel-size sweep (1–8 cm) |
