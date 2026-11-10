"""ScanNet++ laser panoramas (``panocam/``) as ground-truth range data.

Each panorama stores, per pixel, the laser range (uint16 millimeters, 0 = no return) and the
beam direction in the scanner frame as azimuth (uint16 milliradians, 0..2*pi) and elevation
(uint16 milliradians, a polar angle measured from straight up, 0..~150 degrees):

    d_scanner = (sin(el) cos(az), sin(el) sin(az), cos(el))

The scanner poses are not part of the downloaded assets, so ``register_scanner`` recovers
each pose by aligning the panorama's point cloud to the laser mesh, which was built from
these very scans (FPFH + RANSAC global registration, then point-to-plane ICP). On
1ada7a0617 this gives >99% of points within 1 cm of the mesh (RMSE ~4.7 mm).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def panorama_ids(scene_dir: Path) -> list[int]:
    return sorted(int(p.stem) for p in (Path(scene_dir) / "panocam" / "resized_depth").glob("*.png"))


def panorama_points(scene_dir: Path, index: int, resized: bool = True) -> np.ndarray:
    """(N, 3) laser points in the scanner frame (meters); pixels without a return are skipped."""
    pre = "resized_" if resized else ""
    base = Path(scene_dir) / "panocam"
    depth = cv2.imread(str(base / f"{pre}depth" / f"{index}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000
    az = cv2.imread(str(base / f"{pre}azim" / f"{index}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000
    el = cv2.imread(str(base / f"{pre}elev" / f"{index}.png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000
    valid = depth > 0
    d = np.stack([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)], axis=-1)[valid]
    return d * depth[valid, None]


def register_scanner(points: np.ndarray, mesh: o3d.geometry.TriangleMesh, seed: int = 0) -> tuple[np.ndarray, float]:
    """scanner_to_world (4x4) aligning scanner-frame points to the mesh, and fitness at 1 cm."""
    reg = o3d.pipelines.registration
    o3d.utility.random.seed(seed)
    coarse_t = mesh.sample_points_uniformly(1_500_000).voxel_down_sample(0.05)
    fine_t = mesh.sample_points_uniformly(3_000_000).voxel_down_sample(0.01)
    for cloud, r in ((coarse_t, 0.15), (fine_t, 0.05)):
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=r, max_nn=30))
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    coarse_s = src.voxel_down_sample(0.05)
    coarse_s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    feat = lambda c: reg.compute_fpfh_feature(c, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    glob = reg.registration_ransac_based_on_feature_matching(
        coarse_s, coarse_t, feat(coarse_s), feat(coarse_t), True, 0.1,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(0.1)],
        reg.RANSACConvergenceCriteria(400000, 0.999))
    fine_s = src.voxel_down_sample(0.01)
    T = glob.transformation
    for dist in (0.1, 0.03, 0.01):
        T = reg.registration_icp(fine_s, fine_t, dist, T, reg.TransformationEstimationPointToPlane()).transformation
    return np.asarray(T), float(reg.evaluate_registration(fine_s, fine_t, 0.01, T).fitness)


def scanner_poses(scene_dir: Path, mesh: o3d.geometry.TriangleMesh, cache: Path | None = None,
                  min_fitness: float = 0.95) -> dict[int, np.ndarray]:
    """Registered scanner_to_world per panorama, cached as JSON; fails loudly on a bad fit."""
    if cache is not None and Path(cache).exists():
        return {int(k): np.array(v["scanner_to_world"]) for k, v in json.loads(Path(cache).read_text()).items()}
    poses, report = {}, {}
    for i in panorama_ids(scene_dir):
        T, fitness = register_scanner(panorama_points(scene_dir, i), mesh)
        if fitness < min_fitness:
            raise RuntimeError(f"Panorama {i} registered with fitness {fitness:.3f} < {min_fitness}")
        poses[i] = T
        report[i] = {"scanner_to_world": T.tolist(), "fitness_1cm": fitness}
    if cache is not None:
        Path(cache).write_text(json.dumps(report, indent=2) + "\n")
    return poses
