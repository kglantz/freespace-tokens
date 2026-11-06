"""Multi-view pose refinement of RGB-D frames by leave-one-out point-to-plane ICP.

Uses only the input depth maps (no reference geometry). In every round, each frame is
aligned to the union of all *other* frames under the poses of the previous round, so a
single misregistered frame is pulled onto the consensus surface. All frames are updated
simultaneously (Jacobi style), which keeps the result independent of frame order.
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import open3d as o3d

from .backprojection import depth_to_points


def frame_cloud(frame, min_depth: float, max_depth: float, voxel: float) -> o3d.geometry.PointCloud:
    """Camera-space point cloud of one depth map, voxel-downsampled, with normals."""
    points, _ = depth_to_points(frame.depth, frame.K_depth, None, min_depth, max_depth)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(voxel)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    return cloud


def refine_poses(frames: Sequence, rounds: int = 3, voxel: float = 0.02,
                 distances: Sequence[float] = (0.08, 0.05, 0.03), min_depth: float = 0.1,
                 max_depth: float = 4.0, verbose: bool = False) -> tuple[list[np.ndarray], list[dict]]:
    """Return refined cam_to_world poses (frames are not modified) and per-round statistics."""
    clouds = [frame_cloud(f, min_depth, max_depth, voxel) for f in frames]
    poses = [np.asarray(f.cam_to_world, float).copy() for f in frames]
    stats = []
    for r in range(rounds):
        dist = distances[min(r, len(distances) - 1)]
        world = [copy.deepcopy(c).transform(T) for c, T in zip(clouds, poses)]
        new_poses, moves, fitness = [], [], []
        for i, (cloud, T) in enumerate(zip(clouds, poses)):
            target = o3d.geometry.PointCloud()
            for j, w in enumerate(world):
                if j != i:
                    target += w
            target = target.voxel_down_sample(voxel)
            target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
            loss = o3d.pipelines.registration.TukeyLoss(k=dist / 2)
            result = o3d.pipelines.registration.registration_icp(
                cloud, target, dist, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
            # Reject degenerate solutions (too little overlap to constrain the pose).
            if result.fitness < 0.3:
                new_poses.append(T)
                moves.append(0.0)
            else:
                new_poses.append(result.transformation)
                moves.append(float(np.linalg.norm(result.transformation[:3, 3] - T[:3, 3])))
            fitness.append(float(result.fitness))
        poses = new_poses
        stats.append({"round": r + 1, "max_correspondence_m": dist,
                      "median_translation_update_m": float(np.median(moves)),
                      "max_translation_update_m": float(np.max(moves)),
                      "median_fitness": float(np.median(fitness)),
                      "frames_kept_unchanged_low_overlap": int(sum(f < 0.3 for f in fitness))})
        if verbose:
            print(f"  ICP round {r + 1}: {stats[-1]}", flush=True)
    return poses, stats
