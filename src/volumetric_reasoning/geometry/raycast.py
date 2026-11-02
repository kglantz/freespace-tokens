"""Render depth maps of a reference mesh, used to validate poses and intrinsics.

If pose, intrinsics and depth units are interpreted correctly, the depth rendered
from the laser-scan mesh should agree with the measured sensor depth to within a
few centimeters (sensor noise + mesh simplification).
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def make_raycasting_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def render_depth(
    scene: o3d.t.geometry.RaycastingScene,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Z-depth (meters) of the mesh seen by a pinhole camera; inf where no hit.

    Rays go through pixel centers (u + 0.5, v + 0.5). Ray directions are
    K^-1 [u, v, 1] (z-component 1 in camera space), so the hit distance t equals
    the z-depth, which is what depth sensors measure (not the Euclidean range).
    Camera axes: OpenCV (x right, y down, z forward).
    """
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    dirs_cam = np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1
    ).reshape(-1, 3)
    R, t = cam_to_world[:3, :3], cam_to_world[:3, 3]
    dirs_world = dirs_cam @ R.T  # rotate directions only; translation moves the origin
    origins = np.broadcast_to(t, dirs_world.shape)
    rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)
    hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    return hits.reshape(height, width)


def depth_agreement(rendered: np.ndarray, measured: np.ndarray) -> dict:
    """Statistics of |rendered - measured| over pixels valid in both maps."""
    valid = np.isfinite(rendered) & (measured > 0)
    err = np.abs(rendered[valid] - measured[valid])
    return {
        "valid_fraction": float(valid.mean()),
        "median_abs_err_m": float(np.median(err)) if err.size else float("nan"),
        "p90_abs_err_m": float(np.percentile(err, 90)) if err.size else float("nan"),
    }


def render_normals(
    scene: o3d.t.geometry.RaycastingScene,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Simple shaded image (H, W) of a mesh seen from a pinhole camera; NaN where no hit.

    Shading = |cos| of the angle between surface normal and viewing ray (headlight lighting),
    which works for both point-cloud-like and mesh geometry independent of face orientation.
    """
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    dirs_cam = np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1
    ).reshape(-1, 3)
    dirs = dirs_cam @ cam_to_world[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rays = np.concatenate([np.broadcast_to(cam_to_world[:3, 3], dirs.shape), dirs], 1)
    out = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    n = out["primitive_normals"].numpy()
    shade = np.abs((n * dirs).sum(1))
    shade[~np.isfinite(out["t_hit"].numpy())] = np.nan
    return shade.reshape(height, width)


def render_topdown(
    scene: o3d.t.geometry.RaycastingScene,
    xy_min: np.ndarray,
    xy_max: np.ndarray,
    z_start: float,
    resolution: float = 0.01,
) -> np.ndarray:
    """Orthographic view straight down, with rays starting at height z_start.

    Starting below the ceiling gives a 'dollhouse' view into the room. Returns a shaded
    image (rows = y descending, cols = x ascending); NaN where no hit.
    """
    xs = np.arange(xy_min[0], xy_max[0], resolution)
    ys = np.arange(xy_max[1], xy_min[1], -resolution)
    gx, gy = np.meshgrid(xs, ys)
    origins = np.stack([gx, gy, np.full_like(gx, z_start)], -1).reshape(-1, 3)
    dirs = np.broadcast_to([0.0, 0.0, -1.0], origins.shape)
    out = scene.cast_rays(o3d.core.Tensor(np.concatenate([origins, dirs], 1).astype(np.float32)))
    n = out["primitive_normals"].numpy()
    # Light from a fixed oblique direction so that walls and floor are distinguishable.
    light = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
    shade = 0.25 + 0.75 * np.abs(n @ light)
    shade[~np.isfinite(out["t_hit"].numpy())] = np.nan
    return shade.reshape(gx.shape)
