"""Depth back-projection: pixels -> camera coordinates -> world coordinates.

Conventions used throughout the project:

* Camera axes: OpenCV (x right, y down, z forward).
* Pixel (u, v) = (column, row) covers [u, u+1) x [v, v+1); its center is (u + 0.5, v + 0.5).
  This matches `scale_intrinsics`, under which resizing an image scales K exactly.
* Depth is z-depth (distance along the optical axis), not Euclidean ray length.
* Poses are 4x4 camera-to-world matrices T, applied to column vectors:
      p_world = R @ p_cam + t,   with R = T[:3, :3], t = T[:3, 3].
  For row-stacked points P of shape (N, 3) this is  P @ R.T + t.
"""

from __future__ import annotations

import numpy as np


def pixel_centers(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """(H, W) arrays with the u and v coordinates of each pixel center."""
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    return u, v


def backproject_pixels(u: np.ndarray, v: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-space points p = d * K^-1 [u, v, 1]^T for 1-D arrays u, v, depth -> (N, 3).

    Written out explicitly (no matrix inverse) because K is a pinhole matrix:
        x = (u - cx) / fx * d,   y = (v - cy) / fy * d,   z = d
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    return np.stack([x, y, depth], axis=-1)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `backproject_pixels`: camera-space (N, 3) -> continuous (u, v) and depth z."""
    z = points_cam[:, 2]
    uv = (points_cam @ K.T)[:, :2] / z[:, None]
    return uv, z


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to row-stacked points (N, 3)."""
    return points @ T[:3, :3].T + T[:3, 3]


def valid_depth_mask(depth: np.ndarray, min_depth: float = 0.0, max_depth: float = np.inf) -> np.ndarray:
    """Pixels with finite depth in (min_depth, max_depth]. Depth 0 is treated as invalid."""
    return np.isfinite(depth) & (depth > max(min_depth, 0.0)) & (depth <= max_depth)


def depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    cam_to_world: np.ndarray | None = None,
    min_depth: float = 0.0,
    max_depth: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project all valid pixels of a depth map.

    Returns:
        points: (N, 3) in world coordinates if `cam_to_world` is given, else camera coordinates.
        mask:   (H, W) boolean mask of the pixels that produced the points (row-major order),
                e.g. to look up colors: `rgb_at_depth_res[mask]`.
    """
    mask = valid_depth_mask(depth, min_depth, max_depth)
    u, v = pixel_centers(*depth.shape)
    points = backproject_pixels(u[mask], v[mask], depth[mask], K)
    if cam_to_world is not None:
        points = transform_points(points, cam_to_world)
    return points, mask
