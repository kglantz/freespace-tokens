"""Projective TSDF fusion and separate depth-ray visibility evidence.

TSDF samples live at voxel centers and use d - z, normalized by truncation mu.
This projective residual is not an Euclidean distance or an occupancy label.
Voxel occupancy uses the observed zero set of multilinear TSDF interpolation.
Visibility uses depth-ray traversal.

Conventions: OpenCV axes, camera-to-world poses, pixel (u, v) covers
[u, u+1) x [v, v+1); voxel centers are origin + (index + 0.5) * voxel_size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .backprojection import depth_to_points
from .visibility import ray_voxel_evidence
from .states import zero_surface_mask

# Version 3: occupied comes from the observed TSDF zero set, not raw endpoints.
STATE_SCHEMA_VERSION = 3
STATE_DEFINITION = 'observed_tsdf_zero_surface_and_depth_ray_visibility_v1'
UNKNOWN, FREE, OCCUPIED = 0, 1, 2
STATE_NAMES = {UNKNOWN: "unknown", FREE: "free", OCCUPIED: "occupied"}


@dataclass
class DenseTSDF:
    origin: np.ndarray  # (3,) world position of the grid's minimum corner
    dims: tuple[int, int, int]  # number of voxels along x, y, z
    voxel_size: float
    trunc: float  # truncation distance mu in meters
    free_margin: float = 0.02  # Euclidean guard before each measured endpoint

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64)
        if (self.origin.shape != (3,) or not np.isfinite(self.origin).all()
                or len(self.dims) != 3 or any(int(n) != n or n <= 0 for n in self.dims)
                or not np.isfinite(self.voxel_size) or self.voxel_size <= 0
                or not np.isfinite(self.trunc) or self.trunc <= 0
                or not np.isfinite(self.free_margin) or self.free_margin < 0):
            raise ValueError("Grid requires finite origin, positive dimensions/scale and nonnegative free margin")
        self.dims = tuple(int(n) for n in self.dims)
        self.tsdf = np.ones(self.dims, dtype=np.float32)  # normalized by trunc; 1 = "far in front"
        self.weight = np.zeros(self.dims, dtype=np.float32)
        self.n_free = np.zeros(self.dims, dtype=np.uint32)
        self.n_hit = np.zeros(self.dims, dtype=np.uint32)

    # --- construction --------------------------------------------------------
    @classmethod
    def from_bounds(cls, lo, hi, voxel_size: float, trunc_voxels: float = 4.0,
                    free_margin: float = 0.02) -> "DenseTSDF":
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all()
                or np.any(hi <= lo) or not np.isfinite(voxel_size) or voxel_size <= 0):
            raise ValueError("Bounds must be finite and increasing; voxel_size must be positive")
        dims = np.ceil((hi - lo) / voxel_size).astype(int)
        return cls(lo, tuple(dims), voxel_size, trunc_voxels * voxel_size, free_margin)

    @property
    def num_voxels(self) -> int:
        return int(np.prod(self.dims))

    def voxel_centers(self, sl: tuple[slice, slice, slice] | None = None) -> np.ndarray:
        """(nx, ny, nz, 3) world coordinates of voxel centers (optionally of a sub-block)."""
        sl = sl or (slice(0, self.dims[0]), slice(0, self.dims[1]), slice(0, self.dims[2]))
        axes = [
            self.origin[a] + (np.arange(s.start, s.stop) + 0.5) * self.voxel_size
            for a, s in enumerate(sl)
        ]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    # --- integration ---------------------------------------------------------
    def _frustum_block(self, K, cam_to_world, width, height, max_depth):
        """Index ranges of the axis-aligned block of voxels containing the view frustum."""
        corners_px = np.array([[0, 0], [width, 0], [0, height], [width, height]], float)
        rays = np.stack(
            [(corners_px[:, 0] - K[0, 2]) / K[0, 0], (corners_px[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)], 1
        )
        pts_cam = np.vstack([np.zeros((1, 3)), rays * max_depth])
        pts = pts_cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]
        lo = np.floor((pts.min(0) - self.origin) / self.voxel_size).astype(int)
        hi = np.ceil((pts.max(0) - self.origin) / self.voxel_size).astype(int)
        lo, hi = np.clip(lo, 0, self.dims), np.clip(hi, 0, self.dims)
        if np.any(hi <= lo):
            return None
        return tuple(slice(a, b) for a, b in zip(lo, hi))

    def integrate(self, depth: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray,
                  min_depth: float = 0.1, max_depth: float = 4.0, weight: float = 1.0) -> None:
        depth, K, cam_to_world = validate_observation(depth, K, cam_to_world, min_depth, max_depth)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Integration weight must be finite and positive")
        endpoints, _ = depth_to_points(depth, K, cam_to_world, min_depth, max_depth)
        if not len(endpoints):
            return
        free, hits = ray_voxel_evidence(cam_to_world[:3, 3], endpoints, self.origin,
                                       self.dims, self.voxel_size, self.free_margin)
        self.n_free.ravel()[free] += 1
        self.n_hit.ravel()[hits] += 1
        h, w = depth.shape
        # Valid returns at max_depth still support the negative truncation band.
        sl = self._frustum_block(K, cam_to_world, w, h, max_depth + self.trunc)
        if sl is None:
            return
        centers = self.voxel_centers(sl).reshape(-1, 3)

        # World -> camera: p_cam = R^T (p_world - t).
        R, t = cam_to_world[:3, :3], cam_to_world[:3, 3]
        p = (centers - t) @ R
        z = p[:, 2]
        in_front = z > 1e-6
        u = np.full(z.shape, -1.0)
        v = np.full(z.shape, -1.0)
        u[in_front] = K[0, 0] * p[in_front, 0] / z[in_front] + K[0, 2]
        v[in_front] = K[1, 1] * p[in_front, 1] / z[in_front] + K[1, 2]
        # Pixel (i, j) covers [i, i+1) x [j, j+1)  ->  index = floor(coordinate).
        ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        inside = in_front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

        d = np.zeros(z.shape, dtype=np.float32)
        d[inside] = depth[vi[inside], ui[inside]]
        valid = inside & np.isfinite(d) & (d > min_depth) & (d <= max_depth)
        sdf = d - z
        update = valid & (sdf > -self.trunc)  # sdf <= -mu: occluded, no update
        if not update.any():
            return

        tsdf_obs = np.minimum(sdf / self.trunc, 1.0).astype(np.float32)

        # Flat indices into the full grid for the updated voxels of this block.
        local = np.nonzero(update)[0]
        bx, by, bz = (s.stop - s.start for s in sl)
        li, lj, lk = np.unravel_index(local, (bx, by, bz))
        gi, gj, gk = li + sl[0].start, lj + sl[1].start, lk + sl[2].start

        W_old = self.weight[gi, gj, gk]
        D_old = self.tsdf[gi, gj, gk]
        self.tsdf[gi, gj, gk] = (W_old * D_old + weight * tsdf_obs[local]) / (W_old + weight)
        self.weight[gi, gj, gk] = W_old + weight

    def integrate_frames(self, frames: Sequence, min_depth: float = 0.1, max_depth: float = 4.0) -> None:
        for f in frames:
            self.integrate(f.depth, f.K_depth, f.cam_to_world, min_depth, max_depth)

    # --- interpretation ----------------------------------------------------------
    def surface_mask(self) -> np.ndarray:
        """Cells intersecting the observed TSDF zero surface (W>0); no mesh."""
        return zero_surface_mask(self.tsdf, self.weight)

    def classify(self) -> np.ndarray:
        """Proposal states: observed zero surface > ray-traversed free > unknown.

        Raw endpoint counts are retained as diagnostics, but do not determine
        occupancy. Negative TSDF alone does not establish occupancy or free
        space. Labels are finite-resolution observations, not solid completion.
        """
        state = np.full(self.dims, UNKNOWN, dtype=np.uint8)
        state[self.n_free > 0] = FREE
        state[self.surface_mask()] = OCCUPIED
        return state

    @property
    def conflicting_evidence(self) -> np.ndarray:
        """Cells with both traversal and endpoint support (also possible within one frame)."""
        return (self.n_free > 0) & (self.n_hit > 0)



def validate_observation(depth, K, cam_to_world, min_depth, max_depth):
    """Validate the pinhole/rigid-pose assumptions used during fusion."""
    depth = np.asarray(depth)
    K = np.asarray(K, np.float64)
    T = np.asarray(cam_to_world, np.float64)
    if depth.ndim != 2 or not all(depth.shape):
        raise ValueError("Depth must be a nonempty 2-D array")
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8):
        raise ValueError("K must be a finite pinhole intrinsic matrix without skew")
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-4)):
        raise ValueError("cam_to_world must be a finite rigid camera-to-world pose")
    if not (np.isfinite(min_depth) and np.isfinite(max_depth) and 0 <= min_depth < max_depth):
        raise ValueError("Depth range must be finite and satisfy 0 <= min_depth < max_depth")
    return depth, K, T


def count_states(state: np.ndarray) -> dict:
    return {STATE_NAMES[s]: int((state == s).sum()) for s in STATE_NAMES}
