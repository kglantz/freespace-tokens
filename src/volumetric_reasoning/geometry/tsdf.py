"""Dense TSDF fusion with explicit occupied / free / unknown classification.

Integration is the KinectFusion rule (projective signed distance, truncation, running
average with constant weights). What this module adds is an explicit interpretation of
every voxel in a bounded volume, in the spirit of Curless & Levoy's unseen / empty /
near-surface states.

Per frame, for every voxel whose center projects into the image:

    z   = depth of the voxel center in the camera
    d   = measured depth at the pixel containing the projection
    sdf = d - z

    d invalid or d > max_depth  -> no update (no evidence)
    sdf >=  mu                  -> D <- avg(D, +1),       W += 1, n_free += 1   (in front of surface)
    -mu < sdf < mu              -> D <- avg(D, sdf / mu), W += 1, n_surf += 1   (near surface)
    sdf <= -mu                  -> no update (behind the surface: occluded)

D is stored normalized by the truncation distance mu, so D * mu is a metric distance.

Conventions: OpenCV camera axes, camera-to-world poses, pixel (u, v) covers
[u, u+1) x [v, v+1). Voxel (i, j, k) has its center at origin + (i + 0.5, j + 0.5, k + 0.5) * voxel_size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Voxel states.
UNKNOWN, FREE, OCCUPIED, BEHIND = 0, 1, 2, 3
STATE_NAMES = {UNKNOWN: "unknown", FREE: "free", OCCUPIED: "occupied", BEHIND: "behind surface"}


@dataclass
class DenseTSDF:
    origin: np.ndarray  # (3,) world position of the grid's minimum corner
    dims: tuple[int, int, int]  # number of voxels along x, y, z
    voxel_size: float
    trunc: float  # truncation distance mu in meters

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.dims = tuple(int(n) for n in self.dims)
        self.tsdf = np.ones(self.dims, dtype=np.float32)  # normalized by trunc; 1 = "far in front"
        self.weight = np.zeros(self.dims, dtype=np.float32)
        self.n_free = np.zeros(self.dims, dtype=np.uint16)
        self.n_surf = np.zeros(self.dims, dtype=np.uint16)

    # --- construction --------------------------------------------------------
    @classmethod
    def from_bounds(cls, lo, hi, voxel_size: float, trunc_voxels: float = 4.0) -> "DenseTSDF":
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        dims = np.ceil((hi - lo) / voxel_size).astype(int)
        return cls(lo, tuple(dims), voxel_size, trunc_voxels * voxel_size)

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
        h, w = depth.shape
        sl = self._frustum_block(K, cam_to_world, w, h, max_depth)
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
        valid = inside & (d > min_depth) & (d <= max_depth)
        sdf = d - z
        update = valid & (sdf > -self.trunc)  # sdf <= -mu: occluded, no update
        if not update.any():
            return

        tsdf_obs = np.minimum(sdf / self.trunc, 1.0).astype(np.float32)
        free_obs = sdf >= self.trunc

        # Flat indices into the full grid for the updated voxels of this block.
        local = np.nonzero(update)[0]
        bx, by, bz = (s.stop - s.start for s in sl)
        li, lj, lk = np.unravel_index(local, (bx, by, bz))
        gi, gj, gk = li + sl[0].start, lj + sl[1].start, lk + sl[2].start

        W_old = self.weight[gi, gj, gk]
        D_old = self.tsdf[gi, gj, gk]
        self.tsdf[gi, gj, gk] = (W_old * D_old + weight * tsdf_obs[local]) / (W_old + weight)
        self.weight[gi, gj, gk] = W_old + weight
        f = free_obs[local]
        self.n_free[gi[f], gj[f], gk[f]] += 1
        self.n_surf[gi[~f], gj[~f], gk[~f]] += 1

    def integrate_frames(self, frames: Sequence, min_depth: float = 0.1, max_depth: float = 4.0) -> None:
        for f in frames:
            self.integrate(f.depth, f.K_depth, f.cam_to_world, min_depth, max_depth)

    # --- interpretation ----------------------------------------------------------
    def classify(self, w_min: float = 3.0) -> np.ndarray:
        """uint8 state per voxel: UNKNOWN, FREE, OCCUPIED (surface inside voxel) or BEHIND.

        A voxel contains the surface if the (averaged) distance to it is at most half the voxel
        diagonal. BEHIND = observed, just behind a surface (inside the truncation band) - likely
        inside a solid object; whether to treat it as occupied is an open design question.
        """
        dist = self.tsdf * self.trunc
        half_diag = 0.5 * np.sqrt(3.0) * self.voxel_size
        known = self.weight >= w_min
        state = np.full(self.dims, UNKNOWN, dtype=np.uint8)
        state[known & (dist > half_diag)] = FREE
        state[known & (np.abs(dist) <= half_diag)] = OCCUPIED
        state[known & (dist < -half_diag)] = BEHIND
        return state

    def extract_mesh(self, w_min: float = 3.0):
        """Marching Cubes on the zero level set; voxels with weight < w_min are excluded."""
        import open3d as o3d
        from skimage.measure import marching_cubes

        verts_idx, faces, _, _ = marching_cubes(self.tsdf, level=0.0, allow_degenerate=False)

        # Keep only triangles whose vertices lie in cubes with all 8 corner voxels observed
        # (weight >= w_min). Unobserved voxels keep their initial +1, so next to the negative
        # band behind a surface they would create a fake "back side" surface (e.g. a second
        # floor below the real one). Note: skimage's `mask` argument does not enforce this.
        i0 = np.clip(np.floor(verts_idx).astype(int), 0, np.array(self.dims) - 2)
        wmin = np.full(len(verts_idx), np.inf, dtype=np.float32)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wmin = np.minimum(wmin, self.weight[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz])
        faces = faces[(wmin[faces] >= w_min).all(axis=1)]

        # marching_cubes returns coordinates in index space; index i is the voxel center.
        verts = self.origin + (verts_idx + 0.5) * self.voxel_size
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh


def count_states(state: np.ndarray) -> dict:
    return {STATE_NAMES[s]: int((state == s).sum()) for s in STATE_NAMES}


def pool_states(state: np.ndarray, factor: int, unknown_fraction: float = 0.5) -> np.ndarray:
    """Combine factor^3 fine voxels into one coarse voxel.

    Rule (one possible choice, to be discussed):
      occupied  if any fine voxel is OCCUPIED or BEHIND
      unknown   else, if more than `unknown_fraction` of the fine voxels are UNKNOWN
      free      otherwise
    The grid is padded with UNKNOWN to a multiple of `factor`.
    """
    if factor == 1:
        return state.copy()
    pad = [(0, (-n) % factor) for n in state.shape]
    s = np.pad(state, pad, constant_values=UNKNOWN)
    nx, ny, nz = (n // factor for n in s.shape)
    blocks = s.reshape(nx, factor, ny, factor, nz, factor).transpose(0, 2, 4, 1, 3, 5).reshape(nx, ny, nz, -1)
    occ = ((blocks == OCCUPIED) | (blocks == BEHIND)).any(-1)
    unk = (blocks == UNKNOWN).mean(-1) > unknown_fraction
    out = np.full((nx, ny, nz), FREE, dtype=np.uint8)
    out[unk] = UNKNOWN
    out[occ] = OCCUPIED
    return out


def bounds_from_points(points: np.ndarray, margin: float = 0.1, percentile: float = 0.5):
    """Robust axis-aligned bounds of a point cloud (ignores the most extreme outliers)."""
    lo = np.percentile(points, percentile, axis=0) - margin
    hi = np.percentile(points, 100 - percentile, axis=0) + margin
    return lo, hi
