"""Hashed voxel-block map (VoxelHashing layout) with explicit free/unknown states.

Space is split into blocks of B^3 voxels, stored sparsely in a dict keyed by the
integer block coordinate (Niessner et al. 2013). Unlike plain VoxelHashing, the
map keeps three states instead of only the surface band:

* unknown  - default. A block that was never touched is not stored at all.
* free     - voxels traversed by a depth ray before its measured endpoint (3-D DDA,
             see visibility.py). A block whose voxels are all free and that carries
             no TSDF is collapsed to a single flag (no voxel arrays).
* occupied - voxels intersecting the observed TSDF zero surface (zero_surface.py).
             TSDF is only stored in blocks near surfaces.
Label priority: occupied > free > unknown.

Fusion is two-pass: pass 1 carves free space and allocates the TSDF band from all
frames, pass 2 integrates every frame into all allocated TSDF blocks. Conventions:
OpenCV camera axes, camera-to-world poses, projective TSDF (d - z) / trunc sampled at
voxel centers, half-open voxels with centers at origin + (index + 0.5) * voxel_size.

Block categories (for counting / visualisation):
    OCCUPIED  any voxel occupied
    FREE      all voxels free
    UNKNOWN   all voxels unknown (incl. never-stored blocks inside the domain)
    MIXED     no surface, but both free and unknown voxels
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .backprojection import depth_to_points
from .labels import FREE, OCCUPIED, UNKNOWN
from .visibility import ray_voxel_evidence
from .zero_surface import zero_surface_mask

MIXED = 3
CATEGORY_NAMES = {OCCUPIED: "occupied", FREE: "free", UNKNOWN: "unknown", MIXED: "mixed"}


def validate_observation(depth, K, cam_to_world, min_depth, max_depth):
    """Check the pinhole / rigid-pose assumptions used during fusion."""
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


class Block:
    """One B^3 block. `collapsed` blocks are uniformly free and hold no arrays."""

    __slots__ = ("n_free", "tsdf", "weight", "collapsed")

    def __init__(self, B: int):
        self.n_free = np.zeros((B, B, B), np.uint16)
        self.tsdf = None
        self.weight = None
        self.collapsed = False

    def expand(self, B: int) -> None:
        """Undo collapse; free counts become 1 (only `> 0` is ever used)."""
        if self.collapsed:
            self.n_free = np.ones((B, B, B), np.uint16)
            self.collapsed = False

    def allocate_tsdf(self, B: int) -> None:
        self.expand(B)
        if self.tsdf is None:
            self.tsdf = np.ones((B, B, B), np.float32)
            self.weight = np.zeros((B, B, B), np.float32)


class BlockMap:
    def __init__(self, origin, shape_blocks, voxel_size: float = 0.02, block_size: int = 8,
                 trunc: float = 0.08, free_margin: float = 0.02):
        origin = np.asarray(origin, np.float64)
        shape_blocks = tuple(int(n) for n in shape_blocks)
        if (origin.shape != (3,) or not np.isfinite(origin).all() or len(shape_blocks) != 3
                or min(shape_blocks) <= 0 or not voxel_size > 0 or int(block_size) < 1
                or not trunc > 0 or not free_margin >= 0):
            raise ValueError("Invalid block-map geometry")
        self.origin = origin
        self.shape_blocks = shape_blocks
        self.h = float(voxel_size)
        self.B = int(block_size)
        self.trunc = float(trunc)
        self.free_margin = float(free_margin)
        self.blocks: dict[tuple[int, int, int], Block] = {}
        # Local voxel indices of one block, row-major (matches array.ravel()).
        self._local = np.indices((self.B,) * 3).reshape(3, -1).T

    # --- geometry -------------------------------------------------------------
    @classmethod
    def from_bounds(cls, lo, hi, voxel_size=0.02, block_size=8, margin=0.1, **kw) -> "BlockMap":
        """Domain = [lo - margin, hi + margin], snapped outward to world-aligned blocks."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if lo.shape != (3,) or hi.shape != (3,) or np.any(hi < lo) or not np.isfinite([lo, hi]).all():
            raise ValueError("Bounds must be finite with hi >= lo")
        ext = voxel_size * block_size
        b0 = np.floor((lo - margin) / ext).astype(np.int64)
        b1 = np.ceil((hi + margin) / ext).astype(np.int64)
        return cls(b0 * ext, np.maximum(b1 - b0, 1), voxel_size, block_size, **kw)

    @classmethod
    def from_frames(cls, frames, voxel_size=0.02, block_size=8, trunc=0.08, free_margin=0.02,
                    min_depth=0.1, max_depth=4.0, verbose=False) -> "BlockMap":
        """Map over the frames' depth points (0.1-99.9 percentiles, robust to flying pixels), fused."""
        pts = np.concatenate([depth_to_points(f.depth, f.K_depth, f.cam_to_world, min_depth, max_depth)[0]
                              for f in frames])
        bm = cls.from_bounds(np.percentile(pts, 0.1, axis=0), np.percentile(pts, 99.9, axis=0), voxel_size,
                             block_size, margin=0.1, trunc=trunc, free_margin=free_margin)
        bm.fuse(frames, min_depth, max_depth, verbose=verbose)
        return bm

    @property
    def block_extent(self) -> float:
        return self.h * self.B

    @property
    def dims_voxels(self) -> tuple[int, int, int]:
        return tuple(int(n) * self.B for n in self.shape_blocks)

    @property
    def num_blocks_total(self) -> int:
        return int(np.prod(self.shape_blocks))

    def _block(self, key) -> Block:
        blk = self.blocks.get(key)
        if blk is None:
            blk = self.blocks[key] = Block(self.B)
        return blk

    def _voxel_centers(self, keys: np.ndarray) -> np.ndarray:
        """(n, B^3, 3) world-space voxel centers of the given blocks."""
        g = keys[:, None, :] * self.B + self._local[None]
        return self.origin + (g + 0.5) * self.h

    # --- fusion ---------------------------------------------------------------
    def fuse(self, frames: Sequence, min_depth: float = 0.1, max_depth: float = 4.0,
             verbose: bool = False) -> None:
        """Two-pass fusion of frames with .depth, .K_depth, .cam_to_world."""
        for n, f in enumerate(frames, 1):
            self.carve_and_allocate(f.depth, f.K_depth, f.cam_to_world, min_depth, max_depth)
            if verbose and (n % 8 == 0 or n == len(frames)):
                print(f"  pass 1 (free space + allocation): {n}/{len(frames)}", flush=True)
        for n, f in enumerate(frames, 1):
            self.integrate_tsdf(f.depth, f.K_depth, f.cam_to_world, min_depth, max_depth)
            if verbose and (n % 8 == 0 or n == len(frames)):
                print(f"  pass 2 (TSDF): {n}/{len(frames)}", flush=True)

    def carve_and_allocate(self, depth, K, cam_to_world, min_depth=0.1, max_depth=4.0) -> None:
        depth, K, T = validate_observation(depth, K, cam_to_world, min_depth, max_depth)
        endpoints, mask = depth_to_points(depth, K, T, min_depth, max_depth)
        if not len(endpoints):
            return
        cam = T[:3, 3]
        free, _ = ray_voxel_evidence(cam, endpoints, self.origin, self.dims_voxels,
                                     self.h, self.free_margin)
        touched = self._add_free(free)
        z = depth[mask].astype(np.float64)
        for key in self._band_blocks(cam, endpoints, z, min(K[0, 0], K[1, 1])):
            self._block(key).allocate_tsdf(self.B)
        for key in touched:  # collapse uniformly free blocks without TSDF
            blk = self.blocks[key]
            if not blk.collapsed and blk.tsdf is None and blk.n_free.all():
                blk.n_free, blk.collapsed = None, True

    def _add_free(self, flat: np.ndarray) -> list:
        if not len(flat):
            return []
        ijk = np.stack(np.unravel_index(flat, self.dims_voxels), axis=1)
        bk, loc = ijk // self.B, ijk % self.B
        bid = np.ravel_multi_index(bk.T, self.shape_blocks)
        order = np.argsort(bid, kind="stable")
        bid, loc = bid[order], loc[order]
        ub, start = np.unique(bid, return_index=True)
        stop = np.r_[start[1:], len(bid)]
        touched = []
        for b, s, e in zip(ub, start, stop):
            key = tuple(int(v) for v in np.unravel_index(b, self.shape_blocks))
            blk = self._block(key)
            touched.append(key)
            if blk.collapsed:
                continue
            l = loc[s:e]  # unique within a frame
            blk.n_free[l[:, 0], l[:, 1], l[:, 2]] += 1
        return touched

    def _band_blocks(self, cam, endpoints, z, focal, chunk=4096) -> Iterable[tuple]:
        """Blocks that may hold a voxel with |projective SDF| < trunc (or its neighbours).

        Samples each ray over +-(1.5 trunc + 2h) around the endpoint (covers the
        z-to-ray-length factor of the FOV) and pads laterally by 2h + one pixel
        footprint, so every TSDF interpolation corner that dense fusion would mark
        observed near a sign change lies in an allocated block.
        """
        ext = self.block_extent
        band = 1.5 * self.trunc + 2 * self.h
        steps = np.arange(-band, band + self.h / 2, self.h)
        direction = endpoints - cam
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radius = 2 * self.h + (z + band) / focal
        span = int(np.ceil(2 * radius.max() / ext)) + 1
        offs = np.indices((span + 1,) * 3).reshape(3, -1).T
        shape = np.array(self.shape_blocks)
        keys = set()
        for s in range(0, len(endpoints), chunk):
            p = (endpoints[s:s + chunk, None, :] + direction[s:s + chunk, None, :] * steps[None, :, None])
            r = np.broadcast_to(radius[s:s + chunk, None, None], p.shape)
            lo = np.floor((p - r - self.origin) / ext).astype(np.int64).reshape(-1, 3)
            hi = np.floor((p + r - self.origin) / ext).astype(np.int64).reshape(-1, 3)
            # Deduplicate whole boxes (lo AND hi); boxes sharing lo may differ in hi.
            box = np.unique(np.concatenate([lo, hi], axis=1), axis=0)
            lo, hi = box[:, :3], box[:, 3:]
            for o in offs:
                k = lo + o
                ok = (k <= hi).all(1) & (k >= 0).all(1) & (k < shape).all(1)
                keys.update(map(tuple, k[ok].tolist()))
        return keys

    def integrate_tsdf(self, depth, K, cam_to_world, min_depth=0.1, max_depth=4.0,
                       weight: float = 1.0, chunk: int = 2048) -> None:
        depth, K, T = validate_observation(depth, K, cam_to_world, min_depth, max_depth)
        keys = [k for k, b in self.blocks.items() if b.tsdf is not None]
        if not keys:
            return
        kb = np.array(keys, np.int64)
        R, t = T[:3, :3], T[:3, 3]
        h_img, w_img = depth.shape
        # Conservative frustum culling with block bounding spheres.
        ext = self.block_extent
        rad = np.sqrt(3) / 2 * ext
        pc = (self.origin + (kb + 0.5) * ext - t) @ R
        x, y, zc = pc[:, 0], pc[:, 1], pc[:, 2]
        keep = (zc > -rad) & (zc < max_depth + self.trunc + rad)
        far = zc > rad
        for coord, f, c, size in ((x, K[0, 0], K[0, 2], w_img), (y, K[1, 1], K[1, 2], h_img)):
            with np.errstate(divide="ignore", invalid="ignore"):
                lo = np.minimum((coord - rad) / (zc - rad), (coord - rad) / (zc + rad)) * f + c
                hi = np.maximum((coord + rad) / (zc - rad), (coord + rad) / (zc + rad)) * f + c
            keep &= ~far | ((hi >= 0) & (lo <= size))
        kb = kb[keep]
        for s in range(0, len(kb), chunk):
            kc = kb[s:s + chunk]
            p = (self._voxel_centers(kc).reshape(-1, 3) - t) @ R
            z = p[:, 2]
            in_front = z > 1e-6
            u = np.full(z.shape, -1.0)
            v = np.full(z.shape, -1.0)
            u[in_front] = K[0, 0] * p[in_front, 0] / z[in_front] + K[0, 2]
            v[in_front] = K[1, 1] * p[in_front, 1] / z[in_front] + K[1, 2]
            ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
            inside = in_front & (ui >= 0) & (ui < w_img) & (vi >= 0) & (vi < h_img)
            d = np.zeros(z.shape, dtype=np.float32)
            d[inside] = depth[vi[inside], ui[inside]]
            valid = inside & np.isfinite(d) & (d > min_depth) & (d <= max_depth)
            sdf = d - z
            update = (valid & (sdf > -self.trunc)).reshape(len(kc), -1)
            if not update.any():
                continue
            obs = np.minimum(sdf / self.trunc, 1.0).astype(np.float32).reshape(len(kc), -1)
            for i in np.flatnonzero(update.any(1)):
                blk = self.blocks[tuple(int(v) for v in kc[i])]
                m = update[i]
                F, W = blk.tsdf.reshape(-1), blk.weight.reshape(-1)
                W_old, D_old = W[m], F[m]
                F[m] = (W_old * D_old + weight * obs[i, m]) / (W_old + weight)
                W[m] = W_old + weight

    # --- interpretation -------------------------------------------------------
    def _padded(self, key, attr: str, fill: float) -> np.ndarray:
        """(B+2)^3 array with a one-voxel halo from neighbouring blocks."""
        B = self.B
        out = np.full((B + 2,) * 3, fill, np.float32)
        parts = {-1: (slice(0, 1), slice(B - 1, B)), 0: (slice(1, B + 1), slice(0, B)),
                 1: (slice(B + 1, B + 2), slice(0, 1))}
        for d in np.ndindex(3, 3, 3):
            d = tuple(int(v) - 1 for v in d)
            nb = self.blocks.get((key[0] + d[0], key[1] + d[1], key[2] + d[2]))
            if nb is None or getattr(nb, attr) is None:
                continue
            dst = tuple(parts[a][0] for a in d)
            src = tuple(parts[a][1] for a in d)
            out[dst] = getattr(nb, attr)[src]
        return out

    def voxel_states(self, chunk: int = 512) -> dict:
        """Per-voxel states for every stored, non-collapsed block: key -> (B,B,B) uint8."""
        B = self.B
        states = {}
        for key, blk in self.blocks.items():
            if blk.collapsed:
                continue
            s = np.full((B, B, B), UNKNOWN, np.uint8)
            s[blk.n_free > 0] = FREE
            states[key] = s
        # Occupied cells need W>0 at their own center, so only TSDF blocks qualify.
        # Padded blocks are concatenated along x; crossings between two halos only
        # mark halo cells, which are cropped, so batching is exact.
        tkeys = [k for k, b in self.blocks.items() if b.tsdf is not None]
        P = B + 2
        for s in range(0, len(tkeys), chunk):
            ks = tkeys[s:s + chunk]
            F = np.concatenate([self._padded(k, "tsdf", 1.0) for k in ks], axis=0)
            W = np.concatenate([self._padded(k, "weight", 0.0) for k in ks], axis=0)
            occ = zero_surface_mask(F, W)
            for i, k in enumerate(ks):
                o = occ[i * P + 1:i * P + 1 + B, 1:-1, 1:-1]
                states[k][o] = OCCUPIED
        return states

    def classify(self):
        """Return (categories: key -> category, voxel_states) for stored blocks."""
        states = self.voxel_states()
        cats = {}
        for key, blk in self.blocks.items():
            if blk.collapsed:
                cats[key] = FREE
                continue
            s = states[key]
            if (s == OCCUPIED).any():
                cats[key] = OCCUPIED
            elif (s == FREE).all():
                cats[key] = FREE
            elif (s == UNKNOWN).all():
                cats[key] = UNKNOWN
            else:
                cats[key] = MIXED
        return cats, states

    def summary(self, cats: dict, states: dict) -> dict:
        nb = {name: 0 for name in CATEGORY_NAMES.values()}
        for c in cats.values():
            nb[CATEGORY_NAMES[c]] += 1
        stored = len(cats)
        nb["unknown"] += self.num_blocks_total - stored  # never-touched blocks
        bv = self.B ** 3
        n_collapsed = sum(b.collapsed for b in self.blocks.values())
        occ = sum(int((s == OCCUPIED).sum()) for s in states.values())
        free = sum(int((s == FREE).sum()) for s in states.values()) + n_collapsed * bv
        total = self.num_blocks_total * bv
        n_tsdf = sum(b.tsdf is not None for b in self.blocks.values())
        return {
            "domain": {"origin_m": self.origin.tolist(),
                       "upper_m": (self.origin + np.array(self.shape_blocks) * self.block_extent).tolist(),
                       "shape_blocks": list(self.shape_blocks), "voxel_size_m": self.h,
                       "block_size_voxels": self.B, "block_extent_m": self.block_extent,
                       "truncation_m": self.trunc, "free_margin_m": self.free_margin},
            "block_counts": {**nb, "total": self.num_blocks_total},
            "voxel_counts": {"occupied": occ, "free": free, "unknown": total - occ - free, "total": total},
            "storage": {"stored_blocks": stored, "never_touched_blocks": self.num_blocks_total - stored,
                        "collapsed_free_blocks": n_collapsed, "tsdf_blocks": n_tsdf,
                        "approx_bytes": n_tsdf * bv * 8 + (stored - n_collapsed) * bv * 2},
        }

    def to_dense_states(self) -> np.ndarray:
        """Dense (x, y, z) uint8 label grid over the whole domain."""
        out = np.full(self.dims_voxels, UNKNOWN, np.uint8)
        states = self.voxel_states()
        B = self.B
        for key, blk in self.blocks.items():
            sl = tuple(slice(k * B, (k + 1) * B) for k in key)
            out[sl] = FREE if blk.collapsed else states[key]
        return out

    def extract_mesh(self, min_weight: float = 1.0):
        """Marching Cubes mesh (open3d) of the TSDF zero level, for visualization/evaluation.

        Triangles touching a voxel with integration weight < min_weight are dropped, which
        also removes artificial surfaces at unobserved samples (TSDF initialized to +1).
        """
        import open3d as o3d
        from skimage.measure import marching_cubes
        tsdf = np.ones(self.dims_voxels, dtype=np.float32)
        weight = np.zeros_like(tsdf)
        B = self.B
        for key, blk in self.blocks.items():
            if blk.tsdf is not None:
                region = tuple(slice(k * B, (k + 1) * B) for k in key)
                tsdf[region], weight[region] = blk.tsdf, blk.weight
        if min(tsdf.shape) < 2 or not tsdf.min() < 0 < tsdf.max():
            raise ValueError("The block map has no TSDF sign change to mesh")
        verts, faces, _, _ = marching_cubes(tsdf, level=0.0, allow_degenerate=False)
        corner = np.clip(np.floor(verts).astype(int), 0, np.array(tsdf.shape) - 2)
        minimum = np.full(len(verts), np.inf, dtype=np.float32)
        for offset in np.ndindex(2, 2, 2):
            np.minimum(minimum, weight[tuple((corner + offset).T)], out=minimum)
        keep = minimum[faces].min(axis=1) >= min_weight
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.origin + (verts + 0.5) * self.h),
                                         o3d.utility.Vector3iVector(faces[keep]))
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh

    # --- persistence ----------------------------------------------------------
    def save(self, path, **extra) -> None:
        keys = list(self.blocks)
        blks = [self.blocks[k] for k in keys]
        nf = [b for b in blks if not b.collapsed]
        tb = [b for b in blks if b.tsdf is not None]
        B = self.B
        np.savez_compressed(
            path, origin=self.origin, shape_blocks=np.array(self.shape_blocks), voxel_size=self.h,
            block_size=B, trunc=self.trunc, free_margin=self.free_margin,
            keys=np.array(keys, np.int32).reshape(-1, 3),
            collapsed=np.array([b.collapsed for b in blks], bool),
            has_tsdf=np.array([b.tsdf is not None for b in blks], bool),
            n_free=np.stack([b.n_free for b in nf]) if nf else np.zeros((0, B, B, B), np.uint16),
            tsdf=np.stack([b.tsdf for b in tb]) if tb else np.zeros((0, B, B, B), np.float32),
            weight=np.stack([b.weight for b in tb]) if tb else np.zeros((0, B, B, B), np.float32),
            **extra)

    @classmethod
    def load(cls, path) -> "BlockMap":
        with np.load(path, allow_pickle=False) as a:
            m = cls(a["origin"], a["shape_blocks"], float(a["voxel_size"]), int(a["block_size"]),
                    float(a["trunc"]), float(a["free_margin"]))
            n_free, tsdf, weight = a["n_free"], a["tsdf"], a["weight"]  # decompress once
            i_nf = i_t = 0
            for key, col, ht in zip(a["keys"], a["collapsed"], a["has_tsdf"]):
                blk = Block(m.B)
                if col:
                    blk.n_free, blk.collapsed = None, True
                else:
                    blk.n_free = n_free[i_nf].copy()
                    i_nf += 1
                if ht:
                    blk.tsdf, blk.weight = tsdf[i_t].copy(), weight[i_t].copy()
                    i_t += 1
                m.blocks[tuple(int(v) for v in key)] = blk
        return m
