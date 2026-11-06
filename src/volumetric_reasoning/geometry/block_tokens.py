"""Block-level scene tokens from a fine (e.g. 2 cm) occupied/free/unknown voxel grid.

  * the fine grid is grouped into blocks of B^3 voxels (B=8 at 2 cm -> 16 cm blocks);
  * block label: SURFACE (=OCCUPIED) if any occupied voxel, FREE if all free, UNKNOWN if all
    unknown, otherwise (free + unknown, no surface) the majority of free vs. unknown;
  * one surface token per SURFACE block, positioned at the mean of its occupied voxel centers;
  * FREE blocks and (a chosen scope of) UNKNOWN blocks are merged separately into axis-aligned
    boxes with an optional size cap; one token per box (center, size, type).
Only our own observations are used (no ground truth).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy import ndimage

from .labels import FREE, OCCUPIED, UNKNOWN

UNKNOWN_SCOPES = ("all", "room", "frontier")


# --- block labels -----------------------------------------------------------------
def block_labels(state: np.ndarray, B: int) -> tuple[np.ndarray, np.ndarray]:
    """Token label per block and a mask of blocks that were resolved from MIXED.

    The grid is padded with UNKNOWN at its upper end to a whole number of blocks.
    """
    state = np.pad(state, [(0, -s % B) for s in state.shape], constant_values=UNKNOWN)
    nx, ny, nz = (s // B for s in state.shape)
    r = state.reshape(nx, B, ny, B, nz, B)
    occ = (r == OCCUPIED).any(axis=(1, 3, 5))
    nf = (r == FREE).sum(axis=(1, 3, 5))
    nu = (r == UNKNOWN).sum(axis=(1, 3, 5))
    lab = np.where(nf >= nu, FREE, UNKNOWN).astype(np.uint8)
    lab[occ] = OCCUPIED
    return lab, ~occ & (nf > 0) & (nu > 0)


def surface_tokens(state: np.ndarray, B: int, origin: np.ndarray, h: float):
    """(block index (N,3), position (N,3) = mean occupied voxel center, voxel count (N,))."""
    vox = np.argwhere(state == OCCUPIED)
    keys, inverse, counts = np.unique(vox // B, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    centers = origin + (vox + 0.5) * h
    pos = np.stack([np.bincount(inverse, centers[:, k]) for k in range(3)], axis=1) / counts[:, None]
    return keys, pos, counts


def unknown_scope(lab: np.ndarray, scope: str) -> np.ndarray:
    """Which UNKNOWN blocks become tokens.

    all:      every unknown block in the grid (incl. behind walls, above ceiling, below floor)
    room:     unknown blocks inside the room, estimated from OUR free space: within its floor plan
              (holes filled) and between its lowest and highest block layer
    frontier: unknown blocks touching free space (26-neighbourhood)
    """
    unknown, free = lab == UNKNOWN, lab == FREE
    if scope == "all":
        return unknown
    if scope == "frontier":
        return unknown & ndimage.binary_dilation(free, np.ones((3, 3, 3), bool))
    if scope == "room":
        plan = ndimage.binary_fill_holes(ndimage.binary_closing(free.any(axis=2), np.ones((3, 3), bool)))
        layers = np.flatnonzero(free.any(axis=(0, 1)))
        region = np.zeros(lab.shape, bool)
        if len(layers):
            region[:, :, layers[0]:layers[-1] + 1] = plan[:, :, None]
        return unknown & region
    raise ValueError(f"Unknown scope {scope!r}; expected one of {UNKNOWN_SCOPES}")


# --- box merging ------------------------------------------------------------------
def partition_mask(mask, growth_order=(0, 1, 2), max_extent=None):
    """Greedily grow disjoint boxes over a boolean mask; returns x/y/z (start, stop), half-open.

    From the first remaining cell, extend along growth_order[0], then extend that run along
    growth_order[1], then that rectangle along growth_order[2]; every extension needs the whole
    added slab to be available. max_extent: optional per-axis (x, y, z) cap on box size in cells.
    """
    mask = np.asarray(mask, bool)
    if mask.ndim != 3 or not all(mask.shape) or tuple(sorted(growth_order)) != (0, 1, 2):
        raise ValueError("Expected a nonempty 3-D mask and a permutation of (0, 1, 2)")
    limit = np.array(mask.shape, np.int64) if max_extent is None else np.asarray(max_extent, np.int64)
    mx, my, mz = (int(limit[a]) for a in growth_order)
    scan_axes = tuple(reversed(growth_order))
    available = np.ascontiguousarray(mask.transpose(scan_axes)).copy()
    nz, ny, nx = available.shape
    starts, stops = [], []
    for z in range(nz):
        for y in range(ny):
            for x in np.flatnonzero(available[z, y]):
                if not available[z, y, x]:
                    continue
                blocked = np.flatnonzero(~available[z, y, x:])
                x1 = min(int(x + blocked[0]) if len(blocked) else nx, x + mx)
                y1 = y + 1
                while y1 < ny and y1 - y < my and available[z, y1, x:x1].all():
                    y1 += 1
                z1 = z + 1
                while z1 < nz and z1 - z < mz and available[z1, y:y1, x:x1].all():
                    z1 += 1
                available[z:z1, y:y1, x:x1] = False
                starts.append((z, y, x))
                stops.append((z1, y1, x1))
    inverse = np.argsort(scan_axes)
    return (np.asarray(starts, dtype=np.int64).reshape(-1, 3)[:, inverse],
            np.asarray(stops, dtype=np.int64).reshape(-1, 3)[:, inverse])


def merge_boxes(mask: np.ndarray, max_cells: int | None = None):
    """Box partition of a boolean block mask, best (fewest boxes) of the 6 growth orders."""
    if not mask.any():
        return np.zeros((0, 3), np.int64), np.zeros((0, 3), np.int64)
    limit = None if max_cells is None else (max_cells,) * 3
    best = None
    for order in permutations(range(3)):
        start, stop = partition_mask(mask, order, limit)
        if best is None or len(start) < len(best[0]):
            best = (start, stop)
    return best


# --- tokens -----------------------------------------------------------------------
@dataclass
class SceneTokens:
    block_size_m: float
    surface_block: np.ndarray      # (Ns, 3) block indices
    surface_position: np.ndarray   # (Ns, 3) meters
    box_start: np.ndarray          # (Nb, 3) block indices, half-open
    box_stop: np.ndarray
    box_type: np.ndarray           # (Nb,) FREE or UNKNOWN
    origin: np.ndarray

    @property
    def box_center(self):
        return self.origin + (self.box_start + self.box_stop) / 2 * self.block_size_m

    @property
    def box_size(self):
        return (self.box_stop - self.box_start) * self.block_size_m

    def counts(self) -> dict:
        s = len(self.surface_position)
        f = int((self.box_type == FREE).sum())
        u = int((self.box_type == UNKNOWN).sum())
        return {"surface": s, "free": f, "unknown": u, "surface_only": s, "surface_free": s + f,
                "surface_unknown": s + u, "surface_both": s + f + u}


def tokenize(state, B, origin, h, cap_m: float | None = 1.5, unknown: str = "room") -> tuple[SceneTokens, np.ndarray]:
    """Scene tokens for one block size, box size cap (meters, None = no cap) and unknown scope."""
    ext = B * h
    lab, _ = block_labels(state, B)
    keys, pos, _ = surface_tokens(state, B, origin, h)
    max_cells = None if cap_m is None else max(1, int(np.floor(cap_m / ext + 1e-9)))
    fs, fe = merge_boxes(lab == FREE, max_cells)
    us, ue = merge_boxes(unknown_scope(lab, unknown), max_cells)
    types = np.concatenate([np.full(len(fs), FREE, np.uint8), np.full(len(us), UNKNOWN, np.uint8)])
    tokens = SceneTokens(ext, keys, pos, np.concatenate([fs, us]), np.concatenate([fe, ue]), types,
                         np.asarray(origin, float))
    return tokens, lab
