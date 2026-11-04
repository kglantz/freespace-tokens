"""Direct depth-ray evidence, independent of the fused projective TSDF.

A hit marks the cell containing a measured endpoint. Free evidence marks cells
crossed for a positive distance before the hit, with an optional metric guard.
This is a finite-resolution observation model, not a proof that every point in a
cell is empty. Missing rays, occluded regions and the negative TSDF band provide
no free-space evidence. Each cell is counted at most once per frame per type.
"""

from __future__ import annotations

import numpy as np


def ray_voxel_evidence(camera, endpoints, origin, dims, voxel_size, free_margin=0.02,
                       chunk_size=8192):
    """Return unique flat indices of ray-traversed cells and endpoint cells.

    Traversal uses a batched 3-D DDA. Simultaneous boundary crossings advance all
    tied axes; touching only a corner/edge does not carve neighboring cells.
    The endpoint's own cell is never carved by that ray, even with zero guard.
    Cameras and hits may lie outside the grid. The ROI is half-open on each axis.
    """
    dims = np.asarray(dims, dtype=np.int64)
    origin = np.asarray(origin, dtype=np.float64)
    camera = np.asarray(camera, dtype=np.float64)
    endpoints = np.asarray(endpoints, dtype=np.float64).reshape(-1, 3)
    if (dims.shape != (3,) or np.any(dims <= 0) or not np.isfinite(voxel_size)
            or voxel_size <= 0 or not np.isfinite(free_margin) or free_margin < 0
            or chunk_size < 1 or origin.shape != (3,) or camera.shape != (3,)
            or not np.isfinite(origin).all() or not np.isfinite(camera).all()
            or not np.isfinite(endpoints).all()):
        raise ValueError("Invalid grid, ray coordinates, chunk size or free-space margin")
    o = (camera - origin) / voxel_size
    targets = (endpoints - origin) / voxel_size
    hit_cells = np.floor(targets).astype(np.int64)
    hit_inside = ((hit_cells >= 0) & (hit_cells < dims)).all(axis=1)
    hits = np.unique(np.ravel_multi_index(hit_cells[hit_inside].T, tuple(dims)))
    free_chunks = []
    for start in range(0, len(targets), chunk_size):
        target = targets[start:start + chunk_size]
        direction = target - o
        lengths = np.linalg.norm(direction, axis=1) * voxel_size
        stop = np.maximum(0.0, 1.0 - free_margin / np.maximum(lengths, 1e-30))
        nonzero = direction != 0
        low_t = np.full_like(direction, -np.inf)
        high_t = np.full_like(direction, np.inf)
        np.divide(-o, direction, out=low_t, where=nonzero)
        np.divide(dims - o, direction, out=high_t, where=nonzero)
        enter = np.maximum(0.0, np.minimum(low_t, high_t).max(axis=1))
        leave = np.minimum(stop, np.maximum(low_t, high_t).min(axis=1))
        parallel_outside = (~nonzero & ((o < 0) | (o >= dims))).any(axis=1)
        valid = (leave > enter + 1e-12) & ~parallel_outside & (lengths > 0)
        if not valid.any():
            continue
        direction, enter, leave = direction[valid], enter[valid], leave[valid]
        end_cell = np.floor(target[valid]).astype(np.int64)
        pos = o + enter[:, None] * direction
        cell = np.floor(pos).astype(np.int64)
        # At a boundary, the cell entered in the negative direction is the lower one.
        on_boundary = np.abs(pos - np.rint(pos)) < 1e-9
        cell = np.where(on_boundary, np.rint(pos).astype(np.int64), cell)
        cell -= on_boundary & (direction < 0)
        step = np.sign(direction).astype(np.int64)
        boundary = cell + (step > 0)
        t_next = np.full_like(direction, np.inf)
        delta = np.full_like(direction, np.inf)
        np.divide(boundary - o, direction, out=t_next, where=direction != 0)
        np.divide(1.0, np.abs(direction), out=delta, where=direction != 0)
        current = enter.copy()
        active = np.ones(len(cell), dtype=bool)
        visited = []
        # A clipped straight segment crosses at most nx + ny + nz grid planes.
        for _ in range(int(dims.sum()) + 3):
            next_crossing = t_next.min(axis=1)
            inside = ((cell >= 0) & (cell < dims)).all(axis=1)
            use = (active & inside & (np.minimum(next_crossing, leave) > current + 1e-12)
                   & (cell != end_cell).any(axis=1))
            if use.any():
                visited.append(np.ravel_multi_index(cell[use].T, tuple(dims)))
            active &= inside & (next_crossing < leave - 1e-12)
            if not active.any():
                break
            advance = active[:, None] & (t_next <= next_crossing[:, None] + 1e-12)
            cell += advance * step
            # Avoid 0 * inf for axis-parallel rays.
            t_next[advance] += delta[advance]
            current[active] = next_crossing[active]
        else:
            raise RuntimeError("Ray traversal exceeded the grid crossing bound")
        if visited:
            free_chunks.append(np.unique(np.concatenate(visited)))
    free = np.unique(np.concatenate(free_chunks)) if free_chunks else np.empty(0, np.int64)
    return free, hits
