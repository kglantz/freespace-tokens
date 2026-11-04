"""Independent geometric audit of frame-level ray evidence in a state map."""
from itertools import product

import numpy as np


def audit_ray_evidence(grid, states, frames, min_depth, max_depth, sample_count=256, seed=42):
    """Check stratified cells against original depths with ray/AABB intersections.

    Does not call the production backprojection or DDA. Projects each cell's
    corners only to prune candidate pixels, then explicitly intersects each
    measured ray segment with that cell. Counts at most one hit/traversal per
    frame, matching the specified observation semantics. This audits geometric
    computation, not the physical accuracy of the depth sensor.
    """
    if sample_count < 1:
        raise ValueError('sample_count must be positive')
    rng = np.random.default_rng(seed)
    masks = [states == code for code in (0, 1, 2)]
    masks.append((grid.n_free > 0) & (grid.n_hit > 0))
    chosen = []
    for mask in masks:
        candidates = np.flatnonzero(mask)
        chosen.extend(rng.choice(candidates, min(len(candidates), (sample_count + 3) // 4), replace=False).tolist())
    chosen = np.unique(chosen)[:sample_count]
    cells = np.array(np.unravel_index(chosen, grid.dims)).T
    lows = grid.origin + cells * grid.voxel_size
    highs = lows + grid.voxel_size
    corners = np.array(list(product((0, 1), repeat=3)))
    free_counts = np.zeros(len(cells), np.uint32)
    hit_counts = np.zeros(len(cells), np.uint32)
    for frame in frames:
        T, K, depth = frame.cam_to_world, frame.K_depth, frame.depth
        camera = T[:3, 3]
        invK = np.linalg.inv(K)
        for i, (lo, hi) in enumerate(zip(lows, highs)):
            world_corners = lo + corners * grid.voxel_size
            camera_corners = (world_corners - camera) @ T[:3, :3]
            if (camera_corners[:, 2] <= 0).all():
                continue
            if (camera_corners[:, 2] > 0).all():
                homogeneous = camera_corners @ K.T
                uv = homogeneous[:, :2] / homogeneous[:, 2, None]
                a = np.maximum(np.floor(uv.min(0) - .5).astype(int), 0)
                b = np.minimum(np.ceil(uv.max(0) - .5).astype(int) + 1, depth.shape[::-1])
                if (b <= a).any():
                    continue
            else:
                # A cell crossing the camera plane cannot use finite projected bounds.
                a, b = np.zeros(2, int), np.array(depth.shape[::-1])
            u, v = np.meshgrid(np.arange(a[0], b[0]), np.arange(a[1], b[1]))
            d = depth[v, u]
            valid = np.isfinite(d) & (d > min_depth) & (d <= max_depth)
            if not valid.any():
                continue
            rays = np.stack([u[valid] + .5, v[valid] + .5, np.ones(valid.sum())], axis=1)
            direction = ((rays @ invK.T) * d[valid, None]) @ T[:3, :3].T
            endpoints = camera + direction
            hit = ((endpoints >= lo) & (endpoints < hi)).all(axis=1)
            hit_counts[i] += bool(hit.any())
            stop = np.maximum(0, 1 - grid.free_margin / np.maximum(np.linalg.norm(direction, axis=1), 1e-30))
            entry, leave = np.zeros(len(direction)), stop.copy()
            for axis in range(3):
                parallel = direction[:, axis] == 0
                first, second = np.full(len(direction), -np.inf), np.full(len(direction), np.inf)
                np.divide(lo[axis] - camera[axis], direction[:, axis], out=first, where=~parallel)
                np.divide(hi[axis] - camera[axis], direction[:, axis], out=second, where=~parallel)
                entry = np.maximum(entry, np.minimum(first, second))
                leave = np.minimum(leave, np.maximum(first, second))
                leave[parallel & ((camera[axis] < lo[axis]) | (camera[axis] >= hi[axis]))] = -np.inf
            free_counts[i] += bool(((leave > entry + 1e-12) & ~hit).any())
    expected_free = grid.n_free.ravel()[chosen]
    expected_hit = grid.n_hit.ravel()[chosen]
    mismatch = (free_counts != expected_free) | (hit_counts != expected_hit)
    return {'method': 'independent original-depth ray/axis-aligned-cell slab intersections',
            'seed': seed, 'sampled_cells': len(cells), 'frames': len(frames),
            'cell_frame_pairs': len(cells) * len(frames),
            'free_count_mismatches': int((free_counts != expected_free).sum()),
            'hit_count_mismatches': int((hit_counts != expected_hit).sum()),
            'sample_cell_indices': cells.tolist(),
            'mismatches': [{'cell': cells[i].tolist(), 'saved_free': int(expected_free[i]),
                            'oracle_free': int(free_counts[i]), 'saved_hit': int(expected_hit[i]),
                            'oracle_hit': int(hit_counts[i])} for i in np.flatnonzero(mismatch)]}
