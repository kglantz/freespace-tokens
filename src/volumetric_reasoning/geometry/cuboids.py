"""Lossless, deterministic cuboid partitions of free and unknown voxel cells.

This is a greedy partition, not a globally minimum box cover. Six growth orders
are tried independently for each state; the smallest count wins, with lexical
axis-order tie breaking. Occupied cells are never merged or discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .tsdf import FREE, OCCUPIED, UNKNOWN


@dataclass(frozen=True)
class Cuboids:
    """Integer, half-open cell bounds [start, stop), in world-grid x/y/z order."""
    start: np.ndarray
    stop: np.ndarray
    state: np.ndarray

    def descriptors(self, origin, voxel_size):
        origin = np.asarray(origin, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError('Expected finite 3-D origin and positive voxel size')
        center = origin + (self.start + (self.stop - self.start) / 2) * voxel_size
        extent = (self.stop - self.start) * voxel_size
        return center, extent


def validate_states(state):
    state = np.asarray(state)
    if (state.ndim != 3 or not all(state.shape) or state.dtype.kind not in 'iu'
            or not np.isin(state, [UNKNOWN, FREE, OCCUPIED]).all()):
        raise ValueError('Expected a nonempty 3-D integer grid with only unknown/free/occupied states')
    return state


def partition_mask(mask, growth_order=(0, 1, 2)):
    """Greedily grow disjoint boxes over one boolean mask, without changing it.

    From the first remaining cell, extend along growth_order[0], then extend
    that run along growth_order[1], then that rectangle along growth_order[2].
    Each extension requires the entire added slab to be available. Scan order
    uses the growth axes from slowest to fastest. Return x/y/z start and stop.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.dtype != np.bool_ or not all(mask.shape):
        raise ValueError('Expected a nonempty 3-D boolean mask')
    if tuple(sorted(growth_order)) != (0, 1, 2):
        raise ValueError('Growth order must be a permutation of (0, 1, 2)')
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
                x1 = int(x + blocked[0]) if len(blocked) else nx
                y1 = y + 1
                while y1 < ny and available[z, y1, x:x1].all():
                    y1 += 1
                z1 = z + 1
                while z1 < nz and available[z1, y:y1, x:x1].all():
                    z1 += 1
                available[z:z1, y:y1, x:x1] = False
                starts.append((z, y, x))
                stops.append((z1, y1, x1))
    inverse = np.argsort(scan_axes)
    return (np.asarray(starts, dtype=np.int64).reshape(-1, 3)[:, inverse],
            np.asarray(stops, dtype=np.int64).reshape(-1, 3)[:, inverse])


def compress_states(state):
    """Return Cuboids plus counts for every candidate growth order, per state."""
    state = validate_states(state)
    starts, stops, labels, choices = [], [], [], {}
    for code, name in ((FREE, 'free'), (UNKNOWN, 'unknown')):
        mask = state == code
        candidates, best = [], None
        for order in permutations(range(3)):
            start, stop = partition_mask(mask, order)
            candidates.append({'growth_order': list(order), 'cuboids': len(start)})
            if best is None or len(start) < len(best[0]):
                best = (start, stop, order)
        start, stop, order = best
        starts.append(start); stops.append(stop)
        labels.append(np.full(len(start), code, dtype=np.uint8))
        choices[name] = {'selected_growth_order': list(order), 'candidates': candidates}
    start, stop, labels = np.concatenate(starts), np.concatenate(stops), np.concatenate(labels)
    # Canonical serialization independent of the traversal order of the winner.
    sort = np.lexsort((stop[:, 2], stop[:, 1], stop[:, 0], start[:, 2], start[:, 1], start[:, 0], labels))
    return Cuboids(start[sort], stop[sort], labels[sort]), choices


def validate_partition(state, cuboids):
    """Exhaustively check every cell using independent slice expansion.

    This intentionally does not call the merger. Counts expose missing cells,
    overlaps, cross-state boxes, and occupied-cell inclusion separately. Invalid
    bounds or labels raise rather than allowing NumPy clipping/wrapping.
    """
    state = validate_states(state)
    start, stop, labels = map(np.asarray, (cuboids.start, cuboids.stop, cuboids.state))
    if (start.ndim != 2 or start.shape[1:] != (3,) or stop.shape != start.shape
            or labels.shape != (len(start),) or start.dtype.kind not in 'iu'
            or stop.dtype.kind not in 'iu' or labels.dtype.kind not in 'iu'
            or np.any(start < 0) or np.any(stop > state.shape) or np.any(stop <= start)
            or not np.isin(labels, [FREE, UNKNOWN]).all()):
        raise ValueError('Invalid cuboid bounds or labels')
    coverage = np.zeros(state.shape, np.uint32)
    expanded = np.full(state.shape, OCCUPIED, np.uint8)
    mixed = 0
    for a, b, label in zip(start, stop, labels):
        sl = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
        mixed += int(np.any(state[sl] != label))
        coverage[sl] += 1
        expanded[sl] = label
    result = {
        'checked_voxels': int(state.size),
        'missing_free_or_unknown_cells': int(((state != OCCUPIED) & (coverage == 0)).sum()),
        'overlapping_cells': int((coverage > 1).sum()),
        'occupied_cells_in_cuboids': int(((state == OCCUPIED) & (coverage > 0)).sum()),
        'nonhomogeneous_cuboids': mixed,
        'roundtrip_state_mismatches': int((expanded != state).sum()),
    }
    result['passed'] = not any(value for key, value in result.items() if key != 'checked_voxels')
    return result
