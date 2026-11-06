"""Rasterize an observed TSDF zero set without extracting a mesh.

Samples are at voxel centers. Between fully observed neighboring samples the
field is multilinear: linear on edges, bilinear on faces, trilinear in cubes.
Edges/faces also provide surface evidence where a whole cube is not observed.
No interpolation crosses an unobserved sample. A constant-zero primitive alone
is not a sign-change observation; a zero next to a nonzero observed value is.

A voxel owns [i, i+1) on every axis. Interpolation primitives are split at voxel
boundaries, and each part is tested for a zero. Multilinear extrema occur at
corners, so this test needs neither triangles nor an approximate distance band.
"""
from itertools import combinations, product

import numpy as np


def zero_surface_mask(tsdf: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return cells intersecting the supported zero set of center-sampled TSDF.

    Every contributing interpolation corner must have finite F and finite W>0.
    Unobserved values are ignored, including the usual initialized F=+1. The
    grid's outer half-cell has no extrapolated surface support. This function
    also works for singleton grid dimensions using observed edges/faces.
    """
    f, w = np.asarray(tsdf), np.asarray(weight)
    if (f.ndim != 3 or f.shape != w.shape or not all(f.shape)
            or not np.isfinite(w).all() or np.any(w < 0)
            or not np.isfinite(f[w > 0]).all()):
        raise ValueError('Expected equal nonempty 3-D fields, finite nonnegative weights and finite observed TSDF')
    occupied = np.zeros(f.shape, dtype=bool)
    # All 3 edges, 3 faces and 1 cube orientations; tiny/singleton axes are skipped.
    for dimension in (1, 2, 3):
        for axes in combinations(range(3), dimension):
            if any(f.shape[a] < 2 for a in axes):
                continue
            bits = np.array(list(product((0, 1), repeat=dimension)), dtype=int)
            shape = tuple(n - (a in axes) for a, n in enumerate(f.shape))
            offsets = np.zeros((len(bits), 3), dtype=int)
            offsets[:, axes] = bits
            slices = [tuple(slice(int(d), int(d + n)) for d, n in zip(off, shape))
                      for off in offsets]
            known = np.ones(shape, dtype=bool)
            lo, hi = np.full(shape, np.inf), np.full(shape, -np.inf)
            for sl in slices:
                known &= w[sl] > 0
                np.minimum(lo, f[sl], out=lo)
                np.maximum(hi, f[sl], out=hi)
            candidate = known & (lo <= 0) & (hi >= 0) & ((lo < 0) | (hi > 0))
            indices = np.array(np.nonzero(candidate)).T
            del known, lo, hi, candidate
            if not len(indices):
                continue
            # Bound temporary memory even for a noisy, extensively crossing grid.
            for start in range(0, len(indices), 65536):
                base = indices[start:start + 65536]
                values = np.stack([f[tuple((base + off).T)] for off in offsets]).astype(np.float64)
                for half in bits:
                    # A primitive runs between centers i+.5 and i+1.5; its lower
                    # half belongs to cell i and its upper half to cell i+1.
                    positions = (half + bits) / 2.
                    coefficients = np.prod(np.where(bits[None, :, :] == 1,
                                                    positions[:, None, :],
                                                    1 - positions[:, None, :]), axis=2)
                    subvalues = coefficients @ values
                    crosses = (subvalues.min(axis=0) < 0) & (subvalues.max(axis=0) > 0)
                    # Upper endpoints of LOWER halves are excluded by half-open
                    # ownership; upper endpoints of UPPER halves are voxel centers
                    # and are included. This also handles exact zero faces/edges.
                    included = np.all((bits == 0) | (half == 1), axis=1)
                    crosses |= np.any(subvalues[included] == 0, axis=0)
                    offset = np.zeros(3, dtype=int)
                    offset[list(axes)] = half
                    occupied[tuple((base[crosses] + offset).T)] = True
    return occupied
