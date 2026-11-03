"""Synthetic checks for the dense TSDF and voxel classification."""

import numpy as np

from volumetric_reasoning.geometry.tsdf import (
    BEHIND, FREE, OCCUPIED, UNKNOWN, DenseTSDF, pool_states,
)

K = np.array([[60.0, 0, 32.0], [0, 60.0, 24.0], [0, 0, 1]])
PLANE_Z = 2.0


def make_grid():
    # 1 m x 1 m x 3 m column in front of a camera at the origin looking along +z.
    return DenseTSDF.from_bounds([-0.5, -0.5, 0.0], [0.5, 0.5, 3.0], voxel_size=0.05, trunc_voxels=3)


def integrate_plane(grid, n=1, cam_to_world=np.eye(4)):
    depth = np.full((48, 64), PLANE_Z, dtype=np.float32)
    for _ in range(n):
        grid.integrate(depth, K, cam_to_world, min_depth=0.1, max_depth=4.0)


def z_states(grid, state):
    """States along the central column (x = y = 0) as a function of voxel-center z."""
    i, j = grid.dims[0] // 2, grid.dims[1] // 2
    zs = grid.origin[2] + (np.arange(grid.dims[2]) + 0.5) * grid.voxel_size
    return zs, state[i, j]


def test_free_surface_and_unknown_along_a_ray():
    g = make_grid()
    integrate_plane(g, n=3)
    zs, s = z_states(g, g.classify(w_min=3))
    # Very close to the camera the column leaves the (narrow) image cone -> start at 0.5 m.
    assert (s[(zs > 0.5) & (zs < PLANE_Z - g.voxel_size)] == FREE).all()
    assert (s[np.abs(zs - PLANE_Z) < g.voxel_size / 2] == OCCUPIED).all()
    assert set(s[(zs > PLANE_Z + g.voxel_size) & (zs < PLANE_Z + g.trunc)]) <= {BEHIND}
    # Beyond the truncation band behind the surface nothing was updated: unknown, not free.
    assert (s[zs > PLANE_Z + g.trunc + g.voxel_size] == UNKNOWN).all()


def test_tsdf_values_match_signed_distance():
    g = make_grid()
    integrate_plane(g)
    zs, _ = z_states(g, g.classify(w_min=1))
    i, j = g.dims[0] // 2, g.dims[1] // 2
    near = np.abs(PLANE_Z - zs) < g.trunc
    np.testing.assert_allclose(g.tsdf[i, j, near] * g.trunc, (PLANE_Z - zs)[near], atol=1e-5)


def test_weight_threshold_and_outside_frustum():
    g = make_grid()
    integrate_plane(g, n=2)
    assert (g.classify(w_min=3) == UNKNOWN).all()  # seen only twice
    assert (g.classify(w_min=2) != UNKNOWN).any()
    # Voxels outside the field of view stay unknown: camera looking away (along -z).
    g2 = make_grid()
    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # 180 deg rotation about x
    integrate_plane(g2, n=3, cam_to_world=flip)
    assert (g2.weight == 0).all()


def test_invalid_depth_does_not_carve_free_space():
    g = make_grid()
    for _ in range(3):
        g.integrate(np.zeros((48, 64), np.float32), K, np.eye(4))
        g.integrate(np.full((48, 64), 9.0, np.float32), K, np.eye(4), max_depth=4.0)
    assert (g.weight == 0).all()


def test_pool_states():
    s = np.full((4, 4, 4), FREE, dtype=np.uint8)
    s[0, 0, 0] = OCCUPIED
    s[2:, 2:, 2:] = UNKNOWN
    p = pool_states(s, 2)
    assert p.shape == (2, 2, 2)
    assert p[0, 0, 0] == OCCUPIED and p[1, 1, 1] == UNKNOWN and p[1, 0, 0] == FREE
