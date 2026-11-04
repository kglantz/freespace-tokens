"""Analytic geometry and regressions for TSDF fusion and visibility."""
import numpy as np
import pytest

from volumetric_reasoning.geometry.tsdf import (
    FREE, OCCUPIED, UNKNOWN, DenseTSDF,
)

K = np.array([[60., 0, 32.], [0, 60., 24.], [0, 0, 1.]])


def make_grid():
    return DenseTSDF.from_bounds([-.5, -.5, 0], [.5, .5, 3], .05, 3)


def test_plane_surface_free_rays_and_hidden_unknown():
    g = make_grid()
    g.integrate(np.full((48, 64), 2., np.float32), K, np.eye(4))
    state = g.classify()
    z = g.origin[2] + (np.arange(g.dims[2]) + .5) * g.voxel_size
    assert (state[10, 10, (z > .5) & (z < 1.95)] == FREE).all()
    assert (state[:, :, z > 2.05] == UNKNOWN).all()
    hit = np.argwhere(state == OCCUPIED)
    assert len(hit) > 0
    assert np.all(hit[:, 2] == 40)  # fused zero surface z=2, half-open cells
    near = np.abs(2 - z) < g.trunc
    np.testing.assert_allclose(g.tsdf[10, 10, near] * g.trunc, (2 - z)[near], atol=1e-5)
    assert np.any((g.tsdf < 0) & (state == UNKNOWN))  # negative band is not measured occupancy
    assert g.n_free.max() == g.n_hit.max() == 1  # many pixels count as one frame


def test_frustum_includes_negative_band_past_maximum_return_depth():
    g = DenseTSDF.from_bounds([-.05, -.05, 3.9], [.05, .05, 4.2], .1, .8)
    g.integrate(np.full((48, 64), 3.99, np.float32), K, np.eye(4), max_depth=4)
    assert g.weight[0, 0, 1] == 1  # center at z=4.05: still inside mu=.08 of the hit
    assert g.tsdf[0, 0, 1] == pytest.approx((3.99 - 4.05) / .08, abs=1e-5)
    assert g.weight[0, 0, 2] == 0


def test_invalid_depth_and_camera_looking_away():
    for value in (0., -1., 9., np.nan, np.inf):
        g = make_grid()
        g.integrate(np.full((48, 64), value), K, np.eye(4))
        assert not g.weight.any() and not g.n_free.any() and not g.n_hit.any()
    g.integrate(np.full((48, 64), 2.), K, np.diag([1., -1., -1., 1.]))
    assert not g.weight.any() and not g.n_free.any() and not g.n_hit.any()


def test_input_validation():
    with pytest.raises(ValueError):
        DenseTSDF.from_bounds([0, 0, 0], [0, 1, 1], .02)
    with pytest.raises(ValueError):
        make_grid().integrate(np.ones((3, 3)), K, np.zeros((4, 4)))
