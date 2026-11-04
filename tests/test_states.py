"""Independent analytic zero-surface/voxel intersection checks, without meshes."""
from itertools import product

import numpy as np
import pytest

from volumetric_reasoning.geometry.states import zero_surface_mask
from volumetric_reasoning.geometry.tsdf import DenseTSDF, FREE, OCCUPIED, UNKNOWN


def plane_oracle(shape, normal, offset):
    """Plane/AABB test in physical cell coordinates, clipped to sample support.

    Computes an analytic range of n.x-offset on each cell. No TSDF stencils,
    sign-crossing implementation, or rasterizer output are used as the oracle.
    """
    result = np.zeros(shape, bool)
    normal = np.asarray(normal)
    for index in np.ndindex(shape):
        low = np.maximum(index, .5)
        high = np.minimum(np.array(index) + 1, np.array(shape) - .5)
        upper_included = high < np.array(index) + 1
        minimum = normal @ np.where(normal < 0, high, low) - offset
        maximum = normal @ np.where(normal > 0, high, low) - offset
        min_included = np.all((normal >= 0) | upper_included)
        max_included = np.all((normal <= 0) | upper_included)
        result[index] = ((minimum < 0 < maximum)
                         or (minimum == 0 and min_included)
                         or (maximum == 0 and max_included))
    return result


@pytest.mark.parametrize('normal', [(1, 0, 0), (0, -1, 0), (0, 0, 1),
                                    (1, 1, 0), (1, -2, 1), (1, 2, 3)])
@pytest.mark.parametrize('offset', [1., 1.5, 2., 2.25, 3.5, 6.])
def test_analytic_planes_match_exact_half_open_cell_intersections(normal, offset):
    shape = (5, 4, 6)
    coordinates = np.moveaxis(np.indices(shape, dtype=float), 0, -1) + .5
    f = coordinates @ normal - offset
    expected = plane_oracle(shape, normal, offset)
    np.testing.assert_array_equal(zero_surface_mask(f, np.ones(shape)), expected)
    np.testing.assert_array_equal(zero_surface_mask(-f, np.ones(shape)), expected)


def test_trilinear_zero_planes_on_cell_boundaries():
    coordinates = np.indices((3, 3, 3), dtype=float) + .5
    f = np.prod(coordinates - 1, axis=0)
    expected = np.any(np.indices(f.shape) == 1, axis=0)
    np.testing.assert_array_equal(zero_surface_mask(f, np.ones_like(f)), expected)


def test_only_observed_edges_faces_or_cubes_can_support_surface():
    f = np.ones((3, 3, 3))
    w = np.zeros_like(f)
    f[0, 0, 0] = -1
    w[0, 0, 0] = 1
    assert not zero_surface_mask(f, w).any()  # initialized +1 is not an observation
    w[1, 1, 1] = 1
    assert not zero_surface_mask(f, w).any()  # diagonal does not bridge unknown
    w[1, 0, 0] = 1
    expected = np.zeros_like(f, bool)
    expected[1, 0, 0] = True  # exact zero at x=1 belongs only to cell x=1
    np.testing.assert_array_equal(zero_surface_mask(f, w), expected)
    f[w == 0] = np.nan
    np.testing.assert_array_equal(zero_surface_mask(f, w), expected)


@pytest.mark.parametrize('shape', [(1, 1, 1), (1, 2, 3), (3, 4, 5)])
@pytest.mark.parametrize('value', [-1., 0., 1.])
def test_constant_fields_do_not_invent_crossings(shape, value):
    assert not zero_surface_mask(np.full(shape, value), np.ones(shape)).any()


def test_singleton_axes_support_observed_face_and_edge():
    for shape in ((3, 1, 1), (3, 4, 1)):
        f = np.indices(shape)[0] + .5 - 1
        expected = np.indices(shape)[0] == 1
        np.testing.assert_array_equal(zero_surface_mask(f, np.ones(shape)), expected)


def test_exact_zero_samples_require_nonzero_observed_neighbor():
    for values in ([1., 0., -1.], [-1., 0., 1.], [1., 0., 0.], [0., 0., -1.]):
        f = np.array(values)[:, None, None]
        np.testing.assert_array_equal(zero_surface_mask(f, np.ones_like(f)).ravel(), [False, True, False])


def test_proposal_states_use_fused_surface_not_endpoints_or_negative_tsdf():
    g = DenseTSDF(np.zeros(3), (1, 1, 6), 1, 3)
    g.tsdf[0, 0] = [1., .75, .25, -.25, -.75, -1.]
    g.weight[:] = 1
    g.n_free[0, 0, :4] = 1
    g.n_hit[0, 0, 1] = 1  # raw outlier does not turn free cell into occupied
    g.n_hit[0, 0, 5] = 1  # unsupported hit alone is not the fused zero surface
    np.testing.assert_array_equal(g.classify().ravel(), [FREE, FREE, FREE, OCCUPIED, UNKNOWN, UNKNOWN])


def test_fusion_averages_noisy_hits_before_surface_classification():
    g = DenseTSDF(np.array([-.1, -.1, 1.8]), (20, 20, 40), .01, .15)
    K = np.array([[100., 0, 32.], [0, 100., 24.], [0, 0, 1.]])
    for depth in (1.97, 2.03):
        g.integrate(np.full((48, 64), depth, np.float32), K, np.eye(4))
    surface = g.surface_mask()
    centers_z = g.origin[2] + (np.argwhere(surface)[:, 2] + .5) * g.voxel_size
    assert len(centers_z) > 0
    assert np.max(np.abs(centers_z - 2.)) <= .005001
    assert not np.any(surface & (g.n_hit > 0))


def test_invalid_fields_are_rejected():
    f, w = np.ones((2, 2, 2)), np.ones((2, 2, 2))
    for bad in (-1., np.nan, np.inf):
        x = w.copy(); x[0, 0, 0] = bad
        with pytest.raises(ValueError):
            zero_surface_mask(f, x)
    f[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        zero_surface_mask(f, w)
