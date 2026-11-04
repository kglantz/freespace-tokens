"""Independent ray/plane/box rendering provides end-to-end synthetic observations."""
import numpy as np

from volumetric_reasoning.geometry.tsdf import DenseTSDF, FREE, UNKNOWN, OCCUPIED


def render_box_and_wall(K, T, height=96, width=128):
    """Analytic intersections: box [-.25,.25]^2 x [1.2,1.6], wall z=3.

    This renderer does not call the project's backprojection, visibility,
    TSDF, state-classification, or mesh-extraction functions.
    """
    u, v = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
    directions = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ T[:3, :3].T
    origin = T[:3, 3]
    wall = np.full((height, width), np.inf)
    np.divide(3 - origin[2], directions[:, :, 2], out=wall, where=directions[:, :, 2] > 0)
    lo, hi = np.array([-.25, -.25, 1.2]), np.array([.25, .25, 1.6])
    enter, leave = np.full_like(wall, -np.inf), np.full_like(wall, np.inf)
    for axis in range(3):
        d = directions[:, :, axis]
        a, b = np.full_like(d, -np.inf), np.full_like(d, np.inf)
        np.divide(lo[axis] - origin[axis], d, out=a, where=d != 0)
        np.divide(hi[axis] - origin[axis], d, out=b, where=d != 0)
        outside = (d == 0) & ((origin[axis] < lo[axis]) | (origin[axis] > hi[axis]))
        enter = np.maximum(enter, np.minimum(a, b)); leave = np.minimum(leave, np.maximum(a, b))
        leave[outside] = -np.inf
    box = np.where((leave >= enter) & (enter > 0), enter, np.inf)
    depth = np.minimum(wall, box)
    return np.where(np.isfinite(depth), depth, 0).astype(np.float32)


def synthetic_occlusion_scene():
    g = DenseTSDF(np.array([-1.2, -1.2, 0.]), (60, 60, 88), .04, .12, .02)
    K = np.array([[100., 0, 64.], [0, 100., 48.], [0, 0, 1.]])
    first = np.eye(4)
    second = np.eye(4)
    second[:3, 3] = [1., 0., 1.]
    a = 1 / np.sqrt(2)
    second[:3, :3] = [[a, 0, -a], [0, 1, 0], [a, 0, a]]
    g.integrate(render_box_and_wall(K, first), K, first)
    before = g.classify()
    g.integrate(render_box_and_wall(K, second), K, second)
    after = g.classify()
    return g, before, after


def test_box_occlusion_remains_unknown_until_a_second_view_observes_it():
    g, before, after = synthetic_occlusion_scene()
    xyz = g.voxel_centers()
    center = (np.abs(xyz[..., 0]) < .08) & (np.abs(xyz[..., 1]) < .08)
    free = center & (xyz[..., 2] > .4) & (xyz[..., 2] < 1.1)
    hidden = center & (xyz[..., 2] > 2.) & (xyz[..., 2] < 2.3)
    behind_wall = center & (xyz[..., 2] > 3.2)
    assert free.sum() and hidden.sum() and behind_wall.sum()
    assert (before[free] == FREE).all()
    assert (before[hidden] == UNKNOWN).all()
    assert (after[hidden] == FREE).all()
    assert (before[behind_wall] == UNKNOWN).all() and (after[behind_wall] == UNKNOWN).all()
    # A visible front face occupies its own z=1.2 layer, not the solid box interior.
    front = center & (xyz[..., 2] > 1.2) & (xyz[..., 2] < 1.24)
    interior = center & (xyz[..., 2] > 1.35) & (xyz[..., 2] < 1.45)
    assert front.any() and (before[front] == OCCUPIED).all()
    assert interior.any() and (before[interior] == UNKNOWN).all()


def test_invalid_depth_never_carves_free_space_or_creates_surface():
    K = np.array([[30., 0, 16.], [0, 30., 12.], [0, 0, 1.]])
    for value in (0., -1., np.nan, np.inf, 5.):
        g = DenseTSDF(np.array([-.2, -.2, 0]), (10, 10, 80), .04, .12)
        g.integrate(np.full((24, 32), value), K, np.eye(4))
        assert (g.classify() == UNKNOWN).all()
