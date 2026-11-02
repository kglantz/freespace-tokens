"""Synthetic checks for back-projection and pose application."""

import numpy as np

from volumetric_reasoning.geometry.backprojection import (
    backproject_pixels,
    depth_to_points,
    frames_to_points,
    pixel_centers,
    project_points,
    transform_points,
)

K = np.array([[200.0, 0, 64.0], [0, 210.0, 48.0], [0, 0, 1]])


def test_project_backproject_roundtrip():
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0, 128, 100), rng.uniform(0, 96, 100)
    d = rng.uniform(0.3, 5.0, 100)
    pts = backproject_pixels(u, v, d, K)
    uv, z = project_points(pts, K)
    np.testing.assert_allclose(uv, np.stack([u, v], 1), atol=1e-9)
    np.testing.assert_allclose(z, d)


def test_principal_point_lies_on_optical_axis():
    p = backproject_pixels(np.array([K[0, 2]]), np.array([K[1, 2]]), np.array([2.0]), K)
    np.testing.assert_allclose(p, [[0, 0, 2.0]])


def test_constant_depth_gives_plane_with_correct_extent():
    depth = np.full((96, 128), 2.0)
    pts, mask = depth_to_points(depth, K)
    assert mask.all() and pts.shape == (96 * 128, 3)
    np.testing.assert_allclose(pts[:, 2], 2.0)
    # Image x-extent [0, 128] maps to [(0-cx)/fx*d, (128-cx)/fx*d]; outermost pixel centers lie 0.5 px inside.
    assert np.isclose(pts[:, 0].min(), (0.5 - 64) / 200 * 2) and np.isclose(pts[:, 0].max(), (127.5 - 64) / 200 * 2)
    # y grows downwards (OpenCV): first image row has the most negative y.
    u, v = pixel_centers(96, 128)
    assert pts[0, 1] < 0 and pts[-1, 1] > 0


def test_invalid_and_out_of_range_depth_are_dropped():
    depth = np.array([[0.0, 1.0], [np.nan, 9.0]])
    pts, mask = depth_to_points(depth, np.eye(3), max_depth=5.0)
    assert mask.tolist() == [[False, True], [False, False]] and len(pts) == 1


def test_camera_to_world_transform():
    # Camera at (1, 2, 3), rotated 90 deg about world z: camera x -> world y, camera y -> world -x.
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T[:3, 3] = [1, 2, 3]
    p_cam = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    np.testing.assert_allclose(transform_points(p_cam, T), [[1, 3, 3], [0, 2, 3], [1, 2, 4]])
    # Camera center maps to translation; inverse transform recovers the input.
    np.testing.assert_allclose(transform_points(np.zeros((1, 3)), T), [[1, 2, 3]])
    np.testing.assert_allclose(transform_points(transform_points(p_cam, T), np.linalg.inv(T)), p_cam, atol=1e-12)


def test_two_views_of_same_point_agree():
    """A world point seen from two posed cameras back-projects to the same world location."""
    X = np.array([0.4, -0.3, 3.0])
    poses = []
    for angle, t in [(0.0, [0, 0, 0]), (0.3, [0.5, 0.1, -0.2])]:
        c, s = np.cos(angle), np.sin(angle)
        T = np.eye(4)
        T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        T[:3, 3] = t
        poses.append(T)
    recovered = []
    for T in poses:
        p_cam = transform_points(X[None], np.linalg.inv(T))  # world -> camera
        uv, z = project_points(p_cam, K)
        recovered.append(transform_points(backproject_pixels(uv[:, 0], uv[:, 1], z, K), T))
    np.testing.assert_allclose(recovered[0], X[None], atol=1e-9)
    np.testing.assert_allclose(recovered[1], X[None], atol=1e-9)


def test_frames_to_points_combines_and_labels():
    class F:
        def __init__(self, index, t):
            self.index, self.depth, self.K_depth = index, np.full((4, 6), 1.0), K
            self.cam_to_world = np.eye(4)
            self.cam_to_world[:3, 3] = t
            self.rgb = None

    pts, cols, ids = frames_to_points([F(0, [0, 0, 0]), F(7, [1, 0, 0])])
    assert pts.shape == (48, 3) and cols is None
    assert (ids[:24] == 0).all() and (ids[24:] == 7).all()
    np.testing.assert_allclose(pts[24:] - pts[:24], np.tile([1, 0, 0], (24, 1)))
