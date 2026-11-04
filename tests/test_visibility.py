"""Ray evidence checked against an independent segment/AABB intersection oracle."""
import numpy as np

from volumetric_reasoning.geometry.visibility import ray_voxel_evidence


def oracle(camera, endpoints, dims, margin):
    free, hits = set(), set()
    for p in endpoints:
        hit = tuple(np.floor(p).astype(int))
        if all(0 <= c < n for c, n in zip(hit, dims)):
            hits.add(np.ravel_multi_index(hit, dims))
        d = p - camera
        stop = max(0, 1 - margin / max(np.linalg.norm(d), 1e-30))
        for cell in np.ndindex(*dims):
            if cell == hit:
                continue
            lo, hi = 0., stop
            for axis in range(3):
                if d[axis] == 0:
                    if not cell[axis] <= camera[axis] < cell[axis] + 1:
                        hi = -1
                        break
                else:
                    a = (cell[axis] - camera[axis]) / d[axis]
                    b = (cell[axis] + 1 - camera[axis]) / d[axis]
                    lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            if hi > lo + 1e-12:
                free.add(np.ravel_multi_index(cell, dims))
    return np.array(sorted(free), int), np.array(sorted(hits), int)


def test_random_rays_match_slab_intersection_including_external_camera_and_hits():
    rng = np.random.default_rng(42)
    dims = (4, 5, 3)
    points = rng.uniform(-2, 7, (50, 3))
    for camera in (np.array([1.2, 2.3, .4]), np.array([-1., 6., 2.]), np.array([4., 5., 3.])):
        for margin in (0, .2, 2):
            got = ray_voxel_evidence(camera, points, np.zeros(3), dims, 1, margin, chunk_size=7)
            expected = oracle(camera, points, dims, margin)
            for a, b in zip(got, expected):
                np.testing.assert_array_equal(a, b)


def test_parallel_diagonal_boundary_and_zero_length_rays():
    dims = (4, 4, 4)
    for camera in (np.array([0., 0., 0.]), np.array([4., 4., 4.]), np.array([2., 2., 2.])):
        endpoints = np.array([[0., 0., 0.], [4., 4., 4.], [2., 2., 2.],
                              [2., 2., -2.], [2., 2., 6.], [-1., 3., 3.]])
        got = ray_voxel_evidence(camera, endpoints, np.zeros(3), dims, 1, 0)
        expected = oracle(camera, endpoints, dims, 0)
        for a, b in zip(got, expected):
            np.testing.assert_array_equal(a, b)


def test_one_ray_guard_and_endpoint_do_not_carve_behind_surface():
    free, hit = ray_voxel_evidence([.5, .5, -.5], [[.5, .5, 3.2]], [0, 0, 0], (1, 1, 6), 1, .5)
    np.testing.assert_array_equal(free, [0, 1, 2])
    np.testing.assert_array_equal(hit, [3])
    # Traversal evidence and endpoints are invariant under a common world transform
    # consisting of translation and uniform change of length units.
    shifted = ray_voxel_evidence([11, 11, 9], [[11, 11, 16.4]], [10, 10, 10], (1, 1, 6), 2, 1)
    for a, b in zip((free, hit), shifted):
        np.testing.assert_array_equal(a, b)
