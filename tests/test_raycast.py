"""Check render_depth on a plane with known geometry."""

import numpy as np
import open3d as o3d

from volumetric_reasoning.geometry.raycast import make_raycasting_scene, render_depth


def test_fronto_parallel_plane_gives_constant_depth():
    # Large plane at world z = 3, camera at origin looking along +z (OpenCV axes).
    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector([[-10, -10, 3], [10, -10, 3], [10, 10, 3], [-10, 10, 3]])
    plane.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    scene = make_raycasting_scene(plane)
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])

    depth = render_depth(scene, K, np.eye(4), 64, 48)
    np.testing.assert_allclose(depth, 3.0, atol=1e-5)  # z-depth, not ray length

    # Moving the camera 1 m towards the plane must reduce the depth by 1 m.
    c2w = np.eye(4)
    c2w[2, 3] = 1.0
    np.testing.assert_allclose(render_depth(scene, K, c2w, 64, 48), 2.0, atol=1e-5)
