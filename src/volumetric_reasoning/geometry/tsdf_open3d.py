"""Reference TSDF fusion using Open3D (KinectFusion-style integration + voxel hashing).

This is the *reference* implementation used to validate our own TSDF later. What it does:

* The volume is a sparse grid of voxel blocks (block_resolution^3 voxels each), stored in a
  hash map (voxel hashing, Niessner et al. 2013). Blocks are allocated only near observed
  surfaces, so space far from surfaces is not stored: "free" and "unknown" cannot be
  distinguished there.
* Each voxel stores a projective truncated signed distance (measured along the camera ray,
  normalized by the truncation distance), a weight, and a color. Observations are averaged
  with weight 1 each (running weighted average, as in KinectFusion).

Implementation note: we use the tensor API (`o3d.t.geometry.VoxelBlockGrid`). The legacy
`o3d.pipelines.integration.ScalableTSDFVolume` integrates nothing in Open3D 0.20 on macOS
arm64 (0 voxels for valid input), while `UniformTSDFVolume` and `VoxelBlockGrid` work.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import open3d as o3d
import open3d.core as o3c


def _intrinsic_tensor(K: np.ndarray) -> o3c.Tensor:
    # Our convention puts pixel centers at (u + 0.5, v + 0.5); Open3D indexes the depth image
    # at integer pixel coordinates, i.e. it places centers at integers. Shifting the principal
    # point by -0.5 px converts between the two.
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return o3c.Tensor(K, o3c.float64)


def integrate_open3d(
    frames: Sequence,
    voxel_size: float,
    trunc_voxels: float = 4.0,
    min_depth: float = 0.1,
    max_depth: float = 4.0,
    with_color: bool = True,
    block_resolution: int = 8,
    initial_block_count: int = 20_000,
) -> o3d.t.geometry.VoxelBlockGrid:
    """Fuse frames (with .depth, .K_depth, .cam_to_world, optional .rgb) into a TSDF.

    Args:
        voxel_size: edge length of a voxel in meters.
        trunc_voxels: truncation distance in voxels (sdf_trunc = trunc_voxels * voxel_size).
        min_depth, max_depth: depth values outside (min_depth, max_depth] are ignored.
    """
    import cv2

    # Open3D's bindings reject Python ints for float arguments (e.g. `trunc_voxels: 4` from YAML).
    voxel_size, trunc_voxels = float(voxel_size), float(trunc_voxels)
    min_depth, max_depth = float(min_depth), float(max_depth)

    names, dtypes, channels = ["tsdf", "weight"], [o3c.float32, o3c.float32], [1, 1]
    if with_color:
        names.append("color"), dtypes.append(o3c.float32), channels.append(3)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=tuple(names),
        attr_dtypes=tuple(dtypes),
        attr_channels=tuple(channels),
        voxel_size=voxel_size,
        block_resolution=block_resolution,
        block_count=initial_block_count,  # the hash map grows if more blocks are needed
    )
    for f in frames:
        h, w = f.depth.shape
        depth = np.where(f.depth >= min_depth, f.depth, 0.0).astype(np.float32)
        depth_t = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth[..., None])))
        K = _intrinsic_tensor(f.K_depth)
        world_to_cam = o3c.Tensor(np.linalg.inv(f.cam_to_world), o3c.float64)  # Open3D wants w2c

        # Allocate the blocks touched by this frame's truncation band, then integrate.
        coords = vbg.compute_unique_block_coordinates(
            depth_t, K, world_to_cam, depth_scale=1.0, depth_max=max_depth,
            trunc_voxel_multiplier=trunc_voxels,
        )
        if with_color:
            rgb = cv2.resize(f.rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            color_t = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb)))
            vbg.integrate(coords, depth_t, color_t, K, K, world_to_cam, 1.0, max_depth, trunc_voxels)
        else:
            vbg.integrate(coords, depth_t, K, world_to_cam, 1.0, max_depth, trunc_voxels)
    return vbg


def extract_mesh(vbg: o3d.t.geometry.VoxelBlockGrid, weight_threshold: float = 3.0) -> o3d.geometry.TriangleMesh:
    """Marching Cubes on the zero level set; voxels observed fewer than `weight_threshold` times are skipped."""
    mesh = vbg.extract_triangle_mesh(weight_threshold=weight_threshold).to_legacy()
    mesh.compute_vertex_normals()
    return mesh


def voxel_stats(vbg: o3d.t.geometry.VoxelBlockGrid) -> dict:
    """Counts of allocated voxels and of voxels that were actually observed (weight > 0)."""
    weight = vbg.attribute("weight").numpy().ravel()
    tsdf = vbg.attribute("tsdf").numpy().ravel()
    observed = weight > 0
    return {
        "allocated_voxels": int(weight.size),
        "observed_voxels": int(observed.sum()),
        # Inside the truncation band on both sides; |tsdf| < 1 in normalized units.
        "near_surface_voxels": int((observed & (np.abs(tsdf) < 1.0)).sum()),
    }
