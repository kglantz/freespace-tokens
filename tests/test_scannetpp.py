"""Synthetic tests for the ScanNet++ loader helpers (no dataset needed)."""

import struct

import lz4.block
import numpy as np

from volumetric_reasoning.datasets.scannetpp import (
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    decode_depth_chunk,
    index_depth_file,
    scale_intrinsics,
)


def test_depth_bin_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 5000, (DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.uint16) for _ in range(3)]
    path = tmp_path / "depth.bin"
    with open(path, "wb") as f:
        for d in frames:
            chunk = lz4.block.compress(d.astype("<u2").tobytes(), store_size=False)
            f.write(struct.pack("<I", len(chunk)) + chunk)

    index = index_depth_file(path)
    assert len(index) == 3
    with open(path, "rb") as f:
        for (offset, size), expected in zip(index, frames):
            f.seek(offset)
            np.testing.assert_array_equal(decode_depth_chunk(f.read(size)), expected)


def test_scale_intrinsics_maps_image_corners():
    K = np.array([[1400.0, 0, 960.0], [0, 1400.0, 720.0], [0, 0, 1]])
    s = DEPTH_WIDTH / 1920
    Kd = scale_intrinsics(K, s, s)
    # The point projecting to the RGB image center must project to the depth image center.
    p = np.array([0.3, -0.2, 2.0])
    uv_rgb = (K @ p)[:2] / p[2]
    uv_depth = (Kd @ p)[:2] / p[2]
    np.testing.assert_allclose(uv_depth, uv_rgb * s)
