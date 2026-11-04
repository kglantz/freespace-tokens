"""ScanNet++ iPhone RGB-D loader (geometry only).

Data layout of one scene (verified on scene 3e8bba0176, ScanNet++ v2):

    <data_root>/data/<scene_id>/iphone/
        rgb.mkv                   H.264 video, 1920x1440, 60 fps, one video frame per frame index
        depth.bin                 LiDAR depth, 256x192, length-prefixed compressed frames
        pose_intrinsic_imu.json   per-frame ARKit metadata, keys "frame_000000", ...
        colmap/, nerfstudio/      refined poses for a ~every-10th-frame subset (not used here)
    <data_root>/data/<scene_id>/scans/
        mesh_aligned_0.05.ply     laser-scan reference mesh (evaluation only, never an input)

Conventions (verified by rendering the laser mesh with each candidate pose and
comparing against the LiDAR depth; median error ~1-3 cm for the convention below,
~1-2 m for the alternatives):

* ``aligned_pose`` is a 4x4 **camera-to-world** matrix, already in the coordinate
  frame of the laser-scan mesh, in meters.
* The camera axes follow the **OpenCV** convention: x right, y down, z forward.
  (The raw ARKit ``pose`` field lives in a different, unaligned world frame.)
* ``intrinsic`` is a 3x3 pinhole matrix for the **RGB** resolution (1920x1440),
  without lens distortion. The depth map covers the same field of view at lower
  resolution, so for depth it must be scaled by 256/1920 = 192/1440.
* Depth is uint16 in **millimeters**; 0 marks an invalid measurement.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import lz4.block
import numpy as np

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
DEPTH_SCALE = 1000.0  # stored value / DEPTH_SCALE = meters


@dataclass
class Frame:
    """One RGB-D observation. All matrices are float64 numpy arrays."""

    index: int
    depth: np.ndarray  # (192, 256) float32, meters, 0 = invalid
    K_depth: np.ndarray  # (3, 3) intrinsics for the depth resolution
    cam_to_world: np.ndarray  # (4, 4), OpenCV camera axes, world = laser-scan frame
    rgb: np.ndarray | None = None  # (1440, 1920, 3) uint8 RGB, if loaded
    K_rgb: np.ndarray | None = None  # (3, 3) intrinsics for the RGB resolution
    timestamp: float | None = None


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Intrinsics for an image resized by (sx, sy).

    Uses the pixel-corner convention (pixel (0,0) covers [0,1)x[0,1)), under which
    a pure resize scales fx, fy, cx and cy by the same factors.
    """
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def decode_depth_chunk(chunk: bytes) -> np.ndarray:
    """Decode an LZ4 millimeter or raw DEFLATE float-meter frame to uint16 mm."""
    shape = (DEPTH_HEIGHT, DEPTH_WIDTH)
    pixels = DEPTH_WIDTH * DEPTH_HEIGHT
    try:
        raw = lz4.block.decompress(chunk, uncompressed_size=pixels * 2)
    except lz4.block.LZ4BlockError:
        pass
    else:
        if len(raw) == pixels * 2:
            return np.frombuffer(raw, dtype="<u2").reshape(shape)

    # ScanNet++ also stores float32 meters in raw DEFLATE blocks with the same framing.
    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        raw = decoder.decompress(chunk, pixels * 4 + 1)
    except zlib.error as exc:
        raise ValueError("Depth chunk is neither an LZ4 uint16 frame nor a raw DEFLATE float32 frame") from exc
    if len(raw) != pixels * 4 or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid size or framing in raw DEFLATE depth chunk")
    meters = np.frombuffer(raw, dtype="<f4").reshape(shape)
    valid = np.isfinite(meters) & (meters > 0) & (meters <= np.iinfo(np.uint16).max / DEPTH_SCALE)
    millimeters = np.zeros(shape, dtype=np.uint16)
    millimeters[valid] = (meters[valid] * DEPTH_SCALE).astype(np.uint16)
    return millimeters


def index_depth_file(path: Path) -> list[tuple[int, int]]:
    """Return (byte offset, size) of every compressed frame in depth.bin.

    depth.bin layout: repeated [uint32 little-endian chunk size][compressed block].
    Only headers are read, so indexing is fast and allows random access later.
    """
    entries = []
    file_size = Path(path).stat().st_size
    with open(path, "rb") as f:
        while True:
            header = f.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError(f"Truncated depth header at byte {f.tell() - len(header)}")
            (size,) = struct.unpack("<I", header)
            if size == 0 or f.tell() + size > file_size:
                raise ValueError(f"Invalid/truncated depth chunk at byte {f.tell()}")
            entries.append((f.tell(), size))
            f.seek(size, 1)
    return entries


class ScanNetPPIphoneScene:
    """Random access to depth, poses and intrinsics; sequential access to RGB."""

    def __init__(self, data_root: str | Path, scene_id: str):
        self.scene_id = scene_id
        self.scene_dir = Path(data_root) / "data" / scene_id
        self.iphone_dir = self.scene_dir / "iphone"
        if not self.iphone_dir.is_dir():
            raise FileNotFoundError(f"No iPhone data at {self.iphone_dir}")

        self.video_path = self.iphone_dir / "rgb.mkv"
        self.depth_path = self.iphone_dir / "depth.bin"
        with open(self.iphone_dir / "pose_intrinsic_imu.json") as f:
            self._meta = json.load(f)
        self._depth_index = index_depth_file(self.depth_path)

        self.num_frames = len(self._meta)
        if len(self._depth_index) != self.num_frames:
            raise ValueError(
                f"{len(self._depth_index)} depth frames but {self.num_frames} pose entries"
            )

        cap = cv2.VideoCapture(str(self.video_path))
        self.rgb_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.rgb_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n_video != self.num_frames:
            raise ValueError(f"{n_video} video frames but {self.num_frames} pose entries")

    # --- per-frame metadata -------------------------------------------------
    def _entry(self, index: int) -> dict:
        return self._meta[f"frame_{index:06d}"]

    def cam_to_world(self, index: int) -> np.ndarray:
        """4x4 camera-to-world pose (OpenCV camera axes, laser-scan world frame)."""
        return np.asarray(self._entry(index)["aligned_pose"], dtype=np.float64)

    def K_rgb(self, index: int) -> np.ndarray:
        return np.asarray(self._entry(index)["intrinsic"], dtype=np.float64)

    def K_depth(self, index: int) -> np.ndarray:
        return scale_intrinsics(
            self.K_rgb(index), DEPTH_WIDTH / self.rgb_width, DEPTH_HEIGHT / self.rgb_height
        )

    def timestamp(self, index: int) -> float:
        return float(self._entry(index)["timestamp"])

    # --- images -------------------------------------------------------------
    def load_depth(self, index: int) -> np.ndarray:
        """(192, 256) float32 depth in meters; 0 = invalid."""
        if not 0 <= index < self.num_frames:
            raise IndexError(f"Frame index {index} outside [0, {self.num_frames})")
        offset, size = self._depth_index[index]
        with open(self.depth_path, "rb") as f:
            f.seek(offset)
            chunk = f.read(size)
        if len(chunk) != size:
            raise ValueError(f"Truncated depth frame {index} for scene {self.scene_id} at {self.depth_path}")
        try:
            return decode_depth_chunk(chunk).astype(np.float32) / DEPTH_SCALE
        except ValueError as exc:
            raise ValueError(f"Cannot decode depth frame {index} for scene {self.scene_id} at {self.depth_path}: {exc}") from exc

    def iter_rgb(self, indices: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (index, RGB uint8 image) for increasing indices.

        Reads the video sequentially (grab() skips frames without converting them),
        which is reliable for H.264, unlike seeking with CAP_PROP_POS_FRAMES.
        """
        wanted = sorted(set(int(i) for i in indices))
        cap = cv2.VideoCapture(str(self.video_path))
        try:
            current = -1
            for target in wanted:
                while current < target - 1:
                    if not cap.grab():
                        raise IOError(f"Video ended at frame {current}")
                    current += 1
                ok, bgr = cap.read()
                current += 1
                if not ok:
                    raise IOError(f"Could not decode video frame {target}")
                yield target, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    # --- convenience --------------------------------------------------------
    def select_frames(self, start: int = 0, stride: int = 1, max_frames: int | None = None) -> list[int]:
        if not 0 <= start < self.num_frames or stride < 1 or (max_frames is not None and max_frames < 0):
            raise ValueError("Require valid start, positive stride and nonnegative max_frames")
        indices = list(range(start, self.num_frames, stride))
        return indices[:max_frames] if max_frames is not None else indices

    def select_uniform(self, num_frames: int) -> list[int]:
        """`num_frames` indices spread evenly over the whole sequence (first and last included)."""
        if not 1 <= num_frames <= self.num_frames:
            raise ValueError(f"Requested frame count must be in [1, {self.num_frames}]")
        return np.unique(np.linspace(0, self.num_frames - 1, num_frames).round().astype(int)).tolist()

    def load_frames(self, indices: Sequence[int], with_rgb: bool = True) -> list[Frame]:
        frames = {
            i: Frame(
                index=i,
                depth=self.load_depth(i),
                K_depth=self.K_depth(i),
                cam_to_world=self.cam_to_world(i),
                K_rgb=self.K_rgb(i),
                timestamp=self.timestamp(i),
            )
            for i in indices
        }
        if with_rgb:
            for i, rgb in self.iter_rgb(indices):
                frames[i].rgb = rgb
        return [frames[i] for i in indices]
