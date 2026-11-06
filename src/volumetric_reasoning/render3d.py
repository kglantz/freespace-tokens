"""Small CPU renderer for axis-aligned boxes (Open3D ray casting) with baked flat shading.

Used for static figures and, via `box_mesh` + `face_shade`, for plotly views with per-face
colors (plotly's own lighting is avoided: it shaded identical geometry inconsistently).
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from .geometry.labels import FREE, OCCUPIED, UNKNOWN

# Shared palette: orange = occupied/surface, blue = free, gray = unknown.
LABEL_RGB = {OCCUPIED: np.array([0.91, 0.52, 0.22]), FREE: np.array([0.22, 0.53, 0.86]),
             UNKNOWN: np.array([0.66, 0.68, 0.72])}
LABEL_HEX = {k: "#%02x%02x%02x" % tuple(int(255 * c) for c in v) for k, v in LABEL_RGB.items()}
# Occupied blocks are tinted by height within the same orange hue (dark floor -> light top).
OCCUPIED_SCALE = [[0.0, "#a8521c"], [0.35, "#d27030"], [0.7, "#ef9d52"], [1.0, "#fbd3a0"]]

KEY_LIGHT = np.array([0.35, -0.45, 0.82]) / np.linalg.norm([0.35, -0.45, 0.82])
SIDE_LIGHT = np.array([-0.6, 0.3, 0.2]) / np.linalg.norm([-0.6, 0.3, 0.2])
# Cube corners are indexed bx + 2*by + 4*bz; faces in cyclic order, split along corners (0, 2).
_FACES = np.array([[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]])
_BITS = np.array([[(i >> k) & 1 for k in range(3)] for i in range(8)], float)


def block_faces(mask: np.ndarray, origin: np.ndarray, ext: float):
    """Exposed faces of a boolean block grid (faces shared by two filled blocks are skipped).

    Each square is split along its (0, 2) diagonal into triangles (0, 1, 2) and (0, 2, 3).
    Returns V, F and a per-triangle second-half flag (as `box_mesh`).
    """
    verts, tris, halves, n = [], [], [], 0
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for side in (0, 1):
            neighbour = np.zeros_like(mask)
            dst, src = [slice(None)] * 3, [slice(None)] * 3
            dst[axis], src[axis] = (slice(0, -1), slice(1, None)) if side else (slice(1, None), slice(0, -1))
            neighbour[tuple(dst)] = mask[tuple(src)]
            idx = np.argwhere(mask & ~neighbour)
            if not len(idx):
                continue
            quad = np.repeat((origin + idx * ext)[:, None, :], 4, axis=1)
            quad[:, :, axis] += side * ext
            quad[:, :, u] += np.array([0, 1, 1, 0]) * ext
            quad[:, :, v] += np.array([0, 0, 1, 1]) * ext
            verts.append(quad.reshape(-1, 3))
            q = n + 4 * np.arange(len(idx))[:, None]
            tris.append(np.concatenate([q + [0, 1, 2], q + [0, 2, 3]]))
            halves.append(np.repeat([False, True], len(idx)))
            n += 4 * len(idx)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros(0, bool)
    return np.concatenate(verts), np.concatenate(tris), np.concatenate(halves)


def facecolors(rgb: np.ndarray) -> list[str]:
    """plotly 'rgb(r,g,b)' strings from (N, 3) colors in [0, 1]."""
    return [f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in np.clip(rgb, 0, 1) * 255]


def box_mesh(lo: np.ndarray, hi: np.ndarray):
    """Triangles of boxes [lo, hi) (world coordinates, (N, 3) each).

    Returns V (8N, 3), F (12N, 3), box index per triangle (12N,), second-half flag per triangle.
    """
    lo, hi = np.asarray(lo, float).reshape(-1, 3), np.asarray(hi, float).reshape(-1, 3)
    n = len(lo)
    V = (lo[:, None, :] + _BITS[None] * (hi - lo)[:, None, :]).reshape(-1, 3)
    base = 8 * np.arange(n)[:, None, None]
    quads = base + _FACES[None]                                    # (N, 6, 4)
    first = quads[..., [0, 1, 2]].reshape(-1, 3)
    second = quads[..., [0, 2, 3]].reshape(-1, 3)
    F = np.concatenate([first, second])
    box = np.concatenate([np.repeat(np.arange(n), 6)] * 2)
    is_second = np.repeat([False, True], 6 * n)
    return V, F, box, is_second


def face_shade(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-triangle brightness from two fixed lights."""
    if not len(F):
        return np.zeros(0)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return 0.45 + 0.4 * np.abs(nrm @ KEY_LIGHT) + 0.15 * np.abs(nrm @ SIDE_LIGHT)


def scale_colors(values, scale, lo, hi) -> np.ndarray:
    """(N, 3) RGB in [0, 1] from a plotly-style [[t, '#rrggbb'], ...] color scale."""
    t = np.clip((np.asarray(values) - lo) / (hi - lo), 0, 1)
    stops = np.array([s for s, _ in scale])
    rgb = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for _, c in scale])
    return np.stack([np.interp(t, stops, rgb[:, k]) for k in range(3)], axis=1)


def camera(lo, hi, z_focus, az=-125.0, el=52.0, scale=1.25):
    """(center, eye) for an oblique view looking down into a room with bounds lo..hi."""
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z_focus])
    r = scale * np.linalg.norm(np.asarray(hi[:2]) - np.asarray(lo[:2]))
    a, e = np.radians(az), np.radians(el)
    return center, center + r * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])


def render(V, F, tri_rgb, is_second, center, eye, width=900, height=680, fov=42.0, outline=0.045):
    """RGB image (white background) of a triangle mesh with per-triangle base colors.

    Square faces get dark outlines; the shared diagonal of their two triangles is not drawn.
    """
    img = np.ones((height, width, 3))
    if not len(F):
        return img
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    rays = scene.create_rays_pinhole(fov_deg=fov, center=list(map(float, center)), eye=list(map(float, eye)),
                                     up=[0, 0, 1], width_px=width, height_px=height)
    hit = scene.cast_rays(rays)
    ok = np.isfinite(hit["t_hit"].numpy())
    tri = np.where(ok, hit["primitive_ids"].numpy().astype(np.int64), 0)
    shade = face_shade(V, F)[tri]
    uv = hit["primitive_uvs"].numpy()
    w = np.stack([1 - uv[..., 0] - uv[..., 1], uv[..., 0], uv[..., 1]], -1)  # barycentric weights
    sec = is_second[tri]  # diagonal (0, 2) is opposite vertex 1 (first) or vertex 2 (second)
    edge = np.where(sec, np.minimum(w[..., 0], w[..., 1]), np.minimum(w[..., 0], w[..., 2])) < outline
    color = tri_rgb[tri] * np.where(edge, shade * 0.62, shade)[..., None]
    img = np.where(ok[..., None], np.clip(color, 0, 1), 1.0)
    return img[::-1, ::-1]  # Open3D pinhole rays start at the bottom-right pixel
