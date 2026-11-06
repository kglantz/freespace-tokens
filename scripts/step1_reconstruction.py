"""Step 1: 32 iPhone RGB-D frames -> multi-view ICP -> 2 cm VoxelHashing map -> comparison with laser.

  * 32 frames spread uniformly over the COLMAP-registered frames of the capture,
  * ARKit poses refined by leave-one-out point-to-plane ICP between the frames (no ground truth),
  * fused into a BlockMap (2 cm voxels, 8^3 blocks) with occupied / free / unknown labels,
  * Marching Cubes mesh scored against the FARO laser mesh and rendered side by side.

Outputs (outputs/<scene>/step1_reconstruction/): block_map.npz (input to steps 2 and 3), mesh.ply,
reconstruction_vs_gt.png, summary.json

Usage (from the repo root):
    venv/bin/python scripts/step1_reconstruction.py --scene 1ada7a0617
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volumetric_reasoning.datasets.scannetpp import ScanNetPPIphoneScene  # noqa: E402
from volumetric_reasoning.geometry.backprojection import depth_to_points  # noqa: E402
from volumetric_reasoning.geometry.block_map import BlockMap  # noqa: E402
from volumetric_reasoning.geometry.pose_refinement import refine_poses  # noqa: E402


# --- evaluation against the laser mesh --------------------------------------------
def raycasting(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def samples(mesh, n=200_000):
    o3d.utility.random.seed(0)
    return o3d.core.Tensor(np.asarray(mesh.sample_points_uniformly(n).points).astype(np.float32))


def metrics(mesh, reference, taus=(0.02, 0.05)):
    """Accuracy (ours -> laser), completeness (laser -> ours), precision/recall/F at each tau."""
    d_acc = raycasting(reference).compute_distance(samples(mesh)).numpy()
    d_comp = raycasting(mesh).compute_distance(samples(reference)).numpy()
    out = {"triangles": len(mesh.triangles), "accuracy_median_cm": float(np.median(d_acc) * 100),
           "completeness_median_cm": float(np.median(d_comp) * 100)}
    for tau in taus:
        p, r = float((d_acc < tau).mean()), float((d_comp < tau).mean())
        k = f"{round(tau * 100)}cm"
        out.update({f"precision_{k}": p, f"recall_{k}": r, f"fscore_{k}": 2 * p * r / (p + r) if p + r else 0.0})
    return out


# --- rendering ------------------------------------------------------------------
def crop_below(mesh, z_cut):
    """Drop triangles reaching above z_cut (removes the ceiling to look into the room)."""
    m = o3d.geometry.TriangleMesh(mesh)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    m.remove_triangles_by_mask(v[t][:, :, 2].max(axis=1) > z_cut)
    m.remove_unreferenced_vertices()
    return m


def render(mesh, center, eye, width=1100, height=800, fov=42.0):
    """Gray-shaded perspective ray-cast image of a mesh (white background)."""
    scene = raycasting(mesh)
    rays = scene.create_rays_pinhole(fov_deg=fov, center=center.tolist(), eye=eye.tolist(), up=[0, 0, 1],
                                     width_px=width, height_px=height)
    hit = scene.cast_rays(rays)
    n = hit["primitive_normals"].numpy()
    d = rays.numpy()[..., 3:]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    key = np.array([0.35, -0.45, 0.82]) / np.linalg.norm([0.35, -0.45, 0.82])
    shade = 0.18 + 0.47 * np.abs((n * d).sum(-1)) + 0.35 * np.abs(n @ key)
    shade[~np.isfinite(hit["t_hit"].numpy())] = 1.0
    return np.clip(shade, 0, 1)[::-1, ::-1]  # Open3D pinhole rays start at the bottom-right pixel


def save_figure(ours, reference, m, args, n_frames, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ours_c, ref_c = crop_below(ours, args.z_cut), crop_below(reference, args.z_cut)
    lo, hi = reference.get_min_bound(), reference.get_max_bound()
    center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + 0.35 * (args.z_cut - lo[2])])
    radius = 1.25 * np.linalg.norm(hi[:2] - lo[:2])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.4))
    for row, az in enumerate((-125.0, 55.0)):
        a, e = np.radians(az), np.radians(52.0)
        eye = center + radius * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
        for col, mesh in enumerate((ours_c, ref_c)):
            axes[row, col].imshow(render(mesh, center, eye), cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
    axes[0, 0].set_title(f"Ours: {n_frames} iPhone RGB-D frames, ARKit poses + multi-view ICP\n"
                         f"2 cm TSDF (VoxelHashing, 8³ blocks), Marching Cubes", fontsize=12)
    axes[0, 1].set_title("Ground truth: FARO laser-scan mesh", fontsize=12)
    fig.suptitle(f"Scene {args.scene}: reconstruction vs. ground truth (ceiling removed above z = {args.z_cut:g} m)",
                 fontsize=14)
    fig.text(0.5, 0.025, f"Ours vs. laser mesh  |  F-score@5cm {m['fscore_5cm']:.3f}  (precision {m['precision_5cm']:.1%}, "
             f"recall {m['recall_5cm']:.1%})  |  F-score@2cm {m['fscore_2cm']:.3f}  |  median accuracy "
             f"{m['accuracy_median_cm']:.1f} cm, median completeness {m['completeness_median_cm']:.1f} cm",
             ha="center", fontsize=10.5)
    fig.text(0.5, 0.005, "Precision: fraction of our surface within τ of the laser mesh. Recall: fraction of the "
             "laser mesh within τ of ours (incl. parts never seen by the frames).", ha="center", fontsize=9,
             color="#555555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.06, wspace=0.02, hspace=0.06)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- main -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--scene", default="1ada7a0617")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--voxel-size", type=float, default=0.02)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--truncation", type=float, default=0.08)
    p.add_argument("--free-margin", type=float, default=0.02)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=4.0)
    p.add_argument("--z-cut", type=float, default=2.5, help="Ceiling cut for the rendering (m)")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    out = args.output or Path("outputs") / args.scene / "step1_reconstruction"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    scene = ScanNetPPIphoneScene(args.data_root, args.scene)
    frame_ids = scene.select_frames(args.frames)
    frames = scene.load_frames(frame_ids, with_rgb=False)
    print(f"{args.scene}: frames {frame_ids}", flush=True)
    poses, icp = refine_poses(frames, min_depth=args.min_depth, max_depth=args.max_depth, verbose=True)
    for f, T in zip(frames, poses):
        f.cam_to_world = T

    # Grid extent from our own depth points (robust to flying pixels); blocks are world-aligned.
    pts = np.concatenate([depth_to_points(f.depth, f.K_depth, f.cam_to_world, args.min_depth, args.max_depth)[0]
                          for f in frames])
    bm = BlockMap.from_bounds(np.percentile(pts, 0.1, axis=0), np.percentile(pts, 99.9, axis=0), args.voxel_size,
                              args.block_size, margin=0.1, trunc=args.truncation, free_margin=args.free_margin)
    bm.fuse(frames, args.min_depth, args.max_depth, verbose=True)
    cats, states = bm.classify()
    mesh = bm.extract_mesh(min_weight=1.0)
    bm.save(out / "block_map.npz", frame_indices=np.array(frame_ids),
            cam_to_world=np.array([f.cam_to_world for f in frames]))
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)

    reference = o3d.io.read_triangle_mesh(str(args.data_root / "data" / args.scene / "scans" / "mesh_aligned_0.05.ply"))
    reference.compute_vertex_normals()
    m = metrics(mesh, reference)
    save_figure(mesh, reference, m, args, len(frame_ids), out / "reconstruction_vs_gt.png")
    summary = {"scene": args.scene, "frame_indices": frame_ids, "poses": "ARKit aligned_pose + multi-view ICP",
               "settings": {k: getattr(args, k) for k in ("voxel_size", "block_size", "truncation", "free_margin",
                                                           "min_depth", "max_depth")},
               "mesh_vs_laser": m, "icp": icp, "map": bm.summary(cats, states), "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"F@5cm {m['fscore_5cm']:.3f}  P {m['precision_5cm']:.3f}  R {m['recall_5cm']:.3f}  "
          f"F@2cm {m['fscore_2cm']:.3f}\nWrote {out} ({summary['seconds']} s)")


if __name__ == "__main__":
    main()
