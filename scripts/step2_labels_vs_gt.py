"""Step 2: our occupied/free/unknown labels vs. laser ground truth, at 2 cm voxels and 16 cm blocks.

Ground truth on the SAME 2 cm lattice as our step-1 BlockMap:
  occupied = voxels intersected by the FARO laser mesh
  free     = voxels traversed by laser beams from the 4 scanner positions (panocam ranges;
             scanner poses recovered by registering each panorama to the mesh)
  unknown  = neither (inside walls/objects, laser shadows, outside the room)
Labels use the same priority as ours (occupied > free > unknown) and the same block rule.

Outputs (outputs/<scene>/step2_labels/): labels_vs_gt.png, labels_vs_gt_3d.html, summary.json,
gt_states.npz, scanner_poses.json.

Usage (from the repo root, after step1_reconstruction.py):
    venv/bin/python scripts/step2_labels_vs_gt.py --scene 1ada7a0617
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volumetric_reasoning.datasets.scannetpp_laser import panorama_points, scanner_poses  # noqa: E402
from volumetric_reasoning.geometry.block_map import BlockMap  # noqa: E402
from volumetric_reasoning.geometry.block_tokens import block_labels  # noqa: E402
from volumetric_reasoning.geometry.labels import FREE, LABEL_NAMES, OCCUPIED, UNKNOWN  # noqa: E402
from volumetric_reasoning.geometry.visibility import ray_voxel_evidence  # noqa: E402
from volumetric_reasoning.render3d import (LABEL_HEX, LABEL_RGB, OCCUPIED_SCALE, block_faces, camera,  # noqa: E402
                                           face_shade, facecolors, render, scale_colors)

ORDER = (OCCUPIED, FREE, UNKNOWN)


# --- ground truth -----------------------------------------------------------------
def ground_truth(bm, mesh, scene_dir, cache_poses):
    dims, origin, h = np.array(bm.dims_voxels), bm.origin, bm.h
    state = np.full(tuple(dims), UNKNOWN, np.uint8)
    poses = scanner_poses(scene_dir, mesh, cache=cache_poses)
    free = np.zeros(state.size, bool)
    rays = 0
    for i, T in poses.items():
        pts = panorama_points(scene_dir, i) @ T[:3, :3].T + T[:3, 3]
        f, _ = ray_voxel_evidence(T[:3, 3], pts, origin, dims, h, bm.free_margin)
        free[f] = True
        rays += len(pts)
        print(f"  laser panorama {i}: {len(pts):,} beams carved", flush=True)
    state.ravel()[free] = FREE
    # Surface voxels: dense uniform samples (spacing well below h) of the laser mesh.
    n = int(min(2.5e7, mesh.get_surface_area() / (0.25 * h) ** 2))
    o3d.utility.random.seed(0)
    idx = np.floor((np.asarray(mesh.sample_points_uniformly(n).points) - origin) / h).astype(np.int64)
    state[tuple(idx[((idx >= 0) & (idx < dims)).all(1)].T)] = OCCUPIED
    return state, poses, {"laser_beams": rays, "mesh_samples": n}


# --- metrics ----------------------------------------------------------------------
def confusion(ours, gt):
    """C[i, j] = number of cells labelled ORDER[i] by us and ORDER[j] by ground truth."""
    return np.array([[int(((ours == a) & (gt == b)).sum()) for b in ORDER] for a in ORDER])


def label_metrics(C):
    o, f, u = 0, 1, 2  # indices into ORDER
    safe = lambda a, b: float(a / b) if b else None
    return {
        "surface_precision": safe(C[o, o], C[o].sum()), "surface_recall": safe(C[o, o], C[:, o].sum()),
        "free_precision": safe(C[f, f], C[f].sum()), "free_marked_but_gt_occupied": safe(C[f, o], C[f].sum()),
        "free_marked_but_gt_unknown": safe(C[f, u], C[f].sum()), "free_coverage": safe(C[f, f], C[:, f].sum()),
        "unknown_but_gt_free": safe(C[u, f], C[u].sum()), "unknown_but_gt_occupied": safe(C[u, o], C[u].sum()),
        "agreement": safe(np.trace(C), C.sum()),
    }


def voxel_metrics(ours, gt):
    """Exact 3x3 confusion plus surface precision/recall with a +-1 voxel tolerance."""
    C = confusion(ours, gt)
    m = label_metrics(C)
    k = np.ones((3, 3, 3), bool)
    so, sg = ours == OCCUPIED, gt == OCCUPIED
    m["surface_precision_tol1"] = float((so & ndimage.binary_dilation(sg, k)).sum() / so.sum())
    m["surface_recall_tol1"] = float((sg & ndimage.binary_dilation(so, k)).sum() / sg.sum())
    return C, m


# --- views ------------------------------------------------------------------------
def block_z(bm, nz):
    return bm.origin[2] + (np.arange(nz) + 0.5) * bm.block_extent


def room_region(gt_blocks):
    """Blocks inside the room: within the floor plan of ground-truth free space (holes filled)
    and at or above the lowest free block. Display only; shared by both rows."""
    free = gt_blocks == FREE
    plan = ndimage.binary_fill_holes(ndimage.binary_closing(free.any(axis=2), np.ones((3, 3), bool)))
    region = np.zeros(gt_blocks.shape, bool)
    region[:, :, int(np.flatnonzero(free.any(axis=(0, 1)))[0]):] = plan[:, :, None]
    return region


def panel_mask(lab, label, zc, zmax, room):
    m = lab == label
    if label == UNKNOWN:  # unobserved space inside the room, not the shell around it
        m &= room
    return m & (zc[None, None, :] <= zmax)


def heatmap(ax, C, title, scale_note):
    rows = C / np.maximum(C.sum(1, keepdims=True), 1)
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{rows[i, j]:.1%}\n({C[i, j]:,})", ha="center", va="center", fontsize=9,
                    color="white" if rows[i, j] > 0.55 else "black")
    ticks = [LABEL_NAMES[k] for k in ORDER]
    ax.set_xticks(range(3), [f"GT {t}" for t in ticks]); ax.set_yticks(range(3), [f"ours {t}" for t in ticks])
    ax.set_title(f"{title}\nrow-normalized: what our labels are in reality ({scale_note})", fontsize=10)


def save_figure(bm, ours_b, gt_b, Cb, Cv, mb, mv, mixed, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = bm.block_extent
    lo, hi = bm.origin, bm.origin + np.array(bm.shape_blocks) * ext
    zc = block_z(bm, ours_b.shape[2])
    panels = [(OCCUPIED, args.z_cut, f"occupied (surface) blocks, z ≤ {args.z_cut:g} m"),
              (FREE, args.free_z_max, f"free blocks, z ≤ {args.free_z_max:g} m"),
              (UNKNOWN, args.z_cut, f"unknown blocks inside the room, z ≤ {args.z_cut:g} m")]
    room = room_region(gt_b)
    fig = plt.figure(figsize=(19, 15.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.95], hspace=0.12, wspace=0.03)
    for row, (lab, who) in enumerate(((ours_b, "Ours (32 iPhone frames)"), (gt_b, "Ground truth (laser)"))):
        for col, (label, zmax, desc) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            center, eye = camera(lo, hi, z_focus=min(zmax, 1.2) * 0.5)
            V, F, second = block_faces(panel_mask(lab, label, zc, zmax, room), lo, ext)
            ax.imshow(render(V, F, np.tile(LABEL_RGB[label], (len(F), 1)), second, center, eye))
            ax.axis("off")
            extra = (f" ({int((lab == UNKNOWN)[room].sum()):,} inside the room, all heights)"
                     if label == UNKNOWN else "")
            ax.set_title(f"{who}: {desc}\n{int((lab == label).sum()):,} blocks in total{extra}", fontsize=11,
                         color=LABEL_HEX[label] if label != UNKNOWN else "#444444")
    heatmap(fig.add_subplot(gs[2, 0]), Cb, "16 cm blocks (token level)", "blocks")
    ax = fig.add_subplot(gs[2, 1])
    heatmap(ax, Cv, "2 cm voxels (reconstruction level)", "voxels, exact match")
    ax.set_xlabel("Exact match: a surface shifted by one 2 cm voxel counts as wrong (see ±1 voxel numbers)",
                  fontsize=8.5, color="#555555")
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    pct = lambda x: "–" if x is None else f"{x:.1%}"
    lines = [
        ("Key numbers", None),
        ("", "16 cm blocks  |  2 cm voxels"),
        ("Free precision (our free is truly free)", f"{pct(mb['free_precision'])}  |  {pct(mv['free_precision'])}"),
        ("False free (our free is GT occupied)",
         f"{pct(mb['free_marked_but_gt_occupied'])}  |  {pct(mv['free_marked_but_gt_occupied'])}"),
        ("Free coverage (GT free we call free)", f"{pct(mb['free_coverage'])}  |  {pct(mv['free_coverage'])}"),
        ("Surface precision", f"{pct(mb['surface_precision'])}  |  {pct(mv['surface_precision_tol1'])} (±1 vox)"),
        ("Surface recall", f"{pct(mb['surface_recall'])}  |  {pct(mv['surface_recall_tol1'])} (±1 vox)"),
        ("Our unknown that is GT free", f"{pct(mb['unknown_but_gt_free'])}  |  {pct(mv['unknown_but_gt_free'])}"),
        ("Overall label agreement", f"{pct(mb['agreement'])}  |  {pct(mv['agreement'])}"),
        ("Blocks resolved from mixed (ours / GT)", f"{mixed[0]:.1%} / {mixed[1]:.1%}"),
    ]
    y = 0.98
    for k, v in lines:
        if v is None:
            ax.text(0, y, k, fontsize=13, weight="bold", va="top")
        else:
            ax.text(0, y, k, fontsize=10.5, va="top")
            ax.text(1.0, y, v, fontsize=10.5, va="top", ha="right", family="monospace")
        y -= 0.085
    ax.text(0, y - 0.02,
            "GT occupied = laser mesh; GT free = traversed by laser beams from 4 scanner\n"
            "positions; GT unknown = neither (inside walls/objects, laser shadows, outside).\n"
            "Our unknown that is GT free = space the 32 frames did not observe (not an error).",
            fontsize=9, va="top", color="#555555")
    fig.suptitle(f"Scene {args.scene}: occupied / free / unknown labels vs. laser ground truth "
                 f"(blocks of 8³ × {bm.h * 100:g} cm voxels = {ext * 100:g} cm)", fontsize=15, y=0.95)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_html(bm, ours_b, gt_b, args, path):
    """Interactive side-by-side view; cameras are linked; legend entries toggle both views."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    ext = bm.block_extent
    zc = block_z(bm, ours_b.shape[2])
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("Ours (32 iPhone frames)", "Ground truth (laser)"), horizontal_spacing=0.01)
    room = room_region(gt_b)
    floor = float(zc[np.flatnonzero((gt_b == FREE).any(axis=(0, 1)))[0]] - ext / 2)
    # Shading is baked into per-face colors (plotly's lighting shaded identical geometry differently),
    # layers are opaque (transparency blends them into muddy colors), free starts hidden.
    lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    layers = [(OCCUPIED, args.z_cut, True, f"occupied (z ≤ {args.z_cut:g} m, shaded by height)"),
              (UNKNOWN, args.z_cut, True, f"unknown inside the room (z ≤ {args.z_cut:g} m)"),
              (FREE, args.free_z_max, "legendonly", f"free (z ≤ {args.free_z_max:g} m) — click to show")]
    for col, lab in ((1, ours_b), (2, gt_b)):
        for label, zmax, visible, text in layers:
            V, F, _ = block_faces(panel_mask(lab, label, zc, zmax, room), bm.origin, ext)
            base = (scale_colors(V[F].mean(axis=1)[:, 2], OCCUPIED_SCALE, floor, args.z_cut)
                    if label == OCCUPIED else np.tile(LABEL_RGB[label], (len(F), 1)))
            fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                                    facecolor=facecolors(base * face_shade(V, F)[:, None]), flatshading=True,
                                    lighting=lighting, name=text, legendgroup=LABEL_NAMES[label],
                                    showlegend=col == 1, visible=visible, hoverinfo="skip"), row=1, col=col)
    fig.update_scenes(aspectmode="data", camera=dict(eye=dict(x=-0.9, y=-1.3, z=1.5), up=dict(x=0, y=0, z=1)),
                      bgcolor="white", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(title=f"{args.scene}: 16 cm block labels, ceiling removed. Click a legend entry to toggle it "
                            f"in both views; drag to rotate (the views rotate together).",
                      legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, itemsizing="constant",
                                  groupclick="togglegroup"),
                      paper_bgcolor="white", margin=dict(l=0, r=0, t=70, b=40), height=820)
    sync = """
var gd = document.getElementsByClassName('plotly-graph-div')[0]; var busy = false;
gd.on('plotly_relayout', function(e){ if (busy) return; var k = Object.keys(e).find(s => s.endsWith('.camera'));
  if (!k) return; var other = k.startsWith('scene2') ? 'scene.camera' : 'scene2.camera';
  busy = true; var u = {}; u[other] = e[k]; Plotly.relayout(gd, u).then(function(){ busy = false; }); });"""
    fig.write_html(path, include_plotlyjs=True, post_script=sync)


# --- main -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--scene", default="1ada7a0617")
    p.add_argument("--step1", type=Path, help="Step-1 output directory (default outputs/<scene>/step1_reconstruction)")
    p.add_argument("--output", type=Path)
    p.add_argument("--z-cut", type=float, default=2.5, help="Hide blocks above this height (ceiling)")
    p.add_argument("--free-z-max", type=float, default=1.0, help="Height cut for the free-space panels")
    args = p.parse_args()
    step1 = args.step1 or Path("outputs") / args.scene / "step1_reconstruction"
    out = args.output or Path("outputs") / args.scene / "step2_labels"
    out.mkdir(parents=True, exist_ok=True)
    scene_dir = args.data_root / "data" / args.scene

    t0 = time.time()
    bm = BlockMap.load(step1 / "block_map.npz")
    ours = bm.to_dense_states()
    mesh = o3d.io.read_triangle_mesh(str(scene_dir / "scans" / "mesh_aligned_0.05.ply"))
    print("Building laser ground truth ...", flush=True)
    gt, poses, gt_info = ground_truth(bm, mesh, scene_dir, out / "scanner_poses.json")
    np.savez_compressed(out / "gt_states.npz", state=gt, origin=bm.origin, voxel_size=bm.h,
                        scanner_to_world=np.array([poses[i] for i in sorted(poses)]))

    Cv, mv = voxel_metrics(ours, gt)
    ours_b, mixed_o = block_labels(ours, bm.B)
    gt_b, mixed_g = block_labels(gt, bm.B)
    Cb = confusion(ours_b, gt_b)
    mb = label_metrics(Cb)
    mixed = (float(mixed_o.mean()), float(mixed_g.mean()))

    save_figure(bm, ours_b, gt_b, Cb, Cv, mb, mv, mixed, args, out / "labels_vs_gt.png")
    save_html(bm, ours_b, gt_b, args, out / "labels_vs_gt_3d.html")
    count = lambda a: {LABEL_NAMES[k]: int((a == k).sum()) for k in ORDER}
    summary = {"scene": args.scene, "step1": str(step1), "voxel_size_m": bm.h, "block_size_voxels": bm.B,
               "ground_truth": {**gt_info, "scanner_positions": {i: T[:3, 3].tolist() for i, T in poses.items()}},
               "voxels": {"ours": count(ours), "gt": count(gt), "confusion_rows_ours_cols_gt": Cv.tolist(),
                          "order": [LABEL_NAMES[k] for k in ORDER], "metrics": mv},
               "blocks": {"ours": count(ours_b), "gt": count(gt_b), "confusion_rows_ours_cols_gt": Cb.tolist(),
                          "order": [LABEL_NAMES[k] for k in ORDER], "metrics": mb,
                          "mixed_fraction_ours": mixed[0], "mixed_fraction_gt": mixed[1]},
               "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"blocks": mb, "voxels": mv}, indent=1))
    print(f"Wrote {out}/labels_vs_gt.png, labels_vs_gt_3d.html, summary.json ({summary['seconds']} s)")


if __name__ == "__main__":
    main()
