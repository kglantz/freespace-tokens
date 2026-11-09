"""Step 3: block tokenizer and scene-token counts (geometry only; features come in step 4).

From the step-1 2 cm occupied/free/unknown map:
  * one SURFACE token per occupied block (position = mean of its occupied 2 cm voxels),
  * FREE blocks merged into boxes, UNKNOWN blocks (chosen scope) merged into boxes, each box
    one token (center, size, type); boxes are capped at a maximum side length.
Sweeps block size (8/16/32 cm), box cap (none/1.5/1.0 m) and unknown scope (all/room/frontier),
and visualizes the default setting.

Outputs (outputs/<scene>/step3_tokens/): tokens.png, tokens_slices.png, tokens_3d.html, tokens.npz,
summary.json

Usage (from the repo root, after step1_reconstruction.py):
    venv/bin/python scripts/step3_tokens.py --scene 1ada7a0617
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volumetric_reasoning.geometry.block_map import BlockMap  # noqa: E402
from volumetric_reasoning.geometry.block_tokens import UNKNOWN_SCOPES, tokenize  # noqa: E402
from volumetric_reasoning.geometry.labels import FREE, OCCUPIED, UNKNOWN  # noqa: E402
from volumetric_reasoning.render3d import (LABEL_HEX, LABEL_RGB, OCCUPIED_SCALE, box_mesh, camera,  # noqa: E402
                                           face_shade, facecolors, render, scale_colors)

HEX = {"surface": LABEL_HEX[OCCUPIED], "free": LABEL_HEX[FREE], "unknown": LABEL_HEX[UNKNOWN]}
VARIANTS = (("surface_only", "surface only"), ("surface_free", "+ free"),
            ("surface_unknown", "+ unknown"), ("surface_both", "+ free + unknown"))


# --- geometry for drawing -----------------------------------------------------------
def layer_boxes(tok, kind, z_cut, shrink):
    """World-space [lo, hi) boxes of one token layer, clipped at z_cut, shrunk to show separations."""
    ext = tok.block_size_m
    if kind == "surface":
        lo = tok.origin + tok.surface_block * ext
        hi = lo + ext
    else:
        sel = tok.box_type == (FREE if kind == "free" else UNKNOWN)
        lo = tok.origin + tok.box_start[sel] * ext
        hi = tok.origin + tok.box_stop[sel] * ext
    hi = hi.copy()
    hi[:, 2] = np.minimum(hi[:, 2], z_cut)
    keep = hi[:, 2] > lo[:, 2]
    lo, hi = lo[keep], hi[keep]
    if shrink:
        gap = np.minimum(shrink * ext, 0.25 * (hi - lo))  # constant visual gap, not proportional
        lo, hi = lo + gap, hi - gap
    return lo, hi


def layer_colors(kind, lo, hi, floor, z_cut, seed=0):
    """Per-box base colors: surface tinted by height; boxes jittered so neighbours are distinguishable."""
    if kind == "surface":
        return scale_colors((lo[:, 2] + hi[:, 2]) / 2, OCCUPIED_SCALE, floor, z_cut)
    base = LABEL_RGB[FREE if kind == "free" else UNKNOWN]
    jitter = np.random.default_rng(seed).uniform(0.78, 1.18, len(lo))
    return np.clip(base[None] * jitter[:, None], 0, 1)


def mesh_layer(tok, kind, z_cut, floor, shrink):
    lo, hi = layer_boxes(tok, kind, z_cut, shrink)
    V, F, box, second = box_mesh(lo, hi)
    return V, F, layer_colors(kind, lo, hi, floor, z_cut)[box], second


# --- figure -----------------------------------------------------------------------
def save_figure(tok, rows, args, room_floor, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = tok.counts()
    lo = tok.origin
    hi = tok.origin + np.array([s * tok.block_size_m for s in args.grid_blocks])
    fig = plt.figure(figsize=(19, 11.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.16, wspace=0.12)
    titles = {"surface": f"Surface tokens: {c['surface']:,} (one per occupied {tok.block_size_m * 100:g} cm block)",
              "free": f"Free tokens: {c['free']:,} boxes (cap {args.cap:g} m), shown slightly shrunk",
              "unknown": f"Unknown tokens: {c['unknown']:,} boxes (scope: {args.unknown}), shown slightly shrunk"}
    for col, kind in enumerate(("surface", "free", "unknown")):
        ax = fig.add_subplot(gs[0, col])
        V, F, rgb, sec = mesh_layer(tok, kind, args.z_cut, room_floor, shrink=0 if kind == "surface" else 0.12)
        center, eye = camera(lo, hi, z_focus=0.6)
        ax.imshow(render(V, F, rgb, sec, center, eye))
        ax.axis("off")
        ax.set_title(titles[kind] + f"\nceiling removed above z = {args.z_cut:g} m", fontsize=11,
                     color={"surface": "#c0601c", "free": "#2a6fb8", "unknown": "#555555"}[kind])

    # (d) the four matched variants for the default setting
    ax = fig.add_subplot(gs[1, 0])
    names = [n for _, n in VARIANTS]
    parts = {"surface": [c["surface"]] * 4, "free": [0, c["free"], 0, c["free"]],
             "unknown": [0, 0, c["unknown"], c["unknown"]]}
    left = np.zeros(4)
    for k in ("surface", "free", "unknown"):
        ax.barh(names, parts[k], left=left, color=HEX[k], label=k)
        left += parts[k]
    for i, total in enumerate(left):
        ax.text(total + 60, i, f"{int(total):,}", va="center", fontsize=10)
    ax.axvline(args.budget, color="#b03030", ls="--", lw=1)
    ax.invert_yaxis(); ax.set_xlim(0, max(args.budget, left.max()) * 1.15)
    ax.text(args.budget, -0.62, f"budget {args.budget:,}", color="#b03030", fontsize=9, ha="center", va="bottom")
    ax.set_title(f"Scene tokens per variant ({tok.block_size_m * 100:g} cm, cap {args.cap:g} m, "
                 f"unknown: {args.unknown})", fontsize=11, pad=16)
    ax.legend(loc="center right", bbox_to_anchor=(0.97, 0.5), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # (e) block size x cap sweep (unknown scope fixed to the default)
    ax = fig.add_subplot(gs[1, 1])
    sizes = sorted({r["block_cm"] for r in rows})
    caps = ["none", "1.5", "1"]  # same formatting as rows[...]["cap"]
    width = 0.28
    ticks, labels = [], []
    for j, cap in enumerate(caps):
        sel = [next(r for r in rows if r["block_cm"] == s and r["cap"] == cap and r["unknown_scope"] == args.unknown)
               for s in sizes]
        x = np.arange(len(sizes)) + (j - 1) * width
        bottom = np.zeros(len(sizes))
        for k in ("surface", "free", "unknown"):
            v = np.array([r[k] for r in sel])
            ax.bar(x, v, width * 0.9, bottom=bottom, color=HEX[k], edgecolor="white", linewidth=0.5)
            bottom += v
        for xi, b in zip(x, bottom):
            ax.text(xi, b * 1.08, f"{int(b):,}", ha="center", va="bottom", fontsize=8, rotation=90)
        ticks += list(x)
        labels += ["no cap" if cap == "none" else f"{cap} m"] * len(sizes)
    ax.axhline(args.budget, color="#b03030", ls="--", lw=1)
    ax.text(len(sizes) - 0.55, args.budget * 1.08, f"budget {args.budget:,}", color="#b03030", fontsize=9, ha="right")
    ax.set_yscale("log"); ax.set_ylim(500, 1.5e5)
    ax.set_xticks(ticks, labels, fontsize=8)
    for i, s in enumerate(sizes):
        ax.text(i, -0.13, f"{s:g} cm blocks", transform=ax.get_xaxis_transform(), ha="center", fontsize=10)
    ax.set_ylabel("scene tokens (surface + free + unknown, log scale)")
    ax.set_title(f"Block size and box cap (unknown: {args.unknown})", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # (f) unknown scope comparison + notes
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.text(0, 1.0, f"{tok.block_size_m * 100:g} cm blocks: tokens by setting", fontsize=13, weight="bold", va="top")
    hdr = f"{'cap':>5} {'unknown':>9} {'surf':>6} {'free':>5} {'unk':>5} {'total':>6}"
    ax.text(0, 0.9, hdr, family="monospace", fontsize=10, va="top")
    y = 0.83
    for r in rows:
        if r["block_cm"] != round(tok.block_size_m * 100):
            continue
        bold = r["cap"] == f"{args.cap:g}" and r["unknown_scope"] == args.unknown
        ax.text(0, y, f"{r['cap']:>5} {r['unknown_scope']:>9} {r['surface']:>6,} {r['free']:>5,} {r['unknown']:>5,} "
                      f"{r['surface_both']:>6,}" + ("  ← shown" if bold else ""),
                family="monospace", fontsize=10, va="top", weight="bold" if bold else "normal")
        y -= 0.065
    ax.text(0, y - 0.03,
            "unknown scope: all = every unknown block in the grid (behind walls,\n"
            "above ceiling, below floor); room = inside the room, estimated from\n"
            "our own free space (no ground truth); frontier = touching free space.\n"
            "Counts are scene tokens only (no question/answer text).",
            fontsize=9, va="top", color="#555555")
    fig.suptitle(f"Scene {args.scene}: block tokens from the 2 cm map (32 iPhone frames)", fontsize=15, y=0.94)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_html(tok, args, room_floor, path):
    import plotly.graph_objects as go
    fig = go.Figure()
    lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    c = tok.counts()
    for kind, visible, shrink in (("surface", True, 0), ("unknown", True, 0.12), ("free", "legendonly", 0.12)):
        lo, hi = layer_boxes(tok, kind, args.z_cut, shrink)
        V, F, box, _ = box_mesh(lo, hi)
        facecolor = facecolors(layer_colors(kind, lo, hi, room_floor, args.z_cut)[box] * face_shade(V, F)[:, None])
        fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                                facecolor=facecolor, flatshading=True, lighting=lighting, hoverinfo="skip",
                                visible=visible, showlegend=True,  # mesh3d hides legends by default
                                name=f"{kind} tokens ({c[kind]:,})" + (" — click to show" if kind == "free" else "")))
    fig.update_scenes(aspectmode="data", bgcolor="white", camera=dict(eye=dict(x=-0.9, y=-1.3, z=1.5)),
                      xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(title=f"{args.scene}: scene tokens ({tok.block_size_m * 100:g} cm blocks, box cap {args.cap:g} m, "
                            f"unknown: {args.unknown}; ceiling removed). Click legend entries to toggle layers.",
                      legend=dict(x=0.01, y=0.97, bgcolor="rgba(255,255,255,0.85)", font=dict(size=13)),
                      height=820,
                      margin=dict(l=0, r=0, t=60, b=40), paper_bgcolor="white")
    fig.write_html(path, include_plotlyjs=True)


def save_slices(tok, args, path, heights=(0.5, 1.2, 2.0)):
    """Top-down cross-sections: every token box cut by a horizontal plane, one outline per token."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Patch, Rectangle
    ext = tok.block_size_m
    fig, axes = plt.subplots(1, len(heights), figsize=(6.2 * len(heights), 6.4))
    for ax, z in zip(np.atleast_1d(axes), heights):
        k = int(np.floor((z - tok.origin[2]) / ext))
        groups = {}
        sb = tok.surface_block[tok.surface_block[:, 2] == k]
        groups["surface"] = (sb[:, :2], sb[:, :2] + 1)
        for kind, code in (("free", FREE), ("unknown", UNKNOWN)):
            sel = (tok.box_type == code) & (tok.box_start[:, 2] <= k) & (tok.box_stop[:, 2] > k)
            groups[kind] = (tok.box_start[sel, :2], tok.box_stop[sel, :2])
        counts = {}
        for kind, (lo, hi) in groups.items():
            counts[kind] = len(lo)
            rects = [Rectangle(tok.origin[:2] + a * ext, *((b - a) * ext)) for a, b in zip(lo, hi)]
            ax.add_collection(PatchCollection(rects, facecolor=HEX[kind], edgecolor="#1f2a36" if kind != "surface"
                                              else "none", linewidth=0.8, alpha=1.0 if kind != "unknown" else 0.85))
        ax.set_aspect("equal"); ax.autoscale_view()
        ax.set_title(f"z = {z:g} m: {counts['free']} free boxes, {counts['unknown']} unknown boxes,\n"
                     f"{counts['surface']} surface blocks cut by this plane", fontsize=11)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.legend(handles=[Patch(color=HEX[k], label=f"{k} token" + (" (one outlined box = one token)" if k != "surface"
                                                                    else " (one 16 cm block = one token)"))
                        for k in ("surface", "free", "unknown")], loc="lower center", ncol=3, fontsize=10)
    fig.suptitle(f"{args.scene}: horizontal cross-sections through the scene tokens", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- main -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="1ada7a0617")
    p.add_argument("--step1", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--block-voxels", type=int, default=8, help="Default block edge in voxels (8 x 2 cm = 16 cm)")
    p.add_argument("--cap", type=float, default=1.5, help="Default maximum box side (m)")
    p.add_argument("--unknown", choices=UNKNOWN_SCOPES, default="room", help="Default unknown scope")
    p.add_argument("--budget", type=int, default=8000)
    p.add_argument("--z-cut", type=float, default=2.5)
    args = p.parse_args()
    step1 = args.step1 or Path("outputs") / args.scene / "step1_reconstruction"
    out = args.output or Path("outputs") / args.scene / "step3_tokens"
    out.mkdir(parents=True, exist_ok=True)

    bm = BlockMap.load(step1 / "block_map.npz")
    state = bm.to_dense_states()
    rows = []
    for B in (4, 8, 16):
        for cap in (None, 1.5, 1.0):
            for scope in UNKNOWN_SCOPES:
                tok, _ = tokenize(state, B, bm.origin, bm.h, cap, scope)
                rows.append({"block_cm": round(B * bm.h * 100), "cap": "none" if cap is None else f"{cap:g}",
                             "unknown_scope": scope, **tok.counts()})
    print(f"{'block':>6} {'cap':>5} {'unknown':>9} | {'surface':>7} {'free':>5} {'unk':>5} | {'total':>6}")
    for r in rows:
        print(f"{r['block_cm']:>4}cm {r['cap']:>5} {r['unknown_scope']:>9} | {r['surface']:>7} {r['free']:>5} "
              f"{r['unknown']:>5} | {r['surface_both']:>6}")

    tok, lab = tokenize(state, args.block_voxels, bm.origin, bm.h, args.cap, args.unknown)
    args.grid_blocks = lab.shape
    free_layers = np.flatnonzero((lab == FREE).any(axis=(0, 1)))
    room_floor = float(bm.origin[2] + free_layers[0] * tok.block_size_m)
    tok.save(out / "tokens.npz")
    save_figure(tok, rows, args, room_floor, out / "tokens.png")
    save_html(tok, args, room_floor, out / "tokens_3d.html")
    save_slices(tok, args, out / "tokens_slices.png")
    sizes = tok.box_size
    summary = {"scene": args.scene, "step1": str(step1),
               "default": {"block_size_m": tok.block_size_m, "cap_m": args.cap, "unknown_scope": args.unknown,
                           "counts": tok.counts(),
                           "free_box_volume_m3_median": float(np.median(np.prod(sizes[tok.box_type == FREE], 1))),
                           "unknown_box_volume_m3_median": float(np.median(np.prod(sizes[tok.box_type == UNKNOWN], 1)))
                           if (tok.box_type == UNKNOWN).any() else None},
               "sweep": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Default: {tok.counts()}\nWrote {out}/tokens.png, tokens_3d.html, tokens.npz, summary.json")


if __name__ == "__main__":
    main()
