"""Compare a reconstructed mesh with a reference mesh.

Standard surface-reconstruction metrics on uniformly sampled surface points:

* accuracy:      distance from reconstructed points to the reference  (are our surfaces in the right place?)
* completeness:  distance from reference points to the reconstruction (how much of the reference did we recover?)
* precision / recall at threshold tau, and their harmonic mean (F-score).

Caveat: the laser scan also covers regions the iPhone never saw (and vice versa), which
lowers completeness / precision independently of fusion quality.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def _distances(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()


def sample_surface(mesh: o3d.geometry.TriangleMesh, n: int, seed: int = 0) -> np.ndarray:
    o3d.utility.random.seed(seed)
    return np.asarray(mesh.sample_points_uniformly(number_of_points=n).points)


def compare_meshes(
    pred: o3d.geometry.TriangleMesh,
    ref: o3d.geometry.TriangleMesh,
    tau: float = 0.05,
    n_samples: int = 200_000,
) -> dict:
    pred_pts = sample_surface(pred, n_samples)
    ref_pts = sample_surface(ref, n_samples)
    d_acc = _distances(pred_pts, ref)  # pred -> ref
    d_comp = _distances(ref_pts, pred)  # ref -> pred
    precision = float((d_acc < tau).mean())
    recall = float((d_comp < tau).mean())
    return {
        "accuracy_median_m": float(np.median(d_acc)),
        "completeness_median_m": float(np.median(d_comp)),
        "precision": precision,
        "recall": recall,
        "fscore": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        "tau_m": tau,
    }
