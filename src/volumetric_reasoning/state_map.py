"""Build and inspect the proposal's three-state map; no mesh or tokenization.

Fresh fusion requires explicit small-region bounds. --source-volume reuses saved
TSDF/ray evidence but always recomputes labels, including for old schema-2 files.
One sample of TSDF halo preserves surface evidence at the requested ROI boundary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .datasets.scannetpp import ScanNetPPIphoneScene
from .geometry.backprojection import depth_to_points
from .geometry.states import zero_surface_mask
from .geometry.tsdf import (DenseTSDF, FREE, OCCUPIED, UNKNOWN, STATE_SCHEMA_VERSION,
                            STATE_DEFINITION, count_states)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-volume', type=Path, help='Reuse a saved volume.npz; old labels are ignored')
    p.add_argument('--data-root', type=Path, help='Dataset root (otherwise inherited from source manifest, or data)')
    p.add_argument('--scene', help='Scene ID (otherwise inherited, or 1ada7a0617)')
    p.add_argument('--frames', type=int, help='Uniform count for fresh fusion only (default 32)')
    p.add_argument('--bounds', nargs=6, type=float, metavar=('X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1'))
    p.add_argument('--voxel-size', type=float, help='Fresh fusion only (default .02 m)')
    p.add_argument('--truncation', type=float, help='Fresh fusion only (default .08 m)')
    p.add_argument('--free-margin', type=float, help='Fresh fusion only (default .02 m)')
    p.add_argument('--min-depth', type=float, help='Fresh fusion only (default .1 m)')
    p.add_argument('--max-depth', type=float, help='Fresh fusion only (default 4 m)')
    p.add_argument('--output', type=Path, help='New or empty directory')
    p.add_argument('--max-voxels', type=int, default=25_000_000)
    p.add_argument('--skip-depth-check', action='store_true', help='Export without loading input depths for diagnostics')
    return p


def roi_indices(origin, shape, h, bounds):
    """Snap requested bounds outward to existing cell faces; never silently clip."""
    if bounds is None:
        return np.zeros(3, int), np.array(shape, int)
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or np.any(bounds[3:] <= bounds[:3]):
        raise ValueError('Bounds must be six finite coordinates with increasing limits')
    a, b = (bounds[:3] - origin) / h, (bounds[3:] - origin) / h
    # Only cancel floating-point roundoff at an existing grid face.
    a = np.where(np.abs(a - np.rint(a)) < 1e-9, np.rint(a), a)
    b = np.where(np.abs(b - np.rint(b)) < 1e-9, np.rint(b), b)
    start, stop = np.floor(a).astype(int), np.ceil(b).astype(int)
    if np.any(start < 0) or np.any(stop > shape):
        raise ValueError('Requested ROI extends outside the saved volume; fuse fresh observations instead')
    return start, stop


def extract_region(grid, start, stop, tsdf_halo=None, weight_halo=None):
    """Crop observations while preserving adjacent interpolation samples."""
    sl = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
    halo_sl = tuple(slice(int(a), int(b + 2)) for a, b in zip(start, stop))
    fh = np.pad(grid.tsdf, 1, constant_values=1) if tsdf_halo is None else tsdf_halo
    wh = np.pad(grid.weight, 1, constant_values=0) if weight_halo is None else weight_halo
    if fh.shape != tuple(np.array(grid.dims) + 2) or wh.shape != fh.shape:
        raise ValueError('Saved TSDF halo has incorrect dimensions')
    region = DenseTSDF(grid.origin + start * grid.voxel_size, tuple(stop - start),
                       grid.voxel_size, grid.trunc, grid.free_margin)
    for key in ('tsdf', 'weight', 'n_free', 'n_hit'):
        getattr(region, key)[:] = getattr(grid, key)[sl]
    return region, fh[halo_sl].copy(), wh[halo_sl].copy()


def classify_region(grid, tsdf_halo, weight_halo):
    occupied = zero_surface_mask(tsdf_halo, weight_halo)[1:-1, 1:-1, 1:-1]
    if occupied.shape != grid.dims:
        raise ValueError('Surface halo does not match the region')
    states = np.full(grid.dims, UNKNOWN, np.uint8)
    states[grid.n_free > 0] = FREE
    states[occupied] = OCCUPIED
    return states


def state_diagnostics(grid, states):
    """Provenance checks are bookkeeping invariants, not physical ground truth."""
    violations = {
        'free_without_ray_evidence': int(((states == FREE) & (grid.n_free == 0)).sum()),
        'occupied_with_unobserved_center': int(((states == OCCUPIED) & (grid.weight == 0)).sum()),
        'unknown_with_free_ray_evidence': int(((states == UNKNOWN) & (grid.n_free > 0)).sum()),
        'invalid_state_codes': int((~np.isin(states, [UNKNOWN, FREE, OCCUPIED])).sum()),
    }
    return {'total_voxels': int(states.size), 'states': count_states(states), 'invariant_violations': violations,
            'raw_endpoint_cells': int((grid.n_hit > 0).sum()),
            'raw_endpoints_outside_fused_surface': int(((grid.n_hit > 0) & (states != OCCUPIED)).sum()),
            'surface_cells_with_ray_traversal': int(((states == OCCUPIED) & (grid.n_free > 0)).sum()),
            'raw_hit_and_traversal_cells': int(((grid.n_hit > 0) & (grid.n_free > 0)).sum()),
            'negative_tsdf_but_unknown': int(((grid.tsdf < 0) & (grid.weight > 0) & (states == UNKNOWN)).sum())}


def distance_summary(distances):
    d = np.asarray(distances)
    if not len(d):
        return {'count': 0, 'median_m': None, 'p90_m': None, 'fraction_within_5cm': None}
    return {'count': int(len(d)), 'median_m': float(np.median(d)), 'p90_m': float(np.percentile(d, 90)),
            'fraction_within_5cm': float(np.mean(d <= .05))}


def input_depth_diagnostics(grid, state, frames, min_depth, max_depth):
    """Bidirectional distances to INPUT depth points, plus per-view depth support.

    Not held-out accuracy or a ground-truth occupancy metric. Occupied cells are
    represented by centers, introducing up to sqrt(3)*h/2 discretization distance.
    All valid input pixels are used in the point reference; surface probes are
    deterministically limited to 100k. No missing depth is treated as empty.
    """
    from scipy.spatial import cKDTree
    centers = grid.origin + (np.argwhere(state == OCCUPIED) + .5) * grid.voxel_size
    lo, hi = grid.origin, grid.origin + np.array(grid.dims) * grid.voxel_size
    points = []
    for f in frames:
        p, _ = depth_to_points(f.depth, f.K_depth, f.cam_to_world, min_depth, max_depth)
        keep = ((p >= lo - grid.voxel_size) & (p < hi + grid.voxel_size)).all(axis=1)
        points.append(p[keep])
    points = np.concatenate(points) if points else np.empty((0, 3))
    in_roi = ((points >= lo) & (points < hi)).all(axis=1)
    probes = centers[np.linspace(0, len(centers) - 1, min(len(centers), 100_000)).astype(int)]
    endpoints = points[in_roi]
    endpoints = endpoints[np.linspace(0, len(endpoints) - 1, min(len(endpoints), 100_000)).astype(int)]
    result = {'scope': 'Input-observation consistency; not independent accuracy or completion.',
              'frame_indices': [int(f.index) for f in frames], 'all_reference_points_with_one_cell_halo': len(points),
              'all_reference_endpoints_in_roi': int(in_roi.sum()), 'sampling': 'evenly spaced indices, at most 100000 per direction',
              'voxel_center_max_quantization_distance_m': float(np.sqrt(3) * grid.voxel_size / 2)}
    result['surface_centers_to_input_points'] = distance_summary(cKDTree(points).query(probes)[0]) if len(points) else distance_summary([])
    result['input_endpoints_to_surface_centers'] = distance_summary(cKDTree(centers).query(endpoints)[0]) if len(centers) else distance_summary([])
    best = np.full(len(probes), np.inf)
    seen = np.zeros(len(probes), int)
    front = np.zeros(len(probes), int)
    for f in frames:
        cam = (probes - f.cam_to_world[:3, 3]) @ f.cam_to_world[:3, :3]
        z = cam[:, 2]
        valid_z = z > 1e-8
        uv = np.zeros((len(cam), 2))
        uv[valid_z] = (cam[valid_z] @ f.K_depth.T)[:, :2] / z[valid_z, None]
        pixel = np.floor(uv).astype(int)
        valid = valid_z & (pixel[:, 0] >= 0) & (pixel[:, 0] < f.depth.shape[1]) & (pixel[:, 1] >= 0) & (pixel[:, 1] < f.depth.shape[0])
        ids = np.flatnonzero(valid)
        measured = f.depth[pixel[ids, 1], pixel[ids, 0]]
        keep = np.isfinite(measured) & (measured > min_depth) & (measured <= max_depth)
        ids, measured = ids[keep], measured[keep]
        residual = measured - z[ids]
        seen[ids] += 1
        front[ids] += residual > .05
        best[ids] = np.minimum(best[ids], np.abs(residual))
    result['minimum_abs_projective_depth_residual_over_input_views'] = distance_summary(best[np.isfinite(best)])
    result['surface_probes_without_valid_depth_projection'] = int((seen == 0).sum())
    result['surface_probes_in_front_of_every_valid_depth_by_over_5cm'] = int(((seen > 0) & (seen == front)).sum())
    return result


def save_state_views(grid, state, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors = ['#d9dde2', '#66acd0', '#d34843']
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    lo, hi = grid.origin, grid.origin + np.array(grid.dims) * grid.voxel_size
    for column, fraction in enumerate((.2, .5, .8)):
        z = min(int(fraction * grid.dims[2]), grid.dims[2] - 1)
        y = min(int(fraction * grid.dims[1]), grid.dims[1] - 1)
        axes[0, column].imshow(state[:, :, z].T, origin='lower', cmap=ListedColormap(colors), vmin=0, vmax=2,
                              extent=[lo[0], hi[0], lo[1], hi[1]], interpolation='nearest')
        axes[0, column].set(title=f'Horizontal: z={lo[2] + (z + .5) * grid.voxel_size:.2f} m', xlabel='x [m]', ylabel='y [m]')
        axes[1, column].imshow(state[:, y, :].T, origin='lower', cmap=ListedColormap(colors), vmin=0, vmax=2,
                              extent=[lo[0], hi[0], lo[2], hi[2]], interpolation='nearest')
        axes[1, column].set(title=f'Vertical: y={lo[1] + (y + .5) * grid.voxel_size:.2f} m', xlabel='x [m]', ylabel='z [m]')
    fig.legend(handles=[Patch(color=c, label=s) for c, s in zip(colors, ['unknown', 'observed free', 'observed surface'])],
               loc='outside lower center', ncol=3)
    fig.suptitle(title + '\nOccupied = observed TSDF zero surface; gray is unobserved, not solid')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    source_meta, source_hash = {}, None
    if args.source_volume:
        if any(getattr(args, k) is not None for k in ('frames', 'voxel_size', 'truncation', 'free_margin', 'min_depth', 'max_depth')):
            raise ValueError('Fusion settings/frame selection cannot be changed when reusing a volume')
        source = args.source_volume.resolve()
        for candidate in (source.parent / 'manifest.json', source.parent.parent / 'manifest.json'):
            if candidate.exists():
                source_meta = json.loads(candidate.read_text()); break
        if not source_meta:
            raise ValueError('A source manifest is required to establish depth settings and data provenance')
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source, allow_pickle=False) as a:
            if a['schema_version'].item() not in (2, 3):
                raise ValueError('Unsupported source schema; raw TSDF and n_free/n_hit are required')
            dims = tuple(int(v) for v in a['dims'])
            if np.prod(dims) > args.max_voxels:
                raise ValueError('Source exceeds --max-voxels')
            g = DenseTSDF(a['origin'], dims, float(a['voxel_size']), float(a['trunc']), float(a['free_margin']))
            for key in ('tsdf', 'weight', 'n_free', 'n_hit'):
                value = a[key]
                if value.shape != dims:
                    raise ValueError(f'Invalid saved {key} shape')
                getattr(g, key)[:] = value
            indices = [int(v) for v in a['frame_indices']]
            start, stop = roi_indices(g.origin, g.dims, g.voxel_size, args.bounds)
            g, fh, wh = extract_region(g, start, stop, a['tsdf_halo'] if 'tsdf_halo' in a else None,
                                      a['weight_halo'] if 'weight_halo' in a else None)
        settings = source_meta.get('fusion_settings', source_meta.get('arguments', {}))
        if settings.get('min_depth') is None or settings.get('max_depth') is None:
            raise ValueError('Source manifest does not specify the integration depth range')
        min_depth, max_depth = settings['min_depth'], settings['max_depth']
    else:
        if args.bounds is None:
            raise ValueError('Fresh stage-one fusion requires --bounds for the small scene region')
        h = .02 if args.voxel_size is None else args.voxel_size
        trunc = .08 if args.truncation is None else args.truncation
        margin = .02 if args.free_margin is None else args.free_margin
        min_depth = .1 if args.min_depth is None else args.min_depth
        max_depth = 4. if args.max_depth is None else args.max_depth
        lo, hi = np.array(args.bounds[:3]), np.array(args.bounds[3:])
        if (not np.isfinite([*lo, *hi, h, trunc, margin, min_depth, max_depth]).all()
                or np.any(hi <= lo) or h <= 0 or trunc <= 0 or margin < 0 or not 0 <= min_depth < max_depth):
            raise ValueError('Invalid bounds, fusion scales or depth range')
        extent = (hi - lo) / h
        extent = np.where(np.abs(extent - np.rint(extent)) < 1e-9, np.rint(extent), extent)
        dims = np.ceil(extent).astype(int)
        if np.prod(dims + 2) > args.max_voxels:
            raise ValueError('ROI and halo exceed --max-voxels')
        g = DenseTSDF(lo - h, tuple(dims + 2), h, trunc, margin)
    data_root = (args.data_root or Path(source_meta.get('data_root', 'data'))).resolve()
    scene_id = args.scene or source_meta.get('scene', '1ada7a0617')
    if args.source_volume and ((args.scene and args.scene != source_meta.get('scene'))):
        raise ValueError('Cannot validate a saved volume against a different scene')
    frames = []
    if not args.source_volume or not args.skip_depth_check:
        scene = ScanNetPPIphoneScene(data_root, scene_id)
        if not args.source_volume:
            indices = scene.select_uniform(32 if args.frames is None else args.frames)
        frames = scene.load_frames(indices, with_rgb=False)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.output or Path('outputs') / f'state_map_{scene_id}_{timestamp}').resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f'Output is not an empty directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).parent
    manifest = {'status': 'running', 'created_utc': timestamp, 'scene': scene_id, 'data_root': str(data_root),
                'state_schema_version': STATE_SCHEMA_VERSION, 'state_definition': STATE_DEFINITION,
                'source_volume': str(args.source_volume.resolve()) if args.source_volume else None,
                'source_volume_sha256': source_hash, 'source_manifest': source_meta,
                'requested_bounds': args.bounds, 'frame_indices': indices,
                'fusion_settings': {'min_depth': min_depth, 'max_depth': max_depth, 'voxel_size': g.voxel_size,
                                    'truncation': g.trunc, 'free_margin': g.free_margin},
                'source_sha256': {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(code_root.rglob('*.py'))}}
    write_json(out / 'manifest.json', manifest)
    print(f'Output: {out}', flush=True)
    if not args.source_volume:
        for number, f in enumerate(frames, 1):
            g.integrate(f.depth, f.K_depth, f.cam_to_world, min_depth, max_depth)
            if number % 10 == 0 or number == len(frames):
                print(f'Fused {number}/{len(frames)} views', flush=True)
        g, fh, wh = extract_region(g, np.ones(3, int), np.array(g.dims) - 1)
    states = classify_region(g, fh, wh)
    np.savez_compressed(out / 'volume.npz', schema_version=STATE_SCHEMA_VERSION, state_definition=STATE_DEFINITION,
                        origin=g.origin, dims=g.dims, voxel_size=g.voxel_size, trunc=g.trunc, free_margin=g.free_margin,
                        tsdf=g.tsdf, weight=g.weight, n_free=g.n_free, n_hit=g.n_hit, state=states,
                        tsdf_halo=fh, weight_halo=wh, frame_indices=indices)
    result = state_diagnostics(g, states)
    if any(result['invariant_violations'].values()):
        raise RuntimeError(f'State invariant failure: {result["invariant_violations"]}')
    if not args.skip_depth_check:
        result['input_depth_consistency'] = input_depth_diagnostics(g, states, frames, min_depth, max_depth)
    write_json(out / 'summary.json', result)
    save_state_views(g, states, out / 'state_slices.png', f'{scene_id}: {len(indices)} views, {g.voxel_size * 100:g} cm cells')
    manifest.update(status='complete', origin=g.origin.tolist(), dims=list(g.dims),
                    grid_hi=(g.origin + np.array(g.dims) * g.voxel_size).tolist(),
                    halo='One center sample on each side; W=0 padding where no source support exists.',
                    label_priority='observed TSDF zero surface > depth-ray free evidence > unknown')
    write_json(out / 'manifest.json', manifest)
    print(f'Complete: {result["states"]}; invariant violations: {result["invariant_violations"]}', flush=True)
    return out


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == '__main__':
    main()
