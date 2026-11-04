"""Construct and verify proposal free/unknown cuboids and count scene tokens.

Exports geometry descriptors, not learned embeddings or image features. Each
occupied cell is retained as a surface-token candidate. Source labels are never
changed; optional bounds crop the existing grid without resampling or refusion.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .geometry.cuboids import Cuboids, compress_states, validate_partition, validate_states
from .geometry.tsdf import FREE, OCCUPIED, UNKNOWN, STATE_DEFINITION, STATE_SCHEMA_VERSION
from .state_map import roi_indices, write_json


REGION_SCHEMA_VERSION = 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-volume', required=True, type=Path, help='Validated schema-3 volume.npz')
    p.add_argument('--bounds', nargs=6, type=float, metavar=('X0', 'Y0', 'Z0', 'X1', 'Y1', 'Z1'),
                   help='Optional smaller ROI, snapped outward to existing cell faces')
    p.add_argument('--scene-token-budget', type=int, help='Optional budget for scene tokens alone, excluding question/answer')
    p.add_argument('--output', type=Path, help='New or empty directory; otherwise timestamped')
    return p


def token_counts(state, cuboids):
    surface = int((state == OCCUPIED).sum())
    free = int((cuboids.state == FREE).sum())
    unknown = int((cuboids.state == UNKNOWN).sum())
    return {'surface_candidates': surface, 'free_regions': free, 'unknown_regions': unknown,
            'surface_only': surface, 'surface_plus_free': surface + free,
            'surface_plus_unknown': surface + unknown, 'surface_plus_both': surface + free + unknown}


def save_region_views(state, cuboids, origin, h, path):
    """Full slice intersections of the 3-D boxes; no filtered region subset."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    colors = ['#d9dde2', '#66acd0', '#d34843']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    hi = origin + np.array(state.shape) * h
    for row, (plane_axes, normal) in enumerate((((0, 1), 2), ((0, 2), 1))):
        i, j = plane_axes
        k = state.shape[normal] // 2
        slice_data = np.take(state, k, axis=normal).T
        extent = [origin[i], hi[i], origin[j], hi[j]]
        for col in range(2):
            axes[row, col].imshow(slice_data, origin='lower', cmap=ListedColormap(colors), vmin=0, vmax=2,
                                  extent=extent, interpolation='nearest')
            axes[row, col].set(xlabel=f'{"xyz"[i]} [m]', ylabel=f'{"xyz"[j]} [m]',
                              title=f'{"Voxel labels" if col == 0 else "Cuboid cross-sections"}: '
                                    f'{"xyz"[normal]} = {origin[normal] + (k + .5) * h:.2f} m')
        selected = (cuboids.start[:, normal] <= k) & (cuboids.stop[:, normal] > k)
        rectangles = []
        for a, b in zip(cuboids.start[selected], cuboids.stop[selected]):
            rectangles.append(Rectangle((origin[i] + a[i] * h, origin[j] + a[j] * h),
                                        (b[i] - a[i]) * h, (b[j] - a[j]) * h))
        axes[row, 1].add_collection(PatchCollection(rectangles, facecolor='none', edgecolor='#26313a', linewidth=.4))
    counts = token_counts(state, cuboids)
    fig.suptitle(f'Exact free/unknown cuboids at {100*h:g} cm resolution\n'
                 f'{counts["surface_candidates"]:,} surface candidates + {counts["free_regions"]:,} free '
                 f'+ {counts["unknown_regions"]:,} unknown regions = {counts["surface_plus_both"]:,} scene slots')
    fig.legend(handles=[Patch(color=c, label=s) for c, s in zip(colors, ['unknown', 'free', 'occupied surface'])],
               loc='outside lower center', ncol=3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    if args.scene_token_budget is not None and args.scene_token_budget <= 0:
        raise ValueError('Scene token budget must be positive')
    source = args.source_volume.resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as a:
        if (a['schema_version'].item() != STATE_SCHEMA_VERSION
                or a['state_definition'].item() != STATE_DEFINITION):
            raise ValueError('Expected the validated schema-3 state definition; first run state_map on legacy evidence')
        state = validate_states(a['state'])
        origin, h, dims = np.asarray(a['origin'], dtype=float), float(a['voxel_size']), a['dims']
        if (origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(h) or h <= 0
                or dims.shape != (3,) or dims.dtype.kind not in 'iu' or tuple(dims) != state.shape):
            raise ValueError('Invalid source grid geometry')
        start, stop = roi_indices(origin, state.shape, h, args.bounds)
        state = state[tuple(slice(int(x), int(y)) for x, y in zip(start, stop))].copy()
        origin = origin + start * h
        frames = a['frame_indices'].copy()
    out = (args.output or Path('outputs') / ('region_tokens_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f'Output is not an empty directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).parent
    manifest = {'status': 'running', 'region_schema_version': REGION_SCHEMA_VERSION,
                'source_volume': str(source), 'source_sha256': source_hash,
                'state_definition': STATE_DEFINITION, 'source_state_schema_version': STATE_SCHEMA_VERSION,
                'source_crop_start': start.tolist(), 'source_crop_stop': stop.tolist(),
                'requested_bounds': args.bounds, 'origin': origin.tolist(), 'dims': list(state.shape),
                'voxel_size_m': h, 'grid_hi': (origin + np.array(state.shape) * h).tolist(),
                'frame_indices': frames.tolist(),
                'algorithm': 'Greedy positive-axis box growth; best of six axis orders separately per state; not a global minimum.',
                'bounds_convention': 'Integer [start, stop) voxel indices in x/y/z; all metric coordinates use the source world frame.',
                'token_count_convention': 'One candidate per occupied cell and one per cuboid. No image feature availability filtering; no question/answer tokens.',
                'source_code_sha256': {p.relative_to(code).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (code / 'region_tokens.py', code / 'geometry/cuboids.py', code / 'state_map.py', code / 'geometry/tsdf.py')}}
    write_json(out / 'manifest.json', manifest)
    begin = perf_counter()
    cuboids, choices = compress_states(state)
    elapsed = perf_counter() - begin
    check = validate_partition(state, cuboids)
    if not check['passed']:
        raise RuntimeError(f'Cuboid partition failed validation: {check}')
    center, extent = cuboids.descriptors(origin, h)
    surface_index = np.argwhere(state == OCCUPIED)
    surface_center = origin + (surface_index + .5) * h
    np.savez_compressed(out / 'token_geometry.npz', schema_version=REGION_SCHEMA_VERSION,
                        state_schema_version=STATE_SCHEMA_VERSION, state_definition=STATE_DEFINITION,
                        origin=origin, dims=state.shape, voxel_size=h, frame_indices=frames,
                        region_start=cuboids.start, region_stop=cuboids.stop, region_state=cuboids.state,
                        region_center=center, region_extent=extent, surface_index=surface_index, surface_center=surface_center)
    # Validate the serialized artifact as well as the in-memory result.
    with np.load(out / 'token_geometry.npz', allow_pickle=False) as a:
        exported = Cuboids(a['region_start'], a['region_stop'], a['region_state'])
        exported_check = validate_partition(state, exported)
        ec, ee = exported.descriptors(a['origin'], float(a['voxel_size']))
        descriptor_match = np.array_equal(ec, a['region_center']) and np.array_equal(ee, a['region_extent'])
        surface_match = (np.array_equal(a['surface_index'], np.argwhere(state == OCCUPIED))
                         and np.array_equal(a['surface_center'], origin + (a['surface_index'] + .5) * h))
    if not exported_check['passed'] or not descriptor_match or not surface_match:
        raise RuntimeError('Exported geometry failed round-trip verification')
    with (out / 'regions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'state', 'x0_cell', 'y0_cell', 'z0_cell', 'x1_cell', 'y1_cell', 'z1_cell',
                         'center_x_m', 'center_y_m', 'center_z_m', 'width_m', 'height_m', 'depth_m'])
        for i, (a, b, label, c, e) in enumerate(zip(cuboids.start, cuboids.stop, cuboids.state, center, extent)):
            writer.writerow([i, 'free' if label == FREE else 'unknown', *a, *b, *c, *e])
    counts = token_counts(state, cuboids)
    volumes = np.prod(cuboids.stop - cuboids.start, axis=1)
    compression = {}
    for label, name in ((FREE, 'free'), (UNKNOWN, 'unknown')):
        v = volumes[cuboids.state == label]
        cells = int((state == label).sum())
        compression[name] = {'voxels': cells, 'cuboids': len(v),
                             'voxels_per_cuboid': cells / len(v) if len(v) else None,
                             'singleton_cuboids': int((v == 1).sum()),
                             'largest_cuboid_voxels': int(v.max()) if len(v) else 0,
                             'median_cuboid_voxels': float(np.median(v)) if len(v) else None}
    budget = args.scene_token_budget
    summary = {'frame_count': len(frames), 'total_voxels': int(state.size), 'token_counts': counts,
               'count_scope': manifest['token_count_convention'], 'compression': compression,
               'axis_order_search': choices, 'compression_seconds': elapsed,
               'validation': exported_check, 'serialized_descriptors_match': bool(descriptor_match),
               'surface_cells_preserved': bool(surface_match), 'scene_token_budget': budget,
               'fits_scene_token_budget': counts['surface_plus_both'] <= budget if budget is not None else None}
    save_region_views(state, cuboids, origin, h, out / 'cuboid_slices.png')
    write_json(out / 'summary.json', summary)
    manifest['status'] = 'complete'
    manifest['output_sha256'] = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                for name in ('token_geometry.npz', 'regions.csv', 'summary.json', 'cuboid_slices.png')}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'output': str(out), 'token_counts': counts, 'validation': exported_check,
                      'compression_seconds': elapsed}), flush=True)
    return out


def main(argv=None):
    try:
        return run(parser().parse_args(argv))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == '__main__':
    main()
