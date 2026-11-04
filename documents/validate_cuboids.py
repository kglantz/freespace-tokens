"""Reproduce or audit the full-region and fixed-center-crop cuboid experiment.

Run: PYTHONPATH=src venv/bin/python documents/validate_cuboids.py
Existing runs are audited, never overwritten. New runs read only the validated
stage-one maps. Crop selection uses geometry alone, before inspecting labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from volumetric_reasoning import region_tokens
from volumetric_reasoning.geometry.cuboids import Cuboids, validate_partition
from volumetric_reasoning.geometry.tsdf import OCCUPIED
from volumetric_reasoning.state_map import write_json


def audit_export(source, output, expected_start, expected_stop):
    manifest = json.loads((output / 'manifest.json').read_text())
    summary = json.loads((output / 'summary.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest(), 'Source has changed'
    np.testing.assert_array_equal(manifest['source_crop_start'], expected_start)
    np.testing.assert_array_equal(manifest['source_crop_stop'], expected_stop)
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest, f'Modified artifact: {name}'
    with np.load(source, allow_pickle=False) as a, np.load(output / 'token_geometry.npz', allow_pickle=False) as b:
        sl = tuple(slice(int(x), int(y)) for x, y in zip(expected_start, expected_stop))
        state = a['state'][sl]
        h = float(a['voxel_size'])
        origin = a['origin'] + expected_start * h
        boxes = Cuboids(b['region_start'], b['region_stop'], b['region_state'])
        check = validate_partition(state, boxes)
        assert check['passed'], check
        # Independent expressions from source coordinates, including the half-open upper bound.
        lower = origin + boxes.start * h
        upper = origin + boxes.stop * h
        np.testing.assert_allclose(b['region_center'], (lower + upper) / 2, rtol=0, atol=1e-12)
        np.testing.assert_allclose(b['region_extent'], upper - lower, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(b['surface_index'], np.argwhere(state == OCCUPIED))
        np.testing.assert_allclose(b['surface_center'], origin + (b['surface_index'] + .5) * h, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(b['frame_indices'], a['frame_indices'])
        np.testing.assert_array_equal(b['dims'], state.shape)
        np.testing.assert_allclose(b['origin'], origin, rtol=0, atol=1e-12)
        assert float(b['voxel_size']) == h
        counts = region_tokens.token_counts(state, boxes)
        assert counts == summary['token_counts']
        return {'output': str(output.resolve()), 'frames': len(a['frame_indices']),
                'shape': list(state.shape), 'extent_m': (np.array(state.shape) * h).tolist(),
                'origin_m': origin.tolist(), 'token_counts': counts, 'validation': check,
                'scene_budget_examples': {str(n): counts['surface_plus_both'] <= n for n in (4096, 8192, 16384, 32768)},
                'budget_note': 'Illustrative scene-only budgets, not model limits or a selected training budget.'}


def comparison_plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    labels = [f'{r["extent_m"][0]:.2f} × {r["extent_m"][1]:.2f} × {r["extent_m"][2]:.2f} m\n{r["frames"]} views' for r in rows]
    bottom = np.zeros(len(rows))
    for key, label, color in (('surface_candidates', 'Surface candidates', '#d34843'),
                               ('free_regions', 'Free cuboids', '#66acd0'),
                               ('unknown_regions', 'Unknown cuboids', '#b4bdc8')):
        values = [r['token_counts'][key] for r in rows]
        ax.barh(range(len(rows)), values, left=bottom, color=color, label=label)
        for i, n in enumerate(values):
            if n > 1800:  # Keep labels inside their segments; small counts are in validation.json.
                ax.text(bottom[i] + n / 2, i, f'{n:,}', ha='center', va='center', fontsize=10)
        bottom += values
    for i, total in enumerate(bottom):
        ax.text(total + 250, i, f'{int(total):,}', va='center', fontweight='bold')
    ax.axvline(8192, color='#495464', linestyle='--', linewidth=1,
               label='Example 8,192 scene-slot budget')
    ax.set(yticks=range(len(rows)), yticklabels=labels, xlabel='Scene-token slots (one per surface candidate or cuboid)',
           xlim=(0, max(bottom) * 1.13), title='Lossless cuboid compression: full region and fixed central crop\n2 cm cells throughout; question/answer tokens and feature availability excluded')
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path('outputs/stage_one_validation'))
    p.add_argument('--run-dir', type=Path, default=Path('outputs/cuboid_validation'))
    args = p.parse_args()
    rows = []
    for crop in (False, True):
        for subset in ('uniform_32', 'stride_20'):
            source = args.source_root / subset / 'volume.npz'
            with np.load(source, allow_pickle=False) as a:
                dims, origin, h = a['dims'], a['origin'], float(a['voxel_size'])
            start, stop = np.zeros(3, int), dims.copy()
            if crop:
                # Fixed 51 x 51 cell footprint, centered by index, full height.
                # For this validated 2 cm grid this is 1.02 x 1.02 x 2.42 m.
                if min(dims[:2]) < 51:
                    raise ValueError('Center-crop experiment requires at least 51 cells in x and y')
                start[:2] = (dims[:2] - 51) // 2
                stop[:2] = start[:2] + 51
            name = subset + ('_center_51' if crop else '')
            output = args.run_dir / name
            if not output.exists():
                argv = ['--source-volume', str(source), '--output', str(output)]
                if crop:
                    bounds = np.r_[origin + start * h, origin + stop * h]
                    argv += ['--bounds', *map(str, bounds)]
                region_tokens.main(argv)
            row = audit_export(source, output, start, stop)
            rows.append(row)
            print(json.dumps({'name': name, **row['token_counts'], 'validated': row['validation']['passed']}), flush=True)
    write_json(args.run_dir / 'validation.json', {'runs': rows})
    comparison_plot(rows, args.run_dir / 'token_counts.png')


if __name__ == '__main__':
    main()
