"""Reproduce the geometric audit/figures for the completed stage-one runs.

Run from the repository root:
PYTHONPATH=src venv/bin/python documents/validate_stage_one.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

from volumetric_reasoning.datasets.scannetpp import ScanNetPPIphoneScene
from volumetric_reasoning.geometry.tsdf import DenseTSDF
from volumetric_reasoning.geometry.state_validation import audit_ray_evidence
from volumetric_reasoning.state_map import classify_region, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/stage_one_validation'))
    parser.add_argument('--sample-cells', type=int, default=256)
    args = parser.parse_args()
    result = {'runs': {}}
    for name in ('uniform_32', 'stride_20'):
        directory = args.run_dir / name
        m = json.loads((directory / 'manifest.json').read_text())
        with np.load(directory / 'volume.npz') as a:
            g = DenseTSDF(a['origin'], tuple(a['dims']), float(a['voxel_size']), float(a['trunc']), float(a['free_margin']))
            for field in ('tsdf', 'weight', 'n_free', 'n_hit'):
                getattr(g, field)[:] = a[field]
            state = a['state']
            classified = classify_region(g, a['tsdf_halo'], a['weight_halo'])
            np.testing.assert_array_equal(state, classified)
        scene = ScanNetPPIphoneScene(m['data_root'], m['scene'])
        frames = scene.load_frames(m['frame_indices'], with_rgb=False)
        settings = m['fusion_settings']
        audit = audit_ray_evidence(g, state, frames, settings['min_depth'], settings['max_depth'], args.sample_cells)
        write_json(directory / 'ray_audit.json', audit)
        assert not audit['free_count_mismatches'] and not audit['hit_count_mismatches'], audit['mismatches']
        result['runs'][name] = {k: v for k, v in audit.items() if k not in ('sample_cell_indices', 'mismatches')}
        print(name, result['runs'][name], flush=True)
    with np.load(args.run_dir / 'uniform_32' / 'volume.npz') as a, np.load(args.run_dir / 'fresh_32' / 'volume.npz') as b:
        parity = {}
        for field in ('state', 'n_free', 'n_hit', 'weight'):
            np.testing.assert_array_equal(a[field], b[field])
            parity[field + '_mismatches'] = int((a[field] != b[field]).sum())
        parity['max_abs_tsdf_difference'] = float(np.max(np.abs(a['tsdf'] - b['tsdf'])))
        np.testing.assert_allclose(a['tsdf'], b['tsdf'], atol=1e-6, rtol=0)
        result['fresh_vs_cached_32'] = parity
    # Reuse the independent analytic test fixture to make the checked result visible.
    fixture = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tests/test_state_scenes.py'))
    fixture['test_box_occlusion_remains_unknown_until_a_second_view_observes_it']()
    g, before, after = fixture['synthetic_occlusion_scene']()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors = ['#d9dde2', '#66acd0', '#d34843']
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), layout='constrained')
    y = g.dims[1] // 2
    extent = [g.origin[0], g.origin[0] + g.dims[0] * g.voxel_size,
              g.origin[2], g.origin[2] + g.dims[2] * g.voxel_size]
    for ax, states, title in zip(axes, (before, after), ('One camera: hidden region stays unknown', 'Two cameras: newly seen region becomes free')):
        ax.imshow(states[:, y, :].T, origin='lower', cmap=ListedColormap(colors), vmin=0, vmax=2, extent=extent, interpolation='nearest')
        ax.plot([-.25, -.25, .25, .25, -.25], [1.2, 1.6, 1.6, 1.2, 1.2], 'k--', lw=1)
        ax.axhline(3, color='black', ls='--', lw=1)
        ax.plot(0, 0, 'k^', ms=7)
        ax.scatter([0], [2.16], facecolors='none', edgecolors='black', s=120)
        ax.set(title=title, xlabel='x [m]', ylabel='z [m]')
    axes[1].plot(1, 1, 'k^', ms=7)
    fig.legend(handles=[Patch(color=c, label=s) for c, s in zip(colors, ['unknown', 'observed free', 'observed surface'])],
               loc='outside lower center', ncol=3)
    fig.suptitle('Analytic box and wall: depth is computed from known geometry\nDashed lines are true surfaces; circle marks the visibility test region')
    fig.savefig(args.run_dir / 'synthetic_occlusion.png', dpi=150)
    plt.close(fig)
    result['synthetic_occlusion_check'] = 'passed: free foreground, occupied front face, unknown interior/occlusion, second-view revelation, unknown behind wall'
    source = Path(__file__).resolve().parents[1] / 'src' / 'volumetric_reasoning'
    result['audit_source_sha256'] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(source.rglob('*.py'))}
    write_json(args.run_dir / 'validation.json', result)
    print('Fresh/cache parity:', parity, flush=True)


if __name__ == '__main__':
    main()
