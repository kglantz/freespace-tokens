"""Stage-one exports, legacy evidence reuse, and ROI boundary support."""
import json

import numpy as np
import pytest

from volumetric_reasoning import state_map
from volumetric_reasoning.datasets.scannetpp import Frame
from volumetric_reasoning.geometry.tsdf import DenseTSDF, OCCUPIED
from volumetric_reasoning.geometry.state_validation import audit_ray_evidence


def test_crop_keeps_zero_surface_supported_across_roi_boundary():
    g = DenseTSDF(np.zeros(3), (3, 3, 3), 1, 2)
    g.tsdf[:] = (np.indices(g.dims)[0] + .5 - 1)
    g.weight[:] = 1
    core, fh, wh = state_map.extract_region(g, np.array([1, 1, 1]), np.array([2, 2, 2]))
    assert not core.surface_mask().any()  # singleton core alone cannot establish crossing
    assert state_map.classify_region(core, fh, wh).item() == OCCUPIED


def test_roi_snapping_and_outside_rejection():
    a, b = state_map.roi_indices(np.array([.1, .2, .3]), (20, 20, 20), .02, [.12, .22, .32, .3, .4, .5])
    np.testing.assert_array_equal(a, [1, 1, 1]); np.testing.assert_array_equal(b, [10, 10, 10])
    with pytest.raises(ValueError, match='outside'):
        state_map.roi_indices(np.zeros(3), (2, 2, 2), 1, [-1, 0, 0, 2, 2, 2])


def test_fresh_cli_and_cached_cli_match_without_any_mesh(tmp_path, monkeypatch):
    class Scene:
        def __init__(self, *args):
            pass

        def select_uniform(self, n):
            assert n == 1
            return [7]

        def load_frames(self, indices, with_rgb=False):
            return [Frame(i, np.full((12, 16), 2., np.float32),
                          np.array([[20., 0, 8.], [0, 20., 6.], [0, 0, 1.]]), np.eye(4)) for i in indices]

    monkeypatch.setattr(state_map, 'ScanNetPPIphoneScene', Scene)
    fresh = tmp_path / 'fresh'
    args = state_map.parser().parse_args(['--frames', '1', '--voxel-size', '.1', '--truncation', '.3',
        '--bounds', '-.3', '-.3', '.1', '.3', '.3', '2.8', '--output', str(fresh)])
    state_map.run(args)
    m = json.loads((fresh / 'manifest.json').read_text())
    s = json.loads((fresh / 'summary.json').read_text())
    assert m['status'] == 'complete' and m['frame_indices'] == [7]
    assert not any(s['invariant_violations'].values())
    assert s['input_depth_consistency']['surface_centers_to_input_points']['median_m'] < .1
    assert not list(fresh.glob('*.ply'))
    cached = tmp_path / 'cached'
    state_map.main(['--source-volume', str(fresh / 'volume.npz'), '--output', str(cached)])
    with np.load(fresh / 'volume.npz') as a, np.load(cached / 'volume.npz') as b:
        assert int(a['schema_version']) == 3
        for field in ('state', 'tsdf', 'weight', 'n_free', 'n_hit', 'tsdf_halo', 'weight_halo'):
            np.testing.assert_array_equal(a[field], b[field])
    with pytest.raises(ValueError, match='not an empty'):
        state_map.run(args)
    with pytest.raises(SystemExit, match='cannot be changed'):
        state_map.main(['--source-volume', str(fresh / 'volume.npz'), '--frames', '32'])


def test_schema_two_endpoint_labels_are_ignored_and_recomputed(tmp_path):
    source = tmp_path / 'old'
    source.mkdir()
    g = DenseTSDF(np.zeros(3), (2, 2, 4), 1, 2)
    g.tsdf[:] = np.array([1., .5, -.5, -1.])[None, None, :]
    g.weight[:] = 1
    old_state = np.full(g.dims, OCCUPIED, np.uint8)  # deliberately wrong legacy labels
    np.savez_compressed(source / 'volume.npz', schema_version=2, origin=g.origin, dims=g.dims,
                        voxel_size=1, trunc=2, free_margin=.02, tsdf=g.tsdf, weight=g.weight,
                        n_free=g.n_free, n_hit=np.ones(g.dims, np.uint32), state=old_state, frame_indices=[7])
    (source / 'manifest.json').write_text(json.dumps({'scene': 'test', 'data_root': 'data',
                                                   'arguments': {'min_depth': .1, 'max_depth': 4}}))
    out = tmp_path / 'new'
    state_map.main(['--source-volume', str(source / 'volume.npz'), '--skip-depth-check', '--output', str(out)])
    with np.load(out / 'volume.npz') as a:
        expected = np.zeros(g.dims, np.uint8); expected[:, :, 2] = OCCUPIED
        np.testing.assert_array_equal(a['state'], expected)


def test_empty_depth_diagnostics_are_defined():
    g = DenseTSDF(np.zeros(3), (2, 2, 2), .1, .2)
    result = state_map.input_depth_diagnostics(g, g.classify(), [], .1, 4.)
    assert result['input_endpoints_to_surface_centers']['median_m'] is None
    assert result['surface_probes_without_valid_depth_projection'] == 0


def test_independent_ray_audit_matches_fusion_and_detects_corrupt_counts():
    g = DenseTSDF(np.array([-.4, -.4, .5]), (8, 8, 20), .1, .3)
    K = np.array([[20., 0, 8.], [0, 20., 6.], [0, 0, 1.]])
    f = Frame(0, np.full((12, 16), 2., np.float32), K, np.eye(4))
    g.integrate(f.depth, K, f.cam_to_world)
    state = g.classify()
    result = audit_ray_evidence(g, state, [f], .1, 4., sample_count=128)
    assert result['sampled_cells'] > 50
    assert result['free_count_mismatches'] == result['hit_count_mismatches'] == 0
    cell = tuple(result['sample_cell_indices'][0])
    g.n_free[cell] += 1
    result = audit_ray_evidence(g, state, [f], .1, 4., sample_count=128)
    assert result['free_count_mismatches'] >= 1
