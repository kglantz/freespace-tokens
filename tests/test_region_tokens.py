"""CLI export, schema enforcement, cropping, and scene-slot accounting."""
import hashlib
import json

import numpy as np
import pytest

from volumetric_reasoning import region_tokens
from volumetric_reasoning.geometry.cuboids import Cuboids, validate_partition
from volumetric_reasoning.geometry.tsdf import FREE, OCCUPIED, UNKNOWN, STATE_DEFINITION


def save_source(path, schema=3, definition=STATE_DEFINITION):
    state = np.full((6, 5, 4), FREE, np.uint8)
    state[2] = OCCUPIED
    state[3:] = UNKNOWN
    np.savez_compressed(path, schema_version=schema, state_definition=definition,
                        origin=[-1., 2., .5], dims=state.shape, voxel_size=.1,
                        frame_indices=[3, 19], state=state)
    return state


def test_cli_exports_exact_partition_descriptors_all_ablations_and_preserves_input(tmp_path):
    source = tmp_path / 'volume.npz'
    state = save_source(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    out = tmp_path / 'regions'
    region_tokens.main(['--source-volume', str(source), '--output', str(out), '--scene-token-budget', '21'])
    summary = json.loads((out / 'summary.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['source_sha256'] == before
    assert summary['validation']['passed'] and summary['serialized_descriptors_match'] and summary['surface_cells_preserved']
    assert summary['token_counts'] == {'surface_candidates': 20, 'free_regions': 1, 'unknown_regions': 1,
        'surface_only': 20, 'surface_plus_free': 21, 'surface_plus_unknown': 21, 'surface_plus_both': 22}
    assert summary['fits_scene_token_budget'] is False
    with np.load(out / 'token_geometry.npz', allow_pickle=False) as a:
        assert validate_partition(state, Cuboids(a['region_start'], a['region_stop'], a['region_state']))['passed']
        np.testing.assert_array_equal(a['surface_index'], np.argwhere(state == OCCUPIED))
        np.testing.assert_array_equal(a['frame_indices'], [3, 19])
    assert (out / 'cuboid_slices.png').stat().st_size > 1000
    assert len((out / 'regions.csv').read_text().splitlines()) == 3
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with pytest.raises(SystemExit, match='not an empty'):
        region_tokens.main(['--source-volume', str(source), '--output', str(out)])


def test_crop_retains_existing_lattice_and_surface_labels(tmp_path):
    source = tmp_path / 'volume.npz'
    state = save_source(source)
    out = tmp_path / 'cropped'
    region_tokens.main(['--source-volume', str(source), '--output', str(out),
                        '--bounds', '-.9', '2.1', '.6', '-.6', '2.4', '.8'])
    with np.load(out / 'token_geometry.npz', allow_pickle=False) as a:
        np.testing.assert_allclose(a['origin'], [-.9, 2.1, .6])
        np.testing.assert_array_equal(a['dims'], [3, 3, 2])
        sub = state[1:4, 1:4, 1:3]
        assert validate_partition(sub, Cuboids(a['region_start'], a['region_stop'], a['region_state']))['passed']
        np.testing.assert_array_equal(a['surface_index'], np.argwhere(sub == OCCUPIED))


@pytest.mark.parametrize('schema,definition', [(2, STATE_DEFINITION), (3, 'endpoint_occupancy')])
def test_legacy_or_incompatible_state_definitions_rejected(tmp_path, schema, definition):
    source = tmp_path / 'volume.npz'
    save_source(source, schema, definition)
    with pytest.raises(SystemExit, match='schema-3'):
        region_tokens.main(['--source-volume', str(source), '--output', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()
