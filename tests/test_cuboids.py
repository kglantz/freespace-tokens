"""Independent exhaustive voxel checks for lossless region construction."""
from itertools import permutations, product

import numpy as np
import pytest

from volumetric_reasoning.geometry.cuboids import (Cuboids, compress_states, partition_mask,
                                                   validate_partition, validate_states)
from volumetric_reasoning.geometry.tsdf import FREE, OCCUPIED, UNKNOWN


def assert_exact_mask(mask, start, stop):
    # Enumerate integer cells, independently of the production slice validator.
    covered = set()
    for a, b in zip(start, stop):
        assert np.all(a >= 0) and np.all(b <= mask.shape) and np.all(b > a)
        for cell in product(*(range(int(x), int(y)) for x, y in zip(a, b))):
            assert mask[cell] and cell not in covered
            covered.add(cell)
    assert covered == set(map(tuple, np.argwhere(mask)))


@pytest.mark.parametrize('order', list(permutations(range(3))))
def test_all_binary_two_cube_patterns_and_axis_orders(order):
    for bits in product((False, True), repeat=8):
        mask = np.array(bits).reshape(2, 2, 2)
        start, stop = partition_mask(mask, order)
        assert_exact_mask(mask, start, stop)


@pytest.mark.parametrize('order', list(permutations(range(3))))
def test_axis_mapping_and_metric_half_open_extents(order):
    mask = np.zeros((5, 7, 9), bool)
    mask[1:3, 2:5, 3:7] = True
    a, b = partition_mask(mask, order)
    np.testing.assert_array_equal(a, [[1, 2, 3]])
    np.testing.assert_array_equal(b, [[3, 5, 7]])
    c, e = Cuboids(a, b, np.array([FREE])).descriptors([-2., 3., .5], .02)
    np.testing.assert_allclose(c, [[-1.96, 3.07, .60]], atol=1e-14)
    np.testing.assert_allclose(e, [[.04, .06, .08]], atol=1e-14)
    assert_exact_mask(mask, a, b)


@pytest.mark.parametrize('shape', [(1, 1, 1), (1, 7, 3), (7, 1, 3), (7, 3, 1), (5, 7, 9)])
@pytest.mark.parametrize('label', [UNKNOWN, FREE, OCCUPIED])
def test_uniform_empty_and_singleton_dimensions(shape, label):
    s = np.full(shape, label, np.uint8)
    boxes, _ = compress_states(s)
    assert len(boxes.state) == (0 if label == OCCUPIED else 1)
    assert validate_partition(s, boxes)['passed']


def test_random_three_state_grids_exact_and_deterministic():
    rng = np.random.default_rng(907)
    for _ in range(20):
        s = rng.integers(0, 3, size=tuple(rng.integers(1, 8, 3)), dtype=np.uint8)
        original = s.copy()
        boxes, choices = compress_states(s)
        for code, name in ((FREE, 'free'), (UNKNOWN, 'unknown')):
            keep = boxes.state == code
            assert_exact_mask(s == code, boxes.start[keep], boxes.stop[keep])
            assert int(keep.sum()) == min(c['cuboids'] for c in choices[name]['candidates'])
        assert validate_partition(s, boxes)['passed']
        again, choices_again = compress_states(s)
        for field in ('start', 'stop', 'state'):
            np.testing.assert_array_equal(getattr(boxes, field), getattr(again, field))
        np.testing.assert_array_equal(s, original)
        assert choices == choices_again


def test_thin_occupied_wall_unknown_hole_and_diagonal_cells_are_not_bridged():
    s = np.full((7, 9, 11), FREE, np.uint8)
    s[3] = OCCUPIED
    s[1, 2, 3] = s[2, 3, 4] = UNKNOWN
    boxes, _ = compress_states(s)
    assert validate_partition(s, boxes)['passed']
    for code in (FREE, UNKNOWN):
        k = boxes.state == code
        assert_exact_mask(s == code, boxes.start[k], boxes.stop[k])
    assert (boxes.state == UNKNOWN).sum() == 2


def test_checkerboard_requires_single_cell_regions():
    s = (np.indices((4, 5, 6)).sum(axis=0) % 2).astype(np.uint8)
    boxes, _ = compress_states(s)
    assert len(boxes.state) == s.size
    assert np.all(boxes.stop - boxes.start == 1)
    assert validate_partition(s, boxes)['passed']


def test_validator_detects_missing_overlap_wrong_state_and_surface_inclusion():
    s = np.array([FREE, FREE, UNKNOWN, OCCUPIED], np.uint8).reshape(4, 1, 1)
    valid = Cuboids(np.array([[0, 0, 0], [2, 0, 0]]), np.array([[2, 1, 1], [3, 1, 1]]), np.array([FREE, UNKNOWN]))
    assert validate_partition(s, valid)['passed']
    missing = Cuboids(valid.start[:1], valid.stop[:1], valid.state[:1])
    assert validate_partition(s, missing)['missing_free_or_unknown_cells'] == 1
    overlap = Cuboids(np.r_[valid.start, valid.start[:1]], np.r_[valid.stop, valid.stop[:1]], np.r_[valid.state, FREE])
    assert validate_partition(s, overlap)['overlapping_cells'] == 2
    wrong = Cuboids(valid.start, valid.stop, np.array([UNKNOWN, FREE]))
    assert validate_partition(s, wrong)['roundtrip_state_mismatches'] == 3
    crossed = Cuboids(np.array([[0, 0, 0]]), np.array([[4, 1, 1]]), np.array([FREE]))
    assert validate_partition(s, crossed)['occupied_cells_in_cuboids'] == 1
    assert validate_partition(s, crossed)['nonhomogeneous_cuboids'] == 1
    for corrupt in (missing, overlap, wrong, crossed):
        assert not validate_partition(s, corrupt)['passed']


@pytest.mark.parametrize('state', [np.zeros((2, 2)), np.zeros((0, 2, 2), np.uint8),
                                  np.full((2, 2, 2), 3), np.full((2, 2, 2), -1),
                                  np.zeros((2, 2, 2), float)])
def test_invalid_states_rejected(state):
    with pytest.raises(ValueError):
        validate_states(state)


@pytest.mark.parametrize('start,stop,labels', [([[-1, 0, 0]], [[1, 1, 1]], [FREE]),
    ([[0, 0, 0]], [[3, 1, 1]], [FREE]), ([[0, 0, 0]], [[0, 1, 1]], [FREE]),
    ([[0, 0, 0]], [[1, 1, 1]], [OCCUPIED]), ([[0., 0, 0]], [[1, 1, 1]], [FREE])])
def test_invalid_boxes_rejected_before_numpy_can_clip_or_wrap(start, stop, labels):
    with pytest.raises(ValueError):
        validate_partition(np.zeros((2, 2, 2), np.uint8),
                           Cuboids(np.array(start), np.array(stop), np.array(labels)))
