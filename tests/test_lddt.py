"""lDDT-PLI arithmetic: the tolerance average and the reference-defined pair set."""

from __future__ import annotations

import numpy as np
import pytest

from cofolding_tts.lddt import (INCLUSION_RADIUS, TOLERANCES, lddt_from_distances,
                                lddt_pli, youden_threshold)


def test_identical_distances_score_one():
    d = np.array([1.0, 4.0, 9.0])
    assert lddt_from_distances(d, d) == (1.0, 3)


def test_grossly_wrong_distances_score_zero():
    value, n = lddt_from_distances(np.zeros(5), np.full(5, 100.0))
    assert value == 0.0 and n == 5


def test_score_is_the_mean_over_the_four_tolerances():
    # One pair off by 1.5 A: inside the 2 and 4 tolerances, outside 0.5 and 1.
    value, n = lddt_from_distances(np.array([5.0]), np.array([6.5]))
    assert n == 1
    assert value == pytest.approx(0.5)


def test_tolerance_boundary_is_strict():
    # Exactly at a tolerance counts as outside it, matching `error < t`.
    at, _ = lddt_from_distances(np.array([0.0]), np.array([0.5]))
    just_inside, _ = lddt_from_distances(np.array([0.0]), np.array([0.499]))
    assert at == pytest.approx(0.75) and just_inside == pytest.approx(1.0)


def test_empty_pair_set_is_nan_not_zero():
    value, n = lddt_from_distances(np.array([]), np.array([]))
    assert n == 0 and np.isnan(value)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        lddt_from_distances(np.zeros(3), np.zeros(4))


def test_pair_set_comes_from_the_reference_not_the_prediction():
    """A pose that flees the pocket is still scored on the pairs it should have."""

    protein = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ligand = np.array([[2.0, 0.0, 0.0]])
    far = ligand + np.array([[500.0, 0.0, 0.0]])
    value, n = lddt_pli(protein, ligand, protein, far)
    # Both reference pairs are within 15 A, so both are scored, and both are wrong.
    assert n == 2 and value == 0.0


def test_pairs_beyond_the_inclusion_radius_are_dropped():
    protein = np.array([[0.0, 0.0, 0.0]])
    near, far = np.array([[1.0, 0.0, 0.0]]), np.array([[100.0, 0.0, 0.0]])
    ligand = np.vstack([near, far])
    _, n = lddt_pli(protein, ligand, protein, ligand)
    assert n == 1


def test_published_parameters_are_unchanged():
    assert INCLUSION_RADIUS == 15.0
    assert TOLERANCES == (0.5, 1.0, 2.0, 4.0)


def test_a_separable_score_is_cut_at_the_lowest_positive():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    sep = youden_threshold(scores, np.array([False, False, True, True]))
    assert sep.cut == 0.8
    assert sep.youden_j == 1.0
    assert (sep.accepted_negatives, sep.rejected_positives) == (0, 0)


def test_the_cut_is_an_observed_score_and_ties_go_low():
    # 0.5 and 0.6 both separate perfectly; the lower, more inclusive cut wins, and
    # it is a value that actually occurs rather than a midpoint.
    scores = np.array([0.1, 0.5, 0.6])
    sep = youden_threshold(scores, np.array([False, True, True]))
    assert sep.cut == 0.5


def test_overlapping_classes_are_charged_for_both_kinds_of_error():
    #                     neg   neg   pos   neg   pos
    scores = np.array([0.1, 0.4, 0.3, 0.6, 0.9])
    labels = np.array([False, False, True, False, True])
    sep = youden_threshold(scores, labels)
    # Best is 0.9: one positive kept, no negative admitted, J = 1/2 - 0/3.
    assert sep.cut == 0.9
    assert sep.accepted_negatives == 0 and sep.rejected_positives == 1
    assert sep.youden_j == pytest.approx(0.5)


def test_a_score_that_is_backwards_separates_nothing():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    sep = youden_threshold(scores, np.array([False, False, True, True]))
    # Accepting everything is the best an ascending cut can do here: J = 0.
    assert sep.youden_j == 0.0
    assert sep.cut == 0.1


def test_one_class_only_is_an_error():
    with pytest.raises(ValueError, match="need both classes"):
        youden_threshold(np.array([0.1, 0.2]), np.array([True, True]))


def test_mismatched_lengths_are_an_error():
    with pytest.raises(ValueError, match="2 scores vs 3 labels"):
        youden_threshold(np.array([0.1, 0.2]), np.array([True, False, True]))
