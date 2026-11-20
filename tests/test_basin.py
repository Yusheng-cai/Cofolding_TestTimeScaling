"""The basin criterion: the one number every arm is scored by."""
from __future__ import annotations

import numpy as np
import pytest

from cofolding_tts import basin


def test_the_two_centres_classify_as_themselves():
    assert basin.classify([basin.CORRECT_CENTRE])[0] == "correct"
    assert basin.classify([basin.WRONG_CENTRE])[0] == "wrong"


def test_the_deposited_torsion_is_in_the_correct_basin():
    assert basin.classify([basin.CRYSTAL_TORSION])[0] == "correct"


def test_a_pose_between_the_basins_is_neither():
    """-20 deg is roughly the measured dividing surface: nearer to nothing."""
    assert basin.classify([-20.0])[0] == "neither"


def test_the_right_sign_is_not_enough():
    """-180 deg has the crystal's sign but is 67 deg from its centre. Counting it as a
    rescue is exactly the error this module exists to prevent."""
    assert basin.classify([-180.0])[0] == "neither"


def test_wrap_is_used_so_the_circle_closes():
    assert basin.separation(179.0, -179.0) == pytest.approx(2.0)
    assert basin.classify([-175.0])[0] == basin.classify([185.0])[0]


def test_counts_sum_to_the_input_length():
    angles = np.linspace(-180, 180, 73)
    assert sum(basin.counts(angles).values()) == len(angles)


def test_circular_statistics_handle_the_wrap():
    # the mean of 179 and -179 is the antimeridian; +180 and -180 are the same angle
    assert basin.separation(basin.circular_mean([179.0, -179.0]), 180.0) == pytest.approx(0.0, abs=1e-6)
    # identical angles: zero spread, up to the float error in |resultant| = 1
    assert basin.circular_sd([10.0, 10.0, 10.0]) == pytest.approx(0.0, abs=1e-4)
    assert basin.circular_sd(np.linspace(-180, 180, 360)) > 100.0
