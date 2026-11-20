"""Which basin a pose landed in, and the circular statistics that go with it.

ONE criterion, in one place. It was previously defined inside an experiment script,
which is how two parts of the same result came to disagree: a pose at -180 deg is on
the correct SIDE of zero but is 67 deg from the correct basin's centre, and counting
it as a rescue flatters the arm that produced it.

    correct   within BASIN_HALFWIDTH of the crystal rotamer's centre, and nearer to it
    wrong     within BASIN_HALFWIDTH of the model's rotamer, and nearer to it
    neither   further than BASIN_HALFWIDTH from both -- a pose that committed to no
              rotamer at all, which is a failure of a different kind and is reported
              separately rather than folded into one of the other two

Centres are measured, not assumed: 260 poses, zero of them between the basins. The
half-width is half the 178.7 deg separation less a margin.

numpy only, so this stays importable in the inspect tier.
"""
from __future__ import annotations

import numpy as np

CORRECT_CENTRE = -112.6     # the crystal's rotamer
WRONG_CENTRE = 68.7         # the one Boltz-2 builds
BASIN_HALFWIDTH = 40.0      # degrees
CRYSTAL_TORSION = -101.4    # as deposited


def wrap(degrees):
    """Fold an angular difference into (-180, 180]."""
    return (np.asarray(degrees, dtype=float) + 180.0) % 360.0 - 180.0


def separation(a, b):
    """Absolute angular distance in degrees, the short way round."""
    return np.abs(wrap(np.asarray(a, dtype=float) - b))


def classify(torsion_degrees):
    """-> array of 'correct' | 'wrong' | 'neither', one per torsion."""
    to_correct = separation(torsion_degrees, CORRECT_CENTRE)
    to_wrong = separation(torsion_degrees, WRONG_CENTRE)
    out = np.where(to_correct < to_wrong, "correct", "wrong")
    return np.where(np.minimum(to_correct, to_wrong) > BASIN_HALFWIDTH, "neither", out)


def counts(torsion_degrees) -> dict[str, int]:
    """{'correct': n, 'wrong': n, 'neither': n} -- the three always sum to len()."""
    labels = classify(torsion_degrees)
    return {name: int((labels == name).sum()) for name in ("correct", "wrong", "neither")}


def circular_mean(degrees) -> float:
    return float(np.degrees(np.angle(np.exp(1j * np.radians(np.asarray(degrees))).mean())))


def circular_sd(degrees) -> float:
    """Circular SD in degrees; 0 for a single value, large for a scattered set."""
    resultant = abs(np.exp(1j * np.radians(np.asarray(degrees))).mean())
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))))
