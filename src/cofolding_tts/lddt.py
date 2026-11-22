"""lDDT-PLI: the lDDT variant computed on protein-ligand atom pairs.

Implemented to Park et al.'s stated definition: pairs whose *reference* distance is
within 15 A, superposition-free, scored as the fraction of inter-atomic distances
preserved within {0.5, 1, 2, 4} A and averaged over the four tolerances, on
full-atom predictions.

Two properties matter for how it is read here.

It is **superposition-free**, so unlike ligand RMSD it needs no protein alignment
and cannot be flattered or punished by one.

It is **contact-based**, which makes it blind to a particular kind of error. A
ligand that sits in the right pocket the wrong way round keeps most of the same
atoms near most of the same residues, so the preserved-distance fraction barely
moves. On 8SGE the two axial rotamers overlap almost completely on this metric --
AlphaFold3's crystal-rotamer poses span 0.23 to 0.88 -- because having the right
axial sense does not mean being in the right place. That is a property of the
metric, not a bug here, and it is the reason the axial torsion is the primary
label for 8SGE.

Separating the rotamers is not the same question as separating *hits*, and on the
latter the metric does well: a single cut separates RMSD < 2 A from the rest on 284
of the 300 cross-model 8SGE poses (`youden_threshold`). That buys nothing as a
predictor, because the pair set is chosen on the reference distances, so computing
lDDT-PLI at all requires the deposited complex this score is meant to stand in for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

__all__ = ["INCLUSION_RADIUS", "TOLERANCES", "Separation", "lddt_from_distances",
           "lddt_pli", "youden_threshold"]

INCLUSION_RADIUS = 15.0                      # A
TOLERANCES: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)   # A


def lddt_from_distances(
    reference: np.ndarray, predicted: np.ndarray,
    tolerances: Sequence[float] = TOLERANCES,
) -> tuple[float, int]:
    """Mean over tolerances of the fraction of distances preserved within each.

    Both arrays hold distances for the same, already-filtered, set of pairs.
    """

    reference = np.asarray(reference, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    if reference.shape != predicted.shape:
        raise ValueError(f"{reference.size} reference distances vs {predicted.size} predicted")
    if reference.size == 0:
        return float("nan"), 0
    error = np.abs(predicted - reference)
    return float(np.mean([(error < t).mean() for t in tolerances])), int(reference.size)


def lddt_pli(
    reference_protein: np.ndarray, reference_ligand: np.ndarray,
    predicted_protein: np.ndarray, predicted_ligand: np.ndarray,
    *, inclusion_radius: float = INCLUSION_RADIUS,
    tolerances: Sequence[float] = TOLERANCES,
) -> tuple[float, int]:
    """lDDT-PLI for one pose.

    The four arrays must be in corresponding order: protein atom i is the same atom
    in both protein arrays, ligand atom j the same in both ligand arrays. The pair
    set is selected on the REFERENCE distances, as the definition specifies, so the
    same pairs are scored no matter how wrong the prediction is.
    """

    reference_distances = np.linalg.norm(
        np.asarray(reference_protein)[:, None, :] - np.asarray(reference_ligand)[None, :, :],
        axis=2)
    predicted_distances = np.linalg.norm(
        np.asarray(predicted_protein)[:, None, :] - np.asarray(predicted_ligand)[None, :, :],
        axis=2)
    if reference_distances.shape != predicted_distances.shape:
        raise ValueError("reference and predicted distance matrices differ in shape")
    included = reference_distances <= inclusion_radius
    return lddt_from_distances(
        reference_distances[included], predicted_distances[included], tolerances)


class Separation(NamedTuple):
    """How well one cut on a score separates labelled positives from negatives."""

    cut: float
    youden_j: float          # sensitivity + specificity - 1, in [0, 1]
    accepted_negatives: int  # negatives at or above the cut
    rejected_positives: int  # positives below it


def youden_threshold(scores, labels) -> Separation:
    """The cut on `scores` that best separates `labels`, by Youden's J.

    A sample is accepted when ``score >= cut``; candidate cuts are the observed
    scores, so the returned cut is always attained by a sample. Ties go to the
    lowest cut, which is the most inclusive one.

    This answers "how separable are these two sets, at best", not "what cut should
    be used on new data" -- the cut is chosen on the same labels it is scored
    against and is optimistic by construction. For lDDT-PLI in particular it is a
    descriptive statistic and not a selection rule, since the metric needs the
    reference complex.
    """

    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"{scores.size} scores vs {labels.size} labels")
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        raise ValueError(f"need both classes, got {n_pos} positive and {n_neg} negative")

    best = None
    for cut in np.unique(scores):
        accepted = scores >= cut
        tp = int((accepted & labels).sum())
        fp = int((accepted & ~labels).sum())
        j = tp / n_pos - fp / n_neg
        if best is None or j > best.youden_j:
            best = Separation(float(cut), j, fp, n_pos - tp)
    return best
