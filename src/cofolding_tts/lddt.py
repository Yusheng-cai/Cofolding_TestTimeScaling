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
moves. On 8SGE the two basins overlap almost completely on this metric even though
they are 5 A apart in RMSD. That is a property of the metric, not a bug here, and
it is the reason the axial torsion is the primary label for 8SGE.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["INCLUSION_RADIUS", "TOLERANCES", "lddt_from_distances", "lddt_pli"]

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
