"""Small, dependency-light coordinate metrics for pose evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def kabsch(
    mobile: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a proper rotation, translation, and fitted RMSD from mobile to target."""

    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and target must have matching (N, 3) shapes")
    if len(mobile) < 3:
        raise ValueError("at least three coordinate pairs are required")

    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ mobile_center
    aligned = apply_transform(mobile, rotation, translation)
    return rotation, translation, coordinate_rmsd(aligned, target)


def apply_transform(
    coordinates: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply one rigid transform without fitting the supplied coordinates."""

    coordinates = np.asarray(coordinates, dtype=float)
    return (np.asarray(rotation) @ coordinates.T).T + np.asarray(translation)


def coordinate_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    """Return RMS distance between already corresponding coordinates."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("coordinate arrays must have matching (N, 3) shapes")
    if len(first) == 0:
        raise ValueError("coordinate arrays cannot be empty")
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def minimum_mapped_rmsd(
    mobile: np.ndarray,
    target: np.ndarray,
    mappings: Iterable[Sequence[int]],
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    fit_mobile: bool = False,
) -> tuple[float, tuple[int, ...]]:
    """Minimize RMSD over atom mappings, optionally after one fixed transform.

    Each mapping gives the target atom index corresponding to mobile atoms in
    mobile-index order. ``fit_mobile=False`` is essential for protein-aligned
    ligand pose RMSD: no ligand-only fit is then performed.
    """

    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if rotation is not None or translation is not None:
        if rotation is None or translation is None:
            raise ValueError("rotation and translation must be supplied together")
        mobile = apply_transform(mobile, rotation, translation)

    best_value = float("inf")
    best_mapping: tuple[int, ...] | None = None
    for raw_mapping in mappings:
        mapping = tuple(int(index) for index in raw_mapping)
        if len(mapping) != len(mobile):
            raise ValueError("each mapping must contain one target index per mobile atom")
        mapped_target = target[np.asarray(mapping, dtype=int)]
        if fit_mobile:
            _, _, value = kabsch(mobile, mapped_target)
        else:
            value = coordinate_rmsd(mobile, mapped_target)
        if value < best_value:
            best_value = value
            best_mapping = mapping

    if best_mapping is None:
        raise ValueError("at least one atom mapping is required")
    return best_value, best_mapping
