"""Focused tests for protein-aligned ligand pose metrics."""

import unittest

import numpy as np

from cofolding_tts.pose import (
    apply_transform,
    kabsch,
    minimum_mapped_rmsd,
)


class PoseMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mobile = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        )

    def test_kabsch_recovers_proper_rigid_transform(self) -> None:
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([4.0, -2.0, 1.5])
        target = apply_transform(self.mobile, rotation, translation)
        fitted_rotation, fitted_translation, fitted_rmsd = kabsch(self.mobile, target)
        self.assertAlmostEqual(fitted_rmsd, 0.0, places=12)
        np.testing.assert_allclose(
            apply_transform(self.mobile, fitted_rotation, fitted_translation),
            target,
            atol=1e-12,
        )
        self.assertGreater(np.linalg.det(fitted_rotation), 0.0)

    def test_fixed_transform_does_not_refit_ligand(self) -> None:
        shifted = self.mobile + np.array([5.0, 0.0, 0.0])
        value, _ = minimum_mapped_rmsd(
            shifted,
            self.mobile,
            [tuple(range(len(self.mobile)))],
            rotation=np.eye(3),
            translation=np.zeros(3),
        )
        self.assertAlmostEqual(value, 5.0)

    def test_symmetry_mapping_selects_lower_unfitted_rmsd(self) -> None:
        mobile = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        target = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        value, mapping = minimum_mapped_rmsd(mobile, target, [(0, 1), (1, 0)])
        self.assertAlmostEqual(value, 0.0)
        self.assertEqual(mapping, (1, 0))


if __name__ == "__main__":
    unittest.main()
