from __future__ import annotations

import unittest

import numpy as np

from src.gda import choose_alpha, ensemble_logits, fit_gda


class GDATests(unittest.TestCase):
    def test_gda_shapes_and_logits_are_finite(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(12, 5))
        targets = np.repeat(np.arange(3), 4)
        clf = fit_gda(features, targets, num_classes=3, ridge=0.1)
        logits = clf.logits(features)
        self.assertEqual(clf.means.shape, (3, 5))
        self.assertEqual(clf.precision.shape, (5, 5))
        self.assertEqual(logits.shape, (12, 3))
        self.assertTrue(np.isfinite(logits).all())

    def test_alpha_search_returns_grid_value(self):
        clip = np.array([[3, 1], [1, 3], [3, 1]], dtype=float)
        gda = np.array([[1, 3], [3, 1], [1, 3]], dtype=float)
        y = np.array([0, 1, 1])
        alpha, scores = choose_alpha(clip, gda, y, [0.0, 0.5, 1.0])
        self.assertIn(alpha, {0.0, 0.5, 1.0})
        self.assertEqual(set(scores), {"0.000", "0.500", "1.000"})
        self.assertEqual(ensemble_logits(clip, gda, alpha).shape, (3, 2))
