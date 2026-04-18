from __future__ import annotations

import unittest

import numpy as np

from src.tip_adapter import choose_three_way_fusion, combine_tip_logits, prototype_logits, tip_adapter_logits
from src.tip_adapter import choose_tip_hyperparams, fuse_three_way_logits


class TipAdapterTests(unittest.TestCase):
    def test_tip_adapter_logits_shape_and_finiteness(self):
        query = np.array([[1.0, 0.0], [0.0, 1.0]])
        support = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        support_targets = np.array([0, 1, 1])
        logits = tip_adapter_logits(query, support, support_targets, num_classes=2, beta=2.0)
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(np.isfinite(logits).all())
        self.assertGreater(logits[0, 0], logits[0, 1])

    def test_tip_hyperparameter_search_returns_grid_values(self):
        clip = np.array([[2.0, 0.0], [0.0, 2.0]])
        val_features = np.array([[1.0, 0.0], [0.0, 1.0]])
        val_targets = np.array([0, 1])
        support_features = val_features.copy()
        support_targets = val_targets.copy()
        alpha, beta, scores = choose_tip_hyperparams(
            clip,
            val_features,
            val_targets,
            support_features,
            support_targets,
            num_classes=2,
            alpha_grid=[0.1, 1.0],
            beta_grid=[1.0, 5.0],
        )
        self.assertIn(alpha, {0.1, 1.0})
        self.assertIn(beta, {1.0, 5.0})
        self.assertEqual(len(scores), 4)
        self.assertEqual(combine_tip_logits(clip, clip, alpha).shape, (2, 2))

    def test_prototype_fusion_shapes(self):
        query = np.array([[1.0, 0.0], [0.0, 1.0]])
        support = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        support_targets = np.array([0, 0, 1, 1])
        proto = prototype_logits(query, support, support_targets, num_classes=2)
        self.assertEqual(proto.shape, (2, 2))
        self.assertGreater(proto[0, 0], proto[0, 1])
        weights, scores = choose_three_way_fusion(proto, proto, proto, np.array([0, 1]), [0.0, 1.0])
        fused = fuse_three_way_logits(proto, proto, proto, weights)
        self.assertEqual(fused.shape, (2, 2))
        self.assertTrue(scores)
