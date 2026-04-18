from __future__ import annotations

import unittest

import numpy as np

from src.metrics import compute_metrics, predictions_frame


class MetricsTests(unittest.TestCase):
    def test_metrics_for_perfect_predictions(self):
        logits = np.array([[4, 1, 0], [0, 5, 1], [0, 1, 7]], dtype=float)
        targets = np.array([0, 1, 2])
        metrics, per_class, cm = compute_metrics(logits, targets, ["a", "b", "c"])
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(per_class["accuracy"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(cm.values.trace(), 3)

    def test_predictions_frame_contains_top5_fields(self):
        logits = np.eye(6)
        targets = np.arange(6)
        df = predictions_frame(logits, targets, list(range(6)), [f"p{i}" for i in range(6)], [str(i) for i in range(6)])
        self.assertIn("top5_predictions", df.columns)
        self.assertTrue(df["correct"].all())
