from __future__ import annotations

import unittest

from src.fewshot import flatten_support_indices, sample_fewshot_indices


class TinyDataset:
    def __init__(self):
        self.targets = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {"target": self.targets[idx]}


class FewshotTests(unittest.TestCase):
    def test_fewshot_sampling_is_reproducible(self):
        dataset = TinyDataset()
        a = sample_fewshot_indices(dataset, shots=2, seed=7)
        b = sample_fewshot_indices(dataset, shots=2, seed=7)
        self.assertEqual(a, b)
        self.assertTrue(all(len(v) == 2 for v in a.values()))
        self.assertEqual(len(flatten_support_indices(a)), 6)
