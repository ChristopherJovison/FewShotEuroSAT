from __future__ import annotations

import unittest

import pandas as pd

from src.tables import export_fewshot_table, export_main_table


class TableTests(unittest.TestCase):
    def test_table_exports(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            df = pd.DataFrame({
                "Model": ["Zero-shot CLIP ViT-B/16", "CLIP ViT-B/16 + GDA"],
                "method": ["zeroshot_clip", "clip_gda"],
                "Shots": [1, 1],
                "Top-1 Acc": [0.4, 0.5],
                "Top-5 Acc": [0.7, 0.8],
                "Macro F1": [0.3, 0.45],
                "Weighted F1": [0.35, 0.48],
            })
            main = export_main_table(df, tmp_path)
            few = export_fewshot_table(df, tmp_path)
            self.assertTrue((tmp_path / "main_results_table.csv").exists())
            self.assertTrue((tmp_path / "main_results_table.tex").exists())
            self.assertTrue((tmp_path / "fewshot_ablation_table.csv").exists())
            self.assertEqual(len(main), 2)
            self.assertEqual(few.iloc[0]["Shots"], 1)
