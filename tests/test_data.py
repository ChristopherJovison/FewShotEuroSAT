from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import build_datasets, humanize_folder_name


class ImageFolderDatasetTests(unittest.TestCase):
    def test_imagefolder_loader_uses_configured_class_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "imagefolder"
            for split in ("train", "val", "test"):
                for folder in ("AnnualCrop", "SeaLake"):
                    class_dir = root / split / folder
                    class_dir.mkdir(parents=True)
                    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(class_dir / "sample.jpg")

            config = {
                "experiment": {"data_root": "unused"},
                "dataset": {
                    "type": "imagefolder",
                    "root": str(root),
                    "class_names": ["annual crop land", "sea or lake"],
                },
            }
            train_data, val_data, test_data, class_names = build_datasets(config)

            self.assertEqual(class_names, ["annual crop land", "sea or lake"])
            self.assertEqual(train_data.targets, [0, 1])
            self.assertEqual(len(val_data), 2)
            self.assertEqual(test_data[1]["class_name"], "sea or lake")

    def test_humanize_folder_name_handles_camel_case(self):
        self.assertEqual(humanize_folder_name("HerbaceousVegetation"), "herbaceous vegetation")


if __name__ == "__main__":
    unittest.main()
