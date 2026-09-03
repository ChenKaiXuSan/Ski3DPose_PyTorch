from pathlib import Path
import sys
import unittest
from unittest.mock import patch

DUAL2POSE_ROOT = Path(__file__).resolve().parents[1] / "dual2pose"
if str(DUAL2POSE_ROOT) not in sys.path:
    sys.path.insert(0, str(DUAL2POSE_ROOT))

from dual2pose.dataloader.data_loader import UnityDataModule


class TrainingPathRewriteTest(unittest.TestCase):
    """Break caught: fold files load but still point at the retired data root."""

    def test_prepare_data_rewrites_every_nested_index_path(self) -> None:
        module = UnityDataModule.__new__(UnityDataModule)
        module._index_mapping = Path("fold_01.json")
        module._index_path_rewrite_from = "/old/unity"
        module._data_root = "/new/unity"
        fixture = {
            "train": [{"pose": "/old/unity/a.npy", "nested": {"mask": "/old/unity/m.png"}}],
            "val": [],
            "test": [],
        }
        with patch(
            "dual2pose.dataloader.data_loader.load_index_mapping",
            return_value=fixture,
        ):
            module.prepare_data()

        self.assertEqual(module._dataset_idx["train"][0]["pose"], "/new/unity/a.npy")
        self.assertEqual(
            module._dataset_idx["train"][0]["nested"]["mask"],
            "/new/unity/m.png",
        )


if __name__ == "__main__":
    unittest.main()
