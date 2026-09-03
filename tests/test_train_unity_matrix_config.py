import json
from pathlib import Path
import sys
import tempfile
import unittest

from omegaconf import OmegaConf
from torch.utils.data import TensorDataset
import torch

DUAL2POSE_ROOT = Path(__file__).resolve().parents[1] / "dual2pose"
if str(DUAL2POSE_ROOT) not in sys.path:
    sys.path.insert(0, str(DUAL2POSE_ROOT))

from dual2pose.dataloader.data_loader import UnityDataModule
from dual2pose.train_unity import resolve_fold_index_path, validate_fold_metadata


class TrainUnityMatrixConfigTest(unittest.TestCase):
    """Breaks caught: matrix runs silently reuse seed 42 or fold 0."""

    def test_default_config_declares_seed(self) -> None:
        config = OmegaConf.load("configs/dual2pose.yaml")
        self.assertEqual(int(config.train.seed), 42)
        self.assertEqual(str(config.train.test_ckpt_path), "best")

    def test_fold_one_resolves_fold_01(self) -> None:
        actual = resolve_fold_index_path(Path("/data"), 1)
        self.assertEqual(actual.name, "fold_01.json")
        self.assertEqual(
            actual.parent.name,
            "camera_pairs_by_action_folds",
        )

    def test_unsupported_fold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "available folds are 0 and 1"):
            resolve_fold_index_path(Path("/data"), 2)

    def test_metadata_fold_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold_01.json"
            path.write_text(
                json.dumps({"train": [], "val": [], "test": [], "_metadata": {"fold_idx": 0}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "metadata fold 0"):
                validate_fold_metadata(path, fold=1)

    def test_validation_and_test_loaders_keep_partial_batch(self) -> None:
        module = UnityDataModule.__new__(UnityDataModule)
        module._batch_size = 4
        module._num_workers = 0
        module.val_gait_dataset = TensorDataset(torch.arange(10))
        module.test_gait_dataset = TensorDataset(torch.arange(10))

        val_loader = module.val_dataloader()
        test_loader = module.test_dataloader()

        self.assertFalse(val_loader.drop_last)
        self.assertFalse(test_loader.drop_last)
        self.assertEqual([len(batch[0]) for batch in val_loader], [4, 4, 2])
        self.assertEqual([len(batch[0]) for batch in test_loader], [4, 4, 2])


if __name__ == "__main__":
    unittest.main()
