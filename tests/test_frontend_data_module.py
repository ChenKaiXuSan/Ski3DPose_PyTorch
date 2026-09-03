import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from dual2pose.dataloader.frontend_pose_data import FrontEndDataModule, MixedFrontEndDataset
from tests.test_frontend_pose_data import BaseDataset


class FakeDataModule:
    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        self.train_gait_dataset = BaseDataset()
        self.val_gait_dataset = BaseDataset()
        self.test_gait_dataset = BaseDataset()

    def train_dataloader(self):
        return DataLoader(self.train_gait_dataset, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.val_gait_dataset, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test_gait_dataset, batch_size=1)


class FrontEndDataModuleTest(unittest.TestCase):
    """Break caught: mixed training validates on only one front end."""

    def test_mixed_validation_uses_same_balanced_sources_as_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for action in ("action_0", "action_1"):
                for camera in ("cam_a", "cam_b"):
                    pose = root / f"{action}_{camera}.npy"
                    np.save(pose, np.ones((2, 15, 3), np.float32))
                    entries.append(
                        {
                            "person_id": "female",
                            "action_id": action,
                            "camera_id": camera,
                            "pose_path": pose.name,
                        }
                    )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"frontend_name": "video", "entries": entries}),
                encoding="utf-8",
            )
            wrapped = FrontEndDataModule(
                FakeDataModule(),
                train_manifest=None,
                val_manifest=None,
                test_manifest=None,
                mixed_train_sources=[None, manifest],
                mixed_val_sources=[None, manifest],
            )
            wrapped.setup("fit")
        self.assertIsInstance(wrapped.base_dm.train_gait_dataset, MixedFrontEndDataset)
        self.assertIsInstance(wrapped.base_dm.val_gait_dataset, MixedFrontEndDataset)
        self.assertEqual(len(wrapped.base_dm.train_gait_dataset), 4)
        self.assertEqual(len(wrapped.base_dm.val_gait_dataset), 4)


if __name__ == "__main__":
    unittest.main()
