import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

from dual2pose.dataloader.frontend_pose_data import MixedFrontEndDataset
from dual2pose.eval.frontend_manifest import FrontEndManifest


class BaseDataset(Dataset):
    def __init__(self, length: int = 2) -> None:
        self._index_mapping = [
            {
                "person_id": "female",
                "action_id": f"action_{index}",
                "cam1_id": "cam_a",
                "cam2_id": "cam_b",
            }
            for index in range(length)
        ]

    def __len__(self) -> int:
        return len(self._index_mapping)

    def __getitem__(self, index: int) -> dict:
        meta = dict(self._index_mapping[index])
        return {
            "meta": meta,
            "frame_indices": torch.tensor([0, 1]),
            "kpt3d_sam": {
                "cam1": torch.zeros((2, 15, 3)),
                "cam2": torch.zeros((2, 15, 3)),
            },
        }


class FrontEndPoseDataTest(unittest.TestCase):
    """Breaks caught: mixed training is unbalanced or mutates native SAM3D samples."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _manifest(self, name: str, value: float) -> FrontEndManifest:
        entries = []
        for action in ("action_0", "action_1"):
            for camera in ("cam_a", "cam_b"):
                pose = self.root / f"{name}_{action}_{camera}.npy"
                np.save(pose, np.full((2, 15, 3), value, np.float32))
                entries.append(
                    {
                        "person_id": "female",
                        "action_id": action,
                        "camera_id": camera,
                        "pose_path": pose.name,
                    }
                )
        path = self.root / f"{name}.json"
        path.write_text(
            json.dumps({"frontend_name": name, "entries": entries}), encoding="utf-8"
        )
        return FrontEndManifest.load(path)

    def test_mixed_dataset_contains_each_source_once_per_base_sample(self) -> None:
        base = BaseDataset()
        video = self._manifest("videopose3d", 1.0)
        poseformer = self._manifest("poseformer", 2.0)
        motionbert = self._manifest("motionbert", 3.0)
        mixed = MixedFrontEndDataset(base, [None, video, poseformer, motionbert])
        self.assertEqual(len(mixed), 4 * len(base))
        self.assertEqual(
            [mixed[index * len(base)]["_frontend_name"] for index in range(4)],
            ["sam3d", "videopose3d", "poseformer", "motionbert"],
        )

    def test_frontend_replacement_does_not_mutate_base_sample(self) -> None:
        base = BaseDataset(length=1)
        video = self._manifest("videopose3d", 5.0)
        mixed = MixedFrontEndDataset(base, [None, video])
        native = mixed[0]
        replaced = mixed[len(base)]
        self.assertTrue(torch.equal(native["kpt3d_sam"]["cam1"], torch.zeros((2, 15, 3))))
        self.assertTrue(torch.equal(replaced["kpt3d_sam"]["cam1"], torch.full((2, 15, 3), 5.0)))
        self.assertTrue(torch.equal(base[0]["kpt3d_sam"]["cam1"], torch.zeros((2, 15, 3))))

    def test_manifest_coverage_is_validated_at_construction(self) -> None:
        base = BaseDataset()
        incomplete_path = self.root / "incomplete.npy"
        np.save(incomplete_path, np.zeros((2, 15, 3), np.float32))
        manifest_path = self.root / "incomplete.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "frontend_name": "incomplete",
                    "entries": [
                        {
                            "person_id": "female",
                            "action_id": "action_0",
                            "camera_id": "cam_a",
                            "pose_path": incomplete_path.name,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(KeyError, "missing"):
            MixedFrontEndDataset(base, [FrontEndManifest.load(manifest_path)])


if __name__ == "__main__":
    unittest.main()
