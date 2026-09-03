import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dual2pose.eval.export_unity_frontend_predictions import discover_unity_streams
from dual2pose.experiments.export_frontend_splits import merge_split_manifests


def fold_row(root: str, split: str) -> dict[str, str]:
    return {
        "person_id": "female",
        "action_id": f"action_{split}",
        "joint_names_path": f"{root}/{split}/joint_names.json",
        "sequence_meta_path": f"{root}/{split}/sequence.json",
        "cam1_id": "capture_L0_A000",
        "cam2_id": "capture_L0_A090",
        "cam1_kpt2d_dir": f"{root}/{split}/A000",
        "cam2_kpt2d_dir": f"{root}/{split}/A090",
    }


class FrontEndSplitExportTest(unittest.TestCase):
    """Breaks caught: split exports leak actions or merged entries lose membership."""

    def test_all_split_is_disjoint_union_with_membership(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold = root / "fold.json"
            fold.write_text(
                json.dumps({split: [fold_row("/old", split)] for split in ("train", "val", "test")}),
                encoding="utf-8",
            )
            streams = discover_unity_streams(fold, "all", Path("/old"), root)
        self.assertEqual(len(streams), 6)
        self.assertEqual(
            {split: sum(stream.split == split for stream in streams) for split in ("train", "val", "test")},
            {"train": 2, "val": 2, "test": 2},
        )

    def test_camera_stream_in_two_splits_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = fold_row("/old", "train")
            fold = root / "fold.json"
            fold.write_text(
                json.dumps({"train": [duplicate], "val": [duplicate], "test": [fold_row("/old", "test")]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "assigned to both"):
                discover_unity_streams(fold, "all", Path("/old"), root)

    def test_merge_preserves_split_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = []
            for split in ("train", "val", "test"):
                pose = root / f"{split}.npy"
                np.save(pose, np.zeros((2, 15, 3), np.float32))
                manifest = root / f"{split}.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "frontend_name": "videopose3d",
                            "joint_indices": list(range(15)),
                            "metadata": {"split": split},
                            "entries": [
                                {
                                    "person_id": "female",
                                    "action_id": f"action_{split}",
                                    "camera_id": "capture_L0_A000",
                                    "pose_path": pose.name,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                manifests.append(manifest)
            output = root / "all.json"
            merge_split_manifests(manifests, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["metadata"]["split_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual({entry["split"] for entry in payload["entries"]}, {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
