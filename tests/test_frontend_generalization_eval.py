import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dual2pose.eval.extension_experiment_utils import (
    FrontEndManifest,
    replace_frontend_inputs,
)


class FrontEndManifestTest(unittest.TestCase):
    """Break caught: external estimator poses are silently misaligned or incomplete."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_pose(self, name: str, values: list[float], joints: int = 2) -> str:
        pose = np.zeros((len(values), joints, 3), dtype=np.float32)
        pose[:, :, 0] = np.asarray(values, dtype=np.float32)[:, None]
        path = self.root / name
        np.save(path, pose)
        return name

    def _write_manifest(self, entries: list[dict], **extra: object) -> Path:
        payload = {"frontend_name": "fixture_frontend", "entries": entries, **extra}
        path = self.root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def _entry(camera_id: str, pose_path: str) -> dict:
        return {
            "person_id": "female",
            "action_id": "ski",
            "camera_id": camera_id,
            "pose_path": pose_path,
        }

    def _base_sample(self) -> dict:
        return {
            "meta": {
                "person_id": "female",
                "action_id": "ski",
                "cam1_id": "capture_L0_A000",
                "cam2_id": "capture_L0_A090",
            },
            "kpt3d_sam": {
                "cam1": torch.zeros((5, 2, 3), dtype=torch.float32),
                "cam2": torch.zeros((5, 2, 3), dtype=torch.float32),
            },
        }

    def test_replacement_loads_both_views_and_linearly_resamples_time(self) -> None:
        left = self._write_pose("left.npy", [0.0, 1.0, 2.0])
        right = self._write_pose("right.npy", [10.0, 11.0, 12.0])
        manifest = FrontEndManifest.load(
            self._write_manifest(
                [
                    self._entry("capture_L0_A000", left),
                    self._entry("capture_L0_A090", right),
                ]
            )
        )

        actual = replace_frontend_inputs(self._base_sample(), manifest)

        self.assertEqual(tuple(actual["kpt3d_sam"]["cam1"].shape), (5, 2, 3))
        self.assertTrue(
            torch.allclose(
                actual["kpt3d_sam"]["cam1"][:, 0, 0],
                torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                actual["kpt3d_sam"]["cam2"][:, 0, 0],
                torch.tensor([10.0, 10.5, 11.0, 11.5, 12.0]),
            )
        )

    def test_duplicate_manifest_key_is_rejected(self) -> None:
        pose_path = self._write_pose("pose.npy", [0.0, 1.0])
        entry = self._entry("capture_L0_A000", pose_path)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            FrontEndManifest.load(self._write_manifest([entry, entry]))

    def test_missing_camera_coverage_is_rejected_before_replacement(self) -> None:
        pose_path = self._write_pose("left.npy", [0.0, 1.0])
        manifest = FrontEndManifest.load(
            self._write_manifest([self._entry("capture_L0_A000", pose_path)])
        )
        with self.assertRaisesRegex(KeyError, "capture_L0_A090"):
            replace_frontend_inputs(self._base_sample(), manifest)

    def test_nonfinite_pose_is_rejected(self) -> None:
        left = self._write_pose("left.npy", [0.0, float("nan")])
        right = self._write_pose("right.npy", [0.0, 1.0])
        manifest = FrontEndManifest.load(
            self._write_manifest(
                [
                    self._entry("capture_L0_A000", left),
                    self._entry("capture_L0_A090", right),
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            replace_frontend_inputs(self._base_sample(), manifest)

    def test_joint_count_mismatch_is_rejected(self) -> None:
        left = self._write_pose("left.npy", [0.0, 1.0], joints=3)
        right = self._write_pose("right.npy", [0.0, 1.0], joints=3)
        manifest = FrontEndManifest.load(
            self._write_manifest(
                [
                    self._entry("capture_L0_A000", left),
                    self._entry("capture_L0_A090", right),
                ]
            )
        )
        with self.assertRaisesRegex(ValueError, "joint count"):
            replace_frontend_inputs(self._base_sample(), manifest)

    def test_npz_frame_indices_are_used_for_exact_alignment(self) -> None:
        pose = np.zeros((4, 2, 3), dtype=np.float32)
        pose[:, :, 0] = np.asarray([2.0, 4.0, 7.0, 9.0])[:, None]
        for camera_id in ("capture_L0_A000", "capture_L0_A090"):
            np.savez_compressed(
                self.root / f"{camera_id}.npz",
                pose=pose,
                frame_indices=np.asarray([2, 4, 7, 9], dtype=np.int64),
            )
        manifest = FrontEndManifest.load(
            self._write_manifest(
                [
                    self._entry("capture_L0_A000", "capture_L0_A000.npz"),
                    self._entry("capture_L0_A090", "capture_L0_A090.npz"),
                ],
                metadata={"input_2d_source": "unity_gt_h36m17"},
            )
        )
        sample = self._base_sample()
        sample["frame_indices"] = torch.tensor([2, 7, 9], dtype=torch.long)
        sample["kpt3d_sam"]["cam1"] = torch.zeros((3, 2, 3))
        sample["kpt3d_sam"]["cam2"] = torch.zeros((3, 2, 3))
        actual = replace_frontend_inputs(sample, manifest)
        self.assertEqual(manifest.metadata["input_2d_source"], "unity_gt_h36m17")
        self.assertEqual(
            actual["kpt3d_sam"]["cam1"][:, 0, 0].tolist(), [2.0, 7.0, 9.0]
        )


if __name__ == "__main__":
    unittest.main()
