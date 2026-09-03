import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dual2pose.eval.image_occlusion import (
    DISTAL_JOINT_INDICES,
    ImageOcclusionSetting,
    OcclusionFrameKey,
    apply_image_occlusion,
    build_required_frames_manifest,
    selected_joint_mask,
)


class RequiredFramesManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _series(self, directory, prefix, suffix, frame_ids=(0, 1, 2)):
        directory.mkdir(parents=True, exist_ok=True)
        for frame_id in frame_ids:
            (directory / f"{prefix}{frame_id:06d}{suffix}").write_bytes(b"fixture")
        return directory

    def _index_path(self, missing_rgb=False):
        rgb1 = self._series(self.root / "rgb1", "frame_", ".png")
        rgb2 = self._series(self.root / "rgb2", "frame_", ".png")
        kpt1 = self._series(self.root / "kpt1", "kpt2d_", ".npy")
        kpt2 = self._series(self.root / "kpt2", "kpt2d_", ".npy")
        gt = self._series(self.root / "gt", "frame_", ".npy")
        sam1 = self._series(self.root / "sam1", "kpt3d_", ".npy")
        sam2 = self._series(self.root / "sam2", "kpt3d_", ".npy")
        if missing_rgb:
            (rgb1 / "frame_000001.png").unlink()
        row = {
            "person_id": "female",
            "action_id": "turn",
            "cam1_id": "capture_L0_A000",
            "cam2_id": "capture_L0_A010",
            "cam1_frames_dir": str(rgb1),
            "cam2_frames_dir": str(rgb2),
            "cam1_kpt2d_dir": str(kpt1),
            "cam2_kpt2d_dir": str(kpt2),
            "kpt3d_dir": str(gt),
            "sam3d_cam1_kpt3d_dir": str(sam1),
            "sam3d_cam2_kpt3d_dir": str(sam2),
        }
        path = self.root / "fold.json"
        path.write_text(json.dumps({"test": [row]}), encoding="utf-8")
        return path

    def test_preserves_pair_specific_repeated_positions(self):
        result = build_required_frames_manifest(
            self._index_path(),
            target_length=5,
        )

        self.assertEqual(
            result["pair_sequences"][0]["frame_indices"],
            [0, 0, 1, 2, 2],
        )
        self.assertEqual(result["unique_required_frame_count"], 6)
        self.assertEqual(result["unique_source_image_count"], 6)
        self.assertEqual(result["stream_count"], 2)

    def test_rejects_missing_required_rgb_file(self):
        with self.assertRaises(FileNotFoundError):
            build_required_frames_manifest(
                self._index_path(missing_rgb=True),
                target_length=5,
            )


class ImageOcclusionMaskTest(unittest.TestCase):
    def _frame_key(self, frame_id=10):
        return OcclusionFrameKey("female", "turn", "capture_L0_A000", frame_id)

    def test_occluder_uses_twelve_percent_height_and_image_mean(self):
        row = np.arange(200, dtype=np.uint8)[:, None, None]
        image = np.broadcast_to(row, (200, 200, 3)).copy()
        joints = np.zeros((15, 3), dtype=np.float32)
        joints[:, 0] = 100.0
        joints[:, 1] = np.linspace(0.0, 199.0, 15)
        joints[:, 2] = 1.0
        setting = ImageOcclusionSetting(pattern="random", ratio=1.0, seed=42)

        masked, record = apply_image_occlusion(
            image,
            joints,
            setting,
            self._frame_key(),
        )

        self.assertEqual(record["side_px"], 24)
        expected = np.rint(image.mean(axis=(0, 1))).astype(np.uint8)
        self.assertTrue(np.all(masked[88:112, 88:112] == expected))
        self.assertEqual(record["selected_joint_count"], 15)

    def test_random_mask_is_invariant_to_resume_order(self):
        setting = ImageOcclusionSetting(pattern="random", ratio=0.5, seed=42)
        first = selected_joint_mask(setting, self._frame_key(7))
        selected_joint_mask(setting, self._frame_key(8))
        second = selected_joint_mask(setting, self._frame_key(7))
        self.assertTrue(np.array_equal(first, second))

    def test_distal_pattern_uses_archived_pose_mask_indices(self):
        setting = ImageOcclusionSetting(pattern="distal", ratio=1.0, seed=42)
        actual = np.flatnonzero(selected_joint_mask(setting, self._frame_key()))
        self.assertEqual(actual.tolist(), list(DISTAL_JOINT_INDICES))

    def test_temporal_selects_round_ratio_times_fifteen_joints(self):
        setting = ImageOcclusionSetting(
            pattern="temporal",
            ratio=0.5,
            seed=42,
            temporal_span=100,
        )
        actual = selected_joint_mask(
            setting,
            self._frame_key(10),
            source_frame_ids=range(0, 20),
        )
        self.assertEqual(int(actual.sum()), 8)


if __name__ == "__main__":
    unittest.main()
