from pathlib import Path
import tempfile
import unittest

import numpy as np

from dual2pose.eval.nview_protocol import (
    CameraGroup,
    InsufficientCommonFrames,
    load_multiview_sample,
)


class NViewLoadingTest(unittest.TestCase):
    """Breaks caught: views or ground truth use different temporal windows."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.group = CameraGroup(
            "fixture",
            "male",
            "action",
            0,
            tuple(f"capture_L0_A{value:03d}" for value in (0, 90, 180, 270)),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _fixture(self, ranges: dict[int, range]) -> dict[tuple[str, str, str], dict[str, str]]:
        gt_dir = self.root / "gt"
        gt_dir.mkdir()
        for frame in range(50):
            np.save(gt_dir / f"frame_{frame:06d}.npy", np.full((15, 3), frame, np.float32))
        lookup: dict[tuple[str, str, str], dict[str, str]] = {}
        for azimuth, frames in ranges.items():
            camera = f"capture_L0_A{azimuth:03d}"
            pose_dir = self.root / camera
            pose_dir.mkdir()
            for frame in frames:
                np.save(
                    pose_dir / f"kpt3d_{frame:06d}.npy",
                    np.full((15, 3), frame + azimuth, np.float32),
                )
            lookup[("male", "action", camera)] = {
                "sam3d_kpt3d_dir": str(pose_dir),
                "kpt3d_dir": str(gt_dir),
            }
        return lookup

    def test_common_frames_are_intersected_before_subsampling(self) -> None:
        lookup = self._fixture(
            {0: range(0, 40), 90: range(2, 40), 180: range(1, 39), 270: range(3, 41)}
        )
        sample = load_multiview_sample(self.group, lookup, target_t=30)
        self.assertGreaterEqual(int(sample.frame_indices.min()), 3)
        self.assertLessEqual(int(sample.frame_indices.max()), 38)
        self.assertEqual(tuple(sample.ground_truth.shape), (30, 15, 3))
        self.assertTrue(all(tuple(pose.shape) == (30, 15, 3) for pose in sample.poses.values()))
        for camera, pose in sample.poses.items():
            azimuth = int(camera[-3:])
            self.assertTrue(
                np.allclose(
                    pose[:, 0, 0].numpy(),
                    sample.frame_indices.numpy() + azimuth,
                )
            )

    def test_fewer_than_target_common_frames_is_rejected(self) -> None:
        lookup = self._fixture({0: range(20), 90: range(20), 180: range(20), 270: range(20)})
        with self.assertRaisesRegex(InsufficientCommonFrames, "available=20, required=30"):
            load_multiview_sample(self.group, lookup, target_t=30)


if __name__ == "__main__":
    unittest.main()
