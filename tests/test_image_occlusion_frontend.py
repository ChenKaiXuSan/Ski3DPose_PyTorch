import tempfile
import unittest
from pathlib import Path

import numpy as np

from dual2pose.eval.image_occlusion import ImageOcclusionSetting
from dual2pose.eval.run_unity_image_occlusion_frontend import (
    infer_stream,
    validate_stream_npz,
)


def _stream_fixture():
    return {
        "person_id": "female",
        "action_id": "turn",
        "camera_id": "capture_L0_A000",
        "rgb_dir": "/unused/rgb",
        "kpt2d_dir": "/unused/kpt2d",
        "frame_indices": [3],
    }


def _image_loader(_path):
    return np.full((200, 200, 3), 80, dtype=np.uint8)


def _joint_loader(_path):
    joints = np.zeros((98, 3), dtype=np.float32)
    joints[:, 0] = 100.0
    joints[:, 1] = np.linspace(10.0, 190.0, 98)
    joints[:, 2] = 1.0
    return joints


class CountingPredictor:
    def __init__(self):
        self.calls = 0

    def __call__(self, _image):
        self.calls += 1
        pose = np.zeros((70, 3), dtype=np.float32)
        pose[:, 0] = np.arange(70)
        return pose


class ImageOcclusionFrontendTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temporary.name)
        self.setting = ImageOcclusionSetting("random", 0.5, seed=42)

    def tearDown(self):
        self.temporary.cleanup()

    def test_detection_failure_is_recorded_as_zero_pose(self):
        path = infer_stream(
            _stream_fixture(),
            predictor=lambda _image: None,
            setting=self.setting,
            output_root=self.output_root,
            image_loader=_image_loader,
            joint_loader=_joint_loader,
        )

        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data["detection_failed"].tolist(), [True])
            self.assertEqual(np.count_nonzero(data["pose"]), 0)
            self.assertEqual(data["frame_indices"].tolist(), [3])

    def test_resume_skips_only_a_validated_npz(self):
        predictor = CountingPredictor()
        path = infer_stream(
            _stream_fixture(),
            predictor=predictor,
            setting=self.setting,
            output_root=self.output_root,
            image_loader=_image_loader,
            joint_loader=_joint_loader,
        )
        infer_stream(
            _stream_fixture(),
            predictor=predictor,
            setting=self.setting,
            output_root=self.output_root,
            image_loader=_image_loader,
            joint_loader=_joint_loader,
        )

        self.assertEqual(predictor.calls, 1)
        self.assertTrue(validate_stream_npz(path, expected_frames=[3]))

    def test_invalid_existing_npz_is_recomputed(self):
        predictor = CountingPredictor()
        path = infer_stream(
            _stream_fixture(),
            predictor=predictor,
            setting=self.setting,
            output_root=self.output_root,
            image_loader=_image_loader,
            joint_loader=_joint_loader,
        )
        np.savez_compressed(path, pose=np.zeros((1, 15, 3), dtype=np.float32))
        infer_stream(
            _stream_fixture(),
            predictor=predictor,
            setting=self.setting,
            output_root=self.output_root,
            image_loader=_image_loader,
            joint_loader=_joint_loader,
        )

        self.assertEqual(predictor.calls, 2)
        self.assertTrue(validate_stream_npz(path, expected_frames=[3]))


if __name__ == "__main__":
    unittest.main()
