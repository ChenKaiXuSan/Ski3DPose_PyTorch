import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dual2pose.eval.frontend_lifters import (
    H36M17_NAMES,
    OfficialPoseLifter,
    _normalize_checkpoint_key,
    centered_temporal_windows,
    h36m17_to_canonfuse15,
    load_unity_2d_stream,
    normalize_motionbert_pose_2d,
    motionbert_output_scale_m,
    normalize_screen_coordinates,
    temporal_chunks,
    unity_to_h36m17,
)
from dual2pose.eval.export_unity_frontend_predictions import discover_unity_streams


class FrontEndInputTest(unittest.TestCase):
    def test_screen_normalization_preserves_aspect_ratio(self) -> None:
        points = np.asarray([[[0.0, 0.0], [1920.0, 1080.0]]], dtype=np.float32)
        actual = normalize_screen_coordinates(points, width=1920, height=1080)
        expected = np.asarray(
            [[[-1.0, -0.5625], [1.0, 0.5625]]], dtype=np.float32
        )
        np.testing.assert_allclose(actual, expected)

    def test_motionbert_input_matches_official_h36m_bbox_normalization(self) -> None:
        width, height = 1920, 1080
        pixels = np.zeros((1, 17, 3), dtype=np.float32)
        pixels[..., 0] = np.linspace(100.0, 300.0, 17)
        pixels[..., 1] = np.linspace(200.0, 500.0, 17)
        pixels[..., 2] = 0.75
        normalized = pixels.copy()
        normalized[..., :2] = normalize_screen_coordinates(
            normalized[..., :2], width=width, height=height
        )

        actual = normalize_motionbert_pose_2d(
            normalized, image_size=(width, height)
        )
        actual_pixels = actual.copy()
        actual_pixels[..., 0] = (actual[..., 0] + 1.0) * width / 2.0
        actual_pixels[..., 1] = (
            actual[..., 1] + float(height) / float(width)
        ) * width / 2.0

        bbox_min = actual_pixels[0, :, :2].min(axis=0)
        bbox_max = actual_pixels[0, :, :2].max(axis=0)
        np.testing.assert_allclose(
            0.5 * (bbox_min + bbox_max), [528.0, 427.0], atol=1e-4
        )
        self.assertAlmostEqual(float(np.max(bbox_max - bbox_min)), 400.0, places=4)
        np.testing.assert_allclose(actual[..., 2], 0.75)

    def test_motionbert_decode_scale_includes_official_factor_four(self) -> None:
        self.assertAlmostEqual(
            motionbert_output_scale_m((1920, 1080)), 3.84, places=6
        )

    def test_unity_joint_names_are_mapped_to_h36m17_order(self) -> None:
        joint_names = [
            "Hand_R", "calf_l", "spine_02", "Upperarm_R", "Pelvis",
            "spine_03", "Thigh_L", "lowerarm_l", "neck_01", "Foot_R",
            "Thigh_R", "Hand_L", "head", "Upperarm_L", "calf_r", "Foot_L",
            "lowerarm_r",
        ]
        source = np.zeros((2, len(joint_names), 3), dtype=np.float32)
        source[:, :, 0] = np.arange(len(joint_names), dtype=np.float32)
        source[:, :, 2] = 1.0
        actual = unity_to_h36m17(source, joint_names)
        expected_source_names = [
            "Pelvis", "Thigh_R", "calf_r", "Foot_R", "Thigh_L", "calf_l",
            "Foot_L", "spine_02", "spine_03", "neck_01", "head",
            "Upperarm_L", "lowerarm_l", "Hand_L", "Upperarm_R", "lowerarm_r",
            "Hand_R",
        ]
        expected = [joint_names.index(name) for name in expected_source_names]
        self.assertEqual(actual.shape, (2, 17, 3))
        np.testing.assert_array_equal(actual[0, :, 0], expected)

    def test_h36m_output_maps_exactly_to_common13_and_synthesizes_eyes(self) -> None:
        pose = np.zeros((3, 17, 3), dtype=np.float32)
        pose[:, :, 0] = np.arange(17, dtype=np.float32)
        pose[:, 4, :] = [-0.2, 0.0, 0.0]
        pose[:, 1, :] = [0.2, 0.0, 0.0]
        pose[:, 8, :] = [0.0, 0.8, 0.0]
        pose[:, 9, :] = [0.0, 1.0, 0.0]
        pose[:, 10, :] = [0.0, 1.3, 0.0]
        pose[:, 11, :] = [-0.4, 0.9, 0.0]
        pose[:, 14, :] = [0.4, 0.9, 0.0]
        actual = h36m17_to_canonfuse15(pose)
        source_indices = [11, 14, 12, 15, 4, 1, 5, 2, 6, 3, 16, 13, 9]
        self.assertEqual(actual.shape, (3, 15, 3))
        np.testing.assert_allclose(actual[:, 2:, :], pose[:, source_indices, :])
        self.assertTrue(np.isfinite(actual[:, :2]).all())
        self.assertFalse(np.allclose(actual[:, 0], actual[:, 1]))

    def test_centered_windows_repeat_edges_and_preserve_frame_count(self) -> None:
        sequence = torch.arange(4, dtype=torch.float32).view(4, 1, 1)
        actual = centered_temporal_windows(sequence, window=3)
        self.assertEqual(tuple(actual.shape), (4, 3, 1, 1))
        self.assertEqual(
            actual[:, :, 0, 0].tolist(),
            [[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 3]],
        )

    def test_temporal_chunks_cover_sequence_once(self) -> None:
        self.assertEqual(
            temporal_chunks(8, max_length=3), [(0, 3), (3, 6), (6, 8)]
        )

    def test_mmpose_motionbert_checkpoint_keys_are_losslessly_normalized(self) -> None:
        cases = {
            "backbone.spat_embed": "pos_embed",
            "backbone.blocks_st.0.mlp_s.0.weight": "blocks_st.0.mlp_s.fc1.weight",
            "backbone.blocks_ts.4.mlp_t.2.bias": "blocks_ts.4.mlp_t.fc2.bias",
            "head.pre_logits.fc.weight": "pre_logits.fc.weight",
            "head.fc.bias": "head.bias",
            "backbone.attn_regress.3.weight": "ts_attn.3.weight",
        }
        self.assertEqual(
            {key: _normalize_checkpoint_key(key) for key in cases}, cases
        )

    def test_stream_loader_keeps_original_frame_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "joint_names.json").write_text(
                json.dumps({"joint_names": list(H36M17_NAMES)}), encoding="utf-8"
            )
            (root / "sequence.json").write_text(
                json.dumps({"width": 1920, "height": 1080}), encoding="utf-8"
            )
            keypoint_dir = root / "kpt2d"
            keypoint_dir.mkdir()
            for frame_index in (2, 7):
                pose = np.zeros((17, 3), dtype=np.float32)
                pose[:, :2] = [960.0, 540.0]
                pose[:, 2] = 1.0
                np.save(keypoint_dir / f"kpt2d_{frame_index:06d}.npy", pose)
            pose, frame_indices, image_size = load_unity_2d_stream(
                keypoint_dir=keypoint_dir,
                joint_names_path=root / "joint_names.json",
                sequence_meta_path=root / "sequence.json",
            )
            self.assertEqual(pose.shape, (2, 17, 3))
            np.testing.assert_array_equal(frame_indices, [2, 7])
            self.assertEqual(image_size, (1920, 1080))

    def test_fold_stream_discovery_deduplicates_cameras_and_rewrites_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shared = {
                "person_id": "female",
                "action_id": "ski",
                "joint_names_path": "/old/root/meta/joint_names.json",
                "sequence_meta_path": "/old/root/meta/sequence.json",
            }
            rows = [
                {
                    **shared,
                    "cam1_id": "capture_A000",
                    "cam2_id": "capture_A010",
                    "cam1_kpt2d_dir": "/old/root/kpt2d/A000",
                    "cam2_kpt2d_dir": "/old/root/kpt2d/A010",
                },
                {
                    **shared,
                    "cam1_id": "capture_A000",
                    "cam2_id": "capture_A020",
                    "cam1_kpt2d_dir": "/old/root/kpt2d/A000",
                    "cam2_kpt2d_dir": "/old/root/kpt2d/A020",
                },
            ]
            fold = root / "fold.json"
            fold.write_text(json.dumps({"test": rows}), encoding="utf-8")
            streams = discover_unity_streams(
                fold_json=fold,
                split="test",
                rewrite_from=Path("/old/root"),
                data_root=root,
            )
            self.assertEqual([stream.camera_id for stream in streams], [
                "capture_A000", "capture_A010", "capture_A020"
            ])
            self.assertEqual(streams[0].keypoint_dir, root / "kpt2d/A000")

    def test_pose_lifter_adapters_preserve_one_output_per_input_frame(self) -> None:
        class VideoModel(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                xy = value[:, 1:-1]
                return torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)

        class CenterModel(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                xy = value[:, value.shape[1] // 2 : value.shape[1] // 2 + 1]
                return torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)

        class IdentityModel(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value

        inputs = np.zeros((5, 17, 3), dtype=np.float32)
        inputs[..., 0] = np.arange(17, dtype=np.float32)
        for name, model, temporal_length in (
            ("videopose3d", VideoModel(), 3),
            ("poseformer", CenterModel(), 3),
            ("motionbert", IdentityModel(), 3),
        ):
            lifter = OfficialPoseLifter(
                name=name,
                model=model,
                device=torch.device("cpu"),
                temporal_length=temporal_length,
                batch_size=2,
            )
            actual = lifter.predict(inputs, image_size=(1920, 1080))
            self.assertEqual(actual.shape, (5, 17, 3), name)
            np.testing.assert_allclose(actual[:, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
