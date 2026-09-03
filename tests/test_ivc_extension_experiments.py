import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from dual2pose.eval.extension_experiment_utils import (
    assign_angle_bin,
    circular_angle_distance,
    build_experiment_provenance,
    complete_test_dataloader,
    parse_unity_camera_id,
    resample_pose_rate,
    summarize_outputs_by_angle,
)
from dual2pose.eval.eval_unity_sampling_rate import (
    SamplingRateUnityDataModule,
    SamplingRateSetting,
    _apply_sampling_rate_to_batch,
)
from dual2pose.eval.eval_unity_temporal_offset import (
    TemporalOffsetSetting,
    _ensure_legacy_import_path,
    TemporalOffsetUnityDataModule,
)
from dual2pose.eval.eval_unity_masking import (
    MaskedUnityDataModule,
    OcclusionSetting,
    build_study,
    _configure_index_mapping_rewrite,
)


class ViewAngleTest(unittest.TestCase):
    """Break caught: camera-pair geometry is parsed or grouped incorrectly."""

    def test_module_import_does_not_require_image_dependencies(self) -> None:
        module = importlib.import_module("dual2pose.eval.eval_unity_view_angle")
        self.assertTrue(callable(module._parse_bin_edges))

    def test_bin_count_is_not_overwritten_by_total_provenance(self) -> None:
        module = importlib.import_module("dual2pose.eval.eval_unity_view_angle")
        rows = [{"angle_bin": "0-30", "sample_count": 3}]
        module._attach_provenance(
            rows,
            {"sample_count": 10, "checkpoint_sha256": "abc"},
        )
        self.assertEqual(rows[0]["sample_count"], 3)
        self.assertEqual(rows[0]["total_sample_count"], 10)
        self.assertEqual(rows[0]["checkpoint_sha256"], "abc")

    def test_parse_unity_camera_id(self) -> None:
        self.assertEqual(parse_unity_camera_id("capture_L3_A270"), (3, 270.0))
        with self.assertRaises(ValueError):
            parse_unity_camera_id("cam_01")

    def test_circular_angle_distance_wraps_at_360(self) -> None:
        self.assertEqual(circular_angle_distance(350.0, 10.0), 20.0)
        self.assertEqual(circular_angle_distance(0.0, 180.0), 180.0)

    def test_assign_angle_bin_includes_180_in_last_bin(self) -> None:
        edges = [0, 30, 60, 90, 120, 150, 180]
        self.assertEqual(assign_angle_bin(29.999, edges), "0-30")
        self.assertEqual(assign_angle_bin(30.0, edges), "30-60")
        self.assertEqual(assign_angle_bin(180.0, edges), "150-180")

    def test_summarize_outputs_by_angle_keeps_sample_groups_separate(self) -> None:
        gt = torch.zeros((2, 2, 1, 3), dtype=torch.float32)
        fused = gt.clone()
        fused[0, :, :, 0] = 1.0
        fused[1, :, :, 0] = 2.0
        left = gt.clone()
        right = gt.clone()
        left[:, :, :, 0] = 3.0
        right[:, :, :, 0] = 1.0

        rows = summarize_outputs_by_angle(
            [
                {
                    "fused": fused,
                    "left_canonical": left,
                    "right_canonical": right,
                    "ground_truth_canonical": gt,
                    "meta": {
                        "cam1_id": ["capture_L0_A350", "capture_L0_A000"],
                        "cam2_id": ["capture_L0_A010", "capture_L0_A100"],
                    },
                }
            ],
            failure_threshold=1.5,
            bin_edges=[0, 30, 60, 90, 120, 150, 180],
        )

        by_bin = {row["angle_bin"]: row for row in rows}
        self.assertEqual(by_bin["0-30"]["sample_count"], 1)
        self.assertAlmostEqual(by_bin["0-30"]["fused_mpjpe"], 1.0)
        self.assertAlmostEqual(by_bin["0-30"]["failure_rate"], 0.0)
        self.assertEqual(by_bin["90-120"]["sample_count"], 1)
        self.assertAlmostEqual(by_bin["90-120"]["fused_mpjpe"], 2.0)
        self.assertAlmostEqual(by_bin["90-120"]["failure_rate"], 1.0)
        self.assertAlmostEqual(by_bin["90-120"]["canonical_avg_mpjpe"], 2.0)


class SamplingRateTest(unittest.TestCase):
    """Break caught: rate error uses the wrong time direction or perturbs both views."""

    def test_legacy_import_path_is_explicitly_bootstrapped(self) -> None:
        expected = str(Path(__file__).resolve().parents[1] / "dual2pose")
        original = list(__import__("sys").path)
        try:
            __import__("sys").path[:] = [value for value in original if value != expected]
            _ensure_legacy_import_path()
            self.assertEqual(__import__("sys").path[0], expected)
        finally:
            __import__("sys").path[:] = original

    def test_center_anchored_positive_rate_uses_literal_positions(self) -> None:
        pose = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)
        actual = resample_pose_rate(pose, rate_error=1.0, anchor="center")
        expected = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0])
        self.assertTrue(torch.allclose(actual.flatten(), expected))

    def test_center_anchored_negative_rate_clamps_edges(self) -> None:
        pose = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)
        actual = resample_pose_rate(pose, rate_error=-0.5, anchor="center")
        expected = torch.tensor([0.0, 0.0, 2.0, 4.0, 4.0])
        self.assertTrue(torch.allclose(actual.flatten(), expected))

    def test_rate_error_must_keep_sampling_rate_positive(self) -> None:
        pose = torch.zeros((1, 5, 1, 3), dtype=torch.float32)
        with self.assertRaises(ValueError):
            resample_pose_rate(pose, rate_error=-1.0)

    def test_batch_transform_changes_only_selected_view(self) -> None:
        cam1 = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)
        cam2 = cam1 + 10.0
        transformed = _apply_sampling_rate_to_batch(
            {"kpt3d_sam": {"cam1": cam1, "cam2": cam2}},
            SamplingRateSetting(
                name="right_rate_p100", rate_error=1.0, view_mode="right"
            ),
        )
        self.assertTrue(torch.equal(transformed["kpt3d_sam"]["cam1"], cam1))
        self.assertTrue(
            torch.allclose(
                transformed["kpt3d_sam"]["cam2"].flatten(),
                torch.tensor([11.0, 11.5, 12.0, 12.5, 13.0]),
            )
        )


class CompleteTestDataLoaderTest(unittest.TestCase):
    """Break caught: the final partial test batch is silently discarded."""

    def test_complete_test_dataloader_retains_partial_batch(self) -> None:
        base = DataLoader(
            TensorDataset(torch.arange(10)),
            batch_size=4,
            shuffle=False,
            drop_last=True,
        )

        loader = complete_test_dataloader(base)
        batches = list(loader)

        self.assertEqual([len(batch[0]) for batch in batches], [4, 4, 2])
        self.assertFalse(loader.drop_last)

    def test_build_experiment_provenance_hashes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.ckpt"
            checkpoint.write_bytes(b"weights")
            actual = build_experiment_provenance(
                checkpoint,
                sample_count=10,
                fold=0,
                seed=42,
                joint_subset="all15",
                units="meters",
            )

        self.assertEqual(
            actual["checkpoint_sha256"], hashlib.sha256(b"weights").hexdigest()
        )
        self.assertEqual(actual["sample_count"], 10)
        self.assertEqual(actual["joint_subset"], "all15")
        self.assertEqual(actual["units"], "meters")

    def test_perturbation_wrappers_retain_partial_batch(self) -> None:
        class BaseDataModule:
            def test_dataloader(self) -> DataLoader:
                return DataLoader(
                    TensorDataset(torch.arange(10)),
                    batch_size=4,
                    shuffle=False,
                    drop_last=True,
                )

        base = BaseDataModule()
        wrappers = [
            TemporalOffsetUnityDataModule(
                base,
                TemporalOffsetSetting("zero", 0.0, "right"),
            ),
            SamplingRateUnityDataModule(
                base,
                SamplingRateSetting("zero", 0.0, "right"),
            ),
            MaskedUnityDataModule(
                base,
                OcclusionSetting(
                    "zero", "both", "random", 0.0, "noise_masking"
                ),
            ),
        ]

        for wrapper in wrappers:
            loader = wrapper.test_dataloader()
            self.assertFalse(loader.drop_last)
            self.assertEqual(len(loader), 3)


class MaskingStudySelectionTest(unittest.TestCase):
    def test_masking_entry_configures_stale_index_path_rewrite(self) -> None:
        config = SimpleNamespace(
            data=SimpleNamespace(
                unity=SimpleNamespace(root_path="/current/unity")
            )
        )
        with patch(
            "dual2pose.eval.eval_unity_masking.patch_index_mapping_path_rewrite"
        ) as rewriter:
            _configure_index_mapping_rewrite(config)
        rewriter.assert_called_once_with(
            old_root="/home/kaixu_chen/data/skiing/skiing_unity_dataset",
            new_root="/current/unity",
        )

    def test_masking_study_accepts_explicit_factor_grid(self) -> None:
        settings = build_study(
            view_modes=["left", "both"],
            patterns=["random", "temporal"],
            ratios=[0.25, 0.5],
        )

        self.assertEqual(len(settings), 8)
        self.assertEqual(settings[0].name, "left_random_r0p25")
        self.assertEqual(settings[-1].name, "both_temporal_r0p50")
        self.assertTrue(
            all(setting.corruption == "noise_masking" for setting in settings)
        )


if __name__ == "__main__":
    unittest.main()
