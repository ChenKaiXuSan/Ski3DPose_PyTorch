import unittest

import torch

from dual2pose.eval.eval_unity_temporal_offset import _shift_pose_sequence
from dual2pose.eval import temporal_alignment as temporal_alignment_module
from dual2pose.eval.temporal_alignment import (
    CalibrationRow,
    choose_confidence_threshold,
    estimate_temporal_correction,
)


CANDIDATES = [value / 2 for value in range(-10, 11)]


def nonlinear_pose_fixture(batch: int = 2, frames: int = 30) -> torch.Tensor:
    time = torch.arange(frames, dtype=torch.float32)
    pose = torch.zeros((batch, frames, 15, 3), dtype=torch.float32)
    pose[:, :, 6, 0] = -1.0
    pose[:, :, 7, 0] = 1.0
    pose[:, :, 14, 1] = 1.0
    pose[:, :, 0, :] = torch.tensor([-0.2, 1.2, 1.0])
    pose[:, :, 1, :] = torch.tensor([0.2, 1.2, 1.0])
    for joint in [2, 3, 4, 5, 8, 9, 10, 11, 12, 13]:
        pose[:, :, joint, 0] = torch.sin(time * (0.65 + joint * 0.031))
        pose[:, :, joint, 1] = torch.sin(time * (0.93 + joint * 0.017))
        pose[:, :, joint, 2] = torch.cos(time * (1.17 + joint * 0.019))
    return pose


class TemporalEstimatorTest(unittest.TestCase):
    """Breaks caught: correction sign, half-frame interpolation, or ties are wrong."""

    def test_positive_injected_lag_needs_negative_correction(self) -> None:
        left = nonlinear_pose_fixture()
        right = _shift_pose_sequence(left, offset_frames=2.0)
        result = estimate_temporal_correction(left, right, CANDIDATES, 0.0)
        self.assertTrue(torch.equal(result.correction_frames, torch.tensor([-2.0, -2.0])))

    def test_half_frame_lag_is_recovered(self) -> None:
        left = nonlinear_pose_fixture(batch=1)
        right = _shift_pose_sequence(left, offset_frames=-0.5)
        result = estimate_temporal_correction(left, right, CANDIDATES, 0.0)
        self.assertEqual(float(result.correction_frames.item()), 0.5)

    def test_constant_pose_returns_zero(self) -> None:
        pose = torch.zeros((2, 30, 15, 3))
        result = estimate_temporal_correction(pose, pose, CANDIDATES, 0.0)
        self.assertTrue(torch.equal(result.correction_frames, torch.zeros(2)))
        self.assertTrue(torch.equal(result.confidence, torch.zeros(2)))

    def test_high_threshold_gates_a_nonzero_candidate_to_zero(self) -> None:
        left = nonlinear_pose_fixture(batch=1)
        right = _shift_pose_sequence(left, offset_frames=2.0)
        result = estimate_temporal_correction(left, right, CANDIDATES, 10.0)
        self.assertEqual(float(result.correction_frames.item()), 0.0)

    def test_alignment_canonicalization_is_batch_order_invariant(self) -> None:
        first = nonlinear_pose_fixture(batch=1)[0]
        angle = torch.tensor(0.8)
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        rotation = torch.tensor(
            [
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        second = first @ rotation + torch.tensor([3.0, -2.0, 0.5])

        second_alone = temporal_alignment_module._canonical_pose(
            second.unsqueeze(0)
        )[0]
        second_after_first = temporal_alignment_module._canonical_pose(
            torch.stack([first, second])
        )[1]
        second_before_first = temporal_alignment_module._canonical_pose(
            torch.stack([second, first])
        )[0]

        self.assertTrue(torch.allclose(second_alone, second_after_first, atol=1e-5))
        self.assertTrue(torch.allclose(second_alone, second_before_first, atol=1e-5))


class TemporalCorrectionMetricTest(unittest.TestCase):
    """Break caught: predicting zero is counted as correct for a half-frame target."""

    def test_half_frame_boundary_is_not_counted_as_correct(self) -> None:
        accuracy_masks = getattr(
            temporal_alignment_module,
            "correction_accuracy_masks",
            None,
        )
        self.assertTrue(callable(accuracy_masks))

        estimated = torch.tensor([0.0, -0.5, 0.5])
        target = torch.tensor([-0.5, -0.5, 0.5])
        exact, within_half = accuracy_masks(estimated, target)

        self.assertTrue(torch.equal(exact, torch.tensor([False, True, True])))
        self.assertTrue(
            torch.equal(within_half, torch.tensor([False, True, True]))
        )


class TemporalCalibrationTest(unittest.TestCase):
    """Break caught: threshold selection leaks test rows or favors false corrections."""

    def test_threshold_minimizes_validation_offset_mae(self) -> None:
        rows = [
            CalibrationRow("val", 0.1, 2.0, 0.0),
            CalibrationRow("val", 0.3, -1.0, -1.0),
            CalibrationRow("val", 0.6, -2.0, -2.0),
        ]
        result = choose_confidence_threshold(rows, [0.0, 0.2, 0.5, 0.8])
        self.assertEqual(result["selected_threshold"], 0.2)
        self.assertEqual(result["validation_row_count"], 3)

    def test_test_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation rows only"):
            choose_confidence_threshold(
                [CalibrationRow("test", 1.0, -1.0, -1.0)],
                [0.0, 0.5],
            )


if __name__ == "__main__":
    unittest.main()
