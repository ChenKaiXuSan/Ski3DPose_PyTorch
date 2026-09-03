import unittest

import torch

from dual2pose.eval.eval_unity_temporal_offset import (
    TemporalOffsetSetting,
    _apply_temporal_offset_to_batch,
    _format_offset,
    _parse_offsets,
    _shift_pose_sequence,
    _summarize_gate_error_relationship,
)


class TemporalOffsetEvalTest(unittest.TestCase):
    def test_parse_offsets_uses_default_and_csv_values(self) -> None:
        self.assertEqual(_parse_offsets("-1, 0, 0.5"), [-1.0, 0.0, 0.5])
        self.assertIn(0.0, _parse_offsets(None))

    def test_format_offset_is_path_safe(self) -> None:
        self.assertEqual(_format_offset(-0.5), "m0p5")
        self.assertEqual(_format_offset(1.0), "p1")

    def test_shift_pose_sequence_integer_and_fractional(self) -> None:
        pose = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)

        delayed = _shift_pose_sequence(pose, 1.0)
        advanced = _shift_pose_sequence(pose, -1.0)
        half_delayed = _shift_pose_sequence(pose, 0.5)

        self.assertTrue(
            torch.allclose(delayed.flatten(), torch.tensor([0.0, 0.0, 1.0, 2.0]))
        )
        self.assertTrue(
            torch.allclose(advanced.flatten(), torch.tensor([1.0, 2.0, 3.0, 3.0]))
        )
        self.assertTrue(
            torch.allclose(half_delayed.flatten(), torch.tensor([0.0, 0.5, 1.5, 2.5]))
        )

    def test_apply_temporal_offset_only_changes_selected_view(self) -> None:
        cam1 = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
        cam2 = cam1 + 10.0
        batch = {"kpt3d_sam": {"cam1": cam1, "cam2": cam2}}

        out = _apply_temporal_offset_to_batch(
            batch,
            TemporalOffsetSetting(name="right_offset_p1", offset_frames=1.0, view_mode="right"),
        )

        self.assertTrue(torch.equal(out["kpt3d_sam"]["cam1"], cam1))
        self.assertTrue(
            torch.allclose(
                out["kpt3d_sam"]["cam2"].flatten(),
                torch.tensor([10.0, 10.0, 11.0, 12.0]),
            )
        )

    def test_gate_error_summary_rewards_correct_view_preference(self) -> None:
        output = {
            "alpha": torch.tensor([[[[0.9], [0.1]]]], dtype=torch.float32),
            "left_canonical": torch.tensor([[[[0.0], [2.0]]]], dtype=torch.float32),
            "right_canonical": torch.tensor([[[[2.0], [0.0]]]], dtype=torch.float32),
            "ground_truth_canonical": torch.tensor(
                [[[[0.0], [0.0]]]], dtype=torch.float32
            ),
        }

        stats = _summarize_gate_error_relationship([output])

        self.assertAlmostEqual(stats["gate_preference_accuracy"], 1.0)
        self.assertGreater(stats["alpha_when_left_better"], 0.5)
        self.assertLess(stats["alpha_when_right_better"], 0.5)


if __name__ == "__main__":
    unittest.main()
