import unittest

import torch

from dual2pose.eval.view_angle_significance import (
    analyze_angle_rows,
    collapse_action_rows,
    extract_angle_pair_rows,
    holm_adjust,
)


BIN_EDGES = [0, 30, 60, 90, 120, 150, 180]


def _output_fixture():
    ground_truth = torch.zeros((2, 2, 1, 3), dtype=torch.float32)
    fused = ground_truth.clone()
    fused[0, ..., 0] = 1.0
    fused[1, ..., 0] = 2.0
    left = ground_truth.clone()
    right = ground_truth.clone()
    left[..., 0] = 3.0
    right[..., 0] = 1.0
    return {
        "fused": fused,
        "left_canonical": left,
        "right_canonical": right,
        "ground_truth_canonical": ground_truth,
        "meta": {
            "person_id": ["person_01", "person_01"],
            "action_id": ["turn", "jump"],
            "cam1_id": ["capture_L0_A350", "capture_L0_A000"],
            "cam2_id": ["capture_L0_A010", "capture_L0_A100"],
        },
    }


def _pair_row(angle_bin, pair_index, action_index, delta):
    baseline = 2.0 + 0.01 * pair_index
    return {
        "person_id": f"person_{pair_index % 3}",
        "action_id": f"action_{action_index}",
        "camera_pair_id": f"cam_{pair_index:02d}|cam_{pair_index + 1:02d}",
        "cam1_id": f"cam_{pair_index:02d}",
        "cam2_id": f"cam_{pair_index + 1:02d}",
        "angle_bin": angle_bin,
        "separation_deg": float(angle_bin.split("-")[0]) + 5.0,
        "fused_mpjpe": baseline - delta,
        "canonical_avg_mpjpe": baseline,
        "fusion_gain_mpjpe": delta,
    }


class ViewAngleSignificanceTest(unittest.TestCase):
    def test_extract_keeps_action_and_unordered_camera_pair(self):
        rows = extract_angle_pair_rows([_output_fixture()], BIN_EDGES)

        self.assertEqual(rows[0]["action_id"], "turn")
        self.assertEqual(
            rows[0]["camera_pair_id"],
            "capture_L0_A010|capture_L0_A350",
        )
        self.assertEqual(rows[0]["angle_bin"], "0-30")
        self.assertAlmostEqual(rows[0]["fused_mpjpe"], 1.0)
        self.assertAlmostEqual(rows[0]["canonical_avg_mpjpe"], 2.0)

    def test_extract_rejects_nonfinite_tensors(self):
        output = _output_fixture()
        output["fused"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            extract_angle_pair_rows([output], BIN_EDGES)

    def test_collapse_rejects_duplicate_action_pair_records(self):
        row = _pair_row("0-30", 0, 0, 0.1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            collapse_action_rows([row, dict(row)])

    def test_collapse_averages_actions_per_camera_pair(self):
        rows = [
            _pair_row("0-30", 0, 0, 0.1),
            _pair_row("0-30", 0, 1, 0.3),
        ]
        collapsed = collapse_action_rows(rows)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["action_count"], 2)
        self.assertAlmostEqual(collapsed[0]["fusion_gain_mpjpe"], 0.2)

    def test_holm_adjust_preserves_original_order(self):
        actual = holm_adjust([0.04, 0.01, 0.03])
        self.assertEqual(actual, [0.06, 0.03, 0.06])

    def test_analysis_reports_six_adjusted_within_bin_tests(self):
        rows = []
        for bin_index, label in enumerate(
            ["0-30", "30-60", "60-90", "90-120", "120-150", "150-180"]
        ):
            for pair_index in range(8):
                for action_index in range(2):
                    delta = 0.05 + 0.01 * bin_index + 0.001 * pair_index
                    rows.append(_pair_row(label, pair_index + 20 * bin_index, action_index, delta))

        result = analyze_angle_rows(rows, bootstrap_resamples=200, seed=42)

        self.assertEqual(len(result["within_bin"]), 6)
        self.assertTrue(
            all(0.0 <= row["p_holm"] <= 1.0 for row in result["within_bin"])
        )
        self.assertEqual(result["omnibus"]["test"], "kruskal_wallis")
        self.assertEqual(result["cluster_count"], 48)


if __name__ == "__main__":
    unittest.main()
