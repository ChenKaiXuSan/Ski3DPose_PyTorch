import unittest

from dual2pose.eval.summarize_unity_masking import summarize_masking_rows


class MaskingSummaryTest(unittest.TestCase):
    """Break caught: masking degradation or AUC is derived from the wrong baseline."""

    @staticmethod
    def _row(view: str, ratio: float, mpjpe: float, acceleration: float) -> dict:
        return {
            "setting": f"{view}_random_r{ratio}",
            "view_mode": view,
            "pattern": "random",
            "ratio": str(ratio),
            "corruption": "noise_masking",
            "temporal_span": "",
            "fused_mpjpe": str(mpjpe),
            "canonical_avg_mpjpe": "4.0",
            "fused_acceleration_error": str(acceleration),
        }

    def test_selected_rows_and_auc_use_each_curves_zero_ratio_baseline(self) -> None:
        rows = [
            self._row("left", 0.0, 1.0, 0.1),
            self._row("left", 0.5, 2.0, 0.2),
            self._row("left", 1.0, 3.0, 0.3),
            self._row("right", 0.0, 2.0, 0.2),
            self._row("right", 0.5, 2.0, 0.2),
            self._row("right", 1.0, 2.0, 0.2),
        ]

        selected, auc_rows, trend_rows = summarize_masking_rows(
            rows, selected_ratios=[0.0, 0.5, 1.0], max_ratio=1.0
        )

        selected_by_key = {
            (row["view_mode"], row["ratio"]): row for row in selected
        }
        self.assertAlmostEqual(
            selected_by_key[("left", 0.5)]["mpjpe_degradation_percent"], 100.0
        )
        self.assertAlmostEqual(
            selected_by_key[("right", 1.0)]["mpjpe_degradation_percent"], 0.0
        )
        auc_by_view = {row["view_mode"]: row for row in auc_rows}
        self.assertAlmostEqual(auc_by_view["left"]["normalized_mpjpe_auc"], 2.0)
        self.assertAlmostEqual(auc_by_view["right"]["normalized_mpjpe_auc"], 1.0)
        self.assertEqual(len(trend_rows), 6)

    def test_missing_zero_ratio_baseline_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ratio=0"):
            summarize_masking_rows(
                [self._row("left", 0.5, 2.0, 0.2)],
                selected_ratios=[0.5],
                max_ratio=1.0,
            )

    def test_duplicate_curve_ratio_is_rejected(self) -> None:
        row = self._row("left", 0.0, 1.0, 0.1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_masking_rows(
                [row, dict(row)], selected_ratios=[0.0], max_ratio=1.0
            )


if __name__ == "__main__":
    unittest.main()
