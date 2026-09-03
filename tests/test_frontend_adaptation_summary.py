import unittest

from dual2pose.eval.summarize_frontend_adaptation import validate_and_summarize_rows
from dual2pose.experiments.run_frontend_adaptation_matrix import build_evaluation_cells


class FrontEndAdaptationSummaryTest(unittest.TestCase):
    """Breaks caught: duplicate/missing cells or recovery deltas use the wrong baseline."""

    def test_complete_matrix_uses_mmsports_per_frontend_baseline(self) -> None:
        rows = []
        for cell in build_evaluation_cells():
            baseline = {"sam3d": 1.0, "videopose3d": 2.0, "poseformer": 3.0, "motionbert": 4.0}[cell.test_frontend]
            value = baseline if cell.model_name == "mmsports" else baseline - 0.25
            rows.append(
                {
                    "model_name": cell.model_name,
                    "test_frontend": cell.test_frontend,
                    "common13_mpjpe": value,
                }
            )
        enriched = validate_and_summarize_rows(rows)
        target = next(
            row
            for row in enriched
            if row["model_name"] == "mixed_full" and row["test_frontend"] == "motionbert"
        )
        self.assertEqual(target["recovery_vs_mmsports"], 0.25)
        self.assertEqual(target["sam3d_retention_delta"], -0.25)

    def test_duplicate_cell_is_rejected(self) -> None:
        rows = [
            {"model_name": cell.model_name, "test_frontend": cell.test_frontend, "common13_mpjpe": 1.0}
            for cell in build_evaluation_cells()
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_and_summarize_rows(rows + [dict(rows[0])])


if __name__ == "__main__":
    unittest.main()
