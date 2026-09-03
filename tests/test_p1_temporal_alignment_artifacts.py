import unittest

from paper.ivc_draft_20260821.scripts.generate_p1_temporal_alignment_artifacts import (
    INJECTED_OFFSETS,
    render_table,
    validate_summary_rows,
)


class P1TemporalAlignmentArtifactsTest(unittest.TestCase):
    def _rows(self):
        rows = []
        for offset in INJECTED_OFFSETS:
            for condition, value in (
                ("uncorrected", 0.20 + abs(offset) * 0.01),
                ("automatic", 0.19 + abs(offset) * 0.005),
                ("oracle", 0.18),
            ):
                rows.append(
                    {
                        "injected_offset": str(offset),
                        "condition": condition,
                        "sample_count": "64440",
                        "mpjpe_mean": str(value),
                        "mpjpe_std": "0.01",
                        "acceleration_error_mean": "0.02",
                        "degradation_vs_zero_reference": "0.0",
                        "recovery_fraction": "0.5",
                        "offset_mae": "0.25",
                        "offset_signed_bias": "0.0",
                        "offset_accuracy_exact": "0.7",
                        "offset_accuracy_within_0p5": "0.8",
                        "correction_activation_rate": "0.7",
                        "gate_error_correlation": "0.6",
                        "view_preference_accuracy": "0.9",
                    }
                )
        return rows

    def test_exact_11_by_3_matrix_renders(self):
        rows = validate_summary_rows(self._rows())
        table = render_table(rows)
        self.assertIn("Automatic", table)
        self.assertIn("Oracle", table)
        self.assertIn("+5", table)
        self.assertIn(r"$\Delta_{\mathrm{auto}}$", table)
        self.assertIn(r"Exact (\%)", table)

    def test_missing_condition_is_rejected(self):
        rows = self._rows()[:-1]
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            validate_summary_rows(rows)

    def test_inconsistent_sample_count_is_rejected(self):
        rows = self._rows()
        rows[-1]["sample_count"] = "1"
        with self.assertRaisesRegex(ValueError, "sample count"):
            validate_summary_rows(rows)

    def test_missing_exact_offset_accuracy_is_rejected(self):
        rows = self._rows()
        rows[0].pop("offset_accuracy_exact")
        with self.assertRaisesRegex(KeyError, "offset_accuracy_exact"):
            validate_summary_rows(rows)


if __name__ == "__main__":
    unittest.main()
