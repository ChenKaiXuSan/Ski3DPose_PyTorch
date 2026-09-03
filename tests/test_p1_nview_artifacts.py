import unittest

from paper.ivc_draft_20260821.scripts.generate_p1_nview_artifacts import (
    render_efficiency_table,
    render_table,
    validate_efficiency_rows,
    validate_summary_rows,
)


class P1NViewArtifactsTest(unittest.TestCase):
    def _rows(self):
        rows = [
            {
                "n_views": "1",
                "method": "single_view",
                "group_count": "44",
                "mpjpe_mean": "0.3000",
                "mpjpe_std": "0.0100",
                "mpjpe_ci95_low": "0.2970",
                "mpjpe_ci95_high": "0.3030",
                "upper_bound": "False",
            }
        ]
        for n_views in (2, 3, 4):
            for method, value in (
                ("nview_canonical_mean", 0.28),
                ("pairwise_canonfuse_mean", 0.20),
                ("pairwise_oracle_select", 0.18),
            ):
                rows.append(
                    {
                        "n_views": str(n_views),
                        "method": method,
                        "group_count": "44",
                        "mpjpe_mean": str(value - 0.01 * (n_views - 2)),
                        "mpjpe_std": "0.0100",
                        "mpjpe_ci95_low": "0.0000",
                        "mpjpe_ci95_high": "1.0000",
                        "upper_bound": str(method == "pairwise_oracle_select"),
                    }
                )
        return rows

    def test_complete_protocol_renders_table(self):
        rows = validate_summary_rows(self._rows())
        table = render_table(rows)
        self.assertIn("N-view scaling", table)
        self.assertIn("Pairwise CanonFuse3D", table)
        self.assertIn("Upper bound", table)
        self.assertIn(r"\resizebox{\textwidth}{!}{%", table)

    def test_missing_n_row_is_rejected(self):
        rows = [row for row in self._rows() if not (
            row["n_views"] == "4" and row["method"] == "pairwise_canonfuse_mean"
        )]
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            validate_summary_rows(rows)

    def test_oracle_label_must_be_upper_bound(self):
        rows = self._rows()
        target = next(row for row in rows if row["method"] == "pairwise_oracle_select")
        target["upper_bound"] = "False"
        with self.assertRaisesRegex(ValueError, "upper bound"):
            validate_summary_rows(rows)

    def _efficiency_rows(self):
        return [
            {
                "n_views": str(n_views),
                "pair_forward_count": str(pair_count),
                "group_count": "44",
                "mpjpe_mean": str(mpjpe),
                "latency_mean_ms": str(latency),
                "latency_std_ms": "0.5",
                "latency_median_ms": str(latency - 0.1),
                "latency_p95_ms": str(latency + 0.8),
                "relative_latency_vs_two_view": str(relative),
                "throughput_groups_per_second": str(1000.0 / latency),
                "peak_gpu_memory_mib": "20.0",
                "mpjpe_reduction_vs_two_view_pct": str(
                    100.0 * (0.1295 - mpjpe) / 0.1295
                ),
                "warmup_iterations": "10",
                "execution_mode": "serial_all_unique_pairs",
                "device_name": "NVIDIA RTX A6000",
                "torch_version": "2.11.0+cu126",
                "cuda_version": "12.6",
                "sequence_frames": "30",
            }
            for n_views, pair_count, mpjpe, latency, relative in (
                (2, 1, 0.1295, 7.0, 1.0),
                (3, 3, 0.1177, 20.0, 20.0 / 7.0),
                (4, 6, 0.1158, 40.0, 40.0 / 7.0),
            )
        ]

    def test_complete_efficiency_protocol_renders_accuracy_cost_table(self):
        rows = validate_efficiency_rows(self._efficiency_rows())
        table = render_efficiency_table(rows)
        self.assertIn("Accuracy--cost trade-off", table)
        self.assertIn("Serial latency", table)
        self.assertIn("NVIDIA RTX A6000", table)
        self.assertIn("30-frame", table)
        self.assertNotIn("\t", table)
        self.assertIn(chr(92) + "times", table)

    def test_efficiency_summary_missing_view_is_rejected(self):
        rows = [row for row in self._efficiency_rows() if row["n_views"] != "4"]
        with self.assertRaisesRegex(ValueError, "views"):
            validate_efficiency_rows(rows)


if __name__ == "__main__":
    unittest.main()
