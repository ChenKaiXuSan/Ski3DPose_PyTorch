import csv
import json
import tempfile
import unittest
from pathlib import Path

from dual2pose.eval.render_e4_e5_artifacts import render_all


CHECKPOINT_HASH = "869a2217f8676c0ada75ed3c9a3c82a9b8efbb105749f6ffb8bef71e9172f50f"


class E4E5PaperArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.e4_csv = self.root / "e4.csv"
        self.e4_json = self.root / "e4.json"
        self.e5_csv = self.root / "e5.csv"
        self.comparison_csv = self.root / "comparison.csv"
        self.table_root = self.root / "tables"
        self.figure_path = self.root / "figure.pdf"
        labels = ["0-30", "30-60", "60-90", "90-120", "120-150", "150-180"]
        self._write_csv(
            self.e4_csv,
            [
                {
                    "angle_bin": label,
                    "cluster_count": 10,
                    "mean_gain_mpjpe": 0.1234 + index * 0.001,
                    "median_gain_mpjpe": 0.12,
                    "mean_gain_ci95_low": 0.11,
                    "mean_gain_ci95_high": 0.13,
                    "rank_biserial": 0.9,
                    "p_holm": 0.006,
                }
                for index, label in enumerate(labels)
            ],
        )
        self.e4_json.write_text(
            json.dumps(
                {
                    "provenance": {"checkpoint_sha256": CHECKPOINT_HASH},
                    "omnibus": {
                        "test": "kruskal_wallis",
                        "p_value": 0.0123,
                        "epsilon_squared": 0.02,
                    },
                    "cluster_count": 60,
                    "within_bin": [],
                }
            ),
            encoding="utf-8",
        )
        e5_rows = []
        comparison_rows = []
        for view_index, view in enumerate(("left", "right", "both")):
            for pattern_index, pattern in enumerate(("random", "distal", "temporal")):
                for ratio in (0.5, 1.0):
                    fused = 0.2 + view_index * 0.01 + pattern_index * 0.02 + ratio * 0.01
                    e5_rows.append(
                        {
                            "view_mode": view,
                            "pattern": pattern,
                            "ratio": ratio,
                            "sample_count": 64440,
                            "joint_count": 15,
                            "fused_mpjpe": fused,
                            "canonical_avg_mpjpe": fused + 0.05,
                            "fusion_gain_percent": 20.0,
                            "sam3d_detection_failure_rate": 0.01 * ratio,
                            "checkpoint_sha256": CHECKPOINT_HASH,
                        }
                    )
                    comparison_rows.append(
                        {
                            "view_mode": view,
                            "pattern": pattern,
                            "ratio": ratio,
                            "image_fused_mpjpe": fused,
                            "pose_fused_mpjpe": fused + 0.1,
                        }
                    )
        self._write_csv(self.e5_csv, e5_rows)
        self._write_csv(self.comparison_csv, comparison_rows)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_renders_validated_tables_and_pdf_from_sources(self):
        outputs = render_all(
            e4_significance_csv=self.e4_csv,
            e4_statistics_json=self.e4_json,
            e5_summary_csv=self.e5_csv,
            comparison_csv=self.comparison_csv,
            table_root=self.table_root,
            figure_path=self.figure_path,
        )

        e4_text = (self.table_root / "view_angle_significance.tex").read_text()
        e5_text = (self.table_root / "image_occlusion_summary.tex").read_text()
        self.assertIn("Holm-adjusted", e4_text)
        self.assertIn("0.1234", e4_text)
        self.assertIn("Image-level", e5_text)
        self.assertIn("64,440", e5_text)
        self.assertTrue(self.figure_path.is_file())
        self.assertEqual(self.figure_path.read_bytes()[:4], b"%PDF")
        self.assertEqual(len(outputs), 3)

    def test_rejects_checkpoint_hash_mismatch(self):
        with self.e5_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["checkpoint_sha256"] = "wrong"
        self._write_csv(self.e5_csv, rows)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            render_all(
                e4_significance_csv=self.e4_csv,
                e4_statistics_json=self.e4_json,
                e5_summary_csv=self.e5_csv,
                comparison_csv=self.comparison_csv,
                table_root=self.table_root,
                figure_path=self.figure_path,
            )


if __name__ == "__main__":
    unittest.main()
