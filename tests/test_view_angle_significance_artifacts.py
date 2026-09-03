import csv
import json
import tempfile
import unittest
from pathlib import Path

from dual2pose.eval.run_unity_view_angle_significance import (
    validate_protocol_batch_size,
    write_angle_artifacts,
)


def _pair_rows(count):
    return [
        {
            "person_id": "female",
            "action_id": f"action_{index % 4}",
            "camera_pair_id": f"cam_{index // 4}|cam_{index // 4 + 1}",
            "cam1_id": f"cam_{index // 4}",
            "cam2_id": f"cam_{index // 4 + 1}",
            "separation_deg": 10.0,
            "angle_bin": "0-30",
            "sample_count": 1,
            "frame_count": 30,
            "fused_mpjpe": 0.1,
            "canonical_avg_mpjpe": 0.2,
            "fusion_gain_mpjpe": 0.1,
            "fusion_gain_percent": 50.0,
        }
        for index in range(count)
    ]


def _statistics(action_rows, clusters):
    return {
        "analysis_unit": "unordered_camera_pair_averaged_across_test_actions",
        "seed": 42,
        "bootstrap_resamples": 100,
        "action_pair_row_count": action_rows,
        "cluster_count": clusters,
        "angle_bins": ["0-30", "30-60", "60-90", "90-120", "120-150", "150-180"],
        "within_bin": [
            {
                "angle_bin": label,
                "cluster_count": 1,
                "test": "wilcoxon_signed_rank",
                "p_raw": 0.01,
                "p_holm": 0.06,
            }
            for label in ["0-30", "30-60", "60-90", "90-120", "120-150", "150-180"]
        ],
        "omnibus": {
            "test": "kruskal_wallis",
            "p_value": 0.1,
            "significant_0_05": False,
        },
        "pairwise_contrasts": [],
    }


class ViewAngleSignificanceArtifactTest(unittest.TestCase):
    def test_batch_referenced_protocol_requires_archived_batch_size(self):
        validate_protocol_batch_size(256)
        with self.assertRaisesRegex(ValueError, "batch_size=256"):
            validate_protocol_batch_size(4096)

    def test_writer_rejects_incomplete_action_pair_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "64440"):
                write_angle_artifacts(
                    Path(directory),
                    pair_rows=[],
                    statistics=_statistics(0, 0),
                )

    def test_writer_emits_declared_files_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_angle_artifacts(
                Path(directory),
                pair_rows=_pair_rows(8),
                statistics=_statistics(8, 2),
                provenance={"checkpoint_sha256": "abc", "fold": 0},
                expected_action_rows=8,
                expected_clusters=2,
            )
            names = {path.name for path in paths}
            self.assertEqual(
                names,
                {
                    "view_angle_per_pair_last.csv",
                    "view_angle_significance_last.csv",
                    "view_angle_pairwise_contrasts_last.csv",
                    "view_angle_statistics_last.json",
                },
            )
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            payload = json.loads(
                (Path(directory) / "view_angle_statistics_last.json").read_text()
            )
            self.assertEqual(payload["provenance"]["checkpoint_sha256"], "abc")
            with (
                Path(directory) / "view_angle_significance_last.csv"
            ).open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)


if __name__ == "__main__":
    unittest.main()
