import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dual2pose.eval.frontend_comparison import build_comparison_rows
from dual2pose.eval.run_unity_frontend_suite import _manifest_for


class FrontEndComparisonTest(unittest.TestCase):
    def test_comparison_uses_common13_and_rejects_mixed_protocols(self) -> None:
        rows = [
            {
                "frontend_name": "videopose3d",
                "checkpoint": "/tmp/fusion.ckpt",
                "checkpoint_sha256": "abc",
                "joint_subset": "common13",
                "units": "dataset_coordinate_units",
                "fold": 0,
                "sample_count": 10,
                "common13_raw_avg_mpjpe": 0.20,
                "common13_canonical_avg_mpjpe": 0.16,
                "common13_fused_mpjpe": 0.12,
            },
            {
                "frontend_name": "poseformer",
                "checkpoint": "/tmp/fusion.ckpt",
                "checkpoint_sha256": "abc",
                "joint_subset": "common13",
                "units": "dataset_coordinate_units",
                "fold": 0,
                "sample_count": 10,
                "common13_raw_avg_mpjpe": 0.18,
                "common13_canonical_avg_mpjpe": 0.15,
                "common13_fused_mpjpe": 0.10,
            },
        ]
        actual = build_comparison_rows(rows)
        self.assertEqual(
            [row["frontend_name"] for row in actual],
            ["poseformer", "videopose3d"],
        )
        self.assertEqual(
            [row["rank_common13_fused_mpjpe"] for row in actual], [1, 2]
        )
        self.assertAlmostEqual(actual[0]["fusion_gain_common13_mpjpe"], 0.05)
        self.assertAlmostEqual(actual[0]["fusion_gain_common13_percent"], 100.0 / 3.0)
        rows[1]["sample_count"] = 9
        with self.assertRaisesRegex(ValueError, "sample_count"):
            build_comparison_rows(rows)

        rows[1]["sample_count"] = 10
        rows[1]["checkpoint_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            build_comparison_rows(rows)


class FrontEndSuiteTest(unittest.TestCase):
    def test_explicit_manifest_is_reused_without_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pose = root / "pose.npy"
            np.save(pose, np.zeros((2, 15, 3), dtype=np.float32))
            manifest = root / "existing_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "frontend_name": "videopose3d",
                        "entries": [
                            {
                                "person_id": "person",
                                "action_id": "action",
                                "camera_id": "camera",
                                "pose_path": pose.name,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            actual = _manifest_for(
                "videopose3d",
                {"manifest": str(manifest)},
                {"output_root": str(root / "new_results")},
                overwrite=False,
            )
            self.assertEqual(actual, manifest.resolve())


if __name__ == "__main__":
    unittest.main()
