import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dual2pose.eval.eval_unity_image_occlusion import (
    ImageOcclusionManifest,
    build_argument_parser,
    build_image_occlusion_study,
    replace_image_occlusion_inputs,
    summarize_cell,
)
from dual2pose.eval.frontend_manifest import FrontEndManifest


class ImageOcclusionReplacementTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        entries = {}
        for camera, failed in (
            ("capture_L0_A000", np.array([False, True])),
            ("capture_L0_A010", np.array([True, False])),
        ):
            pose = np.zeros((2, 15, 3), dtype=np.float32)
            pose[0, :, 0] = 2.0
            pose[1, :, 0] = 7.0
            path = self.root / f"{camera}.npz"
            np.savez_compressed(
                path,
                pose=pose,
                frame_indices=np.array([2, 7]),
                detection_failed=failed,
            )
            entries[("female", "turn", camera)] = path
        pose_manifest = FrontEndManifest(
            frontend_name="image_occlusion",
            entries=entries,
            source_path=self.root / "manifest.json",
        )
        self.manifest = ImageOcclusionManifest(pose_manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def _sample(self):
        return {
            "meta": {
                "person_id": "female",
                "action_id": "turn",
                "cam1_id": "capture_L0_A000",
                "cam2_id": "capture_L0_A010",
            },
            "frame_indices": torch.tensor([2, 2, 7]),
            "kpt3d_sam": {
                "cam1": torch.full((3, 15, 3), 100.0),
                "cam2": torch.full((3, 15, 3), 200.0),
            },
        }

    def test_left_mode_replaces_only_cam1_and_aligns_duplicates(self):
        sample = self._sample()
        actual = replace_image_occlusion_inputs(sample, self.manifest, "left")

        self.assertEqual(
            actual["kpt3d_sam"]["cam1"][:, 0, 0].tolist(),
            [2.0, 2.0, 7.0],
        )
        self.assertTrue(
            torch.equal(actual["kpt3d_sam"]["cam2"], sample["kpt3d_sam"]["cam2"])
        )
        self.assertEqual(
            actual["image_occlusion_failed"]["cam1"].tolist(),
            [False, False, True],
        )

    def test_both_mode_exposes_pair_aligned_boolean_failure_flags(self):
        actual = replace_image_occlusion_inputs(
            self._sample(), self.manifest, "both"
        )
        self.assertEqual(actual["image_occlusion_failed"]["cam1"].dtype, torch.bool)
        self.assertEqual(actual["image_occlusion_failed"]["cam2"].dtype, torch.bool)
        self.assertEqual(
            actual["image_occlusion_failed"]["cam2"].tolist(),
            [True, True, False],
        )


class ImageOcclusionStudyTest(unittest.TestCase):
    def test_cli_defaults_to_archived_batch_referenced_protocol(self):
        parser = build_argument_parser()
        args = parser.parse_args(
            ["--manifest-root", "/tmp/in", "--output-root", "/tmp/out"]
        )
        self.assertEqual(args.batch_size, 256)

    def test_study_has_exactly_eighteen_cells(self):
        cells = build_image_occlusion_study()
        self.assertEqual(len(cells), 18)
        self.assertEqual({cell.view_mode for cell in cells}, {"left", "right", "both"})
        self.assertEqual({cell.ratio for cell in cells}, {0.5, 1.0})

    def test_summary_keeps_negative_gain_and_failure_rate(self):
        ground_truth = torch.zeros((1, 2, 15, 3), dtype=torch.float32)
        canonical = ground_truth.clone()
        canonical[..., 0] = 1.0
        fused = ground_truth.clone()
        fused[..., 0] = 2.0
        outputs = [
            {
                "fused": fused,
                "alpha": torch.full((1, 2, 15, 1), 0.5),
                "p_left": canonical,
                "p_right": canonical,
                "left_canonical": canonical,
                "right_canonical": canonical,
                "ground_truth": ground_truth,
                "ground_truth_canonical": ground_truth,
            }
        ]
        row = summarize_cell(
            outputs,
            failure_flags={"cam1": torch.tensor([False, True])},
            failure_threshold=0.15,
        )

        self.assertLess(row["fusion_gain_percent"], 0.0)
        self.assertEqual(row["sam3d_detection_failure_rate"], 0.5)
        self.assertEqual(row["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
