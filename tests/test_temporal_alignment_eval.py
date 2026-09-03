import unittest

import torch

from dual2pose.eval import eval_unity_temporal_alignment as alignment_eval
from dual2pose.eval.eval_unity_temporal_alignment import build_right_stream_variants
from dual2pose.eval.eval_unity_temporal_offset import _shift_pose_sequence


class TemporalAlignmentEvalTest(unittest.TestCase):
    """Break caught: automatic correction is applied as one batch-wide lag."""

    def test_three_branches_use_per_sample_automatic_corrections(self) -> None:
        right = torch.arange(8, dtype=torch.float32).view(2, 4, 1, 1)
        estimates = torch.tensor([-1.0, 0.0])
        variants = build_right_stream_variants(
            right,
            injected_offset=1.0,
            estimated_corrections=estimates,
        )
        injected = _shift_pose_sequence(right, 1.0)
        self.assertTrue(torch.equal(variants["uncorrected"], injected))
        self.assertTrue(
            torch.equal(
                variants["automatic"][0:1],
                _shift_pose_sequence(injected[0:1], -1.0),
            )
        )
        self.assertTrue(torch.equal(variants["automatic"][1:2], injected[1:2]))
        self.assertTrue(torch.equal(variants["oracle"], right))

    def test_estimate_shape_must_match_batch(self) -> None:
        right = torch.zeros((2, 4, 1, 3))
        with self.assertRaisesRegex(ValueError, "one correction per sample"):
            build_right_stream_variants(right, 1.0, torch.tensor([-1.0]))

    def test_fixed_metric_slice_excludes_the_maximum_offset_margin(self) -> None:
        metric_slice = getattr(alignment_eval, "fixed_metric_time_slice", None)
        self.assertTrue(callable(metric_slice))
        self.assertEqual(metric_slice(30, 5.0), slice(5, 25))
        self.assertEqual(metric_slice(30, 0.5), slice(1, 29))

    def test_fixed_metric_slice_rejects_an_empty_center(self) -> None:
        metric_slice = getattr(alignment_eval, "fixed_metric_time_slice", None)
        self.assertTrue(callable(metric_slice))
        with self.assertRaisesRegex(ValueError, "no valid center frames"):
            metric_slice(10, 5.0)


if __name__ == "__main__":
    unittest.main()
