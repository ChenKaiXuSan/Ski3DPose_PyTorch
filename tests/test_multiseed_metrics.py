import unittest

import torch

from dual2pose.eval.multiseed_metrics import summarize_test_outputs_by_action


class MultiSeedMetricsTest(unittest.TestCase):
    """Break caught: per-action uncertainty is computed from frames instead of samples."""

    def test_outputs_are_reduced_to_one_mpjpe_per_pair_sample(self) -> None:
        gt = torch.zeros((3, 2, 1, 3))
        fused = gt.clone()
        fused[0, :, :, 0] = 1.0
        fused[1, :, :, 0] = 3.0
        fused[2, :, :, 0] = 2.0
        result = summarize_test_outputs_by_action(
            [
                {
                    "fused": fused,
                    "ground_truth_canonical": gt,
                    "meta": {"action_id": ["a", "a", "b"]},
                }
            ]
        )
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["mpjpe"], 2.0)
        self.assertEqual(result["per_action"]["a"], {"sample_count": 2, "mpjpe": 2.0})
        self.assertEqual(result["per_action"]["b"], {"sample_count": 1, "mpjpe": 2.0})


if __name__ == "__main__":
    unittest.main()
