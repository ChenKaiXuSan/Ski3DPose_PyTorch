import unittest

import torch

from dual2pose.eval.eval_unity_nview import (
    evaluate_nview_group,
    summarize_nview_efficiency,
    warmup_pairwise_composition,
)
from dual2pose.eval.nview_protocol import CameraGroup, MultiViewSample


class FakePairModel:
    def __init__(self, outputs: dict[tuple[str, str], float]) -> None:
        self.outputs = outputs
        self.call_count = 0

    def predict_pair(
        self,
        left_camera: str,
        right_camera: str,
        left_pose: torch.Tensor,
        right_pose: torch.Tensor,
    ) -> torch.Tensor:
        del right_pose
        self.call_count += 1
        value = self.outputs[(left_camera, right_camera)]
        return torch.full_like(left_pose, value)


def make_sample(*cameras: str) -> MultiViewSample:
    padded = tuple(cameras) + tuple(f"unused_{index}" for index in range(4 - len(cameras)))
    group = CameraGroup("fixture", "male", "action", 0, padded[:4])
    poses = {camera: torch.zeros((5, 15, 3)) for camera in group.cameras}
    return MultiViewSample(
        group=group,
        frame_indices=torch.arange(5),
        poses=poses,
        ground_truth=torch.zeros((5, 15, 3)),
    )


class UnityNViewEvalTest(unittest.TestCase):
    """Breaks caught: N-view aggregation omits a pair or oracle-selects per joint."""

    def test_three_views_average_three_pair_predictions(self) -> None:
        model = FakePairModel(
            outputs={
                ("a", "b"): 1.0,
                ("a", "c"): 2.0,
                ("b", "c"): 3.0,
            }
        )
        row = evaluate_nview_group(model, make_sample("a", "b", "c"), n_views=3)
        self.assertTrue(
            torch.allclose(
                row["pairwise_canonfuse_mean"],
                torch.full_like(row["gt"], 2.0),
            )
        )
        self.assertEqual(row["pair_forward_count"], 3)

    def test_oracle_selects_one_complete_pair_prediction(self) -> None:
        model = FakePairModel(
            outputs={
                ("a", "b"): 3.0,
                ("a", "c"): 1.0,
                ("b", "c"): 2.0,
            }
        )
        row = evaluate_nview_group(model, make_sample("a", "b", "c"), n_views=3)
        self.assertEqual(row["pairwise_oracle_pair"], ("a", "c"))
        self.assertTrue(
            torch.equal(row["pairwise_oracle_select"], torch.ones_like(row["gt"]))
        )

    def test_two_views_runs_exactly_the_declared_opposite_pair(self) -> None:
        model = FakePairModel(outputs={("a", "c"): 4.0})
        row = evaluate_nview_group(model, make_sample("a", "b", "c", "d"), n_views=2)
        self.assertEqual(row["selected_cameras"], ("a", "c"))
        self.assertEqual(row["pair_forward_count"], 1)
        self.assertTrue(
            torch.equal(row["pairwise_canonfuse_mean"], torch.full_like(row["gt"], 4.0))
        )

    def test_warmup_executes_every_four_view_pair_before_measurement(self) -> None:
        model = FakePairModel(
            outputs={
                ("a", "b"): 1.0,
                ("a", "c"): 1.0,
                ("a", "d"): 1.0,
                ("b", "c"): 1.0,
                ("b", "d"): 1.0,
                ("c", "d"): 1.0,
            }
        )

        warmup_pairwise_composition(
            model,
            make_sample("a", "b", "c", "d"),
            iterations=3,
        )

        self.assertEqual(model.call_count, 18)

    def test_efficiency_summary_uses_only_deployable_pairwise_rows(self) -> None:
        rows = []
        timings = {
            2: (1, (0.010, 0.020)),
            3: (3, (0.030, 0.030)),
            4: (6, (0.060, 0.060)),
        }
        for n_views, (pair_count, seconds) in timings.items():
            for group_index, elapsed in enumerate(seconds):
                rows.append(
                    {
                        "group_id": f"g{group_index}",
                        "n_views": n_views,
                        "method": "pairwise_canonfuse_mean",
                        "mpjpe": 0.20 - 0.01 * n_views,
                        "pair_forward_count": pair_count,
                        "inference_seconds": elapsed,
                        "peak_gpu_bytes": 20 * 1024 * 1024,
                    }
                )
                rows.append(
                    {
                        "group_id": f"g{group_index}",
                        "n_views": n_views,
                        "method": "pairwise_oracle_select",
                        "mpjpe": 0.01,
                        "pair_forward_count": pair_count,
                        "inference_seconds": elapsed,
                        "peak_gpu_bytes": 20 * 1024 * 1024,
                    }
                )

        summary = summarize_nview_efficiency(
            rows,
            warmup_iterations=10,
            device_name="Fixture GPU",
            torch_version="fixture",
            cuda_version="fixture",
        )

        self.assertEqual([row["n_views"] for row in summary], [2, 3, 4])
        self.assertEqual([row["pair_forward_count"] for row in summary], [1, 3, 6])
        self.assertAlmostEqual(summary[0]["latency_mean_ms"], 15.0)
        self.assertAlmostEqual(summary[0]["latency_p95_ms"], 19.5)
        self.assertAlmostEqual(summary[0]["throughput_groups_per_second"], 1000.0 / 15.0)
        self.assertAlmostEqual(summary[1]["relative_latency_vs_two_view"], 2.0)
        self.assertAlmostEqual(summary[2]["relative_latency_vs_two_view"], 4.0)
        self.assertAlmostEqual(summary[2]["peak_gpu_memory_mib"], 20.0)
        self.assertEqual(summary[2]["warmup_iterations"], 10)

    def test_efficiency_summary_rejects_wrong_pair_count(self) -> None:
        rows = [
            {
                "group_id": "g0",
                "n_views": n_views,
                "method": "pairwise_canonfuse_mean",
                "mpjpe": 0.1,
                "pair_forward_count": pair_count,
                "inference_seconds": 0.01,
                "peak_gpu_bytes": 0,
            }
            for n_views, pair_count in ((2, 1), (3, 2), (4, 6))
        ]
        with self.assertRaisesRegex(ValueError, "pair count"):
            summarize_nview_efficiency(rows)


if __name__ == "__main__":
    unittest.main()
