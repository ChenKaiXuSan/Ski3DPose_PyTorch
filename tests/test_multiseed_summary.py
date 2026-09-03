import math
import unittest

from dual2pose.eval.summarize_multiseed_crossfold import (
    RunMetrics,
    summarize_training_runs,
)


class MultiSeedSummaryTest(unittest.TestCase):
    """Breaks caught: uncertainty uses frames, population std, or an incomplete run matrix."""

    def _rows(self) -> list[RunMetrics]:
        values = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        return [
            RunMetrics(
                fold=fold,
                seed=seed,
                mpjpe=value,
                sample_count=10,
                best_checkpoint=f"fold{fold}_seed{seed}.ckpt",
                checkpoint_sha256=f"sha-{fold}-{seed}",
                best_epoch=5,
                per_action={"action": {"sample_count": 10, "mpjpe": value + 1.0}},
            )
            for fold in (0, 1)
            for seed in (13, 42, 73)
            for value in [next(values)]
        ]

    def test_six_run_mean_sample_std_and_t_interval(self) -> None:
        summary = summarize_training_runs(self._rows())
        aggregate = summary["aggregate"]
        self.assertEqual(aggregate["run_count"], 6)
        self.assertEqual(aggregate["mpjpe_mean"], 3.5)
        self.assertAlmostEqual(aggregate["mpjpe_std"], math.sqrt(3.5))
        margin = 2.5705818366147395 * math.sqrt(3.5) / math.sqrt(6)
        self.assertAlmostEqual(aggregate["mpjpe_ci95_low"], 3.5 - margin)
        self.assertAlmostEqual(aggregate["mpjpe_ci95_high"], 3.5 + margin)
        self.assertEqual(summary["per_action"]["action"]["mpjpe_mean"], 4.5)

    def test_duplicate_or_missing_cell_is_rejected(self) -> None:
        rows = self._rows()
        with self.assertRaisesRegex(ValueError, "matrix"):
            summarize_training_runs(rows[:-1] + [rows[0]])


if __name__ == "__main__":
    unittest.main()
