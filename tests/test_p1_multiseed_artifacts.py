import unittest

from paper.ivc_draft_20260821.scripts.generate_p1_multiseed_artifacts import (
    render_table,
    validate_payload,
)


class P1MultiSeedArtifactsTest(unittest.TestCase):
    def _payload(self):
        runs = []
        for fold in (0, 1):
            for seed in (13, 42, 73):
                runs.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "mpjpe": 0.2 + fold * 0.01 + seed / 100000,
                        "sample_count": 64440,
                        "best_checkpoint": f"fold{fold}-seed{seed}.ckpt",
                        "checkpoint_sha256": f"sha-{fold}-{seed}",
                        "best_epoch": 10,
                        "per_action": {},
                    }
                )
        return {
            "runs": runs,
            "summary": {
                "aggregate": {
                    "run_count": 6,
                    "mpjpe_mean": 0.205,
                    "mpjpe_std": 0.01,
                    "mpjpe_ci95_low": 0.194,
                    "mpjpe_ci95_high": 0.216,
                    "interval": "two-sided t interval over six training runs, df=5",
                },
                "per_fold": {
                    "0": {"run_count": 3, "mpjpe_mean": 0.2, "mpjpe_std": 0.01},
                    "1": {"run_count": 3, "mpjpe_mean": 0.21, "mpjpe_std": 0.01},
                },
                "per_action": {},
            },
        }

    def test_complete_matrix_renders_df5_interval(self):
        payload = validate_payload(self._payload())
        table = render_table(payload)
        self.assertIn("six fresh training runs", table)
        self.assertIn("df=5", table)
        self.assertIn("Fold 1", table)

    def test_missing_run_is_rejected(self):
        payload = self._payload()
        payload["runs"] = payload["runs"][:-1]
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            validate_payload(payload)

    def test_incomplete_test_split_is_rejected(self):
        payload = self._payload()
        payload["runs"][0]["sample_count"] = 1
        with self.assertRaisesRegex(ValueError, "64,440"):
            validate_payload(payload)


if __name__ == "__main__":
    unittest.main()
