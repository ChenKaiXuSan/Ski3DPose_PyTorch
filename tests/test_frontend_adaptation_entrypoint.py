from pathlib import Path
import subprocess
import unittest


class FrontEndAdaptationEntrypointTest(unittest.TestCase):
    """Break caught: legacy and package imports cannot coexist in the script entrypoint."""

    def test_hydra_config_dry_run_imports_successfully(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                "/home/kaixu_chen/miniforge3/envs/dual2pose/bin/python",
                "dual2pose/train_frontend_adaptation.py",
                "--cfg",
                "job",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("adaptation:", result.stdout)
        self.assertIn("epochs: 20", result.stdout)


if __name__ == "__main__":
    unittest.main()
