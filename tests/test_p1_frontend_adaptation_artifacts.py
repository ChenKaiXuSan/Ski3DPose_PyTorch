from pathlib import Path
import subprocess
import sys
import unittest

from dual2pose.experiments.run_frontend_adaptation_matrix import build_evaluation_cells
from paper.ivc_draft_20260821.scripts.generate_p1_frontend_adaptation_artifacts import (
    render_table,
    validate_matrix_rows,
)


class P1FrontendAdaptationArtifactsTest(unittest.TestCase):
    def _rows(self):
        rows = []
        for index, cell in enumerate(build_evaluation_cells()):
            rows.append(
                {
                    "model_name": cell.model_name,
                    "test_frontend": cell.test_frontend,
                    "common13_mpjpe": str(0.1 + index / 1000),
                    "common13_acceleration_error": "0.02",
                    "sample_count": "64440",
                    "checkpoint": "/tmp/model.ckpt",
                    "manifest": "/tmp/manifest.json",
                    "metrics_json": "/tmp/metrics.json",
                    "recovery_vs_mmsports": "0.01",
                    "relative_recovery_percent": "5.0",
                    "sam3d_retention_delta": "0.0",
                }
            )
        return rows

    def test_exact_8_by_4_matrix_renders(self):
        rows = validate_matrix_rows(self._rows())
        table = render_table(rows)
        self.assertIn("Frozen reference", table)
        self.assertIn("Mixed/full", table)
        self.assertIn("MotionBERT", table)

    def test_missing_cell_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            validate_matrix_rows(self._rows()[:-1])

    def test_non_full_test_split_is_rejected(self):
        rows = self._rows()
        rows[-1]["sample_count"] = "0"
        with self.assertRaisesRegex(ValueError, "64,440"):
            validate_matrix_rows(rows)

    def test_script_path_bootstraps_repository_imports(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "paper/ivc_draft_20260821/scripts/generate_p1_frontend_adaptation_artifacts.py"
        code = """
from pathlib import Path
import runpy
import sys

repo_root = Path.cwd().resolve()
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != repo_root
]
runpy.run_path(sys.argv[1], run_name="artifact_import_test")
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(script)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
