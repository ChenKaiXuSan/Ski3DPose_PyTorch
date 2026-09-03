from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dual2pose.experiments.run_frontend_adaptation_matrix import _schedule


@dataclass(frozen=True)
class Job:
    name: str


class FinishedProcess:
    pid = 123

    def poll(self):
        return 0


class FrontEndAdaptationSchedulerTest(unittest.TestCase):
    """Break caught: scheduler writes a reloaded manifest without mutated phase records."""

    def test_terminal_record_is_written_to_the_same_manifest_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"training": {}, "evaluation": {}}
            written = []
            with patch(
                "dual2pose.experiments.run_frontend_adaptation_matrix.gpu_compute_processes",
                return_value=[],
            ), patch(
                "dual2pose.experiments.run_frontend_adaptation_matrix.subprocess.Popen",
                return_value=FinishedProcess(),
            ), patch(
                "dual2pose.experiments.run_frontend_adaptation_matrix._atomic_json_write",
                side_effect=lambda path, payload: written.append(payload.copy()),
            ), patch(
                "dual2pose.experiments.run_frontend_adaptation_matrix.time.sleep"
            ):
                result = _schedule(
                    manifest=manifest,
                    jobs=[Job("one")],
                    records=manifest["training"],
                    job_name=lambda job: job.name,
                    job_directory=lambda job: root / job.name,
                    command_builder=lambda job: (["true"], {}),
                    gpus=[0],
                    poll_seconds=0.01,
                )
        self.assertEqual(result, 0)
        self.assertEqual(manifest["training"]["one"]["status"], "complete")
        self.assertEqual(written[-1]["training"]["one"]["status"], "complete")

    def test_shared_gpu_mode_ignores_existing_compute_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"training": {}, "evaluation": {}}
            with patch(
                "dual2pose.experiments.run_frontend_adaptation_matrix.gpu_compute_processes",
                return_value=[{"pid": "999", "process_name": "unrelated"}],
            ), patch(
                "dual2pose.experiments.run_frontend_adaptation_matrix.subprocess.Popen",
                return_value=FinishedProcess(),
            ) as popen, patch(
                "dual2pose.experiments.run_frontend_adaptation_matrix._atomic_json_write"
            ), patch(
                "dual2pose.experiments.run_frontend_adaptation_matrix.time.sleep"
            ):
                result = _schedule(
                    manifest=manifest,
                    jobs=[Job("one")],
                    records=manifest["training"],
                    job_name=lambda job: job.name,
                    job_directory=lambda job: root / job.name,
                    command_builder=lambda job: (["true"], {}),
                    gpus=[1],
                    poll_seconds=0.01,
                    allow_shared_gpu=True,
                )
        self.assertEqual(result, 0)
        popen.assert_called_once()
        self.assertEqual(manifest["training"]["one"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
