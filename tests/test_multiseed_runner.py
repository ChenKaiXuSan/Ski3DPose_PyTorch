from pathlib import Path
import tempfile
import unittest

from dual2pose.experiments.run_multiseed_crossfold import (
    RunKey,
    build_run_matrix,
    build_training_command,
    prepare_run_directory,
    transition_status,
)


class MultiSeedRunnerTest(unittest.TestCase):
    """Breaks caught: a matrix cell is omitted, overwritten, or targets the wrong fold."""

    def test_matrix_is_three_seeds_by_two_folds(self) -> None:
        self.assertEqual(
            set(build_run_matrix()),
            {
                RunKey(0, 13), RunKey(0, 42), RunKey(0, 73),
                RunKey(1, 13), RunKey(1, 42), RunKey(1, 73),
            },
        )

    def test_command_uses_visible_device_zero_and_unique_log_path(self) -> None:
        command = build_training_command(
            RunKey(1, 73),
            gpu=1,
            repo_root=Path("/repo"),
            python_executable=Path("/env/bin/python"),
            data_root=Path("/data/unity"),
        )
        self.assertEqual(command[:2], ["/env/bin/python", "/repo/dual2pose/train_unity.py"])
        self.assertIn("train.gpu=0", command)
        self.assertIn("train.fold=1", command)
        self.assertIn("train.seed=73", command)
        self.assertIn("train.max_epochs=100", command)
        self.assertIn("data.unity.root_path=/data/unity", command)
        self.assertIn("log_path=/repo/logs/ivc_p1/multiseed/fold_1/seed_73", command)

    def test_only_declared_state_transitions_are_allowed(self) -> None:
        self.assertEqual(transition_status("pending", "running"), "running")
        self.assertEqual(transition_status("running", "complete"), "complete")
        with self.assertRaisesRegex(ValueError, "Invalid run transition"):
            transition_status("failed", "running")

    def test_nonempty_untracked_run_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "fold_0" / "seed_13"
            run_dir.mkdir(parents=True)
            (run_dir / "orphan.ckpt").write_bytes(b"unsafe overwrite")
            with self.assertRaisesRegex(RuntimeError, "untracked files"):
                prepare_run_directory(run_dir, known_record=None)


if __name__ == "__main__":
    unittest.main()
