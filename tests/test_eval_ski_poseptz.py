import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from dual2pose.eval.eval_ski_poseptz import (
    _build_config,
    _build_provenance,
    _summarize_test_outputs,
)
from dual2pose.dataloader.ski_poseptz_dataset_dual_view import (
    load_ski_poseptz_index_mapping,
)


class SkiPosePTZJournalEvaluationTest(unittest.TestCase):
    """Regression coverage for the provenance-controlled Ski-PTZ evaluation."""

    @staticmethod
    def _batch(batch_size: int, final_x: float) -> dict[str, torch.Tensor]:
        ground_truth = torch.zeros((batch_size, 3, 1, 3), dtype=torch.float32)
        prediction = ground_truth.clone()
        prediction[:, 2, 0, 0] = float(final_x)
        return {
            "fused": prediction,
            "p_left": prediction,
            "p_right": prediction,
            "left_canonical": prediction,
            "right_canonical": prediction,
            "ground_truth": ground_truth,
            "ground_truth_canonical": ground_truth,
        }

    def test_summary_weights_every_joint_frame_instead_of_every_batch(self) -> None:
        # Break caught: the final partial batch receives the same weight as a full batch.
        summary = _summarize_test_outputs(
            [self._batch(batch_size=2, final_x=1.0), self._batch(batch_size=1, final_x=4.0)]
        )

        self.assertAlmostEqual(summary["fused"]["mpjpe"], 6.0 / 9.0, places=6)
        self.assertAlmostEqual(summary["fused"]["accel_err"], 6.0 / 3.0, places=6)
        self.assertAlmostEqual(summary["canonical_avg"]["mpjpe"], 6.0 / 9.0, places=6)

    def test_output_path_is_used_for_trainer_and_summary_artifacts(self) -> None:
        # Break caught: the clean re-evaluation overwrites the checkpoint run's old report.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "run" / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            output_path = root / "journal_eval"
            args = SimpleNamespace(
                config_path=Path("configs/dual2pose.yaml"),
                ckpt_path=checkpoint,
                backbone="crossview_fusion",
                batch_size=4,
                num_workers=0,
                time_window=30,
                output_path=output_path,
            )

            config = _build_config(args)

        self.assertEqual(Path(config.log_path), output_path.resolve())

    def test_provenance_hashes_checkpoint_and_fixed_test_index(self) -> None:
        # Break caught: a result row cannot be traced to the evaluated weights and split.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.ckpt"
            index_mapping = root / "index.json"
            checkpoint.write_bytes(b"checkpoint")
            index_mapping.write_bytes(b'{"test": []}')

            provenance = _build_provenance(
                checkpoint=checkpoint,
                index_mapping=index_mapping,
                sample_count=30,
                split="test",
                seed=42,
                joint_subset="common13",
                units="normalized_dataset_coordinates",
                batch_size=4,
                time_window=30,
                path_rewrite_from="/old/ski",
                path_rewrite_to="/new/ski",
            )

        self.assertEqual(
            provenance["checkpoint_sha256"],
            hashlib.sha256(b"checkpoint").hexdigest(),
        )
        self.assertEqual(
            provenance["index_mapping_sha256"],
            hashlib.sha256(b'{"test": []}').hexdigest(),
        )
        self.assertEqual(provenance["sample_count"], 30)
        self.assertEqual(provenance["joint_subset"], "common13")
        self.assertFalse(provenance["drop_last"])
        self.assertEqual(provenance["path_rewrite_from"], "/old/ski")
        self.assertEqual(provenance["path_rewrite_to"], "/new/ski")

    def test_fixed_index_paths_are_rewritten_in_memory(self) -> None:
        # Break caught: an archived absolute root makes the fixed test split unloadable.
        with tempfile.TemporaryDirectory() as directory:
            index_mapping = Path(directory) / "index.json"
            index_mapping.write_text(
                '{"test": [{"cam1_frames_dir": "/old/ski/data/test/cam_00"}], '
                '"_metadata": {"data_root": "/old/ski"}}',
                encoding="utf-8",
            )

            rows = load_ski_poseptz_index_mapping(
                index_mapping,
                split="test",
                path_rewrite_from="/old/ski",
                path_rewrite_to="/new/ski",
            )

        self.assertEqual(
            rows[0]["cam1_frames_dir"],
            "/new/ski/data/test/cam_00",
        )


if __name__ == "__main__":
    unittest.main()
