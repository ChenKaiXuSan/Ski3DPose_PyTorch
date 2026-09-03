from pathlib import Path
import tempfile
import unittest

from dual2pose.eval.eval_unity_nview import resolve_nview_inputs


class NViewDataRootTest(unittest.TestCase):
    """Catch regressions where a relocated Unity root is ignored."""

    def test_explicit_override_replaces_stale_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            current_root = base / "current_unity"
            fold_path = (
                current_root
                / "index_mapping/use_layer_camera_filter_disabled"
                / "camera_pairs_by_action_folds/fold_00.json"
            )
            fold_path.parent.mkdir(parents=True)
            fold_path.write_text(
                '{"train": [], "val": [], "test": [], "_metadata": {"fold_idx": 0}}',
                encoding="utf-8",
            )

            data_root, resolved_fold = resolve_nview_inputs(
                configured_root=base / "stale_unity",
                explicit_root=str(current_root),
            )

        self.assertEqual(data_root, current_root.resolve())
        self.assertEqual(resolved_fold, fold_path.resolve())


if __name__ == "__main__":
    unittest.main()
