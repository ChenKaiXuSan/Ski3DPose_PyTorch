from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dual2pose.eval.frontend_lifters import _checkpoint_state


class FrontEndCheckpointLoadingTest(unittest.TestCase):
    def test_numpy_training_state_can_be_allowlisted_without_unsafe_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "poseformer.bin"
            torch.save(
                {
                    "model_pos": {"weight": torch.tensor([1.0])},
                    "random_state": np.random.RandomState(42),
                },
                checkpoint,
            )
            with self.assertRaises(Exception):
                _checkpoint_state(checkpoint)
            state = _checkpoint_state(
                checkpoint, allow_numpy_checkpoint_state=True
            )
            self.assertEqual(state["weight"].tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
