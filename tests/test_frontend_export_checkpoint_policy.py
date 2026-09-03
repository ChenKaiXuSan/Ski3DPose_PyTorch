import unittest

from dual2pose.experiments.export_frontend_splits import FRONTEND_CONFIGS


class FrontEndExportCheckpointPolicyTest(unittest.TestCase):
    def test_poseformer_uses_numpy_safe_globals_not_unsafe_pickle(self) -> None:
        config = FRONTEND_CONFIGS["poseformer"]
        self.assertTrue(config["allow_numpy_checkpoint_state"])
        self.assertFalse(config.get("allow_unsafe_checkpoint", False))

    def test_unsafe_pickle_is_limited_to_declared_motionbert_checkpoint(self) -> None:
        self.assertTrue(FRONTEND_CONFIGS["motionbert"]["allow_unsafe_checkpoint"])
        self.assertFalse(FRONTEND_CONFIGS["videopose3d"].get("allow_unsafe_checkpoint", False))


if __name__ == "__main__":
    unittest.main()
