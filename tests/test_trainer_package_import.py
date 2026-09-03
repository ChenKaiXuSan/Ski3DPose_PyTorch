import importlib
import unittest


class TrainerPackageImportTest(unittest.TestCase):
    """Break caught: evaluators cannot import the trainer through dual2pose.*."""

    def test_crossview_trainer_supports_package_import(self) -> None:
        module = importlib.import_module("dual2pose.trainer.train_crossview_fusion")
        self.assertTrue(hasattr(module, "CrossViewFusionTrainer"))


if __name__ == "__main__":
    unittest.main()
