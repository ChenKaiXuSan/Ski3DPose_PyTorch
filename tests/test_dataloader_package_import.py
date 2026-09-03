import importlib
import unittest


class DataLoaderPackageImportTest(unittest.TestCase):
    """Break caught: module-style evaluators cannot import the Unity dataloader."""

    def test_unity_dataloader_supports_package_import(self) -> None:
        module = importlib.import_module("dual2pose.dataloader.data_loader")
        self.assertTrue(hasattr(module, "UnityDataModule"))


if __name__ == "__main__":
    unittest.main()
