"""Train/validation/test data wrappers for external 3D pose front ends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from dual2pose.eval.frontend_manifest import (
    FrontEndManifest,
    FrontEndPoseDataset,
    replace_frontend_inputs,
)


def _as_manifest(value: FrontEndManifest | Path | str | None) -> FrontEndManifest | None:
    if value is None or isinstance(value, FrontEndManifest):
        return value
    return FrontEndManifest.load(Path(value))


class MixedFrontEndDataset(Dataset):
    """Balanced block product of one base dataset and named pose sources."""

    def __init__(
        self,
        base_dataset: Dataset,
        sources: Sequence[FrontEndManifest | None],
    ) -> None:
        if len(base_dataset) <= 0 or not sources:
            raise ValueError("Mixed front-end training requires data and at least one source")
        names = ["sam3d" if source is None else source.frontend_name for source in sources]
        if len(names) != len(set(names)):
            raise ValueError(f"Mixed front-end sources must be unique: {names}")
        raw_index = getattr(base_dataset, "_index_mapping", None)
        if not isinstance(raw_index, list):
            raise ValueError("Base front-end dataset must expose its index mapping")
        normalized = [item if isinstance(item, dict) else vars(item) for item in raw_index]
        for source in sources:
            if source is not None:
                source.validate_coverage(normalized)
        self.base_dataset = base_dataset
        self.sources = tuple(sources)
        self.source_names = tuple(names)

    def __len__(self) -> int:
        return len(self.base_dataset) * len(self.sources)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        base_length = len(self.base_dataset)
        source_index, base_index = divmod(index, base_length)
        sample = self.base_dataset[base_index]
        if not isinstance(sample, dict):
            raise TypeError("Unity front-end training expects dictionary samples")
        source = self.sources[source_index]
        if source is not None:
            return replace_frontend_inputs(sample, source)
        output = dict(sample)
        streams = sample.get("kpt3d_sam")
        if isinstance(streams, dict):
            output["kpt3d_sam"] = dict(streams)
        output["_frontend_name"] = "sam3d"
        return output


class FrontEndDataModule(LightningDataModule):
    """Split-specific manifest replacement around an existing UnityDataModule."""

    def __init__(
        self,
        base_dm: LightningDataModule,
        train_manifest: FrontEndManifest | Path | str | None,
        val_manifest: FrontEndManifest | Path | str | None,
        test_manifest: FrontEndManifest | Path | str | None,
        mixed_train_sources: Sequence[FrontEndManifest | Path | str | None] | None = None,
        mixed_val_sources: Sequence[FrontEndManifest | Path | str | None] | None = None,
    ) -> None:
        super().__init__()
        self.base_dm = base_dm
        self.train_manifest = _as_manifest(train_manifest)
        self.val_manifest = _as_manifest(val_manifest)
        self.test_manifest = _as_manifest(test_manifest)
        self.mixed_train_sources = (
            tuple(_as_manifest(source) for source in mixed_train_sources)
            if mixed_train_sources is not None
            else None
        )
        self.mixed_val_sources = (
            tuple(_as_manifest(source) for source in mixed_val_sources)
            if mixed_val_sources is not None
            else None
        )
        self._wrapped = False

    def prepare_data(self) -> None:
        self.base_dm.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.base_dm.setup(stage)
        if self._wrapped:
            return
        if hasattr(self.base_dm, "train_gait_dataset"):
            base_train = self.base_dm.train_gait_dataset
            if self.mixed_train_sources is not None:
                self.base_dm.train_gait_dataset = MixedFrontEndDataset(
                    base_train, self.mixed_train_sources
                )
            elif self.train_manifest is not None:
                self.base_dm.train_gait_dataset = FrontEndPoseDataset(
                    base_train, self.train_manifest
                )
        if hasattr(self.base_dm, "val_gait_dataset"):
            base_val = self.base_dm.val_gait_dataset
            if self.mixed_val_sources is not None:
                self.base_dm.val_gait_dataset = MixedFrontEndDataset(
                    base_val, self.mixed_val_sources
                )
            elif self.val_manifest is not None:
                self.base_dm.val_gait_dataset = FrontEndPoseDataset(
                    base_val, self.val_manifest
                )
        if self.test_manifest is not None and hasattr(self.base_dm, "test_gait_dataset"):
            self.base_dm.test_gait_dataset = FrontEndPoseDataset(
                self.base_dm.test_gait_dataset, self.test_manifest
            )
        self._wrapped = True

    def train_dataloader(self) -> Any:
        return self.base_dm.train_dataloader()

    def val_dataloader(self) -> Any:
        return self.base_dm.val_dataloader()

    def test_dataloader(self) -> Any:
        return self.base_dm.test_dataloader()
