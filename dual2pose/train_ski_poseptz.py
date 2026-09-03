#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Train crossview_fusion on Ski-PosePTZ pseudo-GT supervision.

This entry point is intentionally Ski-only:
- uses Ski-PosePTZ splits from the index mapping JSON
- uses pseudo GT as the only 3D supervision source
- keeps the 13-joint common subset already used by Ski eval
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, default_collate

from dataloader.ski_poseptz_dataset_dual_view import (
    LabeledSkiPosePTZDataset,
)
from map_config import filter_filtered_kpts_to_common, filter_h36m_kpts
from trainer.train_crossview_fusion import CrossViewFusionTrainer

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/configs/dual2pose.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train crossview_fusion on Ski-PosePTZ pseudo GT."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base config file used to populate model and data options.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=30,
        help="Temporal window used by the Ski-PosePTZ dataset.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index to use when GPU is available.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the final test pass after training.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace):
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")

    config = OmegaConf.load(str(args.config_path))
    config.model.backbone = "crossview_fusion"
    config.data.left_hip_idx = 4
    config.data.right_hip_idx = 5
    config.data.neck_idx = 12
    config.data.load_frames = False
    config.data.load_2d_kpt = False
    config.data.load_3d_kpt = True
    config.data.batch_size = int(args.batch_size)
    config.data.num_workers = int(args.num_workers)
    config.data.time_window = int(args.time_window)
    config.train.max_epochs = int(args.max_epochs)
    config.experiment = "train_ski_poseptz"

    run_dir = Path("logs") / "train_ski_poseptz" / "crossview_fusion" / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    config.log_path = str(run_dir)
    setattr(
        config,
        "hydra",
        {
            "run": {"dir": str(run_dir)},
            "sweep": {"dir": "logs/", "subdir": "train_ski_poseptz"},
        },
    )
    return config


def _build_ski_poseptz_loader(
    index_mapping: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    time_window: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
    path_rewrite_from: str | None = None,
    path_rewrite_to: str | None = None,
) -> DataLoader:
    dataset = LabeledSkiPosePTZDataset(
        index_mapping=index_mapping,
        transform=None,
        load_frames=False,
        load_2d_kpt=False,
        load_3d_kpt=True,
        target_t=time_window,
        split=split,
        path_rewrite_from=path_rewrite_from,
        path_rewrite_to=path_rewrite_to,
    )

    def _collate_ski_poseptz_batch(batch):
        collated = default_collate(batch)
        if "kpt3d" in collated:
            kpt3d_sam = collated.pop("kpt3d")
            collated["kpt3d_sam"] = {
                cam_name: torch.stack(
                    [
                        torch.from_numpy(
                            filter_filtered_kpts_to_common(sample.cpu().numpy())
                        )
                        for sample in cam_pose.reshape(
                            -1, cam_pose.shape[-2], cam_pose.shape[-1]
                        )
                    ],
                    dim=0,
                ).view(*cam_pose.shape[:-2], -1, cam_pose.shape[-1])
                for cam_name, cam_pose in kpt3d_sam.items()
            }
        if "gt_kpt3d" in collated:
            gt = collated.pop("gt_kpt3d")
            if gt.ndim == 4 and gt.shape[-2] == 17:
                gt = torch.stack(
                    [
                        torch.from_numpy(filter_h36m_kpts(sample.cpu().numpy()))
                        for sample in gt.reshape(-1, gt.shape[-2], gt.shape[-1])
                    ],
                    dim=0,
                ).view(*gt.shape[:-2], -1, gt.shape[-1])
            collated["kpt3d_gt"] = gt
        return collated

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=_collate_ski_poseptz_batch,
    )


class SkiPosePTZDataModule(LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self._batch_size = int(config.data.batch_size)
        self._num_workers = int(config.data.num_workers)
        self._time_window = int(config.data.time_window)
        self._index_mapping = Path(config.data.ski_pose_ptz.index_mapping_path)
        rewrite_from = getattr(config.data.ski_pose_ptz, "index_path_rewrite_from", None)
        self._path_rewrite_from = str(rewrite_from) if rewrite_from else None
        self._path_rewrite_to = (
            str(config.data.ski_pose_ptz.root_path) if rewrite_from else None
        )
        self._pin_memory = torch.cuda.is_available()

    def train_dataloader(self) -> DataLoader:
        return _build_ski_poseptz_loader(
            index_mapping=self._index_mapping,
            split="train",
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            time_window=self._time_window,
            shuffle=True,
            drop_last=True,
            pin_memory=self._pin_memory,
            path_rewrite_from=self._path_rewrite_from,
            path_rewrite_to=self._path_rewrite_to,
        )

    def val_dataloader(self) -> DataLoader:
        # Ski-PosePTZ does not provide a non-empty validation split in the current index map.
        return _build_ski_poseptz_loader(
            index_mapping=self._index_mapping,
            split="train",
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            time_window=self._time_window,
            shuffle=False,
            drop_last=False,
            pin_memory=self._pin_memory,
            path_rewrite_from=self._path_rewrite_from,
            path_rewrite_to=self._path_rewrite_to,
        )

    def test_dataloader(self) -> DataLoader:
        return _build_ski_poseptz_loader(
            index_mapping=self._index_mapping,
            split="test",
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            time_window=self._time_window,
            shuffle=False,
            drop_last=False,
            pin_memory=self._pin_memory,
            path_rewrite_from=self._path_rewrite_from,
            path_rewrite_to=self._path_rewrite_to,
        )


def main() -> None:
    args = _parse_args()
    seed_everything(42, workers=True)
    config = _build_config(args)

    model = CrossViewFusionTrainer(config)
    datamodule = SkiPosePTZDataModule(config)

    Path(config.log_path).mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(config.log_path, "tb_logs"),
        name="train",
    )
    csv_logger = CSVLogger(
        save_dir=os.path.join(config.log_path, "csv_logs"),
        name="train",
        flush_logs_every_n_steps=100,
    )

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(config.log_path, "checkpoints"),
        filename="{epoch}-{val/loss:.4f}-{val/mpjpe:.4f}",
        auto_insert_metric_name=False,
        monitor="val/mpjpe",
        mode="min",
        save_last=True,
        save_top_k=2,
    )

    use_gpu = torch.cuda.is_available() and not args.cpu
    trainer = Trainer(
        devices=[int(args.gpu)] if use_gpu else 1,
        accelerator="gpu" if use_gpu else "cpu",
        max_epochs=int(config.train.max_epochs),
        logger=[tb_logger, csv_logger],
        callbacks=[
            RichProgressBar(refresh_rate=10, leave=True),
            RichModelSummary(max_depth=3),
            checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(model, datamodule)
    if not args.skip_test:
        trainer.test(model, datamodule, ckpt_path="last")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()