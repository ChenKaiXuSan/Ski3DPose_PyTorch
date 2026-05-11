#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/MultiView_DriverAction_PyTorch/project/dataloader/data_loader.py
Project: /workspace/MultiView_DriverAction_PyTorch/project/dataloader
Created Date: Saturday January 24th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday January 24th 2026 10:51:04 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
)

from project.dataloader.whole_video_dataset_dual_view import (
    whole_video_dataset as whole_video_dataset_dual,
)
from project.dataloader.unity_dataset_single_view import (
    whole_video_dataset as whole_video_dataset_single,
)

from project.dataloader.utils import Div255


class UnityDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._batch_size = opt.data.batch_size

        self._num_workers = opt.data.num_workers
        self._img_size = opt.data.img_size
        self._load_frames = bool(getattr(opt.data, "load_frames", True))
        self._load_2d_kpt = bool(getattr(opt.data, "load_2d_kpt", True))
        self._load_3d_kpt = bool(getattr(opt.data, "load_3d_kpt", True))
        self._time_window = int(getattr(opt.data, "time_window", 32))

        self._view = str(getattr(opt.train, "view", "dual")).lower()
        self._is_single_view = self._view in {"single", "single_view", "cam1", "one"}
        if not self._load_frames and not self._load_2d_kpt and not self._load_3d_kpt:
            raise ValueError(
                "At least one of data.load_frames/data.load_2d_kpt/data.load_3d_kpt/data.load_mask must be true."
            )

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx

        self._experiment = opt.experiment

        self.mapping_transform = Compose(
            [
                Div255(),
                Resize(size=[self._img_size, self._img_size]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _merge_bt_pose(x: torch.Tensor, name: str) -> torch.Tensor:
        """Normalize pose shape: (1,T,J,C) or (T,J,C) -> (T,J,C)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}, got {type(x)}")
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Expected {name} shape (T,J,C), got {tuple(x.shape)}")
        return x

    @staticmethod
    def _merge_bt_video(x: torch.Tensor, name: str) -> torch.Tensor:
        """Normalize video shape: (C,T,H,W) or (1,C,T,H,W) -> (C,T,H,W)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}, got {type(x)}")
        if x.ndim == 5 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 4:
            raise ValueError(
                f"Expected {name} shape (C,T,H,W) or (1,C,T,H,W), got {tuple(x.shape)}"
            )
        return x

    @staticmethod
    def _resize_video_batch_bcthw(
        x: torch.Tensor,
        target_hw: tuple[int, int],
        mode: str,
    ) -> torch.Tensor:
        """Resize spatial size for batched video tensor (B,C,T,H,W)."""
        if x.ndim != 5:
            raise ValueError(f"Expected (B,C,T,H,W), got {tuple(x.shape)}")
        b, c, t, _, _ = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, x.shape[3], x.shape[4])
        if mode == "nearest":
            y = torch.nn.functional.interpolate(y, size=target_hw, mode=mode)
        else:
            y = torch.nn.functional.interpolate(
                y,
                size=target_hw,
                mode=mode,
                align_corners=False,
            )
        return y.reshape(b, t, c, target_hw[0], target_hw[1]).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _temporal_select_indices(src_t: int, dst_t: int) -> torch.Tensor:
        """Uniformly sample temporal indices from src_t to dst_t."""
        if src_t <= 0 or dst_t <= 0:
            raise ValueError(
                f"src_t and dst_t must be > 0, got src_t={src_t}, dst_t={dst_t}"
            )
        if src_t == dst_t:
            return torch.arange(src_t, dtype=torch.long)
        return torch.linspace(0, src_t - 1, steps=dst_t).round().long()

    @classmethod
    def _resample_time_by_index(
        cls,
        x: torch.Tensor,
        target_t: int,
        time_dim: int,
        name: str,
    ) -> torch.Tensor:
        """Resample tensor along a temporal dimension using index selection."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}, got {type(x)}")
        src_t = int(x.shape[time_dim])
        if src_t == target_t:
            return x
        idx = cls._temporal_select_indices(src_t, target_t).to(x.device)
        return x.index_select(time_dim, idx)

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        dataset_builder = (
            whole_video_dataset_single
            if self._is_single_view
            else whole_video_dataset_dual
        )
        if self._is_single_view:
            effective_load_frames = self._load_frames
            effective_load_2d_kpt = self._load_2d_kpt
            effective_load_3d_kpt = self._load_3d_kpt
            target_t = self._time_window
        else:
            # Dual-view policy: only load frames and 3D keypoints.
            effective_load_frames = True
            effective_load_2d_kpt = False
            effective_load_3d_kpt = True
            target_t = self._time_window

        dataset_kwargs = dict(
            transform=self.mapping_transform,
            load_frames=effective_load_frames,
            load_2d_kpt=effective_load_2d_kpt,
            load_3d_kpt=effective_load_3d_kpt,
            target_t=target_t,
        )

        # train dataset
        self.train_gait_dataset = dataset_builder(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["train"],
            **dataset_kwargs,
        )

        # val dataset
        self.val_gait_dataset = dataset_builder(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            **dataset_kwargs,
        )

        # test dataset
        self.test_gait_dataset = dataset_builder(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["test"],
            **dataset_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        train_data_loader = DataLoader(
            self.train_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=False,  # 🚀 GPU内存传输加速（改自True）
            shuffle=True,
            drop_last=True,
            # collate_fn=self._collate_fn,
        )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        val_data_loader = DataLoader(
            self.val_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=False,  # 🚀 GPU内存传输加速（改自True）
            shuffle=False,
            drop_last=True,
            # collate_fn=self._collate_fn,
        )

        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        test_data_loader = DataLoader(
            self.test_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=False,  # 🚀 GPU内存传输加速（改自True）
            shuffle=False,
            drop_last=True,
            # collate_fn=self._collate_fn,
        )

        return test_data_loader
