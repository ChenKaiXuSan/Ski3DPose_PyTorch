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

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
)

from pose2equip.dataloader.unity_dataset_single_view import single_view_dataset

from pose2equip.dataloader.utils import Div255


class UnityDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict):
        super().__init__()

        self._batch_size = opt.data.batch_size

        self._num_workers = opt.data.num_workers
        self._img_size = opt.data.img_size
        self._load_frames = bool(getattr(opt.data, "load_frames", True))
        self._load_2d_kpt = bool(getattr(opt.data, "load_2d_kpt", True))
        self._load_3d_kpt = bool(getattr(opt.data, "load_3d_kpt", True))
        self._time_window = int(getattr(opt.data, "time_window", 32))

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

        # train dataset
        self.train_gait_dataset = single_view_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["train"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            target_t=self._time_window,
        )

        # val dataset
        self.val_gait_dataset = single_view_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            target_t=self._time_window,
        )

        # test dataset
        self.test_gait_dataset = single_view_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["test"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            target_t=self._time_window,
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
        )

        return test_data_loader
