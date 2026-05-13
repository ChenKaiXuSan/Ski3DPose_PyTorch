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
import json
from pathlib import Path

from .unity_dataset_dual_view import DualViewUnityDataset
from .ski_poseptz_dataset_dual_view import LabeledSkiPosePTZDataset

from .utils import Div255
import logging 

logger = logging.getLogger(__name__)

def load_index_mapping(
    index_mapping_path,
) -> Dict[str, list]:
    """加载指定fold的JSON文件
    """

    index_file_path = Path(index_mapping_path)

    with open(index_file_path, "r", encoding="utf-8") as f:
        fold_data = json.load(f)

    fold_data.pop("_metadata", None)

    dataset_idx: Dict[str, list] = {"train": [], "val": [], "test": []}

    # 处理三种split
    for split in ["train", "val", "test"]:
        src_list = fold_data.get(split, [])
        for item in src_list:
            dataset_idx[split].append(item)

    logger.info(
        f"✓ Loaded fold {index_mapping_path}: train={len(dataset_idx['train'])}, val={len(dataset_idx['val'])}, test={len(dataset_idx['test'])}"
    )
    return dataset_idx


class UnityDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()

        self._batch_size = opt.data.batch_size

        self._num_workers = opt.data.num_workers
        self._img_size = opt.data.img_size
        self._load_frames = bool(getattr(opt.data, "load_frames", True))
        self._load_2d_kpt = bool(getattr(opt.data, "load_2d_kpt", True))
        self._load_3d_kpt = bool(getattr(opt.data, "load_3d_kpt", True))
        self._time_window = int(getattr(opt.data, "time_window", 32))

        # * this is the dataset idx, which include the train/val dataset idx.
        self._index_mapping = opt.data.unity.index_mapping_path

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
        self._dataset_idx = load_index_mapping(
            self._index_mapping
        )  # 预加载第0折数据索引，后续在setup中根据fold切换

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        # train dataset
        self.train_gait_dataset = DualViewUnityDataset(
            experiment=self._experiment,
            index_mapping=self._dataset_idx["train"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            target_t=self._time_window,
        )

        # val dataset
        self.val_gait_dataset = DualViewUnityDataset(
            experiment=self._experiment,
            index_mapping=self._dataset_idx["val"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            target_t=self._time_window,
        )

        # test dataset
        self.test_gait_dataset = DualViewUnityDataset(
            experiment=self._experiment,
            index_mapping=self._dataset_idx["test"],
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


class SkiPosePTZDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()

        self._batch_size = opt.data.batch_size

        self._num_workers = opt.data.num_workers
        self._img_size = opt.data.img_size
        self._load_frames = bool(getattr(opt.data, "load_frames", True))
        self._load_2d_kpt = bool(getattr(opt.data, "load_2d_kpt", True))
        self._load_3d_kpt = bool(getattr(opt.data, "load_3d_kpt", True))
        self._time_window = int(getattr(opt.data, "time_window", 32))

        # * this is the dataset idx, which include the train/val dataset idx.
        self._index_mapping = opt.data.ski_pose_ptz.index_mapping

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
        self._dataset_idx = load_index_mapping(
            self._index_mapping
        )  

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        # train dataset
        self.train_gait_dataset = LabeledSkiPosePTZDataset(
            index_mapping=self._dataset_idx["train"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            target_t=self._time_window,
        )

        # val dataset
        self.val_gait_dataset = LabeledSkiPosePTZDataset(
            index_mapping=self._dataset_idx["val"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
            target_t=self._time_window,
        )

        # test dataset
        self.test_gait_dataset = LabeledSkiPosePTZDataset(
            index_mapping=self._dataset_idx["test"],
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
