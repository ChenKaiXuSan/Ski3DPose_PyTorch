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
    Resize,
)

from project.dataloader.whole_video_dataset import whole_video_dataset
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
        if not self._load_frames and not self._load_2d_kpt and not self._load_3d_kpt:
            raise ValueError(
                "At least one of data.load_frames/data.load_2d_kpt/data.load_3d_kpt must be true."
            )

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx

        self._experiment = opt.experiment

        self.mapping_transform = Compose(
            [
                Div255(),
                Resize(size=[self._img_size, self._img_size]),
            ]
        )

    @staticmethod
    def _merge_bt_pose(x: torch.Tensor, name: str) -> torch.Tensor:
        """Merge sample/time dims: (1,T,J,C) or (T,J,C) -> (T,1,J,C)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}, got {type(x)}")
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"Expected {name} shape (T,J,C), got {tuple(x.shape)}")
        return x.unsqueeze(1)

    @staticmethod
    def _merge_bt_video(x: torch.Tensor, name: str) -> torch.Tensor:
        """Merge sample/time dims: (1,C,T,H,W) -> (T,C,H,W)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor for {name}, got {type(x)}")
        if x.ndim != 5 or x.shape[0] != 1:
            raise ValueError(f"Expected {name} shape (1,C,T,H,W), got {tuple(x.shape)}")
        return x[0].permute(1, 0, 2, 3)

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Keep variable T per sample by flattening B and T into pseudo-batch.
        
        Supports variant-based keypoint structure:
        - kpt2d_gt: {character_cam1, character_cam2, pole_cam1, pole_cam2, ski_cam1, ski_cam2}
        - kpt2d_sam: {character_cam1, character_cam2} (SAM only has character)
        - kpt3d_gt: {character, pole, ski}
        - kpt3d_sam: {character_cam1, character_cam2} (SAM only has character)
        """
        if not batch:
            return {}

        first = batch[0]
        has_frames = "frames" in first
        has_2d = "kpt2d_gt" in first and "kpt2d_sam" in first
        has_3d = "kpt3d_gt" in first and "kpt3d_sam" in first

        frames_cam1: List[torch.Tensor] = []
        frames_cam2: List[torch.Tensor] = []
        
        # 2D GT: multiple variants
        gt2d_variant_lists: Dict[str, List[torch.Tensor]] = {}
        
        # 2D SAM: character only
        sam2d_cam1: List[torch.Tensor] = []
        sam2d_cam2: List[torch.Tensor] = []
        
        # 3D GT: multiple variants
        gt3d_variant_lists: Dict[str, List[torch.Tensor]] = {}
        
        # 3D SAM: character only
        sam3d_cam1: List[torch.Tensor] = []
        sam3d_cam2: List[torch.Tensor] = []
        
        frame_indices: List[torch.Tensor] = []
        meta_rows: List[Dict[str, Any]] = []

        for sample in batch:
            if has_frames:
                frames_cam1.append(
                    self._merge_bt_video(sample["frames"]["cam1"], "frames/cam1")
                )
                frames_cam2.append(
                    self._merge_bt_video(sample["frames"]["cam2"], "frames/cam2")
                )

            if has_2d:
                # Process 2D GT: iterate over all keys
                for key, value in sample["kpt2d_gt"].items():
                    if key not in gt2d_variant_lists:
                        gt2d_variant_lists[key] = []
                    gt2d_variant_lists[key].append(
                        self._merge_bt_pose(value, f"kpt2d_gt/{key}")
                    )
                
                # Process 2D SAM: character_cam1 and character_cam2 only
                if "character_cam1" in sample["kpt2d_sam"]:
                    sam2d_cam1.append(
                        self._merge_bt_pose(sample["kpt2d_sam"]["character_cam1"], "kpt2d_sam/character_cam1")
                    )
                if "character_cam2" in sample["kpt2d_sam"]:
                    sam2d_cam2.append(
                        self._merge_bt_pose(sample["kpt2d_sam"]["character_cam2"], "kpt2d_sam/character_cam2")
                    )

            if has_3d:
                # Process 3D SAM: character_cam1 and character_cam2 only
                if "character_cam1" in sample["kpt3d_sam"]:
                    sam3d_cam1.append(
                        self._merge_bt_pose(sample["kpt3d_sam"]["character_cam1"], "kpt3d_sam/character_cam1")
                    )
                if "character_cam2" in sample["kpt3d_sam"]:
                    sam3d_cam2.append(
                        self._merge_bt_pose(sample["kpt3d_sam"]["character_cam2"], "kpt3d_sam/character_cam2")
                    )
                
                # Process 3D GT: iterate over all keys (character, pole, ski)
                for key, value in sample["kpt3d_gt"].items():
                    if key not in gt3d_variant_lists:
                        gt3d_variant_lists[key] = []
                    gt3d_variant_lists[key].append(
                        self._merge_bt_pose(value, f"kpt3d_gt/{key}")
                    )

            idx = sample.get("frame_indices")
            if isinstance(idx, torch.Tensor):
                frame_indices.append(idx.view(-1))

            sample_meta = sample.get("meta", {})
            if isinstance(idx, torch.Tensor):
                num_frames = int(idx.numel())
            elif has_3d:
                # Get first available key from kpt3d_gt to determine num_frames
                first_3d_key = next(iter(sample["kpt3d_gt"].keys()))
                num_frames = int(sample["kpt3d_gt"][first_3d_key].shape[0])
            elif has_2d:
                # Get first available key from kpt2d_gt to determine num_frames
                first_2d_key = next(iter(sample["kpt2d_gt"].keys()))
                num_frames = int(sample["kpt2d_gt"][first_2d_key].shape[0])
            else:
                num_frames = int(sample["frames"]["cam1"].shape[2]) if has_frames else 0
            for t in range(num_frames):
                row = (
                    dict(sample_meta)
                    if isinstance(sample_meta, dict)
                    else {"meta": sample_meta}
                )
                row["time_index_in_sample"] = t
                meta_rows.append(row)

        out: Dict[str, Any] = {
            "frame_indices": torch.cat(frame_indices, dim=0)
            if frame_indices
            else torch.empty(0, dtype=torch.long),
            "meta": meta_rows,
        }

        if has_frames:
            out["frames"] = {
                "cam1": torch.cat(frames_cam1, dim=0),
                "cam2": torch.cat(frames_cam2, dim=0),
            }

        if has_2d:
            # Concat 2D GT: multiple variant keys
            out["kpt2d_gt"] = {
                key: torch.cat(tensors, dim=0) 
                for key, tensors in gt2d_variant_lists.items()
            }
            # Concat 2D SAM: character only
            out["kpt2d_sam"] = {}
            if sam2d_cam1:
                out["kpt2d_sam"]["character_cam1"] = torch.cat(sam2d_cam1, dim=0)
            if sam2d_cam2:
                out["kpt2d_sam"]["character_cam2"] = torch.cat(sam2d_cam2, dim=0)

        if has_3d:
            # Concat 3D GT: multiple variant keys
            out["kpt3d_gt"] = {
                key: torch.cat(tensors, dim=0)
                for key, tensors in gt3d_variant_lists.items()
            }
            # Concat 3D SAM: character only
            out["kpt3d_sam"] = {}
            if sam3d_cam1:
                out["kpt3d_sam"]["character_cam1"] = torch.cat(sam3d_cam1, dim=0)
            if sam3d_cam2:
                out["kpt3d_sam"]["character_cam2"] = torch.cat(sam3d_cam2, dim=0)

        return out


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
        self.train_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["train"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
        )

        # val dataset
        self.val_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["val"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
        )

        # test dataset
        self.test_gait_dataset = whole_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx["test"],
            transform=self.mapping_transform,
            load_frames=self._load_frames,
            load_2d_kpt=self._load_2d_kpt,
            load_3d_kpt=self._load_3d_kpt,
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
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn,
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
            pin_memory=True,  # 🚀 GPU内存传输加速（改自False）
            shuffle=False,
            drop_last=True,
            collate_fn=self._collate_fn,
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
            pin_memory=True,  # 🚀 GPU内存传输加速（改自False）
            shuffle=False,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

        return test_data_loader
