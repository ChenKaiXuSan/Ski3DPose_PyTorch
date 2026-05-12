#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/dataloader/utils.py
Project: /workspace/code/project/dataloader
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Copy from pytorchvideo.

Have a good code time :)
-----
Last Modified: Wednesday June 25th 2025 5:38:56 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch
from torch import Tensor


def uniform_subsample_along_dim(tensor: Tensor, target_t: int, dim: int) -> Tensor:
    """对任意指定维度做均匀采样（不足时重复最近邻帧）。

    Args:
        tensor: 任意形状的输入张量。
        target_t: 目标长度，必须 > 0。
        dim: 需要采样的维度，支持负索引。

    Returns:
        在指定维度上长度为 target_t 的张量。
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be torch.Tensor, got {type(tensor)}")
    if target_t <= 0:
        raise ValueError(f"target_t must be > 0, got {target_t}")
    if tensor.ndim == 0:
        raise ValueError("tensor must have at least 1 dimension")

    dim = dim if dim >= 0 else tensor.ndim + dim
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError(
            f"dim out of range: got {dim}, valid range is [0, {tensor.ndim - 1}]"
        )

    src_t = int(tensor.shape[dim])
    if src_t <= 0:
        raise ValueError(f"source length on dim {dim} must be > 0, got {src_t}")

    idx_float = torch.linspace(
        0,
        max(src_t - 1, 0),
        target_t,
        dtype=torch.float32,
        device=tensor.device,
    )
    idx = torch.round(idx_float).long()
    return torch.index_select(tensor, dim, idx)


class UniformTemporalSubsample:
    """
    等同于 torchvision.transforms.v2.UniformTemporalSubsample，
    但在帧数不足时会 *均匀复制* 最近邻帧进行补齐。
    支持输入形状 (T, C, H, W)   或 (B, T, C, H, W)。
    """

    def __init__(self, num_samples: int):
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        self.num_samples = num_samples

    def _compute_indices(self, t: int, device) -> Tensor:
        """得到 size=[num_samples] 的 long 索引张量。"""
        # 产生 float 索引，范围 [0, t-1]，共 num_samples 个点
        idx_float = torch.linspace(
            0, max(t - 1, 0), self.num_samples, dtype=torch.float32, device=device
        )
        # 四舍五入到最近帧，再转 long
        return torch.round(idx_float).long()

    def __call__(self, video: Tensor) -> Tensor:
        """
        Args:
            video: (T, C, H, W) **或** (B, T, C, H, W)

        Returns:
            Tensor: 与输入批量/通道一致，但时间维被采样/补齐为 `num_samples`
        """
        is_batched = video.ndim == 5
        if not is_batched and video.ndim != 4:
            raise ValueError("Input must be (T, C, H, W) or (B, T, C, H, W)")

        return uniform_subsample_along_dim(video, self.num_samples, dim=-4)


def uniform_temporal_subsample(video: Tensor, num_samples: int) -> Tensor:
    """等同于 torchvision.transforms.v2.uniform_temporal_subsample，
    但在帧数不足时会 *均匀复制* 最近邻帧进行补齐。
    支持输入形状 (T, C, H, W)   或 (B, T, C, H, W)。

    Args:
        video: (T, C, H, W) **或** (B, T, C, H, W)
        num_samples: 采样后的帧数
    Returns:
        Tensor: 与输入批量/通道一致，但时间维被采样/补齐为 `num_samples`
    """
    return UniformTemporalSubsample(num_samples)(video)


class Div255(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.div_255``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale clip frames from [0, 255] to [0, 1].
        Args:
            x (Tensor): A tensor of the clip's RGB frames with shape:
                (C, T, H, W).
        Returns:
            x (Tensor): Scaled tensor by dividing 255.
        """
        return x / 255.0
