#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/models/pose2equip.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/models
Created Date: Tuesday April 28th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday April 28th 2026 2:47:39 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .stgcn import STGCN


# =========================
# Model
# =========================
class DinoFrameEncoder(nn.Module):
    """封装 DINO 视觉骨干的图像预处理、前向推理和投影。

    输入: [B, 3, H, W] float tensor (uint8 范围或已归一化均可)
    输出: [B, out_dim] 特征向量
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        out_dim: int = 256,
        image_size: int = 224,
        freeze: bool = True,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.freeze = bool(freeze)

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "DinoFrameEncoder requires transformers. "
                "Please install transformers>=4.56.0."
            ) from exc

        _proc = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.requires_grad_(not self.freeze)

        model_hidden = int(
            getattr(
                self.encoder.config,
                "hidden_size",
                getattr(self.encoder.config, "projection_dim", out_dim),
            )
        )
        self.proj = nn.Linear(model_hidden, out_dim)

        mean = torch.tensor(_proc.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_proc.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)

    def forward(self, human_frame: torch.Tensor) -> torch.Tensor:
        # human_frame: [B, 3, H, W]
        if human_frame.ndim != 4:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(human_frame.shape)}")
        if human_frame.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {human_frame.shape[1]}")

        x = self._preprocess(human_frame)

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            outputs = self.encoder(pixel_values=x)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            else:
                feat = outputs.last_hidden_state[:, 0]

        return self.proj(feat)


class PoseEncoder(nn.Module):
    """
    ST-GCN-based pose encoder that takes in 3D joint positions and outputs a global feature vector.
    """

    def __init__(
        self,
        num_joints,
        hidden_dim=256,
        target_skeleton_connections_idx=None,
    ):
        super().__init__()
        self.num_joints = int(num_joints)
        if target_skeleton_connections_idx is not None:
            edges = target_skeleton_connections_idx

        self.stgcn = STGCN(
            num_joints=self.num_joints,
            in_channels=3,
            hidden_channels=(64, 64, 128, 128, hidden_dim),
            edges=edges,
            dropout=0.1,
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: [B, J, 3] or [B, T, J, 3]
        if x.ndim == 3:
            if x.shape[1] != self.num_joints or x.shape[2] != 3:
                raise ValueError(
                    f"Expected pose shape [B,{self.num_joints},3], got {tuple(x.shape)}"
                )
            x = x.unsqueeze(1)  # [B, 1, J, 3]
        elif x.ndim == 4:
            if x.shape[2] != self.num_joints or x.shape[3] != 3:
                raise ValueError(
                    f"Expected pose shape [B,T,{self.num_joints},3], got {tuple(x.shape)}"
                )
        else:
            raise ValueError(f"Unsupported pose rank: {tuple(x.shape)}")

        x, _ = self.stgcn(x, return_features=True)  # [B, T, J, hidden_dim]
        x = x.mean(dim=(1, 2))  # global avg pool over T and J
        return self.dropout(x)


class Pose2EquipNet(nn.Module):
    def __init__(
        self,
        num_joints=70,
        hidden_dim=256,
        num_equip_kpts: int = 8,
        target_skeleton_connections_idx=None,
        dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        dino_freeze: bool = True,
        dino_image_size: int = 224,
    ):
        super().__init__()
        self.num_equip_kpts = int(num_equip_kpts)

        self.pose_encoder = PoseEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            target_skeleton_connections_idx=target_skeleton_connections_idx,
        )

        self.frame_encoder = DinoFrameEncoder(
            model_name=dino_model_name,
            out_dim=hidden_dim,
            image_size=dino_image_size,
            freeze=dino_freeze,
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # 直接回归 num_equip_kpts 个装备 3D 关键点
        # 输出顺序: left_ski_tip, left_ski_tail, right_ski_tip, right_ski_tail,
        #           left_pole_grip, left_pole_tip, right_pole_grip, right_pole_tip
        self.equip_head = nn.Linear(hidden_dim, self.num_equip_kpts * 3)

    def forward(
        self,
        human_3d: torch.Tensor,
        human_frame: torch.Tensor,
    ):
        # human_3d: [B, J, 3]
        # human_frame: [B, 3, H, W]

        b = human_3d.shape[0]
        pose_feat = self.pose_encoder(human_3d)
        frame_feat = self.frame_encoder(human_frame)

        fused = self.fuse(torch.cat([pose_feat, frame_feat], dim=-1))

        # 直接回归 8 个装备关键点的 3D 坐标 [B, 8, 3]
        pred_obj = self.equip_head(fused).reshape(b, self.num_equip_kpts, 3)

        return {"object_3d": pred_obj}


class STGCNBaselineNet(nn.Module):
    """STGCN-only baseline: 3D human keypoints -> 3D equipment keypoints."""

    def __init__(
        self,
        num_joints: int = 15,
        hidden_dim: int = 256,
        num_equip_kpts: int = 8,
        target_skeleton_connections_idx=None,
    ):
        super().__init__()
        self.num_equip_kpts = int(num_equip_kpts)

        self.pose_encoder = PoseEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            target_skeleton_connections_idx=target_skeleton_connections_idx,
        )
        self.equip_head = nn.Linear(hidden_dim, self.num_equip_kpts * 3)

    def forward(self, human_3d: torch.Tensor):
        # human_3d: [B, J, 3] or [B, T, J, 3]
        b = human_3d.shape[0]
        pose_feat = self.pose_encoder(human_3d)
        pred_obj = self.equip_head(pose_feat).reshape(b, self.num_equip_kpts, 3)
        return {"object_3d": pred_obj}
