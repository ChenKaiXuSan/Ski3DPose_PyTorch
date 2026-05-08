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
from .stgcn import STGCN


# =========================
# Model
# =========================
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
        left_ankle_idx=13,
        right_ankle_idx=14,
        left_wrist_idx=41,
        right_wrist_idx=62,
        target_skeleton_connections_idx=None,
        dino_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        dino_freeze: bool = True,
        dino_image_size: int = 224,
    ):
        super().__init__()
        self.left_ankle_idx = left_ankle_idx
        self.right_ankle_idx = right_ankle_idx
        self.left_wrist_idx = left_wrist_idx
        self.right_wrist_idx = right_wrist_idx
        self.dino_model_name = str(dino_model_name)
        self.dino_freeze = bool(dino_freeze)
        self.dino_image_size = int(dino_image_size)

        self.pose_encoder = PoseEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            target_skeleton_connections_idx=target_skeleton_connections_idx,
        )

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Pose2EquipNet with DINO backbone requires transformers. "
                "Please install transformers>=4.56.0."
            ) from exc

        self.dino_processor = AutoImageProcessor.from_pretrained(self.dino_model_name)
        self.frame_encoder = AutoModel.from_pretrained(self.dino_model_name)
        self.frame_encoder.requires_grad_(not self.dino_freeze)

        model_hidden = int(
            getattr(
                self.frame_encoder.config,
                "hidden_size",
                getattr(self.frame_encoder.config, "projection_dim", hidden_dim),
            )
        )
        self.frame_proj = nn.Linear(model_hidden, hidden_dim)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # 输出 4 个方向 + 4 个长度
        # 4 directions: left_ski, right_ski, left_pole, right_pole
        self.dir_head = nn.Linear(hidden_dim, 4 * 3)
        self.len_head = nn.Linear(hidden_dim, 4)
        self.dino_len_head = nn.Linear(hidden_dim, 4)

    def _encode_frame_branch(self, human_frame: torch.Tensor) -> torch.Tensor:
        # human_frame: [B, C, H, W]
        if human_frame.ndim != 4:
            raise ValueError(
                f"Expected human_frame shape [B,C,H,W], got {tuple(human_frame.shape)}"
            )

        if human_frame.shape[1] != 3:
            raise ValueError(
                f"Expected human_frame channels to be 3 for DINO processor, got {human_frame.shape[1]}"
            )

        processor_inputs = self.dino_processor(
            images=human_frame,
            return_tensors="pt",
        )
        x = processor_inputs["pixel_values"].to(human_frame.device)

        context = torch.no_grad() if self.dino_freeze else torch.enable_grad()
        with context:
            outputs = self.frame_encoder(pixel_values=x)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                frame_feat = outputs.pooler_output
            else:
                frame_feat = outputs.last_hidden_state[:, 0]

        frame_feat = self.frame_proj(frame_feat)
        return frame_feat

    def forward(
        self,
        human_3d: torch.Tensor,
        human_frame: torch.Tensor,
    ):
        # human_3d: [B, J, 3]
        # human_frame: [B, C, H, W]

        equip_feat = self.pose_encoder(human_3d)
        b, _, _ = human_3d.shape

        frame_feat = self._encode_frame_branch(human_frame=human_frame)
        if frame_feat.shape[0] != b:
            raise ValueError(
                f"Batch mismatch between pose and frame: {b} vs {frame_feat.shape[0]}"
            )

        # Auxiliary branch: supervise DINO features directly with equipment lengths.
        dino_lengths = self.dino_len_head(frame_feat).reshape(b, 4, 1)
        dino_lengths = torch.nn.functional.softplus(dino_lengths) + 1e-4

        equip_feat = self.fuse(torch.cat([equip_feat, frame_feat], dim=-1))

        directions = self.dir_head(equip_feat).reshape(b, 4, 3)
        directions = torch.nn.functional.normalize(directions, dim=-1)

        lengths = self.len_head(equip_feat).reshape(b, 4, 1)
        lengths = torch.nn.functional.softplus(lengths) + 1e-4

        pred_obj = self.build_equipment(human_3d, directions, lengths)

        return {
            "object_3d": pred_obj,
            "directions": directions,
            "lengths": lengths,
            "dino_lengths": dino_lengths,
        }

    def build_equipment(self, human_3d, directions, lengths):
        """
        输出顺序:
        0 left_ski_tip
        1 left_ski_tail
        2 right_ski_tip
        3 right_ski_tail
        4 left_pole_grip
        5 left_pole_tip
        6 right_pole_grip
        7 right_pole_tip
        """
        left_ankle = human_3d[:, self.left_ankle_idx]
        right_ankle = human_3d[:, self.right_ankle_idx]
        left_wrist = human_3d[:, self.left_wrist_idx]
        right_wrist = human_3d[:, self.right_wrist_idx]

        d_left_ski = directions[:, 0]
        d_right_ski = directions[:, 1]
        d_left_pole = directions[:, 2]
        d_right_pole = directions[:, 3]

        l_left_ski = lengths[:, 0]
        l_right_ski = lengths[:, 1]
        l_left_pole = lengths[:, 2]
        l_right_pole = lengths[:, 3]

        left_ski_tip = left_ankle + 0.5 * l_left_ski * d_left_ski
        left_ski_tail = left_ankle - 0.5 * l_left_ski * d_left_ski
        right_ski_tip = right_ankle + 0.5 * l_right_ski * d_right_ski
        right_ski_tail = right_ankle - 0.5 * l_right_ski * d_right_ski

        left_pole_grip = left_wrist
        left_pole_tip = left_wrist + l_left_pole * d_left_pole
        right_pole_grip = right_wrist
        right_pole_tip = right_wrist + l_right_pole * d_right_pole

        obj = torch.stack(
            [
                left_ski_tip,
                left_ski_tail,
                right_ski_tip,
                right_ski_tail,
                left_pole_grip,
                left_pole_tip,
                right_pole_grip,
                right_pole_tip,
            ],
            dim=1,
        )
        return obj
