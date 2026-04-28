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
from torchvision.models import resnet18, ResNet18_Weights

# =========================
# Model
# =========================
class PoseEncoder(nn.Module):
    def __init__(self, num_joints, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_joints * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, J, 3]
        b, j, c = x.shape
        x = x.reshape(b, j * c)
        return self.net(x)


class Pose2EquipNet(nn.Module):
    def __init__(
        self,
        num_joints=70,
        hidden_dim=256,
        left_ankle_idx=13,
        right_ankle_idx=14,
        left_wrist_idx=41,
        right_wrist_idx=62,
    ):
        super().__init__()
        self.left_ankle_idx = left_ankle_idx
        self.right_ankle_idx = right_ankle_idx
        self.left_wrist_idx = left_wrist_idx
        self.right_wrist_idx = right_wrist_idx

        self.pose_encoder = PoseEncoder(num_joints, hidden_dim)
        
        _resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.frame_encoder = nn.Sequential(*list(_resnet.children())[:-1])  # output: [B, 512, 1, 1]
        self.frame_proj = nn.Linear(512, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # 输出 4 个方向 + 4 个长度
        # 4 directions: left_ski, right_ski, left_pole, right_pole
        self.dir_head = nn.Linear(hidden_dim, 4 * 3)
        self.len_head = nn.Linear(hidden_dim, 4)

    def _encode_frame(self, human_frame: torch.Tensor) -> torch.Tensor:
        # human_frame: [B, C, H, W]
        if human_frame.ndim != 4:
            raise ValueError(
                f"Expected human_frame shape [B,C,H,W], got {tuple(human_frame.shape)}"
            )

        if human_frame.shape[1] == 1:
            human_frame = human_frame.repeat(1, 3, 1, 1)
        elif human_frame.shape[1] != 3:
            raise ValueError(
                f"Expected human_frame channels in (1,3), got {human_frame.shape[1]}"
            )

        frame_feat = self.frame_encoder(human_frame).flatten(1)
        frame_feat = self.frame_proj(frame_feat)
        return frame_feat

    def forward(self, human_3d: torch.Tensor, human_frame: torch.Tensor):
        # human_3d: [B, J, 3]
        # human_frame: [B, C, H, W]

        equip_feat = self.pose_encoder(human_3d)
        b, _, _ = human_3d.shape

        frame_feat = self._encode_frame(human_frame)
        if frame_feat.shape[0] != b:
            raise ValueError(
                f"Batch mismatch between pose and frame: {b} vs {frame_feat.shape[0]}"
            )
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
