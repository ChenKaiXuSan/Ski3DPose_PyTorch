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
        edges = self._normalize_edges(target_skeleton_connections_idx, self.num_joints)
        if edges is None:
            edges = self._default_edges(self.num_joints)

        self.stgcn = STGCN(
            num_joints=self.num_joints,
            in_channels=3,
            hidden_channels=(64, 64, 128, 128, hidden_dim),
            edges=edges,
            dropout=0.1,
        )
        self.dropout = nn.Dropout(p=0.1)

    @staticmethod
    def _default_edges(num_joints: int):
        if num_joints == 15:
            return [
                (0, 1),
                (1, 2),
                (2, 3),
                (1, 4),
                (4, 5),
                (5, 6),
                (1, 7),
                (7, 8),
                (8, 9),
                (7, 10),
                (10, 11),
                (11, 12),
                (7, 13),
                (13, 14),
            ]
        # Fallback: chain topology for unknown joint count.
        return [(i, i + 1) for i in range(num_joints - 1)]

    @staticmethod
    def _normalize_edges(edges, num_joints: int):
        """Normalize edge config to List[Tuple[int, int]].

        Supports:
        - [[u, v], [u, v], ...]
        - [(u, v), (u, v), ...]
        - [u0, v0, u1, v1, ...]
        """
        if edges is None:
            return None

        normalized = []
        if (
            isinstance(edges, (list, tuple))
            and len(edges) > 0
            and isinstance(edges[0], (list, tuple))
        ):
            for pair in edges:
                if len(pair) != 2:
                    raise ValueError(f"Invalid edge pair: {pair}")
                u, v = int(pair[0]), int(pair[1])
                if not (0 <= u < num_joints and 0 <= v < num_joints):
                    raise ValueError(
                        f"Edge out of range: ({u},{v}) for num_joints={num_joints}"
                    )
                normalized.append((u, v))
            return normalized

        if isinstance(edges, (list, tuple)) and len(edges) % 2 == 0:
            vals = [int(x) for x in edges]
            for i in range(0, len(vals), 2):
                u, v = vals[i], vals[i + 1]
                if not (0 <= u < num_joints and 0 <= v < num_joints):
                    raise ValueError(
                        f"Edge out of range: ({u},{v}) for num_joints={num_joints}"
                    )
                normalized.append((u, v))
            return normalized

        raise ValueError(
            "target_skeleton_connections_idx must be list of pairs or flattened even-length list"
        )

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
    ):
        super().__init__()
        self.left_ankle_idx = left_ankle_idx
        self.right_ankle_idx = right_ankle_idx
        self.left_wrist_idx = left_wrist_idx
        self.right_wrist_idx = right_wrist_idx

        self.pose_encoder = PoseEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            target_skeleton_connections_idx=target_skeleton_connections_idx,
        )

        # Fuse RGB + two equipment masks into a 3-channel tensor for ResNet.
        self.mask_fuse = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
        )

        _resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.frame_encoder = nn.Sequential(
            *list(_resnet.children())[:-1]
        )  # output: [B, 512, 1, 1]
        self.frame_proj = nn.Linear(512, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # 输出 4 个方向 + 4 个长度
        # 4 directions: left_ski, right_ski, left_pole, right_pole
        self.dir_head = nn.Linear(hidden_dim, 4 * 3)
        self.len_head = nn.Linear(hidden_dim, 4)

    def _encode_frame(
        self,
        human_frame: torch.Tensor,
        pole_mask: torch.Tensor | None = None,
        ski_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        b, _, h, w = human_frame.shape
        if pole_mask is None:
            pole_mask = human_frame.new_zeros((b, 1, h, w))
        if ski_mask is None:
            ski_mask = human_frame.new_zeros((b, 1, h, w))

        if pole_mask.ndim != 4 or pole_mask.shape[1] != 1:
            raise ValueError(
                f"Expected pole_mask shape [B,1,H,W], got {tuple(pole_mask.shape)}"
            )
        if ski_mask.ndim != 4 or ski_mask.shape[1] != 1:
            raise ValueError(
                f"Expected ski_mask shape [B,1,H,W], got {tuple(ski_mask.shape)}"
            )
        if pole_mask.shape[0] != b or ski_mask.shape[0] != b:
            raise ValueError(
                "Mask/frame batch mismatch: "
                f"frame={b}, pole={pole_mask.shape[0]}, ski={ski_mask.shape[0]}"
            )
        if pole_mask.shape[-2:] != (h, w) or ski_mask.shape[-2:] != (h, w):
            raise ValueError(
                "Mask/frame spatial mismatch: "
                f"frame={(h, w)}, pole={tuple(pole_mask.shape[-2:])}, ski={tuple(ski_mask.shape[-2:])}"
            )

        # 5-channel input: [RGB, pole_mask, ski_mask]
        fused_input = torch.cat([human_frame, pole_mask, ski_mask], dim=1)
        fused_rgb = self.mask_fuse(fused_input)

        frame_feat = self.frame_encoder(fused_rgb).flatten(1)
        frame_feat = self.frame_proj(frame_feat)
        return frame_feat

    def forward(
        self,
        human_3d: torch.Tensor,
        human_frame: torch.Tensor,
        pole_mask: torch.Tensor | None = None,
        ski_mask: torch.Tensor | None = None,
    ):
        # human_3d: [B, J, 3]
        # human_frame: [B, C, H, W]
        # pole_mask/ski_mask: [B, 1, H, W]

        equip_feat = self.pose_encoder(human_3d)
        b, _, _ = human_3d.shape

        frame_feat = self._encode_frame(
            human_frame=human_frame,
            pole_mask=pole_mask,
            ski_mask=ski_mask,
        )
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
