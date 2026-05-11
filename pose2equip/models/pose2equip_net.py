#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Human-Guided Equipment Query Transformer

Input:
    RGB sequence + human 3D keypoints

Output:
    Equipment 3D keypoints

Architecture:
    DINOv2 Patch Encoder
    +
    ST-GCN Pose Encoder
    +
    Equipment Query Transformer Decoder
"""

import torch
import torch.nn as nn

from transformers import AutoModel

from .stgcn import STGCN


# =========================================================
# DINO PATCH ENCODER
# =========================================================
class DinoPatchEncoder(nn.Module):

    def __init__(
        self,
        model_name="facebook/dinov2-base",
        out_dim=256,
        freeze=True,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        self.freeze = freeze

        self.encoder.requires_grad_(not freeze)

        hidden_size = self.encoder.config.hidden_size

        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        """
        x:
            B,T,3,H,W

        return:
            B,T,N,C
        """

        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)

        if x.max() > 1:
            x = x / 255.0

        context = torch.no_grad() if self.freeze else torch.enable_grad()

        with context:

            outputs = self.encoder(pixel_values=x)

            # remove CLS token
            feat = outputs.last_hidden_state[:, 1:]

        feat = self.proj(feat)

        N = feat.shape[1]

        feat = feat.reshape(B, T, N, -1)

        return feat


# =========================================================
# POSE ENCODER
# =========================================================
class PoseEncoder(nn.Module):

    def __init__(
        self,
        num_joints=17,
        hidden_dim=256,
        target_skeleton_connections_idx=None,
    ):
        super().__init__()

        self.num_joints = int(num_joints)

        edges = None

        if target_skeleton_connections_idx is not None:
            edges = target_skeleton_connections_idx

        self.stgcn = STGCN(
            num_joints=self.num_joints,
            in_channels=3,
            hidden_channels=(
                64,
                64,
                128,
                128,
                hidden_dim,
            ),
            edges=edges,
            dropout=0.1,
        )

        self.proj = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

    def forward(self, human_3d):
        """
        human_3d:
            B,T,J,3

        return:
            B,T,J,C
        """

        feat, _ = self.stgcn(
            human_3d,
            return_features=True,
        )

        feat = self.proj(feat)

        return feat


# =========================================================
# EQUIPMENT QUERY DECODER
# =========================================================
class EquipmentQueryDecoder(nn.Module):

    def __init__(
        self,
        num_queries=4,
        dim=256,
        num_heads=8,
        num_layers=3,
    ):
        super().__init__()

        self.num_queries = int(num_queries)

        # learnable equipment queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # temporal modeling
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
        )

        self.temporal_attn = nn.TransformerEncoder(
            temporal_layer,
            num_layers=2,
        )

        # each query predicts:
        # endpoint_1(xyz) + endpoint_2(xyz)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 6),
        )

    def forward(self, memory):
        """
        memory:
            B,T,N,C

        return:
            B,T,Q,2,3
        """

        B, T, N, C = memory.shape

        # -------------------------------------------------
        # frame-wise decoder
        # -------------------------------------------------
        memory = memory.reshape(
            B * T,
            N,
            C,
        )

        queries = self.queries.unsqueeze(0)

        queries = queries.repeat(
            B * T,
            1,
            1,
        )

        z = self.decoder(
            tgt=queries,
            memory=memory,
        )

        z = z.reshape(
            B,
            T,
            self.num_queries,
            C,
        )

        # -------------------------------------------------
        # temporal attention
        # -------------------------------------------------
        Q = z.shape[2]

        z = z.permute(
            0,
            2,
            1,
            3,
        )

        z = z.reshape(
            B * Q,
            T,
            C,
        )

        z = self.temporal_attn(z)

        z = z.reshape(
            B,
            Q,
            T,
            C,
        )

        z = z.permute(
            0,
            2,
            1,
            3,
        )

        # -------------------------------------------------
        # regression
        # -------------------------------------------------
        pred = self.reg_head(z)

        pred = pred.reshape(
            B,
            T,
            Q,
            2,
            3,
        )

        return pred


# =========================================================
# FULL MODEL
# =========================================================
class Pose2EquipNet(nn.Module):
    """
    Query definition:

        query 0:
            left ski

        query 1:
            right ski

        query 2:
            left pole

        query 3:
            right pole

    Each query predicts:
        2 endpoints

    Output:
        B,T,4,2,3
    """

    def __init__(
        self,
        num_joints=17,
        hidden_dim=256,
        target_skeleton_connections_idx=None,
        dino_model_name="facebook/dinov2-base",
        dino_freeze=True,
    ):
        super().__init__()

        # ---------------------------------------------
        # image encoder
        # ---------------------------------------------
        self.image_encoder = DinoPatchEncoder(
            model_name=dino_model_name,
            out_dim=hidden_dim,
            freeze=dino_freeze,
        )

        # ---------------------------------------------
        # pose encoder
        # ---------------------------------------------
        self.pose_encoder = PoseEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            target_skeleton_connections_idx=(target_skeleton_connections_idx),
        )

        # ---------------------------------------------
        # decoder
        # ---------------------------------------------
        self.decoder = EquipmentQueryDecoder(
            num_queries=4,
            dim=hidden_dim,
            num_heads=8,
            num_layers=3,
        )

    def forward(
        self,
        human_frame: torch.Tensor,
        human_3d: torch.Tensor,
    ):
        """
        human_frame:
            B,T,3,H,W

        human_3d:
            B,T,J,3
        """

        # ---------------------------------------------
        # image tokens
        # ---------------------------------------------
        image_tokens = self.image_encoder(human_frame)

        # B,T,N_img,C

        # ---------------------------------------------
        # pose tokens
        # ---------------------------------------------
        pose_tokens = self.pose_encoder(human_3d)

        # B,T,J,C

        # ---------------------------------------------
        # multimodal memory
        # ---------------------------------------------
        memory = torch.cat(
            [
                image_tokens,
                pose_tokens,
            ],
            dim=2,
        )

        # B,T,N,C

        # ---------------------------------------------
        # decode equipment
        # ---------------------------------------------
        pred = self.decoder(memory)

        return {"object_3d": pred}


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
        if self.num_equip_kpts % 2 != 0:
            raise ValueError(
                f"num_equip_kpts must be even (pairs of endpoints), got {self.num_equip_kpts}"
            )
        self.num_equip = self.num_equip_kpts // 2

        self.pose_encoder = PoseEncoder(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            target_skeleton_connections_idx=target_skeleton_connections_idx,
        )
        self.equip_head = nn.Linear(hidden_dim, self.num_equip_kpts * 3)

    def forward(self, human_3d: torch.Tensor):
        # human_3d: [B, J, 3] or [B, T, J, 3]
        # Output: [B, 4, 2, 3] (single frame) or [B, T, 4, 2, 3] (sequence)
        if human_3d.ndim == 3:
            # [B, J, 3] -> [B, 1, J, 3]
            human_3d = human_3d.unsqueeze(1)
            single_frame_mode = True
        elif human_3d.ndim == 4:
            single_frame_mode = False
        else:
            raise ValueError(
                f"Expected human_3d shape [B,J,3] or [B,T,J,3], got {tuple(human_3d.shape)}"
            )

        b = human_3d.shape[0]
        t = human_3d.shape[1]

        # PoseEncoder returns [B, T, J, C]
        pose_feat = self.pose_encoder(human_3d)

        # Aggregate joint dimension to get one feature per frame: [B, T, C]
        pose_feat = pose_feat.mean(dim=2)

        pred_obj = self.equip_head(pose_feat).reshape(b, t, self.num_equip, 2, 3)

        if single_frame_mode:
            pred_obj = pred_obj.squeeze(1)  # [B, 4, 2, 3]

        return {"object_3d": pred_obj}
