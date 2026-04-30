#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple as _Tuple

JOINT_FEAT_DIM = 26  # 3+3+3+1 + 3+3+3+3 + 1+1+1+1


def build_velocity_confidence_proxy(pose: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build confidence proxy from per-joint temporal velocity.

    Args:
        pose: (B, T, J, 3)
    Returns:
        confidence: (B, T, J, 1), larger means more stable.
    """
    if pose.ndim != 4 or pose.shape[-1] != 3:
        raise ValueError(f"Expected pose shape (B,T,J,3), got {tuple(pose.shape)}")

    vel = torch.zeros_like(pose)
    vel[:, 1:] = pose[:, 1:] - pose[:, :-1]
    vel_mag = torch.norm(vel, dim=-1, keepdim=True)  # (B,T,J,1)

    # Robust per-sample scaling; faster motion -> lower confidence.
    scale = vel_mag.median(dim=1, keepdim=True).values.median(dim=2, keepdim=True).values
    conf = torch.exp(-vel_mag / (scale + eps))
    return torch.clamp(conf, 0.0, 1.0)


def build_joint_features(
    p_left: torch.Tensor,
    p_right: torch.Tensor,
    bone_edges: List[_Tuple[int, int]],
    lambda_v: float = 1.0,
    lambda_a: float = 0.5,
    lambda_b: float = 0.5,
    lambda_d: float = 1.0,
) -> torch.Tensor:
    """Construct a 26-dim per-joint feature vector from dual-view 3-D poses.

    Feature layout (total 26 per joint per frame):
        P_left          (3)  raw left-view pose
        P_right         (3)  raw right-view pose
        ΔP              (3)  cross-view discrepancy
        ||ΔP||          (1)  discrepancy magnitude
        V_left          (3)  left velocity
        V_right         (3)  right velocity
        A_left          (3)  left acceleration
        A_right         (3)  right acceleration
        E_bone_left     (1)  per-joint mean bone-length deviation (left)
        E_bone_right    (1)  per-joint mean bone-length deviation (right)
        C_left          (1)  pseudo-confidence (left)
        C_right         (1)  pseudo-confidence (right)

    Args:
        p_left, p_right: (B, T, J, 3)
        bone_edges: list of (u, v) joint-index pairs defining skeleton bones.
    Returns:
        features: (B, T, J, 26)
    """
    B, T, J, _ = p_left.shape
    device, dtype = p_left.device, p_left.dtype

    # ── cross-view ──────────────────────────────────────────────────────────
    delta = p_left - p_right                              # (B,T,J,3)
    delta_norm = delta.norm(dim=-1, keepdim=True)         # (B,T,J,1)

    # ── velocity (zero-padded at t=0) ───────────────────────────────────────
    v_left = torch.zeros_like(p_left)
    v_right = torch.zeros_like(p_right)
    v_left[:, 1:] = p_left[:, 1:] - p_left[:, :-1]
    v_right[:, 1:] = p_right[:, 1:] - p_right[:, :-1]

    # ── acceleration (zero-padded at t=0,1) ─────────────────────────────────
    a_left = torch.zeros_like(p_left)
    a_right = torch.zeros_like(p_right)
    a_left[:, 2:] = v_left[:, 2:] - v_left[:, 1:-1]
    a_right[:, 2:] = v_right[:, 2:] - v_right[:, 1:-1]

    # ── per-joint bone-length deviation ─────────────────────────────────────
    bone_err_l = torch.zeros(B, T, J, 1, device=device, dtype=dtype)
    bone_err_r = torch.zeros(B, T, J, 1, device=device, dtype=dtype)
    count = torch.zeros(J, device=device, dtype=dtype)
    for u, v in bone_edges:
        bl_l = (p_left[..., u, :] - p_left[..., v, :]).norm(dim=-1)    # (B,T)
        bl_r = (p_right[..., u, :] - p_right[..., v, :]).norm(dim=-1)
        ref_l = bl_l.mean(dim=1, keepdim=True)   # (B,1)
        ref_r = bl_r.mean(dim=1, keepdim=True)
        err_l = (bl_l - ref_l).abs().unsqueeze(-1)   # (B,T,1)
        err_r = (bl_r - ref_r).abs().unsqueeze(-1)
        bone_err_l[:, :, u] += err_l
        bone_err_l[:, :, v] += err_l
        bone_err_r[:, :, u] += err_r
        bone_err_r[:, :, v] += err_r
        count[u] += 1.0
        count[v] += 1.0
    denom = count.clamp(min=1.0).view(1, 1, J, 1)
    bone_err_l = bone_err_l / denom
    bone_err_r = bone_err_r / denom

    # ── pseudo-confidence ────────────────────────────────────────────────────
    vm_l = v_left.norm(dim=-1, keepdim=True)
    vm_r = v_right.norm(dim=-1, keepdim=True)
    am_l = a_left.norm(dim=-1, keepdim=True)
    am_r = a_right.norm(dim=-1, keepdim=True)
    conf_l = torch.exp(
        -lambda_v * vm_l - lambda_a * am_l - lambda_b * bone_err_l - lambda_d * delta_norm
    )
    conf_r = torch.exp(
        -lambda_v * vm_r - lambda_a * am_r - lambda_b * bone_err_r - lambda_d * delta_norm
    )

    # ── concatenate ──────────────────────────────────────────────────────────
    return torch.cat(
        [p_left, p_right, delta, delta_norm,
         v_left, v_right, a_left, a_right,
         bone_err_l, bone_err_r, conf_l, conf_r],
        dim=-1,
    )   # (B, T, J, 26)


class ViewGating(nn.Module):
    """Joint-wise uncertainty-aware view gating."""

    def __init__(
        self,
        hidden_dim: int = 128,
        use_conf: bool = True,
        predict_logvar: bool = False,
    ) -> None:
        super().__init__()
        self.use_conf = use_conf
        self.predict_logvar = predict_logvar

        in_dim = 9  # pL(3), pR(3), diff(3)
        if use_conf:
            in_dim += 2  # cL(1), cR(1)

        out_dim = 2 if predict_logvar else 1  # alpha, optional logvar
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        p_left: torch.Tensor,
        p_right: torch.Tensor,
        c_left: Optional[torch.Tensor] = None,
        c_right: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            p_left, p_right: (B,T,J,3)
            c_left, c_right: (B,T,J,1)
        Returns:
            p0: (B,T,J,3)
            alpha: (B,T,J,1)
            logvar: (B,T,J,1) or None
        """
        if p_left.shape != p_right.shape:
            raise ValueError(f"Left/right pose shape mismatch: {p_left.shape} vs {p_right.shape}")

        diff = p_left - p_right
        feats = [p_left, p_right, diff]

        if self.use_conf:
            if c_left is None or c_right is None:
                raise ValueError("use_conf=True requires c_left and c_right")
            feats.extend([c_left, c_right])

        x = torch.cat(feats, dim=-1)
        out = self.mlp(x)

        if self.predict_logvar:
            alpha = torch.sigmoid(out[..., :1])
            logvar = out[..., 1:2]
        else:
            alpha = torch.sigmoid(out)
            logvar = None

        p0 = alpha * p_left + (1.0 - alpha) * p_right
        return p0, alpha, logvar


class TemporalSSMBlock(nn.Module):
    """Lightweight Mamba-style temporal mixer (conv-SSM approximation)."""

    def __init__(self, d_model: int, expansion: int = 2, kernel_size: int = 5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        inner = d_model * expansion

        self.in_proj = nn.Linear(d_model, inner)
        self.dw_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=inner,
        )
        self.out_proj = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,D)"""
        h = self.norm(x)
        h = self.in_proj(h)
        h = h.transpose(1, 2)  # (B,D,T)
        h = self.dw_conv(h)
        h = h.transpose(1, 2)  # (B,T,D)
        h = self.act(h)
        h = self.dropout(h)
        h = self.out_proj(h)
        return h


class SSMRefiner(nn.Module):
    """Temporal pose refiner with residual correction."""

    def __init__(
        self,
        num_joints: int,
        d_model: int = 256,
        n_layers: int = 4,
        expansion: int = 2,
        kernel_size: int = 5,
        feat_dim: int = JOINT_FEAT_DIM,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.in_proj = nn.Linear(feat_dim * num_joints, d_model)
        self.blocks = nn.ModuleList(
            [TemporalSSMBlock(d_model, expansion=expansion, kernel_size=kernel_size) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, 3 * num_joints)

    def forward(self, feat: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, T, J, feat_dim)  rich per-joint feature vector
            p0:   (B, T, J, 3)         initial fused pose for residual addition
        Returns:
            p_hat: (B, T, J, 3)
        """
        bsz, t, joints, fd = feat.shape
        if joints != self.num_joints or fd != self.feat_dim:
            raise ValueError(
                f"Expected feat (B,T,{self.num_joints},{self.feat_dim}), got {tuple(feat.shape)}"
            )
        x = feat.reshape(bsz, t, fd * joints)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = x + blk(x)
        delta = self.out_proj(x).reshape(bsz, t, joints, 3)
        return p0 + delta


class Dual2PoseNet(nn.Module):
    """Uncertainty-aware view fusion + temporal SSM refinement."""

    def __init__(
        self,
        num_joints: int,
        d_model: int = 256,
        n_layers: int = 4,
        use_conf: bool = True,
        predict_logvar: bool = False,
        bone_edges: Optional[List[_Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.use_conf = use_conf
        self.bone_edges: List[_Tuple[int, int]] = bone_edges or []
        self.gating = ViewGating(
            hidden_dim=max(64, d_model // 2),
            use_conf=use_conf,
            predict_logvar=predict_logvar,
        )
        self.refiner = SSMRefiner(
            num_joints=num_joints,
            d_model=d_model,
            n_layers=n_layers,
            feat_dim=JOINT_FEAT_DIM,
        )

    def forward(
        self,
        p_left: torch.Tensor,
        p_right: torch.Tensor,
        c_left: Optional[torch.Tensor] = None,
        c_right: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with keys: p_hat, p0, alpha, and optional logvar."""
        if self.use_conf and (c_left is None or c_right is None):
            c_left = build_velocity_confidence_proxy(p_left)
            c_right = build_velocity_confidence_proxy(p_right)

        p0, alpha, logvar = self.gating(p_left, p_right, c_left, c_right)
        feat = build_joint_features(p_left, p_right, self.bone_edges)
        p_hat = self.refiner(feat, p0)

        out: Dict[str, torch.Tensor] = {"p_hat": p_hat, "p0": p0, "alpha": alpha}
        if logvar is not None:
            out["logvar"] = logvar
        return out


@dataclass
class PoseLossWeights:
    mpjpe: float = 1.0
    bone: float = 0.2
    vel: float = 0.05
    acc: float = 0.02
    agree: float = 0.1
    bone_stab: float = 0.05


class PoseRefineLoss(nn.Module):
    """Loss bundle for supervised + self-supervised adaptation."""

    def __init__(
        self,
        bone_edges: Optional[Sequence[Tuple[int, int]]] = None,
        weights: Optional[PoseLossWeights] = None,
    ) -> None:
        super().__init__()
        self.bone_edges = list(bone_edges) if bone_edges is not None else []
        self.w = weights or PoseLossWeights()

    @staticmethod
    def _temporal_velocity(x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:] - x[:, :-1]

    @staticmethod
    def _temporal_acceleration(x: torch.Tensor) -> torch.Tensor:
        return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]

    def _bone_lengths(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,J,3) -> (B,T,E)
        if not self.bone_edges:
            return x.new_zeros((x.shape[0], x.shape[1], 0))

        segs: List[torch.Tensor] = []
        for u, v in self.bone_edges:
            seg = torch.norm(x[..., u, :] - x[..., v, :], dim=-1, keepdim=True)
            segs.append(seg)
        return torch.cat(segs, dim=-1)

    def forward(
        self,
        p_hat: torch.Tensor,
        p_gt: Optional[torch.Tensor] = None,
        p_left: Optional[torch.Tensor] = None,
        p_right: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Unified forward for training loops.

        - If p_gt is provided: supervised loss
        - Else requires p_left/p_right/alpha: self-supervised loss
        """
        if p_gt is not None:
            return self.supervised(p_hat=p_hat, p_gt=p_gt, logvar=logvar)
        if p_left is None or p_right is None or alpha is None:
            raise ValueError("Self-supervised mode requires p_left, p_right and alpha")
        return self.self_supervised(p_hat=p_hat, p_left=p_left, p_right=p_right, alpha=alpha)

    def supervised(
        self,
        p_hat: torch.Tensor,
        p_gt: torch.Tensor,
        logvar: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if logvar is None:
            l_mpjpe = F.l1_loss(p_hat, p_gt)
        else:
            # heteroscedastic regression
            sq = (p_hat - p_gt).pow(2).mean(dim=-1, keepdim=True)
            l_mpjpe = torch.mean(torch.exp(-logvar) * sq + logvar)

        if self.bone_edges:
            b_hat = self._bone_lengths(p_hat)
            b_gt = self._bone_lengths(p_gt)
            l_bone = F.l1_loss(b_hat, b_gt)
        else:
            l_bone = p_hat.new_tensor(0.0)

        vel = self._temporal_velocity(p_hat)
        acc = self._temporal_acceleration(p_hat)
        l_vel = vel.abs().mean() if vel.numel() > 0 else p_hat.new_tensor(0.0)
        l_acc = acc.abs().mean() if acc.numel() > 0 else p_hat.new_tensor(0.0)

        total = (
            self.w.mpjpe * l_mpjpe
            + self.w.bone * l_bone
            + self.w.vel * l_vel
            + self.w.acc * l_acc
        )
        return {
            "loss": total,
            "loss/mpjpe": l_mpjpe,
            "loss/bone": l_bone,
            "loss/vel": l_vel,
            "loss/acc": l_acc,
        }

    def self_supervised(
        self,
        p_hat: torch.Tensor,
        p_left: torch.Tensor,
        p_right: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # agreement: alpha decides which view to trust more
        l_agree = (alpha * (p_hat - p_left).abs() + (1.0 - alpha) * (p_hat - p_right).abs()).mean()

        vel = self._temporal_velocity(p_hat)
        acc = self._temporal_acceleration(p_hat)
        l_vel = vel.abs().mean() if vel.numel() > 0 else p_hat.new_tensor(0.0)
        l_acc = acc.abs().mean() if acc.numel() > 0 else p_hat.new_tensor(0.0)

        if self.bone_edges:
            b = self._bone_lengths(p_hat)
            b_prev = b[:, :-1]
            b_next = b[:, 1:]
            l_bone_stab = (b_next - b_prev).abs().mean() if b_next.numel() > 0 else p_hat.new_tensor(0.0)
        else:
            l_bone_stab = p_hat.new_tensor(0.0)

        total = (
            self.w.agree * l_agree
            + self.w.vel * l_vel
            + self.w.acc * l_acc
            + self.w.bone_stab * l_bone_stab
        )

        return {
            "loss": total,
            "loss/agree": l_agree,
            "loss/vel": l_vel,
            "loss/acc": l_acc,
            "loss/bone_stab": l_bone_stab,
        }
