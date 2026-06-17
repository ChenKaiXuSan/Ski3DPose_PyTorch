#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/dataloader/canonicalize.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/dual2pose/dataloader
Created Date: Sunday May 10th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday May 11th 2026 6:59:18 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import numpy as np
import torch


def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)


def _normalize_torch(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def canonicalize_pose_numpy(
    x,
    left_hip=6,
    right_hip=7,
    neck=14,
    mode="first_frame",
    eps=1e-8,
    enforce_face_z_positive=True,
    left_eye=0,
    right_eye=1,
):
    """
    x: (J, 3) or (T, J, 3) or (B, T, J, 3)
    left_hip, right_hip, neck: joint indices for pelvis canonicalization
    mode:
        "per_frame"
        "first_frame"
    enforce_face_z_positive:
        当提供 left_eye/right_eye 且为 True 时，强制 +Z 与(双眼中点-颈部)同向
    return:
        x_canon: same shape as x
        transform dict with pelvis and R
    """

    x = np.asarray(x, dtype=np.float32)

    squeeze_time = False
    if x.ndim == 2:
        x = x[None, ...]
        squeeze_time = True
    elif x.ndim == 3 and x.shape[1] != 3:
        pass
    elif x.ndim == 4:
        b, t, j, c = x.shape
        x = x.reshape(b * t, j, c)
    elif x.ndim != 3 and x.ndim != 4:
        raise ValueError("x must have shape (J, 3) or (T, J, 3) or (B, T, J, 3)")

    left_hip_pos = x[:, left_hip]
    right_hip_pos = x[:, right_hip]
    neck_pos = x[:, neck]

    pelvis = (left_hip_pos + right_hip_pos) / 2.0
    x_centered = x - pelvis[:, None, :]

    x_axis = normalize(right_hip_pos - left_hip_pos, eps)  # body right
    y_axis = normalize(neck_pos - pelvis, eps)  # body up

    z_axis = normalize(np.cross(x_axis, y_axis), eps)  # forward/back (unsigned)

    if enforce_face_z_positive and left_eye is not None and right_eye is not None:
        eye_mid = (x[:, left_eye] + x[:, right_eye]) / 2.0
        face_dir = normalize(eye_mid - neck_pos, eps)
        sign = np.sign(np.sum(z_axis * face_dir, axis=-1, keepdims=True))
        sign[sign == 0] = 1.0
        z_axis = z_axis * sign

    # y_axis = normalize(np.cross(z_axis, x_axis), eps)      # re-orthogonalized up
    R = np.stack([x_axis, y_axis, z_axis], axis=-1)

    if mode == "first_frame":
        ref_pelvis = pelvis[0]
        ref_R = R[0]
        x_canon = np.matmul(x - ref_pelvis[None, None, :], ref_R)
        transform = {"pelvis": ref_pelvis, "R": ref_R}
    elif mode == "per_frame":
        x_canon = np.einsum("tjc,tck->tjk", x_centered, R)
        transform = {"pelvis": pelvis, "R": R}
    else:
        raise ValueError("mode must be first_frame or per_frame")

    if squeeze_time:
        x_canon = x_canon[0]

    x_canon = x_canon.reshape(b, t, j, c)

    return x_canon, transform


def apply_canonical_transform_numpy(points, pelvis, R):
    """Apply a precomputed canonical transform to arbitrary 3D points."""
    pts = np.asarray(points, dtype=np.float32)
    return np.matmul(pts - pelvis[None, :], R)


# ---------------------------------------------------------------------------
# PyTorch version
# ---------------------------------------------------------------------------


def canonicalize_pose_torch(
    x: torch.Tensor,
    left_hip: int = 6,
    right_hip: int = 7,
    neck: int = 14,
    mode: str = "first_frame",
    eps: float = 1e-8,
    enforce_face_z_positive: bool = True,
    left_eye: int = 0,
    right_eye: int = 1,
):
    """
    PyTorch version of canonicalize_pose_numpy.

    x: (J, 3) or (T, J, 3) or (B, T, J, 3)
    mode: "per_frame" | "first_frame"
    Returns:
        x_canon: same shape as x
        transform: dict with "pelvis" and "R"
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    else:
        x = x.float()

    original_shape = x.shape
    squeeze_time = False

    if x.ndim == 2:  # (J, 3) → (1, J, 3)
        x = x.unsqueeze(0)
        squeeze_time = True
        b, t = 1, 1
        j, c = original_shape
    elif x.ndim == 3:  # (T, J, 3)
        b, t = 1, x.shape[0]
        j, c = x.shape[1], x.shape[2]
    elif x.ndim == 4:  # (B, T, J, 3)
        b, t, j, c = x.shape
        x = x.reshape(b * t, j, c)
    else:
        raise ValueError("x must have shape (J,3), (T,J,3) or (B,T,J,3)")

    # x is now (N, J, 3) where N = B*T or T or 1
    left_hip_pos = x[:, left_hip]  # (N, 3)
    right_hip_pos = x[:, right_hip]  # (N, 3)
    neck_pos = x[:, neck]  # (N, 3)

    pelvis = (left_hip_pos + right_hip_pos) / 2.0  # (N, 3)
    x_centered = x - pelvis[:, None, :]  # (N, J, 3)

    x_axis = _normalize_torch(right_hip_pos - left_hip_pos, eps)  # (N, 3)
    y_axis = _normalize_torch(neck_pos - pelvis, eps)  # (N, 3)
    z_axis = _normalize_torch(torch.linalg.cross(x_axis, y_axis), eps)  # (N, 3)

    if enforce_face_z_positive and left_eye is not None and right_eye is not None:
        eye_mid = (x[:, left_eye] + x[:, right_eye]) / 2.0  # (N, 3)
        face_dir = _normalize_torch(eye_mid - neck_pos, eps)  # (N, 3)
        sign = torch.sign((z_axis * face_dir).sum(dim=-1, keepdim=True))  # (N,1)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        z_axis = z_axis * sign

    # R: (N, 3, 3), columns are [x_axis, y_axis, z_axis]
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N, 3, 3)

    if mode == "first_frame":
        ref_pelvis = pelvis[0]  # (3,)
        ref_R = R[0]  # (3, 3)
        x_canon = (x - ref_pelvis[None, None, :]) @ ref_R  # (N, J, 3)
        transform = {"pelvis": ref_pelvis, "R": ref_R}
    elif mode == "per_frame":
        x_canon = torch.einsum("njc,nck->njk", x_centered, R)  # (N, J, 3)
        transform = {"pelvis": pelvis, "R": R}
    else:
        raise ValueError("mode must be 'first_frame' or 'per_frame'")

    if squeeze_time:
        x_canon = x_canon[0]  # (J, 3)
    else:
        x_canon = x_canon.reshape(b, t, j, c)

    return x_canon, transform


def apply_canonical_transform_torch(
    points: torch.Tensor,
    pelvis: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    """Apply a precomputed canonical transform to arbitrary 3D points.

    points: (..., 3)
    pelvis: (3,)
    R:      (3, 3)
    """
    return (points - pelvis) @ R
