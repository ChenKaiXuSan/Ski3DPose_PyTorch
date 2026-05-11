#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/dataloader/canonicalize.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/dataloader
Created Date: Sunday May 10th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Sunday May 10th 2026 10:34:01 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import numpy as np

def normalize(v, eps=1e-8):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)

def canonicalize_pose_numpy(
    x,
    left_hip,
    right_hip,
    neck,
    mode="first_frame",
    eps=1e-8,
    enforce_face_z_positive=False,
    left_eye=None,
    right_eye=None,
):

    """
    x: (J, 3) or (T, J, 3)
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
    elif x.ndim != 3:
        raise ValueError("x must have shape (J, 3) or (T, J, 3)")

    left_hip_pos = x[:, left_hip]
    right_hip_pos = x[:, right_hip]
    neck_pos = x[:, neck]

    pelvis = (left_hip_pos + right_hip_pos) / 2.0
    x_centered = x - pelvis[:, None, :]

    x_axis = normalize(right_hip_pos - left_hip_pos, eps)  # body right
    y_axis = normalize(neck_pos - pelvis, eps)             # body up

    z_axis = normalize(np.cross(x_axis, y_axis), eps)      # forward/back (unsigned)

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

    return x_canon, transform

def apply_canonical_transform_numpy(points, pelvis, R):
    """Apply a precomputed canonical transform to arbitrary 3D points."""
    pts = np.asarray(points, dtype=np.float32)
    return np.matmul(pts - pelvis[None, :], R)