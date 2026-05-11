#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/map_config.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project
Created Date: Monday March 9th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday March 9th 2026 11:22:51 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, Optional

import numpy as np

# * 这里定义了sam 3d body需要过滤的关节点序号和名字
SAM3D_BODY_MAPPING = {
    1: "Bone_Eye_L",
    2: "Bone_Eye_R",
    5: "Upperarm_L",
    6: "Upperarm_R",
    7: "lowerarm_l",
    8: "lowerarm_r",
    9: "Thigh_L",
    10: "Thigh_R",
    11: "calf_l",
    12: "calf_r",
    13: "Foot_L",
    14: "Foot_R",
    41: "Hand_R",
    62: "Hand_L",
    69: "neck_01",
}

# * 这里定义了unity需要过滤的关节点序号和名字
# * key 是 joint_names_character.json 中的0-based索引，value 与 SAM3D_BODY_MAPPING 保持一致
UNITY_MALE_MAPPING = {
    51: "Bone_Eye_L",
    52: "Bone_Eye_R",
    4: "Upperarm_L",
    27: "Upperarm_R",
    5: "lowerarm_l",
    28: "lowerarm_r",
    78: "Thigh_L",
    87: "Thigh_R",
    79: "calf_l",
    88: "calf_r",
    83: "Foot_L",
    92: "Foot_R",
    30: "Hand_R",
    7: "Hand_L",
    49: "neck_01",
}

UNITY_FEMALE_MAPPING = {
    51: "Bone_Eye_L",
    52: "Bone_Eye_R",
    4: "Upperarm_L",
    27: "Upperarm_R",
    5: "lowerarm_l",
    28: "lowerarm_r",
    79: "Thigh_L",
    88: "Thigh_R",
    80: "calf_l",
    89: "calf_r",
    82: "Foot_L",
    91: "Foot_R",
    29: "Hand_R",
    6: "Hand_L",
    49: "neck_01",
}

# * 过滤后的index和关节点名字
FILTERED_KPTS_MAPPING = {
    0: "Bone_Eye_L",
    1: "Bone_Eye_R",
    2: "Upperarm_L",
    3: "Upperarm_R",
    4: "lowerarm_l",
    5: "lowerarm_r",
    6: "Thigh_L",
    7: "Thigh_R",
    8: "calf_l",
    9: "calf_r",
    10: "Foot_L",
    11: "Foot_R",
    12: "Hand_R",
    13: "Hand_L",
    14: "neck_01",
}

SKI_MAPPING = {
    0: "ski_left_center",
    1: "ski_left_front",
    2: "ski_left_back",
    3: "ski_right_center",
    4: "ski_right_front",
    5: "ski_right_back",
}

POLE_MAPPING = {
    0: "pole_left_handle",
    1: "pole_left_tip",
    2: "pole_right_handle",
    3: "pole_right_tip",
}

OBJ_MAPPING = {
    0: "left_ski_front",
    1: "left_ski_back",
    2: "right_ski_front",
    3: "right_ski_back",
    4: "left_pole_handle",
    5: "left_pole_tip",
    6: "right_pole_handle",
    7: "right_pole_tip",
}

# Skeleton connections after filtering, represented by contiguous joint indices.

FILTER_SKELETON_CONNECTIONS = [
    # left arm
    (14, 2),  # neck -> shoulder_l
    (2, 4),  # shoulder_l -> elbow_l
    (4, 13),  # elbow_l -> hand_l
    # right arm
    (14, 3),  # neck -> shoulder_r
    (3, 5),  # shoulder_r -> elbow_r
    (5, 12),  # elbow_r -> hand_r
    # spine
    (14, 6),  # neck -> hip_l
    (14, 7),  # neck -> hip_r
    # left leg
    (6, 8),  # hip_l -> knee_l
    (8, 10),  # knee_l -> foot_l
    # right leg
    (7, 9),  # hip_r -> knee_r
    (9, 11),  # knee_r -> foot_r
]


@dataclass
class UnityDataConfig:
    """全局映射配置类，包含与 Unity MHR70 骨骼结构相关的映射和配置。"""

    person_id: str
    action_id: str
    cam1_id: str
    cam2_id: str

    label_path: str

    cam1_frames_dir: str
    cam2_frames_dir: str

    sequence_meta_path: str
    joint_names_path: str

    # root dir, maybe use?
    cam1_kpt2d_dir: str
    cam2_kpt2d_dir: str
    kpt3d_dir: str
    # ground true from unity
    cam1_kpt2d_dirs: Optional[Dict[str, str]]
    cam2_kpt2d_dirs: Optional[Dict[str, str]]
    kpt3d_dirs: Optional[Dict[str, str]]

    # sam3d body results
    sam3d_cam1_kpt2d_dir: str
    sam3d_cam2_kpt2d_dir: str
    sam3d_cam1_kpt3d_dir: str
    sam3d_cam2_kpt3d_dir: str

    # sam3 mask
    sam3_cam1_mask_ski_dir: str
    sam3_cam2_mask_ski_pole_dir: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "UnityDataConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrueDataConfig:
    """真实数据的全局映射配置类，包含与真实数据相关的路径和标识符。"""

    person_id: str

    left_cam_frames_dir: str
    right_cam_frames_dir: str

    left_cam_sam3d_kpt2d_dir: str
    right_cam_sam3d_kpt2d_dir: str

    left_cam_sam3d_kpt3d_dir: str
    right_cam_sam3d_kpt3d_dir: str


def _normalize_kpts_array(kpts: np.ndarray) -> np.ndarray:
    arr = np.asarray(kpts, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected keypoints shape (J,C), got {arr.shape}")
    if arr.shape[1] not in (2, 3):
        raise ValueError(f"Expected C in (2,3), got {arr.shape}")
    return arr


def filter_sam3d_body_kpts(kpts: np.ndarray) -> np.ndarray:
    """Filter SAM3D body keypoints to FILTERED_KPTS_MAPPING order.

    Directly uses SAM3D_BODY_MAPPING keys as source indices, so output order
    matches FILTERED_KPTS_MAPPING exactly.
    """
    arr = _normalize_kpts_array(kpts)
    selected = list(SAM3D_BODY_MAPPING.keys())
    if max(selected) >= arr.shape[0]:
        raise IndexError(
            f"SAM3D_BODY_MAPPING index out of range for source shape {arr.shape}."
        )
    return arr[selected]


def filter_unity_kpts(kpts: np.ndarray, flag: str, gender: str = "male") -> np.ndarray:
    """Filter Unity keypoints to FILTERED_KPTS_MAPPING order.

    Directly uses UNITY_MAPPING keys as source indices, output order matches
    filter_sam3d_body_kpts exactly.
    """
    arr = _normalize_kpts_array(kpts)
    if flag == "3d":
        if gender == "male":
            selected = list(UNITY_MALE_MAPPING.keys())
        else:
            selected = list(UNITY_FEMALE_MAPPING.keys())
    elif flag == "2d":
        # 2d keypoints index + 1
        if gender == "male":
            selected = [idx + 1 for idx in UNITY_MALE_MAPPING.keys()]
        else:
            selected = [idx + 1 for idx in UNITY_FEMALE_MAPPING.keys()]
    else:
        raise ValueError(
            f"Invalid flag {flag} for filter_unity_kpts, expected '2d' or '3d'."
        )

    if max(selected) >= arr.shape[0]:
        raise IndexError(
            f"UNITY_MAPPING index out of range for source shape {arr.shape}."
        )
    return arr[selected]
