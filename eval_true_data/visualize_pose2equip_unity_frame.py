#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.map_config import (
    ID_TO_INDEX,
    SKELETON_CONNECTIONS,
    TARGET_IDS,
    UNITY_MHR70_MAPPING,
)
from project.models.pose2equip_net import Pose2EquipNet

EQUIP_LABELS = [
    "left_ski_tip",
    "left_ski_tail",
    "right_ski_tip",
    "right_ski_tail",
    "left_pole_grip",
    "left_pole_tip",
    "right_pole_grip",
    "right_pole_tip",
]

EQUIP_SEGMENTS = [
    (0, 1),  # left ski
    (2, 3),  # right ski
    (4, 5),  # left pole
    (6, 7),  # right pole
]

TARGET_JOINT_NAMES = [UNITY_MHR70_MAPPING[jid] for jid in TARGET_IDS]


def _extract_float_token(token: str) -> Optional[float]:
    try:
        return float(token)
    except Exception:
        return None


def _parse_ckpt_name(
    ckpt_name: str,
) -> Optional[Tuple[int, Optional[float], Optional[float]]]:
    # epoch-loss-mpjpe.ckpt or epoch-loss.ckpt
    m = re.match(r"^(\d+)-([^\-]+)(?:-([^\-]+))?\.ckpt$", ckpt_name)
    if not m:
        return None
    epoch = int(m.group(1))
    m1 = _extract_float_token(m.group(2))
    m2 = _extract_float_token(m.group(3)) if m.group(3) is not None else None
    return epoch, m1, m2


def _select_best_ckpt(ckpt_dir: Path) -> Path:
    all_ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not all_ckpts:
        all_ckpts = sorted(ckpt_dir.rglob("*.ckpt"))
    if not all_ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    candidates = [
        p for p in all_ckpts if p.name.lower() not in {"last.ckpt", "last-v1.ckpt"}
    ]
    if not candidates:
        for name in ["last.ckpt", "last-v1.ckpt"]:
            p = ckpt_dir / name
            if p.exists():
                return p
        return max(all_ckpts, key=lambda p: p.stat().st_mtime)

    parsed: List[Tuple[Path, int, float]] = []
    for p in candidates:
        rec = _parse_ckpt_name(p.name)
        if rec is None:
            continue
        epoch, m1, m2 = rec
        metric = m2 if m2 is not None else m1
        if metric is None:
            continue
        # metric lower is better (loss/mpjpe)
        parsed.append((p, epoch, float(metric)))

    if not parsed:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    parsed.sort(key=lambda x: (x[2], -x[1]))
    return parsed[0][0]


def _extract_last_int(name: str) -> int:
    nums = re.findall(r"(\d+)", name)
    if not nums:
        raise ValueError(f"No frame index found in filename: {name}")
    six_digits = [x for x in nums if len(x) >= 6]
    if six_digits:
        return int(six_digits[0])
    return int(nums[-1])


def _build_idx_file_map(root: Path, patterns: Iterable[str]) -> Dict[int, Path]:
    if not root.exists() or not root.is_dir():
        return {}
    out: Dict[int, Path] = {}
    for pattern in patterns:
        for p in sorted(root.glob(pattern)):
            idx = _extract_last_int(p.stem)
            out[idx] = p
    return out


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _frame_to_model_tensor(
    frame_rgb: np.ndarray, image_size: int, device: torch.device
) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frame_rgb, dtype=np.float32)).permute(
        2, 0, 1
    )
    x = x / 255.0
    x = x.unsqueeze(0)
    x = F.interpolate(
        x, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.to(device)


def _load_single_mask_npz(npz_path: Path) -> np.ndarray:
    """Load one mask npz and return merged binary mask with shape [1,H,W]."""
    data = np.load(npz_path, allow_pickle=True)
    if "masks" not in data.files:
        raise KeyError(f"'masks' not found in {npz_path}")
    masks = np.asarray(data["masks"])
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks ndim=3 after normalize, got {masks.shape}")
    if masks.shape[0] == 0:
        merged = np.zeros(masks.shape[-2:], dtype=np.float32)
    else:
        merged = np.any(masks > 0, axis=0).astype(np.float32, copy=False)
    return merged[None, ...]


def _mask_to_model_tensor(mask_chw: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    """Convert [1,H,W] mask to [1,1,image_size,image_size] float tensor."""
    x = torch.from_numpy(np.ascontiguousarray(mask_chw, dtype=np.float32)).unsqueeze(0)
    x = F.interpolate(x, size=(image_size, image_size), mode="nearest")
    return x.to(device)


def _to_model_joint_count(human_3d: np.ndarray, expected_joints: int) -> np.ndarray:
    if human_3d.ndim != 2 or human_3d.shape[1] != 3:
        raise ValueError(f"Expected human_3d shape [J,3], got {human_3d.shape}")
    if human_3d.shape[0] == expected_joints:
        return human_3d.astype(np.float32)

    # Use a stable mapping policy here to avoid coupling to dataloader-specific
    # index rules (single-view and dual-view loaders differ in candidate priority).
    src_idx: List[int] = []
    for jid in [jid for jid, _ in sorted(ID_TO_INDEX.items(), key=lambda kv: kv[1])]:
        candidates = (jid, ID_TO_INDEX[jid])
        mapped = next((c for c in candidates if 0 <= c < human_3d.shape[0]), None)
        if mapped is None:
            raise IndexError(
                f"Target joint id {jid} cannot be mapped for source joint count {human_3d.shape[0]}."
            )
        src_idx.append(int(mapped))

    filtered = human_3d[src_idx]
    if filtered.shape[0] != expected_joints:
        raise ValueError(
            f"Joint count mismatch after filtering: expected {expected_joints}, got {filtered.shape[0]}"
        )
    return filtered.astype(np.float32)


def _load_joint_name_to_index(path: Path) -> Optional[Dict[str, int]]:
    """Load joint_names json and build name->index map.

    Some metadata files are saved with BOM, so we use utf-8-sig.
    """
    if not path.exists() or not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

    names: Optional[List[str]] = None
    if isinstance(payload, dict):
        raw = payload.get("joint_names")
        if isinstance(raw, list):
            names = [str(x) for x in raw]
    elif isinstance(payload, list):
        names = [str(x) for x in payload]

    if not names:
        return None

    return {name: idx for idx, name in enumerate(names)}


def _to_model_joint_count_from_names(
    human_3d: np.ndarray,
    expected_joints: int,
    name_to_index: Optional[Dict[str, int]],
) -> np.ndarray:
    """Select 15 target joints by semantic names from GT character arrays."""
    if human_3d.ndim != 2 or human_3d.shape[1] != 3:
        raise ValueError(f"Expected human_3d shape [J,3], got {human_3d.shape}")

    if name_to_index is None:
        return _to_model_joint_count(human_3d, expected_joints)

    missing = [name for name in TARGET_JOINT_NAMES if name not in name_to_index]
    if missing:
        raise KeyError(f"Missing target joint names in metadata: {missing}")

    src_idx = [int(name_to_index[name]) for name in TARGET_JOINT_NAMES]
    if max(src_idx) >= human_3d.shape[0]:
        raise IndexError(
            f"Joint index out of range for shape {human_3d.shape}: idx={src_idx}"
        )

    filtered = human_3d[src_idx]
    if filtered.shape[0] != expected_joints:
        raise ValueError(
            f"Joint count mismatch after name mapping: expected {expected_joints}, got {filtered.shape[0]}"
        )
    return filtered.astype(np.float32)


def _draw_human_skeleton_3d(ax, human_3d: np.ndarray) -> None:
    ax.scatter(
        human_3d[:, 0], human_3d[:, 1], human_3d[:, 2], s=10, c="tab:blue", alpha=0.7
    )

    edges: List[Tuple[int, int]] = []
    for a, b in SKELETON_CONNECTIONS:
        if a in ID_TO_INDEX and b in ID_TO_INDEX:
            edges.append((ID_TO_INDEX[a], ID_TO_INDEX[b]))

    for i, j in edges:
        if i < human_3d.shape[0] and j < human_3d.shape[0]:
            ax.plot(
                [human_3d[i, 0], human_3d[j, 0]],
                [human_3d[i, 1], human_3d[j, 1]],
                [human_3d[i, 2], human_3d[j, 2]],
                color="tab:blue",
                linewidth=1.2,
                alpha=0.8,
            )


def _draw_equipment_3d(
    ax,
    equip_obj: np.ndarray,
    color: str = "tab:red",
    alpha: float = 0.95,
    linewidth: float = 2.4,
    linestyle: str = "-",
    label: Optional[str] = None,
) -> None:
    ax.scatter(
        equip_obj[:, 0],
        equip_obj[:, 1],
        equip_obj[:, 2],
        s=18,
        c=color,
        alpha=alpha,
        label=label,
    )

    for i, j in EQUIP_SEGMENTS:
        ax.plot(
            [equip_obj[i, 0], equip_obj[j, 0]],
            [equip_obj[i, 1], equip_obj[j, 1]],
            [equip_obj[i, 2], equip_obj[j, 2]],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )


def _draw_equipment_ski_pole(
    ax,
    equip_obj: np.ndarray,
    ski_color: str,
    pole_color: str,
    label_prefix: str,
    linestyle: str = "-",
) -> None:
    """Draw ski and pole parts with different colors.

    Order in equip_obj:
      ski: 0..3, pole: 4..7
    """
    ski_pts = equip_obj[:4]
    pole_pts = equip_obj[4:]

    ax.scatter(
        ski_pts[:, 0],
        ski_pts[:, 1],
        ski_pts[:, 2],
        s=18,
        c=ski_color,
        alpha=0.95,
        label=f"{label_prefix}_ski",
    )
    ax.scatter(
        pole_pts[:, 0],
        pole_pts[:, 1],
        pole_pts[:, 2],
        s=18,
        c=pole_color,
        alpha=0.95,
        label=f"{label_prefix}_pole",
    )

    for i, j in EQUIP_SEGMENTS[:2]:
        ax.plot(
            [equip_obj[i, 0], equip_obj[j, 0]],
            [equip_obj[i, 1], equip_obj[j, 1]],
            [equip_obj[i, 2], equip_obj[j, 2]],
            color=ski_color,
            linewidth=2.4,
            linestyle=linestyle,
        )
    for i, j in EQUIP_SEGMENTS[2:]:
        ax.plot(
            [equip_obj[i, 0], equip_obj[j, 0]],
            [equip_obj[i, 1], equip_obj[j, 1]],
            [equip_obj[i, 2], equip_obj[j, 2]],
            color=pole_color,
            linewidth=2.4,
            linestyle=linestyle,
        )


def _build_gt_segments_by_config(
    ski_kpt3d: np.ndarray,
    pole_kpt3d: np.ndarray,
    ski_idx: List[int],
    pole_idx: List[int],
) -> Dict[str, np.ndarray]:
    """Build GT segments using exactly the configured connection indices."""
    ski = np.asarray(ski_kpt3d, dtype=np.float32)
    pole = np.asarray(pole_kpt3d, dtype=np.float32)
    return {
        "ski_left": ski[[ski_idx[0], ski_idx[1]]],
        "ski_right": ski[[ski_idx[2], ski_idx[3]]],
        "pole_left": pole[[pole_idx[0], pole_idx[1]]],
        "pole_right": pole[[pole_idx[2], pole_idx[3]]],
    }


def _draw_gt_segments_by_config(ax, gt_segments: Dict[str, np.ndarray]) -> None:
    """Draw GT ski/pole segments with explicit configured connections."""
    ski_color = "tab:green"
    pole_color = "tab:olive"

    sl = gt_segments["ski_left"]
    sr = gt_segments["ski_right"]
    pl = gt_segments["pole_left"]
    pr = gt_segments["pole_right"]

    ski_pts = np.concatenate([sl, sr], axis=0)
    pole_pts = np.concatenate([pl, pr], axis=0)

    ax.scatter(
        ski_pts[:, 0],
        ski_pts[:, 1],
        ski_pts[:, 2],
        s=22,
        c=ski_color,
        alpha=0.95,
        label="gt_ski",
    )
    ax.scatter(
        pole_pts[:, 0],
        pole_pts[:, 1],
        pole_pts[:, 2],
        s=22,
        c=pole_color,
        alpha=0.95,
        label="gt_pole",
    )

    for seg in [sl, sr]:
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=ski_color, linewidth=2.8)
    for seg in [pl, pr]:
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=pole_color, linewidth=2.8)


def _anchor_consistency_metrics(
    human_3d: np.ndarray,
    equip_obj: np.ndarray,
    left_ankle_idx: int,
    right_ankle_idx: int,
    left_wrist_idx: int,
    right_wrist_idx: int,
) -> Dict[str, float]:
    """Compute anchor consistency diagnostics between body anchors and equipment anchors."""
    la = int(left_ankle_idx)
    ra = int(right_ankle_idx)
    lw = int(left_wrist_idx)
    rw = int(right_wrist_idx)

    if max(la, ra, lw, rw) >= human_3d.shape[0]:
        return {}

    left_ski_center = 0.5 * (equip_obj[0] + equip_obj[1])
    right_ski_center = 0.5 * (equip_obj[2] + equip_obj[3])
    left_pole_grip = equip_obj[4]
    right_pole_grip = equip_obj[6]

    return {
        "ankleL_to_skiL_center": float(np.linalg.norm(human_3d[la] - left_ski_center)),
        "ankleR_to_skiR_center": float(np.linalg.norm(human_3d[ra] - right_ski_center)),
        "wristL_to_poleL_grip": float(np.linalg.norm(human_3d[lw] - left_pole_grip)),
        "wristR_to_poleR_grip": float(np.linalg.norm(human_3d[rw] - right_pole_grip)),
    }


def _compose_gt_equipment_points(
    ski_kpt3d: np.ndarray,
    pole_kpt3d: np.ndarray,
    ski_idx: List[int],
    pole_idx: List[int],
) -> np.ndarray:
    """Compose 8-point equipment GT layout from ski/pole arrays.

    Output order:
      [left_ski_tip, left_ski_tail, right_ski_tip, right_ski_tail,
       left_pole_grip, left_pole_tip, right_pole_grip, right_pole_tip]
    """
    ski = np.asarray(ski_kpt3d, dtype=np.float32)
    pole = np.asarray(pole_kpt3d, dtype=np.float32)
    if ski.ndim != 2 or ski.shape[1] != 3:
        raise ValueError(f"Expected ski GT shape [J,3], got {ski.shape}")
    if pole.ndim != 2 or pole.shape[1] != 3:
        raise ValueError(f"Expected pole GT shape [J,3], got {pole.shape}")
    if len(ski_idx) != 4:
        raise ValueError(f"Expected 4 ski_gt_idx values, got {ski_idx}")
    if len(pole_idx) != 4:
        raise ValueError(f"Expected 4 pole_gt_idx values, got {pole_idx}")

    if max(ski_idx) >= ski.shape[0]:
        raise ValueError(
            f"ski_gt_idx out of range for ski shape {ski.shape}: idx={ski_idx}"
        )
    if max(pole_idx) >= pole.shape[0]:
        raise ValueError(
            f"pole_gt_idx out of range for pole shape {pole.shape}: idx={pole_idx}"
        )

    ski_sel = ski[ski_idx]  # [4,3]
    pole_sel = pole[pole_idx]  # [4,3]
    return np.concatenate([ski_sel, pole_sel], axis=0).astype(np.float32)


def _set_equal_3d_axes(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).max() * 0.6)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _compute_equipment_lengths(equip_obj: np.ndarray) -> Dict[str, float]:
    """Compute 3D lengths of ski and pole segments.
    
    equip_obj: [8, 3] array with order:
      [left_ski_tip, left_ski_tail, right_ski_tip, right_ski_tail,
       left_pole_grip, left_pole_tip, right_pole_grip, right_pole_tip]
    
    Returns dict with keys: left_ski_len, right_ski_len, left_pole_len, right_pole_len
    """
    if equip_obj.shape != (8, 3):
        raise ValueError(f"Expected equip_obj shape [8,3], got {equip_obj.shape}")
    
    left_ski_len = float(np.linalg.norm(equip_obj[0] - equip_obj[1]))
    right_ski_len = float(np.linalg.norm(equip_obj[2] - equip_obj[3]))
    left_pole_len = float(np.linalg.norm(equip_obj[4] - equip_obj[5]))
    right_pole_len = float(np.linalg.norm(equip_obj[6] - equip_obj[7]))
    
    return {
        "left_ski_len": left_ski_len,
        "right_ski_len": right_ski_len,
        "left_pole_len": left_pole_len,
        "right_pole_len": right_pole_len,
        "avg_ski_len": (left_ski_len + right_ski_len) / 2.0,
        "avg_pole_len": (left_pole_len + right_pole_len) / 2.0,
    }


def _draw_masks_2d(
    ax,
    frame_rgb: np.ndarray,
    ski_mask: Optional[np.ndarray],
    pole_mask: Optional[np.ndarray],
) -> None:
    """Draw frame with overlaid ski and pole masks.
    
    ski_mask: [1,H,W] or [H,W], binary mask for ski
    pole_mask: [1,H,W] or [H,W], binary mask for ski+pole
    """
    frame_h, frame_w = frame_rgb.shape[:2]

    # Display frame as background
    ax.imshow(frame_rgb)
    
    # Normalize masks to [H,W] if needed
    if ski_mask is not None:
        ski_m = ski_mask.squeeze() if ski_mask.ndim > 2 else ski_mask
        if ski_m.shape[:2] != (frame_h, frame_w):
            ski_m = cv2.resize(
                ski_m.astype(np.float32), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
            )
        if ski_m.size > 0 and ski_m.max() > 0:
            # Create a colored mask for ski (blue)
            ski_colored = np.zeros((*ski_m.shape, 4), dtype=np.float32)
            ski_colored[..., 2] = ski_m  # Blue channel
            ski_colored[..., 3] = ski_m * 0.6  # Alpha = 60% where ski_mask > 0
            ax.imshow(ski_colored, alpha=1.0)
    
    if pole_mask is not None:
        pole_m = pole_mask.squeeze() if pole_mask.ndim > 2 else pole_mask
        if pole_m.shape[:2] != (frame_h, frame_w):
            pole_m = cv2.resize(
                pole_m.astype(np.float32), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
            )
        if pole_m.size > 0 and pole_m.max() > 0:
            # Create a colored mask for pole_only (pole - ski)
            if ski_mask is not None:
                ski_m = ski_mask.squeeze() if ski_mask.ndim > 2 else ski_mask
                if ski_m.shape[:2] != pole_m.shape[:2]:
                    ski_m = cv2.resize(
                        ski_m.astype(np.float32),
                        (pole_m.shape[1], pole_m.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                pole_only = np.maximum(0, pole_m - ski_m)
            else:
                pole_only = pole_m
            
            if pole_only.max() > 0:
                # Create a colored mask for pole (yellow/orange)
                pole_colored = np.zeros((*pole_only.shape, 4), dtype=np.float32)
                pole_colored[..., 0] = pole_only  # Red channel
                pole_colored[..., 1] = pole_only * 0.8  # Green channel (for yellow tint)
                pole_colored[..., 3] = pole_only * 0.6  # Alpha = 60% where pole_only > 0
                ax.imshow(pole_colored, alpha=1.0)
    
    ax.set_title("masks (blue=ski, orange=pole)")
    ax.axis("off")


def _render_one_figure(
    frame_rgb: np.ndarray,
    human_pred_3d: np.ndarray,
    human_gt_3d: Optional[np.ndarray],
    pred_obj: np.ndarray,
    gt_obj: Optional[np.ndarray],
    gt_segments: Optional[Dict[str, np.ndarray]],
    title: str,
    out_path: Path,
    ski_mask_np: Optional[np.ndarray] = None,
    pole_mask_np: Optional[np.ndarray] = None,
) -> None:
    fig = plt.figure(figsize=(28, 7))

    # Panel 1: frame
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(frame_rgb)
    ax1.set_title("frame")
    ax1.axis("off")

    gt_human = human_gt_3d if human_gt_3d is not None else human_pred_3d

    gt_xyz_for_scale = [gt_human]
    if gt_obj is not None:
        gt_xyz_for_scale.append(gt_obj)
    if gt_segments is not None:
        gt_xyz_for_scale.extend(list(gt_segments.values()))
    gt_xyz_ref = np.concatenate(gt_xyz_for_scale, axis=0)

    pred_xyz_ref = np.concatenate([human_pred_3d, pred_obj], axis=0)

    # Panel 2: GT
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    _draw_human_skeleton_3d(ax2, gt_human)
    if gt_segments is not None:
        _draw_gt_segments_by_config(ax2, gt_segments)
        ax2.legend(loc="upper right")
    elif gt_obj is not None:
        _draw_equipment_ski_pole(
            ax2,
            gt_obj,
            ski_color="tab:green",
            pole_color="tab:olive",
            label_prefix="gt",
            linestyle="-",
        )
        ax2.legend(loc="upper right")
    else:
        ax2.text2D(0.05, 0.95, "GT unavailable", transform=ax2.transAxes, color="tab:gray")
    _set_equal_3d_axes(ax2, gt_xyz_ref)
    ax2.set_title("gt")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Panel 3: Pred
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    _draw_human_skeleton_3d(ax3, human_pred_3d)
    _draw_equipment_ski_pole(
        ax3,
        pred_obj,
        ski_color="crimson",
        pole_color="goldenrod",
        label_prefix="pred",
        linestyle="-",
    )
    ax3.legend(loc="upper right")
    _set_equal_3d_axes(ax3, pred_xyz_ref)
    ax3.set_title("pred")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    # Panel 4: Masks (2D)
    ax4 = fig.add_subplot(1, 4, 4)
    _draw_masks_2d(ax4, frame_rgb, ski_mask_np, pole_mask_np)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_pose2equip_from_ckpt(
    ckpt_path: Path, cfg: Any, device: torch.device
) -> Pose2EquipNet:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model.") :]] = value

    model = Pose2EquipNet(
        num_joints=15,
        left_ankle_idx=int(cfg.pose2equip.left_ankle_idx),
        right_ankle_idx=int(cfg.pose2equip.right_ankle_idx),
        left_wrist_idx=int(cfg.pose2equip.left_wrist_idx),
        right_wrist_idx=int(cfg.pose2equip.right_wrist_idx),
    )
    model.load_state_dict(model_state, strict=True)
    model.eval()
    model.to(device)
    return model


def _load_fold_items(cfg: Any, fold: int, split: str) -> List[Dict[str, Any]]:
    fold_dir = Path(str(cfg.data.index_mapping_path))
    fold_file = fold_dir / f"fold_{fold:02d}.json"
    if not fold_file.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_file}")

    with open(fold_file, "r", encoding="utf-8") as f:
        fold_data = json.load(f)

    fold_data.pop("_metadata", None)
    items = fold_data.get(split, [])
    if not items and split == "test":
        items = fold_data.get("val", [])
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use best pose2equip checkpoint to infer equipment keypoints on true dataset and save visualizations.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/pose2equip/2026-05-01/fold_0/checkpoints/fold_0"
        ),
        help="Checkpoint directory containing *.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/configs/train.yaml"
        ),
        help="Hydra train.yaml path",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold id used to load true dataset index",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to run inference on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Max number of sample sequences to process",
    )
    parser.add_argument(
        "--max-frames-per-sample",
        type=int,
        default=10,
        help="Max frames per sample sequence",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Frame stride within one sample",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_true_data/pose2equip_unity_frame"
        ),
        help="Output root directory",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {args.ckpt_dir}")

    cfg = OmegaConf.load(str(args.config))
    image_size = int(cfg.data.img_size)

    ckpt_path = _select_best_ckpt(args.ckpt_dir)
    print(f"[INFO] Best ckpt: {ckpt_path}")

    model = _load_pose2equip_from_ckpt(ckpt_path, cfg, device)
    expected_joints = int(getattr(model.pose_encoder, "num_joints", len(ID_TO_INDEX)))

    split_items = _load_fold_items(cfg, int(args.fold), args.split)
    if len(split_items) == 0:
        raise ValueError(f"Empty split: {args.split}")

    run_out = args.out_dir / f"fold_{args.fold}" / Path(ckpt_path).stem
    run_out.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, Any]] = []

    max_samples = max(1, int(args.max_samples))
    max_frames = max(1, int(args.max_frames_per_sample))
    stride = max(1, int(args.frame_stride))

    print(
        f"[INFO] split={args.split}, total_samples={len(split_items)}, process_samples={min(len(split_items), max_samples)}"
    )

    for sample_idx, sample in enumerate(split_items[:max_samples]):
        sample_dict = asdict(sample) if not isinstance(sample, dict) else sample

        cam1_frames_dir = Path(str(sample_dict["cam1_frames_dir"]))
        sam3d_cam1_kpt3d_dir = Path(str(sample_dict["sam3d_cam1_kpt3d_dir"]))
        mask_ski_dir = sample_dict.get("sam3_cam1_mask_ski_dir")
        mask_ski_pole_dir = sample_dict.get("sam3_cam1_mask_ski_pole_dir")

        frame_map = _build_idx_file_map(cam1_frames_dir, ["*.png", "*.jpg", "*.jpeg"])
        sam3d_map = _build_idx_file_map(sam3d_cam1_kpt3d_dir, ["kpt3d_*.npy", "*.npy"])
        ski_mask_map = (
            _build_idx_file_map(Path(str(mask_ski_dir)), ["*.npz"])
            if mask_ski_dir is not None
            else {}
        )
        ski_pole_mask_map = (
            _build_idx_file_map(Path(str(mask_ski_pole_dir)), ["*.npz"])
            if mask_ski_pole_dir is not None
            else {}
        )

        common_indices = sorted(set(frame_map.keys()) & set(sam3d_map.keys()))
        if not common_indices:
            print(f"[WARN] no common frame indices for sample {sample_idx}, skip")
            continue

        picked_indices = common_indices[::stride][:max_frames]

        person_id = str(sample_dict.get("person_id", "unknown"))
        action_id = str(sample_dict.get("action_id", "unknown"))
        cam1_id = str(sample_dict.get("cam1_id", "unknown"))
        cam2_id = str(sample_dict.get("cam2_id", "unknown"))
        sample_tag = (
            f"sample_{sample_idx:03d}_{person_id}_{action_id}_{cam1_id}_{cam2_id}"
        )

        gt_ski_map: Dict[int, Path] = {}
        gt_pole_map: Dict[int, Path] = {}
        gt_character_map: Dict[int, Path] = {}
        gt_name_to_index: Optional[Dict[str, int]] = None
        kpt3d_dirs = sample_dict.get("kpt3d_dirs")
        joint_names_path = sample_dict.get("joint_names_path")
        if joint_names_path is not None:
            gt_name_to_index = _load_joint_name_to_index(Path(str(joint_names_path)))
        if isinstance(kpt3d_dirs, dict):
            ski_dir = kpt3d_dirs.get("ski")
            pole_dir = kpt3d_dirs.get("pole")
            character_dir = kpt3d_dirs.get("character")
            if ski_dir is not None and pole_dir is not None:
                gt_ski_map = _build_idx_file_map(
                    Path(str(ski_dir)), ["frame_*.npy", "*.npy"]
                )
                gt_pole_map = _build_idx_file_map(
                    Path(str(pole_dir)), ["frame_*.npy", "*.npy"]
                )
            if character_dir is not None:
                gt_character_map = _build_idx_file_map(
                    Path(str(character_dir)), ["frame_*.npy", "*.npy"]
                )

        ski_gt_idx = [
            int(x) for x in list(getattr(cfg.pose2equip, "ski_gt_idx", [1, 2, 4, 5]))
        ]
        pole_gt_idx = [
            int(x) for x in list(getattr(cfg.pose2equip, "pole_gt_idx", [0, 1, 2, 3]))
        ]
        left_ankle_idx = int(getattr(cfg.pose2equip, "left_ankle_idx", 10))
        right_ankle_idx = int(getattr(cfg.pose2equip, "right_ankle_idx", 11))
        left_wrist_idx = int(getattr(cfg.pose2equip, "left_wrist_idx", 12))
        right_wrist_idx = int(getattr(cfg.pose2equip, "right_wrist_idx", 13))

        sample_out = run_out / sample_tag
        vis_dir = sample_out / "vis"
        pred_dir = sample_out / "pred"
        vis_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        printed_offset_diagnosis = False

        for frame_idx in picked_indices:
            frame_path = frame_map[frame_idx]
            sam3d_path = sam3d_map[frame_idx]

            frame_rgb = _read_rgb(frame_path)
            human_3d_raw = np.asarray(np.load(sam3d_path), dtype=np.float32)
            human_pred_3d = _to_model_joint_count(human_3d_raw, expected_joints)

            human_gt_3d: Optional[np.ndarray] = None
            if frame_idx in gt_character_map:
                human_gt_raw = np.asarray(np.load(gt_character_map[frame_idx]), dtype=np.float32)
                try:
                    human_gt_3d = _to_model_joint_count_from_names(
                        human_gt_raw,
                        expected_joints=expected_joints,
                        name_to_index=gt_name_to_index,
                    )
                except Exception as e:
                    # Keep backward compatibility for datasets without valid joint_names mapping.
                    print(
                        f"[WARN] GT name-based joint mapping failed at frame {frame_idx}, fallback to index mapping: {e}"
                    )
                    human_gt_3d = _to_model_joint_count(human_gt_raw, expected_joints)

            human_3d_t = torch.from_numpy(human_pred_3d).unsqueeze(0).to(device)
            frame_t = _frame_to_model_tensor(
                frame_rgb, image_size=image_size, device=device
            )

            ski_mask_t: Optional[torch.Tensor] = None
            pole_mask_t: Optional[torch.Tensor] = None
            ski_mask_np: Optional[np.ndarray] = None
            pole_mask_np: Optional[np.ndarray] = None
            
            if frame_idx in ski_mask_map:
                ski_mask = _load_single_mask_npz(ski_mask_map[frame_idx])
                ski_mask_np = ski_mask.copy()  # Save numpy version for visualization
                ski_mask_t = _mask_to_model_tensor(ski_mask, image_size=image_size, device=device)
            if frame_idx in ski_pole_mask_map:
                ski_pole_mask = _load_single_mask_npz(ski_pole_mask_map[frame_idx])
                pole_mask_np = ski_pole_mask.copy()  # Same-view mask for visualization
                pole_mask_t = _mask_to_model_tensor(
                    ski_pole_mask, image_size=image_size, device=device
                )

            with torch.no_grad():
                out = model(
                    human_3d=human_3d_t,
                    human_frame=frame_t,
                    pole_mask=pole_mask_t,
                    ski_mask=ski_mask_t,
                )

            pred_obj = out["object_3d"][0].detach().cpu().numpy().astype(np.float32)
            directions = out["directions"][0].detach().cpu().numpy().astype(np.float32)
            lengths = out["lengths"][0].detach().cpu().numpy().astype(np.float32)

            # Compute 3D equipment lengths
            pred_lengths_dict = _compute_equipment_lengths(pred_obj)
            
            # Print pred length diagnostics
            print(
                f"[PRED] sample={sample_idx} frame={frame_idx} "
                f"ski_len=(L:{pred_lengths_dict['left_ski_len']:.4f}, R:{pred_lengths_dict['right_ski_len']:.4f}, avg:{pred_lengths_dict['avg_ski_len']:.4f}) "
                f"pole_len=(L:{pred_lengths_dict['left_pole_len']:.4f}, R:{pred_lengths_dict['right_pole_len']:.4f}, avg:{pred_lengths_dict['avg_pole_len']:.4f})"
            )

            gt_obj: Optional[np.ndarray] = None
            gt_segments: Optional[Dict[str, np.ndarray]] = None
            if frame_idx in gt_ski_map and frame_idx in gt_pole_map:
                ski_gt_raw = np.asarray(
                    np.load(gt_ski_map[frame_idx]), dtype=np.float32
                )
                pole_gt_raw = np.asarray(
                    np.load(gt_pole_map[frame_idx]), dtype=np.float32
                )
                try:
                    gt_obj = _compose_gt_equipment_points(
                        ski_kpt3d=ski_gt_raw,
                        pole_kpt3d=pole_gt_raw,
                        ski_idx=ski_gt_idx,
                        pole_idx=pole_gt_idx,
                    )
                    gt_segments = _build_gt_segments_by_config(
                        ski_kpt3d=ski_gt_raw,
                        pole_kpt3d=pole_gt_raw,
                        ski_idx=ski_gt_idx,
                        pole_idx=pole_gt_idx,
                    )
                    # Compute GT lengths
                    gt_lengths_dict = _compute_equipment_lengths(gt_obj)
                    print(
                        f"[GT] sample={sample_idx} frame={frame_idx} "
                        f"ski_len=(L:{gt_lengths_dict['left_ski_len']:.4f}, R:{gt_lengths_dict['right_ski_len']:.4f}, avg:{gt_lengths_dict['avg_ski_len']:.4f}) "
                        f"pole_len=(L:{gt_lengths_dict['left_pole_len']:.4f}, R:{gt_lengths_dict['right_pole_len']:.4f}, avg:{gt_lengths_dict['avg_pole_len']:.4f})"
                    )
                except Exception as e:
                    print(
                        f"[WARN] failed to compose GT equipment at frame {frame_idx}: {e}"
                    )

            if gt_obj is not None and not printed_offset_diagnosis:
                sam_metrics = _anchor_consistency_metrics(
                    human_3d=human_pred_3d,
                    equip_obj=gt_obj,
                    left_ankle_idx=left_ankle_idx,
                    right_ankle_idx=right_ankle_idx,
                    left_wrist_idx=left_wrist_idx,
                    right_wrist_idx=right_wrist_idx,
                )
                gt_metrics = (
                    _anchor_consistency_metrics(
                        human_3d=human_gt_3d,
                        equip_obj=gt_obj,
                        left_ankle_idx=left_ankle_idx,
                        right_ankle_idx=right_ankle_idx,
                        left_wrist_idx=left_wrist_idx,
                        right_wrist_idx=right_wrist_idx,
                    )
                    if human_gt_3d is not None
                    else {}
                )
                if sam_metrics:
                    print(f"[DIAG] sample={sample_idx} frame={frame_idx} anchor consistency with SAM human: {sam_metrics}")
                if gt_metrics:
                    print(f"[DIAG] sample={sample_idx} frame={frame_idx} anchor consistency with GT human: {gt_metrics}")
                printed_offset_diagnosis = True

            vis_path = vis_dir / f"frame_{frame_idx:06d}.png"
            title = f"fold={args.fold} sample={sample_idx} frame={frame_idx}"
            _render_one_figure(
                frame_rgb=frame_rgb,
                human_pred_3d=human_pred_3d,
                human_gt_3d=human_gt_3d,
                pred_obj=pred_obj,
                gt_obj=gt_obj,
                gt_segments=gt_segments,
                title=title,
                out_path=vis_path,
                ski_mask_np=ski_mask_np,
                pole_mask_np=pole_mask_np,
            )

            pred_payload = {
                "frame_index": int(frame_idx),
                "frame_path": str(frame_path),
                "sam3d_cam1_kpt3d_path": str(sam3d_path),
                "equipment_labels": EQUIP_LABELS,
                "pred_object_3d": pred_obj.tolist(),
                "pred_directions": directions.tolist(),
                "pred_lengths": lengths.tolist(),
                "pred_equipment_lengths": pred_lengths_dict,
                "gt_object_3d": gt_obj.tolist() if gt_obj is not None else None,
                "gt_equipment_lengths": (
                    _compute_equipment_lengths(gt_obj)
                    if gt_obj is not None
                    else None
                ),
            }
            pred_json_path = pred_dir / f"frame_{frame_idx:06d}.json"
            with open(pred_json_path, "w", encoding="utf-8") as f:
                json.dump(pred_payload, f, ensure_ascii=False, indent=2)

            summary_records.append(
                {
                    "sample_idx": sample_idx,
                    "person_id": person_id,
                    "action_id": action_id,
                    "cam1_id": cam1_id,
                    "cam2_id": cam2_id,
                    "frame_index": int(frame_idx),
                    "vis_path": str(vis_path),
                    "pred_json_path": str(pred_json_path),
                }
            )

    summary_path = run_out / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt_path": str(ckpt_path),
                "fold": int(args.fold),
                "split": args.split,
                "records": summary_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] Saved {len(summary_records)} frame predictions")
    print(f"[DONE] Output root: {run_out}")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
