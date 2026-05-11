#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/eval_true_data/visualize_pose2equip_unity_frame copy.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/eval_true_data
Created Date: Monday May 11th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday May 11th 2026 12:39:38 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

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
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.map_config import (
    FILTER_SKELETON_CONNECTIONS,
    filter_unity_kpts,
)
from project.models.pose2equip_net import STGCNBaselineNet
from project.dataloader.canonicalize import (
    canonicalize_pose_numpy,
    apply_canonical_transform_numpy,
)

EQUIP_LABELS = [
    "left_ski_tip",
    "left_ski_tail",
    "right_ski_tip",
    "right_ski_tail",
    "left_pole_grip",
    "right_pole_tip",
    "right_pole_grip",
    "left_pole_tip",
]

EQUIP_SEGMENTS = [
    (0, 1),  # left ski
    (2, 3),  # right ski
    (4, 7),  # left pole
    (6, 5),  # right pole
]


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


def _ensure_joint_count(
    human_3d: np.ndarray, expected_joints: int, source: str
) -> np.ndarray:
    """Validate filtered human joints shape against model expectation."""
    arr = np.asarray(human_3d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected {source} shape [J,3], got {arr.shape}")
    if arr.shape[0] != expected_joints:
        raise ValueError(
            f"{source} joint count mismatch: expected {expected_joints}, got {arr.shape[0]}"
        )
    return arr


def _draw_human_skeleton_3d(ax, human_3d: np.ndarray) -> None:
    ax.scatter(
        human_3d[:, 0], human_3d[:, 1], human_3d[:, 2], s=10, c="tab:blue", alpha=0.7
    )

    for i, j in FILTER_SKELETON_CONNECTIONS:
        if i < human_3d.shape[0] and j < human_3d.shape[0]:
            ax.plot(
                [human_3d[i, 0], human_3d[j, 0]],
                [human_3d[i, 1], human_3d[j, 1]],
                [human_3d[i, 2], human_3d[j, 2]],
                color="tab:blue",
                linewidth=1.2,
                alpha=0.8,
            )

    # Draw joint indices to make skeleton connectivity debugging easier.
    for idx, p in enumerate(human_3d):
        ax.text(
            float(p[0]),
            float(p[1]),
            float(p[2]),
            str(idx),
            color="navy",
            fontsize=7,
            alpha=0.9,
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

    # Draw equipment point indices and short labels.
    for idx, p in enumerate(equip_obj):
        label = EQUIP_LABELS[idx] if idx < len(EQUIP_LABELS) else f"pt{idx}"
        text_color = ski_color if idx < 4 else pole_color
        ax.text(
            float(p[0]),
            float(p[1]),
            float(p[2]),
            f"{idx}:{label}",
            color=text_color,
            fontsize=7,
            alpha=0.95,
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

    # Keep GT labels consistent with model output order:
    # 0 left_ski_tip, 1 left_ski_tail, 2 right_ski_tip, 3 right_ski_tail,
    # 4 left_pole_grip, 5 left_pole_tip, 6 right_pole_grip, 7 right_pole_tip.
    gt_points_for_labels = [
        sl[0],
        sl[1],
        sr[0],
        sr[1],
        pl[0],
        pl[1],
        pr[0],
        pr[1],
    ]
    for idx, p in enumerate(gt_points_for_labels):
        label = EQUIP_LABELS[idx] if idx < len(EQUIP_LABELS) else f"pt{idx}"
        text_color = ski_color if idx < 4 else pole_color
        ax.text(
            float(p[0]),
            float(p[1]),
            float(p[2]),
            f"{idx}:{label}",
            color=text_color,
            fontsize=7,
            alpha=0.95,
        )


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


def _compute_pred_gt_metrics(
    pred_obj: np.ndarray, gt_obj: np.ndarray
) -> Dict[str, float]:
    """Compute point and length errors between predicted and GT equipment."""
    pred = np.asarray(pred_obj, dtype=np.float32)
    gt = np.asarray(gt_obj, dtype=np.float32)
    if pred.shape != (8, 3) or gt.shape != (8, 3):
        raise ValueError(
            f"Expected pred/gt shapes [8,3], got pred={pred.shape}, gt={gt.shape}"
        )

    point_err = np.linalg.norm(pred - gt, axis=1)
    pred_len = _compute_equipment_lengths(pred)
    gt_len = _compute_equipment_lengths(gt)

    return {
        "mpjpe_equip": float(point_err.mean()),
        "mae_equip_x": float(np.abs(pred[:, 0] - gt[:, 0]).mean()),
        "mae_equip_y": float(np.abs(pred[:, 1] - gt[:, 1]).mean()),
        "mae_equip_z": float(np.abs(pred[:, 2] - gt[:, 2]).mean()),
        "mpjpe_ski": float(point_err[:4].mean()),
        "mpjpe_pole": float(point_err[4:].mean()),
        "left_ski_len_err": float(
            abs(pred_len["left_ski_len"] - gt_len["left_ski_len"])
        ),
        "right_ski_len_err": float(
            abs(pred_len["right_ski_len"] - gt_len["right_ski_len"])
        ),
        "left_pole_len_err": float(
            abs(pred_len["left_pole_len"] - gt_len["left_pole_len"])
        ),
        "right_pole_len_err": float(
            abs(pred_len["right_pole_len"] - gt_len["right_pole_len"])
        ),
    }


def _render_one_figure(
    frame_rgb: np.ndarray,
    human_pred_3d: np.ndarray,
    human_gt_3d: Optional[np.ndarray],
    pred_obj: np.ndarray,
    gt_obj: Optional[np.ndarray],
    gt_segments: Optional[Dict[str, np.ndarray]],
    title: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(21, 7))

    # Panel 1: frame
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(frame_rgb)
    ax1.set_title("frame")
    ax1.axis("off")

    # Use Unity body as the skeleton shown in the GT panel.
    gt_human = human_pred_3d

    gt_xyz_for_scale = [gt_human]
    if gt_obj is not None:
        gt_xyz_for_scale.append(gt_obj)
    if gt_segments is not None:
        gt_xyz_for_scale.extend(list(gt_segments.values()))
    gt_xyz_ref = np.concatenate(gt_xyz_for_scale, axis=0)

    pred_xyz_ref = np.concatenate([human_pred_3d, pred_obj], axis=0)

    # Panel 2: GT
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
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
        ax2.text2D(
            0.05, 0.95, "GT unavailable", transform=ax2.transAxes, color="tab:gray"
        )
    _set_equal_3d_axes(ax2, gt_xyz_ref)
    ax2.set_title("gt")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Panel 3: Pred
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
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

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    # Also save 3-view (front/side/top) figure for easier spatial inspection.
    view_specs: List[Tuple[str, float, float]] = [
        ("front", 12.0, -90.0),
        ("side", 12.0, 0.0),
        ("top", 90.0, -90.0),
    ]
    fig3 = plt.figure(figsize=(18, 10))

    for col, (view_name, elev, azim) in enumerate(view_specs, start=1):
        ax_gt = fig3.add_subplot(2, 3, col, projection="3d")
        _draw_human_skeleton_3d(ax_gt, gt_human)
        if gt_segments is not None:
            _draw_gt_segments_by_config(ax_gt, gt_segments)
        elif gt_obj is not None:
            _draw_equipment_ski_pole(
                ax_gt,
                gt_obj,
                ski_color="tab:green",
                pole_color="tab:olive",
                label_prefix="gt",
                linestyle="-",
            )
        _set_equal_3d_axes(ax_gt, gt_xyz_ref)
        ax_gt.view_init(elev=elev, azim=azim)
        ax_gt.set_title(f"gt-{view_name}")
        ax_gt.set_xlabel("X")
        ax_gt.set_ylabel("Y")
        ax_gt.set_zlabel("Z")

        ax_pred = fig3.add_subplot(2, 3, col + 3, projection="3d")
        _draw_human_skeleton_3d(ax_pred, human_pred_3d)
        _draw_equipment_ski_pole(
            ax_pred,
            pred_obj,
            ski_color="crimson",
            pole_color="goldenrod",
            label_prefix="pred",
            linestyle="-",
        )
        _set_equal_3d_axes(ax_pred, pred_xyz_ref)
        ax_pred.view_init(elev=elev, azim=azim)
        ax_pred.set_title(f"pred-{view_name}")
        ax_pred.set_xlabel("X")
        ax_pred.set_ylabel("Y")
        ax_pred.set_zlabel("Z")

    fig3.suptitle(f"{title} | 3 views")
    fig3.tight_layout()
    out_3views = out_path.with_name(f"{out_path.stem}_3views{out_path.suffix}")
    fig3.savefig(out_3views, dpi=180)
    plt.close(fig3)


def _load_stgcn_from_ckpt(
    ckpt_path: Path, cfg: Any, device: torch.device
) -> STGCNBaselineNet:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model.") :]] = value
        else:
            model_state[key] = value

    model = STGCNBaselineNet(
        num_joints=15,
        target_skeleton_connections_idx=FILTER_SKELETON_CONNECTIONS,
        hidden_dim=256,
        num_equip_kpts=8,
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
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/stgcn/2026-05-11/fold_0/checkpoints/fold_0"
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
        default=10,
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
        default=3,
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
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_true_data/stgcn/stgcn_unity_frame"
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
    ckpt_path = _select_best_ckpt(args.ckpt_dir)
    print(f"[INFO] Best ckpt: {ckpt_path}")

    model = _load_stgcn_from_ckpt(ckpt_path, cfg, device)
    expected_joints = int(getattr(model.pose_encoder, "num_joints", 15))

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
        unity_cam1_kpt3d_dir = Path(str(sample_dict.get("unity_cam1_kpt3d_dir", "")))

        frame_map = _build_idx_file_map(cam1_frames_dir, ["*.png", "*.jpg", "*.jpeg"])
        unity_input_map = _build_idx_file_map(
            unity_cam1_kpt3d_dir, ["frame_*.npy", "kpt3d_*.npy", "*.npy"]
        )

        person_id = str(sample_dict.get("person_id", "unknown"))
        action_id = str(sample_dict.get("action_id", "unknown"))
        cam1_id = str(sample_dict.get("cam1_id", "unknown"))
        cam2_id = str(sample_dict.get("cam2_id", "unknown"))
        gender = "female" if "female" in person_id.lower() else "male"
        sample_tag = (
            f"sample_{sample_idx:03d}_{person_id}_{action_id}_{cam1_id}_{cam2_id}"
        )

        gt_ski_map: Dict[int, Path] = {}
        gt_pole_map: Dict[int, Path] = {}
        gt_character_map: Dict[int, Path] = {}
        kpt3d_dirs = sample_dict.get("kpt3d_dirs")
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

        # Fallback: when unity_cam1_kpt3d_dir is unavailable, use character GT directory.
        if not unity_input_map and gt_character_map:
            unity_input_map = dict(gt_character_map)

        common_indices = sorted(set(frame_map.keys()) & set(unity_input_map.keys()))
        if not common_indices:
            print(
                f"[WARN] no common frame indices between frame and Unity 3D for sample {sample_idx}, skip"
            )
            continue

        picked_indices = common_indices[::stride][:max_frames]

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
        sample_txt_log_path = sample_out / "metrics_log.txt"
        sample_txt_fp = sample_txt_log_path.open("w", encoding="utf-8")

        def _sample_log(msg: str) -> None:
            print(msg)
            sample_txt_fp.write(msg + "\n")
            sample_txt_fp.flush()

        _sample_log(
            f"[INFO] sample={sample_idx} tag={sample_tag} selected_frames={len(picked_indices)}"
        )
        printed_offset_diagnosis = False

        for frame_idx in picked_indices:
            frame_path = frame_map[frame_idx]
            unity_human_path = unity_input_map[frame_idx]

            frame_rgb = _read_rgb(frame_path)
            human_3d_raw = np.asarray(np.load(unity_human_path), dtype=np.float32)
            human_pred_3d_filtered = _ensure_joint_count(
                filter_unity_kpts(human_3d_raw, flag="3d", gender=gender),
                expected_joints,
                source="Unity human input",
            )
            # Apply canonicalize transform to human pose (using hip indices 6,7 and neck index 14)
            human_pred_3d, transform_info = canonicalize_pose_numpy(
                human_pred_3d_filtered,
                left_hip=6,
                right_hip=7,
                neck=14,
                mode="first_frame",
            )

            human_gt_3d: Optional[np.ndarray] = None
            if frame_idx in gt_character_map:
                human_gt_raw = np.asarray(
                    np.load(gt_character_map[frame_idx]), dtype=np.float32
                )
                # human_gt_raw = _rotate_x_180_deg(human_gt_raw)
                human_gt_3d_filtered = _ensure_joint_count(
                    filter_unity_kpts(human_gt_raw, flag="3d", gender=gender),
                    expected_joints,
                    source="Unity GT human",
                )
                # Apply same canonicalize transform from human_pred_3d
                human_gt_3d = apply_canonical_transform_numpy(
                    human_gt_3d_filtered,
                    transform_info["pelvis"],
                    transform_info["R"],
                )

            human_3d_t = torch.from_numpy(human_pred_3d).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(human_3d=human_3d_t)

            pred_obj_raw = out["object_3d"][0].detach().cpu().numpy().astype(np.float32)
            # Apply same canonicalize transform to predicted equipment points
            pred_obj = apply_canonical_transform_numpy(
                pred_obj_raw,
                transform_info["pelvis"],
                transform_info["R"],
            )

            # Compute 3D equipment lengths
            pred_lengths_dict = _compute_equipment_lengths(pred_obj)

            # Print pred length diagnostics
            _sample_log(
                f"[PRED] sample={sample_idx} frame={frame_idx} "
                f"ski_len=(L:{pred_lengths_dict['left_ski_len']:.4f}, R:{pred_lengths_dict['right_ski_len']:.4f}, avg:{pred_lengths_dict['avg_ski_len']:.4f}) "
                f"pole_len=(L:{pred_lengths_dict['left_pole_len']:.4f}, R:{pred_lengths_dict['right_pole_len']:.4f}, avg:{pred_lengths_dict['avg_pole_len']:.4f})"
            )

            gt_obj: Optional[np.ndarray] = None
            gt_segments: Optional[Dict[str, np.ndarray]] = None
            pred_gt_metrics: Optional[Dict[str, float]] = None
            if frame_idx in gt_ski_map and frame_idx in gt_pole_map:
                ski_gt_raw = np.asarray(
                    np.load(gt_ski_map[frame_idx]), dtype=np.float32
                )
                pole_gt_raw = np.asarray(
                    np.load(gt_pole_map[frame_idx]), dtype=np.float32
                )

                try:
                    gt_obj_raw = _compose_gt_equipment_points(
                        ski_kpt3d=ski_gt_raw,
                        pole_kpt3d=pole_gt_raw,
                        ski_idx=ski_gt_idx,
                        pole_idx=pole_gt_idx,
                    )
                    # Apply same canonicalize transform to GT equipment points
                    gt_obj = apply_canonical_transform_numpy(
                        gt_obj_raw,
                        transform_info["pelvis"],
                        transform_info["R"],
                    )
                    gt_segments = _build_gt_segments_by_config(
                        ski_kpt3d=ski_gt_raw,
                        pole_kpt3d=pole_gt_raw,
                        ski_idx=ski_gt_idx,
                        pole_idx=pole_gt_idx,
                    )
                    # Apply canonicalize transform to gt_segments points
                    gt_segments = {
                        key: apply_canonical_transform_numpy(
                            pts,
                            transform_info["pelvis"],
                            transform_info["R"],
                        )
                        for key, pts in gt_segments.items()
                    }
                    # Compute GT lengths
                    gt_lengths_dict = _compute_equipment_lengths(gt_obj)
                    _sample_log(
                        f"[GT] sample={sample_idx} frame={frame_idx} "
                        f"ski_len=(L:{gt_lengths_dict['left_ski_len']:.4f}, R:{gt_lengths_dict['right_ski_len']:.4f}, avg:{gt_lengths_dict['avg_ski_len']:.4f}) "
                        f"pole_len=(L:{gt_lengths_dict['left_pole_len']:.4f}, R:{gt_lengths_dict['right_pole_len']:.4f}, avg:{gt_lengths_dict['avg_pole_len']:.4f})"
                    )

                    pred_gt_metrics = _compute_pred_gt_metrics(
                        pred_obj=pred_obj,
                        gt_obj=gt_obj,
                    )
                    _sample_log(
                        f"[METRIC] sample={sample_idx} frame={frame_idx} "
                        f"mpjpe_equip={pred_gt_metrics['mpjpe_equip']:.4f} "
                        f"(ski={pred_gt_metrics['mpjpe_ski']:.4f}, pole={pred_gt_metrics['mpjpe_pole']:.4f}) "
                        f"mae_xyz=({pred_gt_metrics['mae_equip_x']:.4f}, {pred_gt_metrics['mae_equip_y']:.4f}, {pred_gt_metrics['mae_equip_z']:.4f}) "
                        f"len_err=(skiL:{pred_gt_metrics['left_ski_len_err']:.4f}, skiR:{pred_gt_metrics['right_ski_len_err']:.4f}, "
                        f"poleL:{pred_gt_metrics['left_pole_len_err']:.4f}, poleR:{pred_gt_metrics['right_pole_len_err']:.4f})"
                    )
                except Exception as e:
                    _sample_log(
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
                    _sample_log(
                        f"[DIAG] sample={sample_idx} frame={frame_idx} anchor consistency with Unity human input: {sam_metrics}"
                    )
                if gt_metrics:
                    _sample_log(
                        f"[DIAG] sample={sample_idx} frame={frame_idx} anchor consistency with GT human: {gt_metrics}"
                    )
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
            )

            pred_payload = {
                "frame_index": int(frame_idx),
                "frame_path": str(frame_path),
                "unity_cam1_kpt3d_path": str(unity_human_path),
                "equipment_labels": EQUIP_LABELS,
                "pred_object_3d": pred_obj.tolist(),
                "pred_equipment_lengths": pred_lengths_dict,
                "gt_object_3d": gt_obj.tolist() if gt_obj is not None else None,
                "gt_equipment_lengths": (
                    _compute_equipment_lengths(gt_obj) if gt_obj is not None else None
                ),
                "pred_gt_metrics": pred_gt_metrics,
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

        _sample_log(f"[DONE] sample={sample_idx} metrics_txt={sample_txt_log_path}")
        sample_txt_fp.close()

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
    print(f"[DONE] Metrics txt saved per sample under sample directories")


if __name__ == "__main__":
    main()
