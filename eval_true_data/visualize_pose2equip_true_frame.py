#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.map_config import (
    FILTERED_KPTS_MAPPING,
    FILTER_SKELETON_CONNECTIONS,
    filter_sam3d_body_kpts,
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


def _extract_frame_idx(path: Path) -> int:
    m = re.search(r"frame_(\d+)", path.name)
    if m is None:
        raise ValueError(f"Cannot parse frame index from: {path.name}")
    return int(m.group(1))


def _load_record(npz_path: Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    if "outputs" not in data.files:
        raise KeyError(f"Missing 'outputs' in npz: {npz_path}")
    outs = data["outputs"]
    if len(outs) == 0:
        raise ValueError(f"Empty 'outputs' in npz: {npz_path}")
    rec = outs[0]
    if isinstance(rec, np.ndarray) and rec.shape == ():
        rec = rec.item()
    if not isinstance(rec, dict):
        raise TypeError(f"Unexpected output type in {npz_path}: {type(rec)}")
    return rec


def _ensure_joint_count(kpt: np.ndarray, expected_joints: int, source: str) -> np.ndarray:
    arr = np.asarray(kpt, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"Expected {source} shape [J,2/3], got {arr.shape}")
    if arr.shape[0] != expected_joints:
        raise ValueError(
            f"{source} joint count mismatch: expected {expected_joints}, got {arr.shape[0]}"
        )
    return arr


def _build_edges() -> List[Tuple[int, int]]:
    return list(FILTER_SKELETON_CONNECTIONS)


def _build_labels() -> List[str]:
    return [f"{i}:{name}" for i, name in FILTERED_KPTS_MAPPING.items()]


def _extract_float_token(token: str) -> Optional[float]:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _parse_ckpt_name(
    ckpt_name: str,
) -> Optional[Tuple[int, Optional[float], Optional[float]]]:
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
        parsed.append((p, epoch, float(metric)))

    if not parsed:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    parsed.sort(key=lambda x: (x[2], -x[1]))
    return parsed[0][0]


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


def _set_equal_axes_3d(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).max() * 0.55)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_2d(
    ax,
    frame: np.ndarray,
    kpt2d: np.ndarray,
    edges: List[Tuple[int, int]],
    labels: List[str],
    show_labels: bool,
) -> None:
    ax.imshow(frame)
    ax.set_title("Frame + SAM 2D")
    ax.axis("off")

    ax.scatter(kpt2d[:, 0], kpt2d[:, 1], s=14, c="yellow")
    for i, j in edges:
        ax.plot(
            [kpt2d[i, 0], kpt2d[j, 0]],
            [kpt2d[i, 1], kpt2d[j, 1]],
            color="cyan",
            linewidth=1.5,
        )

    if show_labels:
        for i in range(min(len(kpt2d), len(labels))):
            ax.text(
                float(kpt2d[i, 0]) + 2,
                float(kpt2d[i, 1]) - 2,
                labels[i],
                fontsize=7,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 0.5},
            )


def _draw_3d(
    ax,
    kpt3d: np.ndarray,
    edges: List[Tuple[int, int]],
    labels: List[str],
    show_labels: bool,
) -> None:
    ax.set_title("SAM 3D")
    ax.scatter(kpt3d[:, 0], kpt3d[:, 1], kpt3d[:, 2], s=16, c="tab:red", alpha=0.9)

    for i, j in edges:
        ax.plot(
            [kpt3d[i, 0], kpt3d[j, 0]],
            [kpt3d[i, 1], kpt3d[j, 1]],
            [kpt3d[i, 2], kpt3d[j, 2]],
            color="tab:red",
            linewidth=2.0,
        )

    if show_labels:
        for i in range(min(len(kpt3d), len(labels))):
            ax.text(
                float(kpt3d[i, 0]),
                float(kpt3d[i, 1]),
                float(kpt3d[i, 2]),
                labels[i],
                fontsize=8,
            )

    _set_equal_axes_3d(ax, kpt3d)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _draw_equipment_3d(ax, pred_obj: np.ndarray) -> None:
    ax.scatter(
        pred_obj[:, 0], pred_obj[:, 1], pred_obj[:, 2], s=24, c="tab:green", alpha=0.95
    )

    segments = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for i, j in segments:
        ax.plot(
            [pred_obj[i, 0], pred_obj[j, 0]],
            [pred_obj[i, 1], pred_obj[j, 1]],
            [pred_obj[i, 2], pred_obj[j, 2]],
            color="tab:green",
            linewidth=2.6,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize frame and SAM 3D keypoints from /workspace/data/sam3d_body_results/person",
    )
    parser.add_argument(
        "--sam-root",
        type=Path,
        default=Path("/workspace/data/sam3d_body_results/person"),
        help="Root containing <run_id>/<left|right>/frame_xxxxx_sam_3d_body_outputs.npz",
    )
    parser.add_argument(
        "--run-id", type=str, default="pro_2", help="Run/person id under sam-root"
    )
    parser.add_argument(
        "--side",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Camera side",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Specific frame index. Default: middle one",
    )
    parser.add_argument(
        "--max-frames", type=int, default=20, help="How many frames to visualize"
    )
    parser.add_argument(
        "--stride", type=int, default=30, help="Frame stride when max-frames > 1"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_true_data/pose2equip/pose2equip_true_frame"
        ),
        help="Output directory",
    )
    parser.add_argument(
        "--show-joint-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to draw joint labels",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/pose2equip/2026-05-02/fold_0/checkpoints/fold_0"
        ),
        help="Pose2Equip checkpoint directory (or its parent).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/configs/train.yaml"
        ),
        help="Config file used for pose2equip joint index settings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device: cuda/cpu",
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
    model = _load_pose2equip_from_ckpt(ckpt_path, cfg, device)
    expected_joints = int(getattr(model.pose_encoder, "num_joints", 15))

    print(f"[INFO] Best ckpt: {ckpt_path}")

    npz_dir = args.sam_root / args.run_id / args.side
    if not npz_dir.exists():
        raise FileNotFoundError(f"NPZ directory not found: {npz_dir}")

    npz_files = sorted(npz_dir.glob("frame_*_sam_3d_body_outputs.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No frame_*_sam_3d_body_outputs.npz under: {npz_dir}")

    frame_to_npz = {_extract_frame_idx(p): p for p in npz_files}
    all_indices = sorted(frame_to_npz.keys())

    if args.frame_index is not None:
        if args.frame_index not in frame_to_npz:
            raise KeyError(f"frame_index {args.frame_index} not found in {npz_dir}")
        picked = [args.frame_index]
    else:
        mid = len(all_indices) // 2
        start = all_indices[mid]
        picked = all_indices[all_indices.index(start) :: max(1, int(args.stride))][
            : max(1, int(args.max_frames))
        ]

    out_dir = args.out_dir / args.run_id / args.side / Path(ckpt_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = _build_edges()
    labels = _build_labels()
    records: List[Dict[str, Any]] = []

    for idx in picked:
        npz_path = frame_to_npz[idx]
        rec = _load_record(npz_path)

        if "frame" not in rec:
            raise KeyError(f"Missing 'frame' in record: {npz_path}")
        frame = np.asarray(rec["frame"])

        if "pred_keypoints_3d" in rec:
            kpt3d = np.asarray(rec["pred_keypoints_3d"], dtype=np.float32)
        elif "pred_joint_coords" in rec:
            kpt3d = np.asarray(rec["pred_joint_coords"], dtype=np.float32)
        else:
            raise KeyError(f"Missing 3D keypoint fields in record: {npz_path}")

        if "pred_keypoints_2d" in rec:
            kpt2d = np.asarray(rec["pred_keypoints_2d"], dtype=np.float32)
        else:
            kpt2d = kpt3d[:, :2].astype(np.float32)

        kpt2d_t = _ensure_joint_count(
            filter_sam3d_body_kpts(kpt2d),
            expected_joints,
            source="SAM 2D keypoints",
        )
        kpt3d_t = _ensure_joint_count(
            filter_sam3d_body_kpts(kpt3d),
            expected_joints,
            source="SAM 3D keypoints",
        )

        if kpt3d_t.shape[0] != expected_joints:
            raise ValueError(
                f"Model expects {expected_joints} joints, but got {kpt3d_t.shape[0]} from SAM output"
            )

        human_3d_t = (
            torch.from_numpy(kpt3d_t.astype(np.float32)).unsqueeze(0).to(device)
        )
        frame_t = _frame_to_model_tensor(
            frame.astype(np.float32), image_size=image_size, device=device
        )
        with torch.no_grad():
            out = model(human_3d=human_3d_t, human_frame=frame_t)

        pred_obj = out["object_3d"][0].detach().cpu().numpy().astype(np.float32)
        pred_dirs = out["directions"][0].detach().cpu().numpy().astype(np.float32)
        pred_lens = out["lengths"][0].detach().cpu().numpy().astype(np.float32)

        fig = plt.figure(figsize=(22, 7))
        ax1 = fig.add_subplot(1, 3, 1)
        _draw_2d(ax1, frame, kpt2d_t, edges, labels, args.show_joint_labels)

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        _draw_3d(ax2, kpt3d_t, edges, labels, args.show_joint_labels)

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        _draw_3d(ax3, kpt3d_t, edges, labels, False)
        _draw_equipment_3d(ax3, pred_obj)
        ax3.set_title("SAM human 3D + Pose2Equip")
        _set_equal_axes_3d(ax3, np.concatenate([kpt3d_t, pred_obj], axis=0))

        fig.suptitle(
            f"run={args.run_id} side={args.side} frame={idx} | ckpt={ckpt_path.name}"
        )
        fig.tight_layout()

        png_path = out_dir / f"frame_{idx:06d}_vis.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

        json_path = out_dir / f"frame_{idx:06d}_kpt3d.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": args.run_id,
                    "side": args.side,
                    "frame_index": int(idx),
                    "npz_path": str(npz_path),
                    "target_joint_names": [
                        FILTERED_KPTS_MAPPING[i] for i in range(len(FILTERED_KPTS_MAPPING))
                    ],
                    "target_kpt3d": kpt3d_t.tolist(),
                    "pose2equip_ckpt": str(ckpt_path),
                    "equipment_labels": EQUIP_LABELS,
                    "pred_object_3d": pred_obj.tolist(),
                    "pred_directions": pred_dirs.tolist(),
                    "pred_lengths": pred_lens.tolist(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        records.append(
            {
                "frame_index": int(idx),
                "vis": str(png_path),
                "kpt3d_json": str(json_path),
            }
        )

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": args.run_id,
                "side": args.side,
                "pose2equip_ckpt": str(ckpt_path),
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] Saved {len(records)} frame visualizations")
    print(f"[DONE] Output: {out_dir}")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
