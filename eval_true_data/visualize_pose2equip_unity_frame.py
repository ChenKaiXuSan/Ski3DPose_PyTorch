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

from project.dataloader.whole_video_dataset_dual_view import LabeledUnityDataset
from project.map_config import ID_TO_INDEX, SKELETON_CONNECTIONS
from project.models.pose2equip import Pose2EquipNet

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
    return x.to(device)


def _to_model_joint_count(human_3d: np.ndarray, expected_joints: int) -> np.ndarray:
    if human_3d.ndim != 2 or human_3d.shape[1] != 3:
        raise ValueError(f"Expected human_3d shape [J,3], got {human_3d.shape}")
    if human_3d.shape[0] == expected_joints:
        return human_3d.astype(np.float32)

    src_idx = LabeledUnityDataset._select_source_joint_indices(human_3d.shape[0])
    filtered = human_3d[src_idx]
    if filtered.shape[0] != expected_joints:
        raise ValueError(
            f"Joint count mismatch after filtering: expected {expected_joints}, got {filtered.shape[0]}"
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


def _draw_equipment_3d(ax, pred_obj: np.ndarray) -> None:
    ax.scatter(
        pred_obj[:, 0], pred_obj[:, 1], pred_obj[:, 2], s=18, c="tab:red", alpha=0.95
    )

    segments = [
        (0, 1),  # left ski
        (2, 3),  # right ski
        (4, 5),  # left pole
        (6, 7),  # right pole
    ]
    for i, j in segments:
        ax.plot(
            [pred_obj[i, 0], pred_obj[j, 0]],
            [pred_obj[i, 1], pred_obj[j, 1]],
            [pred_obj[i, 2], pred_obj[j, 2]],
            color="tab:red",
            linewidth=2.4,
        )


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


def _render_one_figure(
    frame_rgb: np.ndarray,
    human_3d: np.ndarray,
    pred_obj: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(frame_rgb)
    ax1.set_title("cam1 frame")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_human_skeleton_3d(ax2, human_3d)
    _draw_equipment_3d(ax2, pred_obj)
    _set_equal_3d_axes(ax2, np.concatenate([human_3d, pred_obj], axis=0))
    ax2.set_title("human pose + predicted equipment")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

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
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/pose2equip/2026-04-29/fold_0/checkpoints/fold_0"
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
    expected_joints = int(model.pose_encoder.net[0].in_features // 3)

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

        frame_map = _build_idx_file_map(cam1_frames_dir, ["*.png", "*.jpg", "*.jpeg"])
        sam3d_map = _build_idx_file_map(sam3d_cam1_kpt3d_dir, ["kpt3d_*.npy", "*.npy"])

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

        sample_out = run_out / sample_tag
        vis_dir = sample_out / "vis"
        pred_dir = sample_out / "pred"
        vis_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in picked_indices:
            frame_path = frame_map[frame_idx]
            sam3d_path = sam3d_map[frame_idx]

            frame_rgb = _read_rgb(frame_path)
            human_3d_raw = np.asarray(np.load(sam3d_path), dtype=np.float32)
            human_3d = _to_model_joint_count(human_3d_raw, expected_joints)

            human_3d_t = torch.from_numpy(human_3d).unsqueeze(0).to(device)
            frame_t = _frame_to_model_tensor(
                frame_rgb, image_size=image_size, device=device
            )

            with torch.no_grad():
                out = model(human_3d=human_3d_t, human_frame=frame_t)

            pred_obj = out["object_3d"][0].detach().cpu().numpy().astype(np.float32)
            directions = out["directions"][0].detach().cpu().numpy().astype(np.float32)
            lengths = out["lengths"][0].detach().cpu().numpy().astype(np.float32)

            vis_path = vis_dir / f"frame_{frame_idx:06d}.png"
            title = f"fold={args.fold} sample={sample_idx} frame={frame_idx}"
            _render_one_figure(
                frame_rgb=frame_rgb,
                human_3d=human_3d,
                pred_obj=pred_obj,
                title=title,
                out_path=vis_path,
            )

            pred_payload = {
                "frame_index": int(frame_idx),
                "frame_path": str(frame_path),
                "sam3d_cam1_kpt3d_path": str(sam3d_path),
                "equipment_labels": EQUIP_LABELS,
                "pred_object_3d": pred_obj.tolist(),
                "pred_directions": directions.tolist(),
                "pred_lengths": lengths.tolist(),
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
