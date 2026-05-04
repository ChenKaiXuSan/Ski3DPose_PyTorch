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

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.map_config import (
    ID_TO_INDEX,
    SAM3D_BODY_SKELETON_CONNECTIONS,
    TARGET_IDS,
    SAM3D_BODY_MAPPING,
)
from project.models.dual2pose_net import Dual2PoseNet


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
    img = plt.imread(path)
    if not isinstance(img, np.ndarray):
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    if img.dtype.kind == "f":
        scale = 255.0 if img.max() <= 1.5 else 1.0
        img = np.clip(img * scale, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


def _frame_to_video_tensor(frame_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frame_rgb, dtype=np.float32)).permute(2, 0, 1)
    if x.max() <= 1.5:
        x = x * 255.0
    x = x.unsqueeze(0).unsqueeze(2)
    return x.to(device)


def _to_model_joint_count(
    kpt: np.ndarray,
    expected_joints: int,
    source_indices: Optional[List[int]] = None,
) -> np.ndarray:
    if kpt.ndim != 2 or kpt.shape[1] not in (2, 3):
        raise ValueError(f"Expected keypoint shape [J,2/3], got {kpt.shape}")

    if source_indices is not None:
        if len(source_indices) != int(expected_joints):
            raise ValueError(
                f"source_indices len mismatch: expected {expected_joints}, got {len(source_indices)}"
            )
        if max(source_indices) >= int(kpt.shape[0]):
            raise ValueError(
                "source_indices out of bounds for source keypoints: "
                f"max_index={max(source_indices)}, source_joints={kpt.shape[0]}"
            )
        return kpt[source_indices].astype(np.float32)

    # Case 1: already in target compact layout [15,2/3].
    if int(kpt.shape[0]) == int(expected_joints):
        return kpt.astype(np.float32)

    # Case 2: source is full Unity-MHR70-like layout, use target joint IDs as indices.
    max_target_id = max(TARGET_IDS)
    if int(kpt.shape[0]) > int(max_target_id):
        return kpt[TARGET_IDS].astype(np.float32)

    raise ValueError(
        "Unsupported keypoint layout for remapping: "
        f"source_joints={kpt.shape[0]}, expected_joints={expected_joints}, "
        f"required_source_joints>{max_target_id} for id-based remap"
    )


def _load_target_source_indices_by_joint_names(
    joint_names_path: Optional[str],
) -> Optional[List[int]]:
    if not joint_names_path:
        return None

    path = Path(str(joint_names_path))
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    names_obj = data.get("joint_names") if isinstance(data, dict) else None
    if not isinstance(names_obj, list) or len(names_obj) == 0:
        return None

    names: List[str] = [str(x) for x in names_obj]
    exact = {name: idx for idx, name in enumerate(names)}
    lower = {name.lower(): idx for idx, name in enumerate(names)}

    src_idx: List[int] = []
    for jid in TARGET_IDS:
        target_name = str(SAM3D_BODY_MAPPING.get(jid, ""))
        idx = exact.get(target_name)
        if idx is None:
            idx = lower.get(target_name.lower())
        if idx is None:
            return None
        src_idx.append(int(idx))

    return src_idx


def _build_edges() -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for a, b in SAM3D_BODY_SKELETON_CONNECTIONS:
        if a in ID_TO_INDEX and b in ID_TO_INDEX:
            edges.append((ID_TO_INDEX[a], ID_TO_INDEX[b]))
    return edges


def _build_labels() -> List[str]:
    labels: List[str] = []
    for i, jid in enumerate(TARGET_IDS):
        labels.append(f"{i}:{SAM3D_BODY_MAPPING.get(jid, str(jid))}")
    return labels


def _load_dual2pose_from_ckpt(
    ckpt_path: Path, cfg: Any, device: torch.device
) -> Dual2PoseNet:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("models.character."):
            model_state[key[len("models.character.") :]] = value

    if not model_state:
        raise KeyError("No 'models.character.*' weights found in checkpoint")

    out_proj_w = model_state["refiner.out_proj.weight"]
    num_joints = int(out_proj_w.shape[0] // 3)

    block_ids = set()
    block_pat = re.compile(r"^refiner\.blocks\.(\d+)\.")
    for key in model_state.keys():
        m = block_pat.match(key)
        if m is not None:
            block_ids.add(int(m.group(1)))
    n_layers = (max(block_ids) + 1) if block_ids else int(getattr(cfg.dual2pose, "n_layers", 4))

    gate_in = int(model_state["gating.mlp.0.weight"].shape[1])
    use_conf = gate_in > 9
    gate_out = int(model_state["gating.mlp.4.weight"].shape[0])
    predict_logvar = gate_out == 2

    use_dino_features = any(k.startswith("dino_") for k in model_state.keys())

    model = Dual2PoseNet(
        num_joints=num_joints,
        d_model=int(getattr(cfg.dual2pose, "d_model", 256)),
        n_layers=n_layers,
        use_conf=use_conf,
        predict_logvar=predict_logvar,
        bone_edges=_build_edges(),
        use_dino_features=use_dino_features,
        dino_model_name=str(
            getattr(
                cfg.dual2pose,
                "dino_model_name",
                "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            )
        ),
        dino_freeze=bool(getattr(cfg.dual2pose, "dino_freeze", True)),
        dino_image_size=int(getattr(cfg.dual2pose, "dino_image_size", 224)),
        dino_feature_dim=int(getattr(cfg.dual2pose, "dino_feature_dim", 768)),
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


def _filter_edges_by_length(
    kpts: np.ndarray, edges: List[Tuple[int, int]], max_ratio: float
) -> List[Tuple[int, int]]:
    if kpts.ndim != 2 or kpts.shape[0] == 0:
        return []

    lengths: List[float] = []
    valid_edges: List[Tuple[int, int]] = []
    for i, j in edges:
        if i >= kpts.shape[0] or j >= kpts.shape[0]:
            continue
        a = kpts[i]
        b = kpts[j]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            continue
        dist = float(np.linalg.norm(a - b))
        if dist <= 1e-8:
            continue
        lengths.append(dist)
        valid_edges.append((i, j))

    if not lengths:
        return []

    median_len = float(np.median(np.asarray(lengths, dtype=np.float32)))
    if median_len <= 1e-8:
        return valid_edges

    upper = median_len * max_ratio
    return [e for e, d in zip(valid_edges, lengths) if d <= upper]


def _draw_3d(ax, xyz: np.ndarray, edges: List[Tuple[int, int]], title: str, color: str) -> None:
    ax.set_title(title)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=16, c=color, alpha=0.9)
    draw_edges = _filter_edges_by_length(xyz, edges, max_ratio=2.5)
    for i, j in draw_edges:
        ax.plot(
            [xyz[i, 0], xyz[j, 0]],
            [xyz[i, 1], xyz[j, 1]],
            [xyz[i, 2], xyz[j, 2]],
            color=color,
            linewidth=2.0,
        )
    _set_equal_axes_3d(ax, xyz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _draw_frame_with_gt_2d(
    ax,
    frame: np.ndarray,
    gt_2d: Optional[np.ndarray],
    edges: List[Tuple[int, int]],
    title: str,
) -> None:
    ax.imshow(frame)
    ax.set_title(title)
    ax.axis("off")

    if gt_2d is None or gt_2d.ndim != 2 or gt_2d.shape[0] == 0:
        return

    ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=14, c="yellow", alpha=0.95)
    draw_edges = _filter_edges_by_length(gt_2d, edges, max_ratio=3.0)
    for i, j in draw_edges:
        ax.plot(
            [gt_2d[i, 0], gt_2d[j, 0]],
            [gt_2d[i, 1], gt_2d[j, 1]],
            color="cyan",
            linewidth=1.4,
        )


def _draw_3d_compare(
    ax,
    left_3d: np.ndarray,
    right_3d: np.ndarray,
    fused_3d: np.ndarray,
    gt_3d: Optional[np.ndarray],
    edges: List[Tuple[int, int]],
) -> None:
    ax.set_title("Dual2Pose fused")
    ax.scatter(left_3d[:, 0], left_3d[:, 1], left_3d[:, 2], s=10, c="tab:blue", alpha=0.3)
    ax.scatter(
        right_3d[:, 0], right_3d[:, 1], right_3d[:, 2], s=10, c="tab:orange", alpha=0.3
    )
    ax.scatter(
        fused_3d[:, 0], fused_3d[:, 1], fused_3d[:, 2], s=16, c="tab:red", alpha=0.9
    )
    if gt_3d is not None:
        ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], s=10, c="tab:green", alpha=0.45)

    draw_edges = _filter_edges_by_length(fused_3d, edges, max_ratio=2.5)
    for i, j in draw_edges:
        ax.plot(
            [fused_3d[i, 0], fused_3d[j, 0]],
            [fused_3d[i, 1], fused_3d[j, 1]],
            [fused_3d[i, 2], fused_3d[j, 2]],
            color="tab:red",
            linewidth=2.0,
        )

    all_xyz = [left_3d, right_3d, fused_3d]
    if gt_3d is not None:
        all_xyz.append(gt_3d)
    _set_equal_axes_3d(ax, np.concatenate(all_xyz, axis=0))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _build_alpha_diagnostics(
    p_hat: np.ndarray,
    p0: np.ndarray,
    p_left: np.ndarray,
    p_right: np.ndarray,
    alpha: np.ndarray,
) -> Dict[str, float]:
    return {
        "alpha_min": float(np.min(alpha)),
        "alpha_max": float(np.max(alpha)),
        "alpha_mean": float(np.mean(alpha)),
        "alpha_std": float(np.std(alpha)),
        "l2_hat_to_left": float(np.linalg.norm(p_hat - p_left, axis=1).mean()),
        "l2_hat_to_right": float(np.linalg.norm(p_hat - p_right, axis=1).mean()),
        "l2_p0_to_left": float(np.linalg.norm(p0 - p_left, axis=1).mean()),
        "l2_p0_to_right": float(np.linalg.norm(p0 - p_right, axis=1).mean()),
    }


def _rigid_align_right_to_left_single(
    left: np.ndarray,
    right: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Align one right-view pose to left-view pose using Kabsch rigid transform."""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: left={left.shape}, right={right.shape}")
    if left.ndim != 2 or left.shape[1] != 3:
        raise ValueError(f"Expected [J,3], got {left.shape}")

    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    n_valid = int(valid.sum())
    if n_valid < 3:
        return right.copy(), {
            "applied": 0.0,
            "valid_points": float(n_valid),
            "rmse_before": float("nan"),
            "rmse_after": float("nan"),
        }

    x = right[valid]
    y = left[valid]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x0 = x - x_mean
    y0 = y - y_mean

    h = x0.T @ y0
    u, _, vh = np.linalg.svd(h, full_matrices=False)
    r = vh.T @ u.T
    if np.linalg.det(r) < 0:
        vh = vh.copy()
        vh[-1, :] *= -1.0
        r = vh.T @ u.T

    t = y_mean - x_mean @ r.T
    right_aligned = right @ r.T + t

    diff_before = right[valid] - left[valid]
    diff_after = right_aligned[valid] - left[valid]
    rmse_before = float(np.sqrt(np.sum(diff_before * diff_before, axis=1).mean()))
    rmse_after = float(np.sqrt(np.sum(diff_after * diff_after, axis=1).mean()))

    return right_aligned.astype(np.float32), {
        "applied": 1.0,
        "valid_points": float(n_valid),
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
    }


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
        description="Use best dual2pose checkpoint to fuse dual-view SAM 3D poses on unity frames"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/dual2pose/2026-05-03/fold_0/checkpoints/fold_0"
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
    parser.add_argument("--fold", type=int, default=0, help="Fold id to load index")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to run inference on",
    )
    parser.add_argument("--max-samples", type=int, default=5, help="Max number of sample sequences")
    parser.add_argument(
        "--max-frames-per-sample", type=int, default=10, help="Max frames per sample sequence"
    )
    parser.add_argument("--frame-stride", type=int, default=5, help="Frame stride within one sample")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument(
        "--rigid-align-right-to-left",
        action="store_true",
        help="Apply rigid alignment to right-view pose before fusion",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_true_data/dual2pose/dual2pose_unity_frame"
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

    rigid_align_right_to_left = bool(
        args.rigid_align_right_to_left
        or bool(getattr(cfg.dual2pose, "rigid_align_right_to_left", False))
    )

    ckpt_path = _select_best_ckpt(args.ckpt_dir)
    print(f"[INFO] Best ckpt: {ckpt_path}")
    print(f"[INFO] rigid_align_right_to_left={rigid_align_right_to_left}")

    model = _load_dual2pose_from_ckpt(ckpt_path, cfg, device)
    expected_joints = int(getattr(model, "num_joints", len(ID_TO_INDEX)))

    split_items = _load_fold_items(cfg, int(args.fold), args.split)
    if len(split_items) == 0:
        raise ValueError(f"Empty split: {args.split}")

    run_out = args.out_dir / f"fold_{args.fold}" / Path(ckpt_path).stem
    run_out.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, Any]] = []
    edges = _build_edges()

    max_samples = max(1, int(args.max_samples))
    max_frames = max(1, int(args.max_frames_per_sample))
    stride = max(1, int(args.frame_stride))

    print(
        f"[INFO] split={args.split}, total_samples={len(split_items)}, process_samples={min(len(split_items), max_samples)}"
    )

    for sample_idx, sample in enumerate(split_items[:max_samples]):
        sample_dict = asdict(sample) if not isinstance(sample, dict) else sample

        cam1_frames_dir = Path(str(sample_dict["cam1_frames_dir"]))
        cam2_frames_dir = Path(str(sample_dict["cam2_frames_dir"]))
        sam3d_cam1_kpt3d_dir = Path(str(sample_dict["sam3d_cam1_kpt3d_dir"]))
        sam3d_cam2_kpt3d_dir = Path(str(sample_dict["sam3d_cam2_kpt3d_dir"]))

        frame1_map = _build_idx_file_map(cam1_frames_dir, ["*.png", "*.jpg", "*.jpeg"])
        frame2_map = _build_idx_file_map(cam2_frames_dir, ["*.png", "*.jpg", "*.jpeg"])
        sam3d_1_map = _build_idx_file_map(sam3d_cam1_kpt3d_dir, ["kpt3d_*.npy", "*.npy"])
        sam3d_2_map = _build_idx_file_map(sam3d_cam2_kpt3d_dir, ["kpt3d_*.npy", "*.npy"])

        common_indices = sorted(
            set(frame1_map.keys())
            & set(frame2_map.keys())
            & set(sam3d_1_map.keys())
            & set(sam3d_2_map.keys())
        )
        if not common_indices:
            print(f"[WARN] no common frame indices for sample {sample_idx}, skip")
            continue

        picked_indices = common_indices[::stride][:max_frames]

        person_id = str(sample_dict.get("person_id", "unknown"))
        action_id = str(sample_dict.get("action_id", "unknown"))
        cam1_id = str(sample_dict.get("cam1_id", "unknown"))
        cam2_id = str(sample_dict.get("cam2_id", "unknown"))
        sample_tag = f"sample_{sample_idx:03d}_{person_id}_{action_id}_{cam1_id}_{cam2_id}"

        gt_character_map: Dict[int, Path] = {}
        kpt3d_dirs = sample_dict.get("kpt3d_dirs")
        if isinstance(kpt3d_dirs, dict) and "character" in kpt3d_dirs:
            gt_character_map = _build_idx_file_map(
                Path(str(kpt3d_dirs["character"])), ["frame_*.npy", "*.npy"]
            )

        gt_source_indices = _load_target_source_indices_by_joint_names(
            sample_dict.get("joint_names_path")
        )

        gt_cam1_2d_map: Dict[int, Path] = {}
        gt_cam2_2d_map: Dict[int, Path] = {}

        cam1_kpt2d_dirs = sample_dict.get("cam1_kpt2d_dirs")
        cam2_kpt2d_dirs = sample_dict.get("cam2_kpt2d_dirs")
        if isinstance(cam1_kpt2d_dirs, dict) and "character" in cam1_kpt2d_dirs:
            gt_cam1_2d_map = _build_idx_file_map(
                Path(str(cam1_kpt2d_dirs["character"])), ["kpt2d_*.npy", "*.npy"]
            )
        elif "cam1_kpt2d_dir" in sample_dict:
            gt_cam1_2d_map = _build_idx_file_map(
                Path(str(sample_dict["cam1_kpt2d_dir"])), ["kpt2d_*.npy", "*.npy"]
            )

        if isinstance(cam2_kpt2d_dirs, dict) and "character" in cam2_kpt2d_dirs:
            gt_cam2_2d_map = _build_idx_file_map(
                Path(str(cam2_kpt2d_dirs["character"])), ["kpt2d_*.npy", "*.npy"]
            )
        elif "cam2_kpt2d_dir" in sample_dict:
            gt_cam2_2d_map = _build_idx_file_map(
                Path(str(sample_dict["cam2_kpt2d_dir"])), ["kpt2d_*.npy", "*.npy"]
            )

        sample_out = run_out / sample_tag
        vis_dir = sample_out / "vis"
        pred_dir = sample_out / "pred"
        vis_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in picked_indices:
            frame1 = _read_rgb(frame1_map[frame_idx])
            frame2 = _read_rgb(frame2_map[frame_idx])
            left_raw = np.asarray(np.load(sam3d_1_map[frame_idx]), dtype=np.float32)
            right_raw = np.asarray(np.load(sam3d_2_map[frame_idx]), dtype=np.float32)

            left_3d = _to_model_joint_count(left_raw, expected_joints)
            right_3d = _to_model_joint_count(right_raw, expected_joints)
            rigid_diag: Dict[str, float] = {
                "applied": 0.0,
                "valid_points": 0.0,
                "rmse_before": float("nan"),
                "rmse_after": float("nan"),
            }
            if rigid_align_right_to_left:
                right_3d, rigid_diag = _rigid_align_right_to_left_single(
                    left=left_3d,
                    right=right_3d,
                )

            gt_3d: Optional[np.ndarray] = None
            gt_cam1_2d: Optional[np.ndarray] = None
            gt_cam2_2d: Optional[np.ndarray] = None
            mpjpe_to_gt: Optional[float] = None
            if frame_idx in gt_character_map:
                gt_raw = np.asarray(np.load(gt_character_map[frame_idx]), dtype=np.float32)
                gt_3d = _to_model_joint_count(
                    gt_raw,
                    expected_joints,
                    source_indices=gt_source_indices,
                )
            if frame_idx in gt_cam1_2d_map:
                gt_cam1_raw = np.asarray(np.load(gt_cam1_2d_map[frame_idx]), dtype=np.float32)
                gt_cam1_2d = _to_model_joint_count(
                    gt_cam1_raw,
                    expected_joints,
                    source_indices=gt_source_indices,
                )
            if frame_idx in gt_cam2_2d_map:
                gt_cam2_raw = np.asarray(np.load(gt_cam2_2d_map[frame_idx]), dtype=np.float32)
                gt_cam2_2d = _to_model_joint_count(
                    gt_cam2_raw,
                    expected_joints,
                    source_indices=gt_source_indices,
                )

            p_left_t = torch.from_numpy(left_3d).unsqueeze(0).unsqueeze(0).to(device)
            p_right_t = torch.from_numpy(right_3d).unsqueeze(0).unsqueeze(0).to(device)

            img_l_t = None
            img_r_t = None
            if bool(getattr(model, "use_dino_features", False)):
                img_l_t = _frame_to_video_tensor(frame1, device)
                img_r_t = _frame_to_video_tensor(frame2, device)

            with torch.no_grad():
                out = model(
                    img_l=img_l_t,
                    img_r=img_r_t,
                    p_left=p_left_t,
                    p_right=p_right_t,
                )

            p_hat = out["p_hat"][0, 0].detach().cpu().numpy().astype(np.float32)
            p0 = out["p0"][0, 0].detach().cpu().numpy().astype(np.float32)
            alpha = out["alpha"][0, 0, :, 0].detach().cpu().numpy().astype(np.float32)
            diag = _build_alpha_diagnostics(p_hat, p0, left_3d, right_3d, alpha)

            print(
                "[DIAG] "
                f"sample={sample_idx} frame={frame_idx} "
                f"alpha(min/mean/max/std)=({diag['alpha_min']:.4f}/{diag['alpha_mean']:.4f}/{diag['alpha_max']:.4f}/{diag['alpha_std']:.4f}) "
                f"hat_to(L/R)=({diag['l2_hat_to_left']:.4f}/{diag['l2_hat_to_right']:.4f}) "
                f"p0_to(L/R)=({diag['l2_p0_to_left']:.4f}/{diag['l2_p0_to_right']:.4f}) "
                f"rigid(applied/rmse_before/rmse_after)=({rigid_diag['applied']:.0f}/{rigid_diag['rmse_before']:.4f}/{rigid_diag['rmse_after']:.4f})"
            )

            if gt_3d is not None:
                mpjpe_to_gt = float(np.linalg.norm(p_hat - gt_3d, axis=1).mean())

            fig = plt.figure(figsize=(30, 7))

            ax1 = fig.add_subplot(1, 5, 1)
            _draw_frame_with_gt_2d(
                ax1,
                frame1,
                gt_cam1_2d,
                edges,
                "cam1 frame + GT 2D" if gt_cam1_2d is not None else "cam1 frame",
            )

            ax2 = fig.add_subplot(1, 5, 2)
            _draw_frame_with_gt_2d(
                ax2,
                frame2,
                gt_cam2_2d,
                edges,
                "cam2 frame + GT 2D" if gt_cam2_2d is not None else "cam2 frame",
            )

            ax3 = fig.add_subplot(1, 5, 3, projection="3d")
            _draw_3d(ax3, left_3d, edges, "sam3d cam1", "tab:blue")

            ax4 = fig.add_subplot(1, 5, 4, projection="3d")
            _draw_3d_compare(ax4, left_3d, right_3d, p_hat, gt_3d, edges)

            ax5 = fig.add_subplot(1, 5, 5)
            ax5.set_title("alpha per joint")
            ax5.bar(np.arange(len(alpha)), alpha, color="tab:green")
            ax5.set_xlabel("joint index")
            ax5.set_ylabel("alpha")
            ax5.set_ylim(0.0, 1.0)
            ax5.grid(True, alpha=0.25)

            title = (
                f"fold={args.fold} sample={sample_idx} frame={frame_idx} "
                f"| alpha_mean={diag['alpha_mean']:.3f} "
                f"| hat_to(L/R)=({diag['l2_hat_to_left']:.3f}/{diag['l2_hat_to_right']:.3f})"
            )
            if mpjpe_to_gt is not None:
                title += f" | mpjpe={mpjpe_to_gt:.4f}"
            fig.suptitle(title)
            fig.tight_layout()

            vis_path = vis_dir / f"frame_{frame_idx:06d}.png"
            fig.savefig(vis_path, dpi=180)
            plt.close(fig)

            pred_payload = {
                "frame_index": int(frame_idx),
                "frame_cam1": str(frame1_map[frame_idx]),
                "frame_cam2": str(frame2_map[frame_idx]),
                "sam3d_cam1_kpt3d": str(sam3d_1_map[frame_idx]),
                "sam3d_cam2_kpt3d": str(sam3d_2_map[frame_idx]),
                "target_joint_ids": TARGET_IDS,
                "dual2pose_ckpt": str(ckpt_path),
                "p_left": left_3d.tolist(),
                "p_right": right_3d.tolist(),
                "p0": p0.tolist(),
                "p_hat": p_hat.tolist(),
                "alpha": alpha.tolist(),
                "diagnostics": diag,
                "rigid_alignment": rigid_diag,
                "gt_character": gt_3d.tolist() if gt_3d is not None else None,
                "mpjpe_to_gt": mpjpe_to_gt,
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
