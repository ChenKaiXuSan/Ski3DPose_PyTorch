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
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual2pose.map_config import (
    FILTERED_KPTS_MAPPING,
    FILTER_SKELETON_CONNECTIONS,
)
from dual2pose.models.crossview_fusion import CrossViewCanonicalFusion
from dual2pose.dataloader.data_loader import UnityDataModule
from dual2pose.trainer.canonicalize import canonicalize_pose_numpy


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


def _build_edges() -> List[Tuple[int, int]]:
    return list(FILTER_SKELETON_CONNECTIONS)


def _load_from_ckpt(
    ckpt_path: Path, _cfg: Any, device: torch.device
) -> CrossViewCanonicalFusion:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("models.character."):
            model_state[key[len("models.character.") :]] = value

    if not model_state:
        raise KeyError("No 'models.character.*' weights found in checkpoint")

    model = CrossViewCanonicalFusion()

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


def _draw_3d(
    ax, xyz: np.ndarray, edges: List[Tuple[int, int]], title: str, color: str
) -> None:
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
    frame: Optional[np.ndarray],
    gt_2d: Optional[np.ndarray],
    edges: List[Tuple[int, int]],
    title: str,
) -> None:
    ax.set_title(title)
    ax.axis("off")
    if frame is None:
        ax.text(0.5, 0.5, "frame unavailable", ha="center", va="center")
        return

    ax.imshow(frame)

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
    ax.scatter(
        left_3d[:, 0], left_3d[:, 1], left_3d[:, 2], s=10, c="tab:blue", alpha=0.3
    )
    ax.scatter(
        right_3d[:, 0], right_3d[:, 1], right_3d[:, 2], s=10, c="tab:orange", alpha=0.3
    )
    ax.scatter(
        fused_3d[:, 0], fused_3d[:, 1], fused_3d[:, 2], s=16, c="tab:red", alpha=0.9
    )
    if gt_3d is not None:
        ax.scatter(
            gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], s=10, c="tab:green", alpha=0.45
        )

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


def _to_frame_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    arr = frame_chw.detach().cpu().numpy().astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame shape (C,H,W), got {arr.shape}")

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = arr * std + mean
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))


def _meta_value(meta: Any, key: str, idx: int, default: str = "unknown") -> str:
    if not isinstance(meta, dict):
        return default
    if key not in meta:
        return default
    value = meta[key]
    if isinstance(value, (list, tuple)):
        if 0 <= idx < len(value):
            return str(value[idx])
        return default
    return str(value)


def _load_fold_data(cfg: Any, fold: int) -> Dict[str, Any]:
    fold_dir = Path(str(cfg.data.index_mapping_path))
    fold_file = fold_dir / f"fold_{fold:02d}.json"
    if not fold_file.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_file}")

    with open(fold_file, "r", encoding="utf-8") as f:
        fold_data = json.load(f)

    fold_data.pop("_metadata", None)
    return fold_data


def _mpjpe(pred_xyz: np.ndarray, gt_xyz: Optional[np.ndarray]) -> Optional[float]:
    if gt_xyz is None:
        return None
    if pred_xyz.shape != gt_xyz.shape:
        return None
    return float(np.linalg.norm(pred_xyz - gt_xyz, axis=-1).mean())


def _rigid_align_right_to_left(
    left_xyz: np.ndarray, right_xyz: np.ndarray
) -> np.ndarray:
    """Align right pose to left pose with rigid transform (Kabsch, no scale)."""
    if (
        left_xyz.shape != right_xyz.shape
        or left_xyz.ndim != 2
        or left_xyz.shape[1] != 3
    ):
        return right_xyz

    src = right_xyz.astype(np.float64)
    dst = left_xyz.astype(np.float64)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = dst_mean - src_mean @ R
    aligned = src @ R + t
    return aligned.astype(np.float32)


def _compute_pose_metrics(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    fused_xyz: np.ndarray,
    gt_xyz: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    """Compute requested metrics against GT from left/right/fused inputs."""
    right_aligned_to_left = _rigid_align_right_to_left(left_xyz, right_xyz)
    rigid_avg_xyz = 0.5 * (left_xyz + right_aligned_to_left)

    return {
        "mpjpe_left_to_gt": _mpjpe(left_xyz, gt_xyz),
        "mpjpe_right_to_gt": _mpjpe(right_xyz, gt_xyz),
        "mpjpe_fused_to_gt": _mpjpe(fused_xyz, gt_xyz),
        "mpjpe_rigid_avg_to_gt": _mpjpe(rigid_avg_xyz, gt_xyz),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize dual2pose fusion output on unity dual-view test batches"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/crossview_fusion/2026-05-12/fold_0/checkpoints/fold_0"
        ),
        help="Checkpoint directory containing *.ckpt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/configs/dual2pose.yaml"
        ),
        help="Hydra config path",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold id")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--max-samples", type=int, default=5, help="Max samples to save"
    )
    parser.add_argument(
        "--max-frames-per-sample",
        type=int,
        default=10,
        help="Max frames per sample",
    )
    parser.add_argument("--frame-stride", type=int, default=5, help="Frame stride")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_true_data/dual2pose/dual2pose_unity_frame"
        ),
        help="Output root",
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

    cfg.data.load_frames = True

    ckpt_path = _select_best_ckpt(args.ckpt_dir)
    print(f"[INFO] Best ckpt: {ckpt_path}")

    model = _load_from_ckpt(ckpt_path, cfg, device)

    fold_data = _load_fold_data(cfg, int(args.fold))
    split_items = fold_data.get(args.split, [])
    if not isinstance(split_items, list) or len(split_items) == 0:
        raise RuntimeError(f"No items in split={args.split}, fold={args.fold}")

    dataset = UnityDataModule(opt=cfg, dataset_idx=fold_data)
    dataset.setup(stage="test")
    if args.split == "train":
        loader = dataset.train_dataloader()
    elif args.split == "val":
        loader = dataset.val_dataloader()
    else:
        loader = dataset.test_dataloader()

    run_out = args.out_dir / f"fold_{args.fold}" / Path(ckpt_path).stem
    run_out.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, Any]] = []
    global_metrics: Dict[str, List[float]] = {
        "mpjpe_left_to_gt": [],
        "mpjpe_right_to_gt": [],
        "mpjpe_rigid_avg_to_gt": [],
        "mpjpe_fused_to_gt": [],
    }
    edges = _build_edges()

    max_samples = max(1, int(args.max_samples))
    max_frames = max(1, int(args.max_frames_per_sample))
    stride = max(1, int(args.frame_stride))

    print(
        f"[INFO] split={args.split}, total_samples={len(split_items)}, process_samples={min(len(split_items), max_samples)}"
    )

    saved_samples = 0

    for batch_idx, batch in enumerate(loader):
        left = batch["kpt3d_sam"]["cam1"].float()
        right = batch["kpt3d_sam"]["cam2"].float()
        gt = batch.get("kpt3d_gt", None)
        if isinstance(gt, dict):
            gt = gt.get("character", None)
        if isinstance(gt, torch.Tensor):
            gt = gt.float()

        left_canonical, _ = canonicalize_pose_numpy(left.cpu().numpy())
        right_canonical, _ = canonicalize_pose_numpy(right.cpu().numpy())
        gt_canonical = None
        if isinstance(gt, torch.Tensor):
            gt_canonical, _ = canonicalize_pose_numpy(gt.cpu().numpy())

        left_canonical_t = torch.from_numpy(left_canonical).to(device)
        right_canonical_t = torch.from_numpy(right_canonical).to(device)

        with torch.no_grad():
            fused, out = model(left_canonical_t, right_canonical_t)

        alpha = out["alpha"].detach().cpu()
        p_hat = fused.detach().cpu()
        left_can = left_canonical_t.detach().cpu()
        right_can = right_canonical_t.detach().cpu()
        gt_can = (
            torch.from_numpy(gt_canonical).float() if gt_canonical is not None else None
        )

        frame_indices = batch.get("frame_indices", None)
        frames = batch.get("frames", {})
        frames_cam1 = frames.get("cam1") if isinstance(frames, dict) else None
        frames_cam2 = frames.get("cam2") if isinstance(frames, dict) else None
        meta = batch.get("meta", {})

        bsz = p_hat.shape[0]
        for b in range(bsz):

            person_id = _meta_value(meta, "person_id", b)
            action_id = _meta_value(meta, "action_id", b)
            cam1_id = _meta_value(meta, "cam1_id", b)
            cam2_id = _meta_value(meta, "cam2_id", b)
            sample_tag = (
                f"{person_id}_{action_id}_{cam1_id}_{cam2_id}_s{saved_samples:04d}"
            )

            sample_out = run_out / sample_tag
            vis_dir = sample_out / "vis"
            pred_dir = sample_out / "pred"
            vis_dir.mkdir(parents=True, exist_ok=True)
            pred_dir.mkdir(parents=True, exist_ok=True)

            t_total = int(p_hat.shape[1])
            frame_positions = list(range(0, t_total, stride))[:max_frames]

            sample_mpjpe_fused: List[float] = []
            sample_mpjpe_left: List[float] = []
            sample_mpjpe_right: List[float] = []
            sample_mpjpe_rigid_avg: List[float] = []
            for t_pos in frame_positions:
                frame_idx = (
                    int(frame_indices[b, t_pos].item())
                    if isinstance(frame_indices, torch.Tensor)
                    else t_pos
                )

                left_3d = left_can[b, t_pos].numpy()
                right_3d = right_can[b, t_pos].numpy()
                fused_3d = p_hat[b, t_pos].numpy()
                p0_3d = 0.5 * (left_3d + right_3d)
                gt_3d = (
                    gt_can[b, t_pos].numpy()
                    if isinstance(gt_can, torch.Tensor)
                    else None
                )

                frame1 = (
                    _to_frame_uint8(frames_cam1[b, t_pos])
                    if isinstance(frames_cam1, torch.Tensor)
                    else None
                )
                frame2 = (
                    _to_frame_uint8(frames_cam2[b, t_pos])
                    if isinstance(frames_cam2, torch.Tensor)
                    else None
                )

                alpha_t = alpha[b, t_pos, :, 0].numpy()
                diag = {
                    "alpha_mean": float(alpha_t.mean()),
                    "alpha_std": float(alpha_t.std()),
                    "l2_hat_to_left": float(
                        np.linalg.norm(fused_3d - left_3d, axis=-1).mean()
                    ),
                    "l2_hat_to_right": float(
                        np.linalg.norm(fused_3d - right_3d, axis=-1).mean()
                    ),
                }

                metric_pack = _compute_pose_metrics(
                    left_xyz=left_3d,
                    right_xyz=right_3d,
                    fused_xyz=fused_3d,
                    gt_xyz=gt_3d,
                )
                mpjpe_left_to_gt = metric_pack["mpjpe_left_to_gt"]
                mpjpe_right_to_gt = metric_pack["mpjpe_right_to_gt"]
                mpjpe_fused_to_gt = metric_pack["mpjpe_fused_to_gt"]
                mpjpe_rigid_avg_to_gt = metric_pack["mpjpe_rigid_avg_to_gt"]

                if mpjpe_left_to_gt is not None:
                    sample_mpjpe_left.append(mpjpe_left_to_gt)
                    global_metrics["mpjpe_left_to_gt"].append(mpjpe_left_to_gt)
                if mpjpe_right_to_gt is not None:
                    sample_mpjpe_right.append(mpjpe_right_to_gt)
                    global_metrics["mpjpe_right_to_gt"].append(mpjpe_right_to_gt)
                if mpjpe_rigid_avg_to_gt is not None:
                    sample_mpjpe_rigid_avg.append(mpjpe_rigid_avg_to_gt)
                    global_metrics["mpjpe_rigid_avg_to_gt"].append(
                        mpjpe_rigid_avg_to_gt
                    )
                if mpjpe_fused_to_gt is not None:
                    sample_mpjpe_fused.append(mpjpe_fused_to_gt)
                    global_metrics["mpjpe_fused_to_gt"].append(mpjpe_fused_to_gt)

                fig = plt.figure(figsize=(36, 7))

                ax1 = fig.add_subplot(1, 6, 1)
                _draw_frame_with_gt_2d(ax1, frame1, None, edges, "cam1 frame")

                ax2 = fig.add_subplot(1, 6, 2)
                _draw_frame_with_gt_2d(ax2, frame2, None, edges, "cam2 frame")

                ax3 = fig.add_subplot(1, 6, 3, projection="3d")
                _draw_3d(ax3, left_3d, edges, "sam3d cam1 (canonical)", "tab:blue")

                ax4 = fig.add_subplot(1, 6, 4, projection="3d")
                _draw_3d(ax4, right_3d, edges, "sam3d cam2 (canonical)", "tab:orange")

                ax5 = fig.add_subplot(1, 6, 5, projection="3d")
                _draw_3d_compare(ax5, left_3d, right_3d, fused_3d, gt_3d, edges)

                ax6 = fig.add_subplot(1, 6, 6)
                ax6.set_title("alpha per joint")
                ax6.bar(np.arange(len(alpha_t)), alpha_t, color="tab:green")
                ax6.set_xlabel("joint index")
                ax6.set_ylabel("alpha")
                ax6.set_ylim(0.0, 1.0)
                ax6.grid(True, alpha=0.25)

                title = (
                    f"fold={args.fold} sample={saved_samples} frame={frame_idx} "
                    f"| alpha_mean={diag['alpha_mean']:.3f} "
                    f"| hat_to(L/R)=({diag['l2_hat_to_left']:.3f}/{diag['l2_hat_to_right']:.3f})"
                )
                if mpjpe_fused_to_gt is not None:
                    title += (
                        f" | mpjpe(fused/rigid_avg/l/r)=({mpjpe_fused_to_gt:.4f}/"
                        f"{mpjpe_rigid_avg_to_gt:.4f}/{mpjpe_left_to_gt:.4f}/{mpjpe_right_to_gt:.4f})"
                    )
                fig.suptitle(title)
                fig.tight_layout()

                vis_path = vis_dir / f"frame_{frame_idx:06d}.png"
                fig.savefig(vis_path, dpi=180)
                plt.close(fig)

                pred_payload = {
                    "batch_idx": int(batch_idx),
                    "sample_index": int(saved_samples),
                    "frame_index": int(frame_idx),
                    "target_joint_names": [
                        FILTERED_KPTS_MAPPING[i]
                        for i in range(len(FILTERED_KPTS_MAPPING))
                    ],
                    "dual2pose_ckpt": str(ckpt_path),
                    "p_left_canonical": left_3d.tolist(),
                    "p_right_canonical": right_3d.tolist(),
                    "p0_canonical": p0_3d.tolist(),
                    "p_hat_canonical": fused_3d.tolist(),
                    "alpha": alpha_t.tolist(),
                    "diagnostics": diag,
                    "gt_character_canonical": (
                        gt_3d.tolist() if gt_3d is not None else None
                    ),
                    "mpjpe_to_gt": {
                        "left": mpjpe_left_to_gt,
                        "right": mpjpe_right_to_gt,
                        "rigid_avg": mpjpe_rigid_avg_to_gt,
                        "fused": mpjpe_fused_to_gt,
                        "fused_minus_rigid_avg": (
                            (mpjpe_fused_to_gt - mpjpe_rigid_avg_to_gt)
                            if (
                                mpjpe_fused_to_gt is not None
                                and mpjpe_rigid_avg_to_gt is not None
                            )
                            else None
                        ),
                    },
                    "meta": {
                        "person_id": person_id,
                        "action_id": action_id,
                        "cam1_id": cam1_id,
                        "cam2_id": cam2_id,
                    },
                }
                pred_json_path = pred_dir / f"frame_{frame_idx:06d}.json"
                with open(pred_json_path, "w", encoding="utf-8") as f:
                    json.dump(pred_payload, f, ensure_ascii=False, indent=2)

                summary_records.append(
                    {
                        "sample_idx": int(saved_samples),
                        "person_id": person_id,
                        "action_id": action_id,
                        "cam1_id": cam1_id,
                        "cam2_id": cam2_id,
                        "frame_index": int(frame_idx),
                        "vis_path": str(vis_path),
                        "pred_json_path": str(pred_json_path),
                        "mpjpe_to_gt": {
                            "left": mpjpe_left_to_gt,
                            "right": mpjpe_right_to_gt,
                            "rigid_avg": mpjpe_rigid_avg_to_gt,
                            "fused": mpjpe_fused_to_gt,
                        },
                    }
                )

            if sample_mpjpe_fused:
                sample_left = float(np.mean(sample_mpjpe_left))
                sample_right = float(np.mean(sample_mpjpe_right))
                sample_rigid_avg = float(np.mean(sample_mpjpe_rigid_avg))
                sample_fused = float(np.mean(sample_mpjpe_fused))
                sample_best_input = min(sample_left, sample_right)
                print(
                    "[SAMPLE] "
                    f"sample={saved_samples} tag={sample_tag} "
                    f"mean_mpjpe(fused/rigid_avg/l/r)=({sample_fused:.4f}/{sample_rigid_avg:.4f}/{sample_left:.4f}/{sample_right:.4f}) "
                    f"improve_vs_rigid_avg={(sample_rigid_avg - sample_fused):+.4f} "
                    f"improve_vs_best_input={(sample_best_input - sample_fused):+.4f} "
                    f"frames_with_gt={len(sample_mpjpe_fused)}"
                )

            saved_samples += 1

        if saved_samples >= max_samples:
            break

    metric_mean = {
        key: (float(np.mean(values)) if values else None)
        for key, values in global_metrics.items()
    }
    improve_vs_rigid_avg = None
    if (
        metric_mean["mpjpe_fused_to_gt"] is not None
        and metric_mean["mpjpe_rigid_avg_to_gt"] is not None
    ):
        improve_vs_rigid_avg = (
            metric_mean["mpjpe_rigid_avg_to_gt"] - metric_mean["mpjpe_fused_to_gt"]
        )
    improve_vs_best_input = None
    if (
        metric_mean["mpjpe_fused_to_gt"] is not None
        and metric_mean["mpjpe_left_to_gt"] is not None
        and metric_mean["mpjpe_right_to_gt"] is not None
    ):
        improve_vs_best_input = (
            min(metric_mean["mpjpe_left_to_gt"], metric_mean["mpjpe_right_to_gt"])
            - metric_mean["mpjpe_fused_to_gt"]
        )

    summary_path = run_out / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ckpt_path": str(ckpt_path),
                "fold": int(args.fold),
                "split": args.split,
                "global_mpjpe_to_gt": {
                    **metric_mean,
                    "improve_vs_rigid_avg": improve_vs_rigid_avg,
                    "improve_vs_best_input": improve_vs_best_input,
                },
                "records": summary_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if metric_mean["mpjpe_fused_to_gt"] is not None:
        print(
            "[GLOBAL] "
            f"mpjpe(fused/rigid_avg/l/r)=({metric_mean['mpjpe_fused_to_gt']:.4f}/"
            f"{metric_mean['mpjpe_rigid_avg_to_gt']:.4f}/{metric_mean['mpjpe_left_to_gt']:.4f}/{metric_mean['mpjpe_right_to_gt']:.4f}) "
            f"improve_vs_rigid_avg={improve_vs_rigid_avg:+.4f} "
            f"improve_vs_best_input={improve_vs_best_input:+.4f}"
        )

    print(f"[DONE] Saved {len(summary_records)} frame predictions")
    print(f"[DONE] Output root: {run_out}")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
