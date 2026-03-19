#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.dataloader.whole_video_dataset import LabeledUnityDataset
from project.map_config import ID_TO_INDEX, SKELETON_CONNECTIONS, TARGET_IDS, UNITY_MHR70_MAPPING
from project.trainer.train_fusion_SSM import FusionSSMTrainer


def _extract_float_token(token: str) -> Optional[float]:
    try:
        return float(token)
    except Exception:
        return None


def _parse_ckpt_name(ckpt_name: str) -> Optional[Tuple[int, Optional[float], Optional[float]]]:
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
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    candidates = [p for p in all_ckpts if p.name.lower() not in {"last.ckpt", "last-v1.ckpt"}]
    if not candidates:
        for name in ["last.ckpt", "last-v1.ckpt"]:
            p = ckpt_dir / name
            if p.exists():
                return p
        return max(all_ckpts, key=lambda p: p.stat().st_mtime)

    parsed: List[Tuple[Path, int, float, Optional[float]]] = []
    for p in candidates:
        rec = _parse_ckpt_name(p.name)
        if rec is None:
            continue
        epoch, m1, m2 = rec
        if m1 is None:
            continue
        parsed.append((p, epoch, m1, m2))

    if not parsed:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    parsed.sort(key=lambda x: (x[2], -x[1]))
    return parsed[0][0]


def _discover_fold_ckpts(logs_root: Path, target_folds: Optional[Iterable[int]]) -> Dict[int, Path]:
    fold_pat = re.compile(r"fold_(\d+)$")
    fold_to_ckpt: Dict[int, Path] = {}
    allowed = set(target_folds) if target_folds is not None else None

    for fold_dir in sorted(logs_root.glob("*/fold_*")):
        if not fold_dir.is_dir():
            continue
        m = fold_pat.search(fold_dir.name)
        if not m:
            continue
        fold = int(m.group(1))
        if allowed is not None and fold not in allowed:
            continue

        ckpt_dir = fold_dir / "checkpoints" / f"fold_{fold}"
        if not ckpt_dir.exists():
            continue

        try:
            fold_to_ckpt[fold] = _select_best_ckpt(ckpt_dir)
        except FileNotFoundError:
            continue

    if not fold_to_ckpt:
        raise FileNotFoundError(
            f"No fold checkpoints discovered under {logs_root}. "
            "Expected layout: <logs_root>/<date>/fold_X/checkpoints/fold_X/*.ckpt"
        )
    return dict(sorted(fold_to_ckpt.items(), key=lambda kv: kv[0]))


def _choose_frame_index(n_frames: int, given_index: Optional[int]) -> int:
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0")
    if given_index is None:
        return n_frames // 2
    if given_index < 0:
        return max(0, n_frames + given_index)
    if given_index >= n_frames:
        return n_frames - 1
    return given_index


def _extract_frame_payload(frame_rec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "frame" not in frame_rec:
        raise KeyError("Frame record missing 'frame'")
    frame = np.asarray(frame_rec["frame"])

    if "pred_keypoints_3d" in frame_rec:
        kpt3d = np.asarray(frame_rec["pred_keypoints_3d"], dtype=np.float32)
    elif "pred_joint_coords" in frame_rec:
        kpt3d = np.asarray(frame_rec["pred_joint_coords"], dtype=np.float32)
    else:
        raise KeyError("Frame record missing both 'pred_keypoints_3d' and 'pred_joint_coords'")

    if "pred_keypoints_2d" in frame_rec:
        kpt2d = np.asarray(frame_rec["pred_keypoints_2d"], dtype=np.float32)
    else:
        kpt2d = np.zeros((kpt3d.shape[0], 2), dtype=np.float32)

    return frame, kpt2d, kpt3d


def _target_source_indices(num_joints: int) -> List[int]:
    return LabeledUnityDataset._select_source_joint_indices(num_joints)


def _filter_joints(kpts: np.ndarray, src_idx: List[int]) -> np.ndarray:
    if kpts.ndim != 2:
        raise ValueError(f"Expected 2D keypoint array (J,C), got {kpts.shape}")
    if max(src_idx) >= kpts.shape[0]:
        raise ValueError(f"Joint index out of range: max(src_idx)={max(src_idx)}, kpts={kpts.shape}")
    return kpts[src_idx]


def _build_edges() -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for a, b in SKELETON_CONNECTIONS:
        if a in ID_TO_INDEX and b in ID_TO_INDEX:
            edges.append((ID_TO_INDEX[a], ID_TO_INDEX[b]))
    return edges


def _build_joint_labels() -> List[str]:
    labels: List[str] = []
    for idx, jid in enumerate(TARGET_IDS):
        name = UNITY_MHR70_MAPPING.get(jid, f"jid_{jid}")
        labels.append(f"{idx}:{name}")
    return labels


def _filter_edges_by_length(kpts: np.ndarray, edges: List[Tuple[int, int]], max_ratio: float) -> List[Tuple[int, int]]:
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
    out: List[Tuple[int, int]] = []
    for (i, j), dist in zip(valid_edges, lengths):
        if dist <= upper:
            out.append((i, j))
    return out


def _draw_2d(
    ax,
    frame: np.ndarray,
    kpt2d: np.ndarray,
    edges: List[Tuple[int, int]],
    title: str,
    joint_labels: Optional[List[str]] = None,
    show_joint_labels: bool = False,
) -> None:
    ax.imshow(frame)
    ax.set_title(title)
    ax.axis("off")
    if kpt2d.size == 0:
        return

    ax.scatter(kpt2d[:, 0], kpt2d[:, 1], s=12, c="yellow")
    draw_edges = _filter_edges_by_length(kpt2d, edges, max_ratio=3.0)
    for i, j in draw_edges:
        ax.plot(
            [kpt2d[i, 0], kpt2d[j, 0]],
            [kpt2d[i, 1], kpt2d[j, 1]],
            color="cyan",
            linewidth=1.5,
        )

    if show_joint_labels:
        labels = joint_labels or [str(i) for i in range(kpt2d.shape[0])]
        for i in range(min(kpt2d.shape[0], len(labels))):
            ax.text(
                float(kpt2d[i, 0]) + 2.0,
                float(kpt2d[i, 1]) - 2.0,
                labels[i],
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 0.5},
            )


def _set_equal_3d_axes(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float((maxs - mins).max() * 0.55)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_3d(
    ax,
    left: np.ndarray,
    right: np.ndarray,
    fused: np.ndarray,
    edges: List[Tuple[int, int]],
    title: str,
    joint_labels: Optional[List[str]] = None,
    show_joint_labels: bool = False,
) -> None:
    ax.set_title(title)
    ax.scatter(left[:, 0], left[:, 1], left[:, 2], s=10, c="tab:blue", alpha=0.35)
    ax.scatter(right[:, 0], right[:, 1], right[:, 2], s=10, c="tab:orange", alpha=0.35)
    ax.scatter(fused[:, 0], fused[:, 1], fused[:, 2], s=12, c="tab:red", alpha=0.9)

    draw_edges = _filter_edges_by_length(fused, edges, max_ratio=2.5)
    for i, j in draw_edges:
        ax.plot(
            [fused[i, 0], fused[j, 0]],
            [fused[i, 1], fused[j, 1]],
            [fused[i, 2], fused[j, 2]],
            color="tab:red",
            linewidth=2.0,
        )

    if show_joint_labels:
        labels = joint_labels or [str(i) for i in range(fused.shape[0])]
        for i in range(min(fused.shape[0], len(labels))):
            ax.text(
                float(fused[i, 0]),
                float(fused[i, 1]),
                float(fused[i, 2]),
                labels[i],
                color="black",
                fontsize=8,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_equal_3d_axes(ax, fused)


def _build_model_from_ckpt(ckpt_path: Path, device: torch.device) -> FusionSSMTrainer:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    in_proj_w = state_dict["model.refiner.in_proj.weight"]
    d_model = int(in_proj_w.shape[0])
    num_joints = int(in_proj_w.shape[1] // 3)

    block_pat = re.compile(r"^model\.refiner\.blocks\.(\d+)\.")
    block_ids = set()
    for key in state_dict.keys():
        m = block_pat.match(key)
        if m:
            block_ids.add(int(m.group(1)))
    n_layers = (max(block_ids) + 1) if block_ids else 4

    gate_in = int(state_dict["model.gating.mlp.0.weight"].shape[1])
    use_conf = gate_in == 11

    gate_out = int(state_dict["model.gating.mlp.4.weight"].shape[0])
    predict_logvar = gate_out == 2

    hparams = OmegaConf.create(
        {
            "loss": {
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "lambda_mpjpe": 1.0,
                "lambda_bone": 0.2,
                "lambda_vel": 0.05,
                "lambda_acc": 0.02,
                "lambda_agree": 0.1,
                "lambda_bone_stab": 0.05,
            },
            "model": {
                "d_model": d_model,
                "n_layers": n_layers,
                "use_conf": use_conf,
                "predict_logvar": predict_logvar,
            },
            "log_path": str(ckpt_path.parent.parent.parent),
        }
    )

    module = FusionSSMTrainer(hparams=hparams)
    if num_joints != len(ID_TO_INDEX):
        raise ValueError(
            f"Checkpoint joint count mismatch: ckpt has {num_joints}, current target has {len(ID_TO_INDEX)}"
        )
    module.load_state_dict(state_dict, strict=True)

    module.eval()
    module.to(device)
    return module


def _run_infer_one_frame(
    module: FusionSSMTrainer,
    left_3d: np.ndarray,
    right_3d: np.ndarray,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    p_left = torch.from_numpy(left_3d.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    p_right = torch.from_numpy(right_3d.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = module.model(p_left=p_left, p_right=p_right)

    return {
        "p_hat": out["p_hat"][0, 0].detach().cpu().numpy(),
        "p0": out["p0"][0, 0].detach().cpu().numpy(),
        "alpha": out["alpha"][0, 0, :, 0].detach().cpu().numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load best ckpt for each fold, infer one side_raw dual-view frame, and visualize outputs.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train/3dcnn_fuse_method_mamba_ssm"),
        help="Root dir that contains <date>/fold_x/checkpoints/fold_x/*.ckpt",
    )
    parser.add_argument(
        "--sam-person-root",
        type=Path,
        default=Path("/workspace/data/sam3d_body_results/person"),
        help="Root dir that contains run_x/osmo_1_sam_3d_body_outputs.npz and osmo_2_*.npz",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="run_3",
        help="Side raw run id, e.g. run_3 / run_4 / pro_1 ...",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index for inference; default is middle frame. Negative index is supported.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Optional fold list, e.g. --folds 0 1 2",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device: cuda/cpu",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_side_raw"),
        help="Visualization output directory",
    )
    parser.add_argument(
        "--show-joint-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to draw joint index/name labels on 2D/3D subplots (default: True)",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable, fallback to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    left_npz_path = args.sam_person_root / args.run_id / "osmo_1_sam_3d_body_outputs.npz"
    right_npz_path = args.sam_person_root / args.run_id / "osmo_2_sam_3d_body_outputs.npz"
    if not left_npz_path.exists() or not right_npz_path.exists():
        raise FileNotFoundError(
            f"Missing SAM outputs for {args.run_id}.\n"
            f"Expected:\n  {left_npz_path}\n  {right_npz_path}"
        )

    left_outputs = np.load(left_npz_path, allow_pickle=True)["outputs"]
    right_outputs = np.load(right_npz_path, allow_pickle=True)["outputs"]

    n = min(len(left_outputs), len(right_outputs))
    frame_idx = _choose_frame_index(n, args.frame_index)
    left_frame, left_2d_raw, left_3d_raw = _extract_frame_payload(left_outputs[frame_idx])
    right_frame, right_2d_raw, right_3d_raw = _extract_frame_payload(right_outputs[frame_idx])

    src_idx = _target_source_indices(min(left_3d_raw.shape[0], right_3d_raw.shape[0]))
    left_2d = _filter_joints(left_2d_raw, src_idx)
    right_2d = _filter_joints(right_2d_raw, src_idx)
    left_3d = _filter_joints(left_3d_raw, src_idx)
    right_3d = _filter_joints(right_3d_raw, src_idx)

    edges = _build_edges()
    joint_labels = _build_joint_labels()
    fold_ckpts = _discover_fold_ckpts(args.logs_root, args.folds)

    run_out = args.out_dir / args.run_id / f"frame_{frame_idx:06d}"
    run_out.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    print(f"[INFO] run_id={args.run_id}, frame_index={frame_idx}, folds={list(fold_ckpts.keys())}")
    print(f"[INFO] output dir: {run_out}")

    for fold, ckpt_path in fold_ckpts.items():
        print(f"[INFO] fold={fold}: loading ckpt {ckpt_path}")
        module = _build_model_from_ckpt(ckpt_path, device)

        out = _run_infer_one_frame(module, left_3d, right_3d, device)
        p_hat = out["p_hat"]
        alpha = out["alpha"]

        fig = plt.figure(figsize=(32, 8))
        ax1 = fig.add_subplot(1, 4, 1)
        _draw_2d(
            ax1,
            left_frame,
            left_2d,
            edges,
            f"Left view (run={args.run_id}, frame={frame_idx})",
            joint_labels=joint_labels,
            show_joint_labels=args.show_joint_labels,
        )

        ax2 = fig.add_subplot(1, 4, 2)
        _draw_2d(
            ax2,
            right_frame,
            right_2d,
            edges,
            "Right view",
            joint_labels=joint_labels,
            show_joint_labels=args.show_joint_labels,
        )

        ax3 = fig.add_subplot(1, 4, 3, projection="3d")
        _draw_3d(
            ax3,
            left_3d,
            right_3d,
            p_hat,
            edges,
            f"Fold {fold} fused 3D",
            joint_labels=joint_labels,
            show_joint_labels=args.show_joint_labels,
        )

        ax4 = fig.add_subplot(1, 4, 4)
        ax4.set_title("Alpha per joint")
        ax4.bar(np.arange(len(alpha)), alpha, color="tab:green")
        ax4.set_xlabel("Joint index")
        ax4.set_ylabel("Alpha")
        ax4.set_ylim(0.0, 1.0)
        ax4.grid(True, alpha=0.2)

        fig.suptitle(f"Fold {fold} | ckpt={ckpt_path.name}")
        fig.tight_layout()

        out_path = run_out / f"fold_{fold}_vis.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

        records.append(
            {
                "fold": fold,
                "ckpt": str(ckpt_path),
                "vis_path": str(out_path),
                "alpha_mean": float(np.mean(alpha)),
                "alpha_std": float(np.std(alpha)),
            }
        )

    summary_path = run_out / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": args.run_id,
                "frame_index": int(frame_idx),
                "num_frames_used": int(n),
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] Saved {len(records)} fold visualizations")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
