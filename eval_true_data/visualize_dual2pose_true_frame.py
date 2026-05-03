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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.map_config import (
    ID_TO_INDEX,
    SKELETON_CONNECTIONS,
    TARGET_IDS,
    UNITY_MHR70_MAPPING,
)
from project.models.dual2pose_net import Dual2PoseNet


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
    edges: List[Tuple[int, int]] = []
    for a, b in SKELETON_CONNECTIONS:
        if a in ID_TO_INDEX and b in ID_TO_INDEX:
            edges.append((ID_TO_INDEX[a], ID_TO_INDEX[b]))
    return edges


def _build_labels() -> List[str]:
    labels: List[str] = []
    for i, jid in enumerate(TARGET_IDS):
        labels.append(f"{i}:{UNITY_MHR70_MAPPING.get(jid, str(jid))}")
    return labels


def _to_model_joint_count(kpt: np.ndarray, expected_joints: int) -> np.ndarray:
    if kpt.ndim != 2 or kpt.shape[1] not in (2, 3):
        raise ValueError(f"Expected keypoint shape [J,2/3], got {kpt.shape}")

    src_idx: List[int] = []
    for jid in [jid for jid, _ in sorted(ID_TO_INDEX.items(), key=lambda kv: kv[1])]:
        candidates = (jid, ID_TO_INDEX[jid])
        mapped = next((c for c in candidates if 0 <= c < kpt.shape[0]), None)
        if mapped is None:
            raise IndexError(
                f"Target joint id {jid} cannot be mapped for source joint count {kpt.shape[0]}."
            )
        src_idx.append(int(mapped))

    filtered = kpt[src_idx]
    if filtered.shape[0] != expected_joints:
        raise ValueError(
            f"Joint count mismatch after filtering: expected {expected_joints}, got {filtered.shape[0]}"
        )
    return filtered.astype(np.float32)


def _frame_to_video_tensor(frame_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(frame_rgb, dtype=np.float32)).permute(2, 0, 1)
    if x.max() <= 1.5:
        x = x * 255.0
    x = x.unsqueeze(0).unsqueeze(2)
    return x.to(device)


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


def _draw_2d(
    ax,
    frame: np.ndarray,
    kpt2d: np.ndarray,
    edges: List[Tuple[int, int]],
    labels: List[str],
    show_labels: bool,
    title: str,
) -> None:
    ax.imshow(frame)
    ax.set_title(title)
    ax.axis("off")

    if kpt2d.shape[0] > 0:
        ax.scatter(kpt2d[:, 0], kpt2d[:, 1], s=14, c="yellow")
    for i, j in edges:
        if i < kpt2d.shape[0] and j < kpt2d.shape[0]:
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
    left_3d: np.ndarray,
    right_3d: np.ndarray,
    fused_3d: np.ndarray,
    edges: List[Tuple[int, int]],
    labels: List[str],
    show_labels: bool,
) -> None:
    ax.set_title("Dual2Pose fused 3D")
    ax.scatter(left_3d[:, 0], left_3d[:, 1], left_3d[:, 2], s=10, c="tab:blue", alpha=0.35)
    ax.scatter(
        right_3d[:, 0], right_3d[:, 1], right_3d[:, 2], s=10, c="tab:orange", alpha=0.35
    )
    ax.scatter(
        fused_3d[:, 0], fused_3d[:, 1], fused_3d[:, 2], s=16, c="tab:red", alpha=0.9
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

    if show_labels:
        for i in range(min(len(fused_3d), len(labels))):
            ax.text(
                float(fused_3d[i, 0]),
                float(fused_3d[i, 1]),
                float(fused_3d[i, 2]),
                labels[i],
                fontsize=8,
            )

    _set_equal_axes_3d(ax, np.concatenate([left_3d, right_3d, fused_3d], axis=0))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Dual2Pose fusion on true frame dual-view SAM3D outputs"
    )
    parser.add_argument(
        "--sam-root",
        type=Path,
        default=Path("/workspace/data/sam3d_body_results/person"),
        help="Root containing <run_id>/<left|right>/frame_xxxxx_sam_3d_body_outputs.npz",
    )
    parser.add_argument("--run-id", type=str, default="pro_1", help="Run/person id")
    parser.add_argument("--left-side", type=str, default="left", help="Left camera folder")
    parser.add_argument("--right-side", type=str, default="right", help="Right camera folder")
    parser.add_argument(
        "--frame-index", type=int, default=None, help="Specific frame index. Default: middle"
    )
    parser.add_argument("--max-frames", type=int, default=10, help="How many frames to visualize")
    parser.add_argument("--stride", type=int, default=30, help="Frame stride when max-frames > 1")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_true_data/dual2pose/dual2pose_true_frame"
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
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/dual2pose/2026-05-02/fold_0/checkpoints/fold_0"
        ),
        help="Dual2Pose checkpoint directory (or its parent)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/configs/train.yaml"
        ),
        help="Config file used for dual2pose settings",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device: cuda/cpu")
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
    model = _load_dual2pose_from_ckpt(ckpt_path, cfg, device)
    expected_joints = int(getattr(model, "num_joints", len(ID_TO_INDEX)))

    print(f"[INFO] Best ckpt: {ckpt_path}")

    left_dir = args.sam_root / args.run_id / args.left_side
    right_dir = args.sam_root / args.run_id / args.right_side
    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"Missing side directories: {left_dir}, {right_dir}")

    left_npz = sorted(left_dir.glob("frame_*_sam_3d_body_outputs.npz"))
    right_npz = sorted(right_dir.glob("frame_*_sam_3d_body_outputs.npz"))
    if not left_npz or not right_npz:
        raise FileNotFoundError("No SAM3D frame npz files found for one/both sides")

    left_map = {_extract_frame_idx(p): p for p in left_npz}
    right_map = {_extract_frame_idx(p): p for p in right_npz}
    all_indices = sorted(set(left_map.keys()) & set(right_map.keys()))
    if not all_indices:
        raise ValueError("No common frame index between left and right sides")

    if args.frame_index is not None:
        if args.frame_index not in left_map or args.frame_index not in right_map:
            raise KeyError(f"frame_index {args.frame_index} not found in both sides")
        picked = [args.frame_index]
    else:
        mid = len(all_indices) // 2
        start = all_indices[mid]
        picked = all_indices[all_indices.index(start) :: max(1, int(args.stride))][
            : max(1, int(args.max_frames))
        ]

    out_dir = args.out_dir / args.run_id / Path(ckpt_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    edges = _build_edges()
    labels = _build_labels()
    records: List[Dict[str, Any]] = []

    for idx in picked:
        left_rec = _load_record(left_map[idx])
        right_rec = _load_record(right_map[idx])

        if "frame" not in left_rec or "frame" not in right_rec:
            raise KeyError(f"Missing 'frame' in records at frame {idx}")

        left_frame = np.asarray(left_rec["frame"])
        right_frame = np.asarray(right_rec["frame"])

        left_3d_raw = np.asarray(
            left_rec.get("pred_keypoints_3d", left_rec.get("pred_joint_coords")),
            dtype=np.float32,
        )
        right_3d_raw = np.asarray(
            right_rec.get("pred_keypoints_3d", right_rec.get("pred_joint_coords")),
            dtype=np.float32,
        )

        if left_3d_raw.ndim != 2 or right_3d_raw.ndim != 2:
            raise ValueError(f"Invalid raw 3D shapes at frame {idx}: {left_3d_raw.shape}, {right_3d_raw.shape}")

        left_2d_raw = np.asarray(left_rec.get("pred_keypoints_2d", left_3d_raw[:, :2]), dtype=np.float32)
        right_2d_raw = np.asarray(right_rec.get("pred_keypoints_2d", right_3d_raw[:, :2]), dtype=np.float32)

        left_3d = _to_model_joint_count(left_3d_raw, expected_joints)
        right_3d = _to_model_joint_count(right_3d_raw, expected_joints)
        left_2d = _to_model_joint_count(left_2d_raw, expected_joints)
        right_2d = _to_model_joint_count(right_2d_raw, expected_joints)

        p_left_t = torch.from_numpy(left_3d).unsqueeze(0).unsqueeze(0).to(device)
        p_right_t = torch.from_numpy(right_3d).unsqueeze(0).unsqueeze(0).to(device)

        img_l_t = None
        img_r_t = None
        if bool(getattr(model, "use_dino_features", False)):
            img_l_t = _frame_to_video_tensor(left_frame, device)
            img_r_t = _frame_to_video_tensor(right_frame, device)

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
            f"frame={idx} "
            f"alpha(min/mean/max/std)=({diag['alpha_min']:.4f}/{diag['alpha_mean']:.4f}/{diag['alpha_max']:.4f}/{diag['alpha_std']:.4f}) "
            f"hat_to(L/R)=({diag['l2_hat_to_left']:.4f}/{diag['l2_hat_to_right']:.4f}) "
            f"p0_to(L/R)=({diag['l2_p0_to_left']:.4f}/{diag['l2_p0_to_right']:.4f})"
        )

        fig = plt.figure(figsize=(26, 7))
        ax1 = fig.add_subplot(1, 4, 1)
        _draw_2d(ax1, left_frame, left_2d, edges, labels, args.show_joint_labels, "Left frame")

        ax2 = fig.add_subplot(1, 4, 2)
        _draw_2d(
            ax2,
            right_frame,
            right_2d,
            edges,
            labels,
            args.show_joint_labels,
            "Right frame",
        )

        ax3 = fig.add_subplot(1, 4, 3, projection="3d")
        _draw_3d(ax3, left_3d, right_3d, p_hat, edges, labels, args.show_joint_labels)

        ax4 = fig.add_subplot(1, 4, 4)
        ax4.set_title("alpha per joint")
        ax4.bar(np.arange(len(alpha)), alpha, color="tab:green")
        ax4.set_xlabel("joint index")
        ax4.set_ylabel("alpha")
        ax4.set_ylim(0.0, 1.0)
        ax4.grid(True, alpha=0.25)

        fig.suptitle(
            f"run={args.run_id} frame={idx} | ckpt={ckpt_path.name} | "
            f"alpha_mean={diag['alpha_mean']:.3f} | hat_to(L/R)=({diag['l2_hat_to_left']:.3f}/{diag['l2_hat_to_right']:.3f})"
        )
        fig.tight_layout()

        png_path = out_dir / f"frame_{idx:06d}_vis.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)

        json_path = out_dir / f"frame_{idx:06d}_dual2pose.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": args.run_id,
                    "frame_index": int(idx),
                    "left_npz_path": str(left_map[idx]),
                    "right_npz_path": str(right_map[idx]),
                    "dual2pose_ckpt": str(ckpt_path),
                    "target_joint_ids": TARGET_IDS,
                    "left_kpt3d": left_3d.tolist(),
                    "right_kpt3d": right_3d.tolist(),
                    "p0": p0.tolist(),
                    "p_hat": p_hat.tolist(),
                    "alpha": alpha.tolist(),
                    "diagnostics": diag,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        records.append(
            {
                "frame_index": int(idx),
                "vis": str(png_path),
                "pred_json": str(json_path),
            }
        )

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": args.run_id,
                "dual2pose_ckpt": str(ckpt_path),
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
