#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Visualize real-world samples for paper figures.

Each output figure shows five panels for the same real sample frame:
  1. real frame
  2. left-side SAM3D
  3. right-side SAM3D
  4. average of left/right SAM3D
  5. proposed method output

The 3D poses are displayed in canonical coordinates so the comparison is direct.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent
for repo_path in (REPO_ROOT, REPO_ROOT / "dual2pose"):
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


DEFAULT_CKPT_UNITY = REPO_ROOT / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
DEFAULT_CKPT_SKI = REPO_ROOT / "logs/train_ski_poseptz/crossview_fusion/2026-05-25/14-13-23/checkpoints/last.ckpt"
DEFAULT_RUNS = ["pro_1", "pro_2", "run_3", "run_4", "run_5", "run_6"]

# For visualization, use the stable 13 body joints plus a synthetic head point.
# Raw eye joints are often noisy in Unity/real-world SAM3D outputs, so the head
# is approximated from neck and shoulder/body scale instead of drawing eyes.
VIS_15_TO_BODY_13 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
BONE_PAIRS = [
    (12, 13),
    (12, 0), (0, 2), (2, 11),
    (12, 1), (1, 3), (3, 10),
    (12, 4), (12, 5), (4, 5),
    (4, 6), (6, 8),
    (5, 7), (7, 9),
]


@lru_cache(maxsize=1)
def _symbols():
    fusion_mod = importlib.import_module("trainer.train_crossview_fusion")
    map_mod = importlib.import_module("map_config")
    canon_mod = importlib.import_module("trainer.canonicalize")
    return (
        fusion_mod.CrossViewFusionTrainer,
        map_mod.FILTERED_KPTS_MAPPING,
        map_mod.filter_sam3d_body_kpts,
        canon_mod.canonicalize_pose_torch,
    )


def _joint_names() -> list[str]:
    _, mapping, _, _ = _symbols()
    if isinstance(mapping, dict) and mapping:
        return [str(mapping[idx]) for idx in sorted(mapping)]
    return [f"joint_{idx}" for idx in range(15)]


def _filter_15(kp3d: np.ndarray) -> np.ndarray:
    _, _, filter_sam3d_body_kpts, _ = _symbols()
    try:
        return np.asarray(filter_sam3d_body_kpts(kp3d), dtype=np.float32)
    except Exception:
        n_joints = min(kp3d.shape[0], 15)
        out = np.zeros((15, 3), dtype=np.float32)
        out[:n_joints] = kp3d[:n_joints, :3]
        return out


def _frame_id_from_path(path: Path) -> int:
    for part in path.stem.split("_"):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot parse frame id from {path.name}")


def _index_npz_files(view_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(view_dir.glob("*.npz")):
        try:
            files[_frame_id_from_path(path)] = path
        except ValueError:
            continue
    return files


def _load_npz_kpt3d(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    for key in ("output", "outputs"):
        if key not in data:
            continue
        obj = data[key]
        if isinstance(obj, np.ndarray) and obj.ndim == 0:
            item = obj.item()
            if isinstance(item, dict) and "pred_keypoints_3d" in item:
                return np.asarray(item["pred_keypoints_3d"], dtype=np.float32)
        if isinstance(obj, np.ndarray):
            for item in obj.reshape(-1):
                if isinstance(item, dict) and "pred_keypoints_3d" in item:
                    return np.asarray(item["pred_keypoints_3d"], dtype=np.float32)
    for key in data.keys():
        value = data[key]
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3:
            return value.astype(np.float32)
    raise KeyError(f"No pred_keypoints_3d found in {npz_path}")


def _load_npz_frame(npz_path: Path) -> np.ndarray | None:
    data = np.load(npz_path, allow_pickle=True)
    for key in ("frame", "image", "img"):
        if key in data and isinstance(data[key], np.ndarray):
            return _normalize_image(data[key])
    for key in ("output", "outputs"):
        if key not in data:
            continue
        obj = data[key]
        items = [obj.item()] if isinstance(obj, np.ndarray) and obj.ndim == 0 else list(obj.reshape(-1))
        for item in items:
            if not isinstance(item, dict):
                continue
            for frame_key in ("frame", "image", "img"):
                if frame_key in item and isinstance(item[frame_key], np.ndarray):
                    return _normalize_image(item[frame_key])
    return None


def _normalize_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        max_value = float(np.nanmax(img)) if img.size else 1.0
        if max_value <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _load_model(ckpt_path: Path, backbone: str = "crossview_fusion"):
    CrossViewFusionTrainer, _, _, _ = _symbols()

    class AttrDict(dict):
        def __getattr__(self, key):
            return self.get(key)

        def __setattr__(self, key, value):
            self[key] = value

        def get(self, key, default=None):
            if "." not in str(key):
                return super().get(key, default)
            current: Any = self
            for part in str(key).split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current

    raw = OmegaConf.load(str(REPO_ROOT / "configs" / "dual2pose.yaml"))
    config = AttrDict(OmegaConf.to_container(raw, resolve=False))
    config["model"] = {"backbone": backbone}
    config["data"] = {
        "load_frames": False,
        "load_2d_kpt": False,
        "load_3d_kpt": True,
        "time_window": 30,
        "batch_size": 1,
        "num_workers": 0,
    }
    config["log_path"] = str(ckpt_path.parent / "paper_vis")

    model = CrossViewFusionTrainer(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _squeeze_to_time_pose(tensor: torch.Tensor) -> torch.Tensor:
    while tensor.ndim > 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected pose shape (T,J,3), got {tuple(tensor.shape)}")
    return tensor


def _canonicalize_sequence(model, pose_btj3: torch.Tensor) -> torch.Tensor:
    _, _, _, canonicalize_pose_torch = _symbols()
    canonical, _ = canonicalize_pose_torch(
        pose_btj3.squeeze(0),
        left_hip=model.left_hip_idx,
        right_hip=model.right_hip_idx,
        neck=model.neck_idx,
    )
    return _squeeze_to_time_pose(canonical)


def _run_model_on_window(model, left_seq: np.ndarray, right_seq: np.ndarray, device: str):
    left = torch.from_numpy(left_seq).float().unsqueeze(0).to(device)
    right = torch.from_numpy(right_seq).float().unsqueeze(0).to(device)
    left_canon = _canonicalize_sequence(model, left).to(device)
    right_canon = _canonicalize_sequence(model, right).to(device)
    with torch.no_grad():
        model_output = model.models(left_canon.unsqueeze(0), right_canon.unsqueeze(0))
    if isinstance(model_output, tuple):
        fused = model_output[0]
        aux = model_output[1] if len(model_output) > 1 and isinstance(model_output[1], dict) else {}
    else:
        fused = model_output
        aux = {}
    alpha = aux.get("alpha")
    fused = _squeeze_to_time_pose(fused)
    if isinstance(alpha, torch.Tensor):
        while alpha.ndim > 3 and alpha.shape[0] == 1:
            alpha = alpha.squeeze(0)
    return (
        left_canon.detach().cpu().numpy(),
        right_canon.detach().cpu().numpy(),
        fused.detach().cpu().numpy(),
        None if alpha is None else alpha.detach().cpu().numpy(),
    )


def _load_window(run_dir: Path, center_frame: int, time_window: int):
    left_files = _index_npz_files(run_dir / "left")
    right_files = _index_npz_files(run_dir / "right")
    shared = sorted(set(left_files) & set(right_files))
    if len(shared) < time_window:
        raise ValueError(f"{run_dir.name}: only {len(shared)} aligned frames, need {time_window}")

    if center_frame not in shared:
        center_frame = min(shared, key=lambda value: abs(value - center_frame))
    center_pos = shared.index(center_frame)
    start = max(0, min(center_pos - time_window // 2, len(shared) - time_window))
    frame_ids = shared[start : start + time_window]
    offset = frame_ids.index(center_frame)

    left_seq = np.stack([_filter_15(_load_npz_kpt3d(left_files[fid])) for fid in frame_ids], axis=0)
    right_seq = np.stack([_filter_15(_load_npz_kpt3d(right_files[fid])) for fid in frame_ids], axis=0)
    left_frame = _load_npz_frame(left_files[center_frame])
    right_frame = _load_npz_frame(right_files[center_frame])
    return frame_ids, offset, left_frame, right_frame, left_seq, right_seq


def _select_frames(run_dir: Path, count: int, requested: list[int] | None) -> list[int]:
    left_files = _index_npz_files(run_dir / "left")
    right_files = _index_npz_files(run_dir / "right")
    shared = sorted(set(left_files) & set(right_files))
    if requested:
        return [min(shared, key=lambda value: abs(value - frame_id)) for frame_id in requested]
    if count <= 1:
        return [shared[len(shared) // 2]]
    idx = np.linspace(0, len(shared) - 1, num=count, dtype=int)
    return [shared[int(i)] for i in idx]


def _pose_for_plot(pose: np.ndarray) -> np.ndarray:
    if pose.shape[0] == 15:
        body = pose[VIS_15_TO_BODY_13]
        neck = body[12]
        shoulder_mid = 0.5 * (body[0] + body[1])
        hip_mid = 0.5 * (body[4] + body[5])
        body_up = neck - hip_mid
        up_norm = np.linalg.norm(body_up)
        if up_norm < 1e-6:
            body_up = neck - shoulder_mid
            up_norm = np.linalg.norm(body_up)
        if up_norm < 1e-6:
            body_up = np.array([0.0, 1.0, 0.0], dtype=pose.dtype)
            up_norm = 1.0
        shoulder_width = np.linalg.norm(body[0] - body[1])
        head_len = max(0.18 * up_norm, 0.55 * shoulder_width, 0.08)
        synthetic_head = neck + body_up / up_norm * head_len
        pose = np.concatenate([body, synthetic_head[None, :]], axis=0)
    # Matplotlib's z axis is vertical. Canonical poses use y as the body-up
    # direction, so rotate axes for display only: (x, y, z) -> (x, z, y).
    return pose[:, [0, 2, 1]]


def _pose_limits(poses: list[np.ndarray]) -> tuple[np.ndarray, float]:
    points = np.concatenate([_pose_for_plot(pose).reshape(-1, 3) for pose in poses], axis=0)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return np.zeros(3), 1.0
    center = finite.mean(axis=0)
    radius = float(np.max(np.abs(finite - center)))
    return center, max(radius, 0.5)


def _draw_pose_3d(ax, pose: np.ndarray, label: str, color: str, alpha: float = 0.95) -> None:
    pose = _pose_for_plot(pose)
    for i, j in BONE_PAIRS:
        if i >= pose.shape[0] or j >= pose.shape[0]:
            continue
        ax.plot(
            [pose[i, 0], pose[j, 0]],
            [pose[i, 1], pose[j, 1]],
            [pose[i, 2], pose[j, 2]],
            color=color,
            linewidth=2.3,
            alpha=alpha,
        )
    ax.scatter(
        pose[:, 0],
        pose[:, 1],
        pose[:, 2],
        color=color,
        s=18,
        edgecolors="black",
        linewidths=0.35,
        alpha=alpha,
        label=label,
    )


def _style_pose_axis(ax, center: np.ndarray, radius: float) -> None:
    ax.set_title("3D Pose Comparison", fontsize=11, fontweight="bold", pad=3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=16, azim=-72)
    ticks = np.linspace(-radius, radius, 3)
    ax.set_xticks(center[0] + ticks)
    ax.set_yticks(center[1] + ticks)
    ax.set_zticks(center[2] + ticks)
    ax.set_xlabel("X", labelpad=-5)
    ax.set_ylabel("Z", labelpad=-5)
    ax.set_zlabel("Y", labelpad=-5)
    ax.tick_params(axis="both", which="major", labelsize=7, pad=-2)
    ax.set_box_aspect((1, 1, 1.15))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9, frameon=False)


def _draw_frame_panel(ax, frame: np.ndarray | None, title: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
    if frame is None:
        ax.set_facecolor("#111318")
        ax.text(0.5, 0.5, "frame unavailable", ha="center", va="center", color="white", fontsize=10)
    else:
        ax.imshow(frame)
    ax.axis("off")


def _make_figure(
    run_name: str,
    frame_id: int,
    ckpt_label: str,
    left_frame: np.ndarray | None,
    right_frame: np.ndarray | None,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    avg_pose: np.ndarray,
    fused_pose: np.ndarray,
    alpha_pose: np.ndarray | None,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(13.5, 6.2), dpi=180)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.06, 2.25], wspace=0.04, hspace=0.04)

    ax_left = fig.add_subplot(gs[0, 0])
    _draw_frame_panel(ax_left, left_frame, "Left Frame")

    ax_right = fig.add_subplot(gs[1, 0])
    _draw_frame_panel(ax_right, right_frame, "Right Frame")

    ax_pose = fig.add_subplot(gs[:, 2], projection="3d")
    center, radius = _pose_limits([left_pose, right_pose, avg_pose, fused_pose])
    pose_items = [
        (left_pose, "Left SAM3D", "#2f80ed", 0.78),
        (right_pose, "Right SAM3D", "#eb5757", 0.78),
        (avg_pose, "Left/Right Avg", "#7a7f87", 0.72),
        (fused_pose, "Proposed", "#219653", 1.0),
    ]
    for pose, label, color, alpha in pose_items:
        _draw_pose_3d(ax_pose, pose, label, color, alpha=alpha)
    _style_pose_axis(ax_pose, center, radius)
    fig.tight_layout(pad=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _resolve_ckpts(args) -> list[tuple[str, Path]]:
    if args.ckpt_path:
        return [(args.ckpt_label, args.ckpt_path)]
    ckpts: list[tuple[str, Path]] = []
    if DEFAULT_CKPT_UNITY.exists():
        ckpts.append(("unity", DEFAULT_CKPT_UNITY))
    if DEFAULT_CKPT_SKI.exists():
        ckpts.append(("ski_poseptz", DEFAULT_CKPT_SKI))
    return ckpts


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize real-world SAM3D samples with model outputs.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/kaixu_chen/data/skiing/sam3d_body_results/person"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/eval_realworld_direct/paper_vis"))
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--frames", nargs="*", type=int, default=None, help="Specific frame ids to visualize; nearest aligned frame is used.")
    parser.add_argument("--num-frames", type=int, default=3, help="Frames per run when --frames is not provided.")
    parser.add_argument("--time-window", type=int, default=30)
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--ckpt-label", type=str, default="custom")
    parser.add_argument("--backbone", type=str, default="crossview_fusion")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    device = "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{args.gpu}"
    ckpts = _resolve_ckpts(args)
    if not ckpts:
        raise RuntimeError("No checkpoint found. Pass --ckpt-path explicitly.")

    for ckpt_label, ckpt_path in ckpts:
        if not ckpt_path.exists():
            print(f"[SKIP] checkpoint not found: {ckpt_path}")
            continue
        print(f"Loading {ckpt_label}: {ckpt_path}")
        model = _load_model(ckpt_path, backbone=args.backbone).to(device)

        for run_name in args.runs:
            run_dir = args.data_root / run_name
            if not run_dir.exists():
                print(f"[SKIP] missing run: {run_dir}")
                continue
            try:
                frame_ids = _select_frames(run_dir, count=args.num_frames, requested=args.frames)
            except Exception as exc:
                print(f"[SKIP] {run_name}: {exc}")
                continue

            for frame_id in frame_ids:
                try:
                    window_ids, offset, left_frame, right_frame, left_seq, right_seq = _load_window(run_dir, frame_id, args.time_window)
                    model_result = _run_model_on_window(model, left_seq, right_seq, device=device)
                    if len(model_result) == 4:
                        left_canon, right_canon, fused, alpha = model_result
                    elif len(model_result) == 3:
                        left_canon, right_canon, fused = model_result
                        alpha = None
                    else:
                        raise ValueError(f"Unexpected model result length: {len(model_result)}")
                except Exception as exc:
                    print(f"[WARN] {run_name} frame {frame_id}: {exc}")
                    continue

                left_pose = left_canon[offset]
                right_pose = right_canon[offset]
                avg_pose = 0.5 * (left_pose + right_pose)
                fused_pose = fused[offset]
                alpha_pose = None if alpha is None else alpha[offset]
                actual_frame = window_ids[offset]
                output_path = args.output_dir / ckpt_label / f"{run_name}_frame{actual_frame:04d}_real_compare.png"
                _make_figure(
                    run_name=run_name,
                    frame_id=actual_frame,
                    ckpt_label=ckpt_label,
                    left_frame=left_frame,
                    right_frame=right_frame,
                    left_pose=left_pose,
                    right_pose=right_pose,
                    avg_pose=avg_pose,
                    fused_pose=fused_pose,
                    alpha_pose=alpha_pose,
                    output_path=output_path,
                )
                print(f"Generated: {output_path}")

    print(f"Done. Outputs are under: {args.output_dir}")


if __name__ == "__main__":
    main()
