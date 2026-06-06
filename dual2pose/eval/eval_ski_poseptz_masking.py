#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Evaluate Ski-PosePTZ checkpoints with masking sweeps.

The masking is applied to the input 3D keypoint streams before the trainer
sees them, mirroring the Unity masking evaluation style.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from torch.utils.data import DataLoader, default_collate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DUAL2POSE_ROOT = REPO_ROOT / "dual2pose"
if str(DUAL2POSE_ROOT) not in sys.path:
    sys.path.insert(0, str(DUAL2POSE_ROOT))


@lru_cache(maxsize=1)
def _repo_symbols():
    dataset_module = importlib.import_module(
        "dual2pose.dataloader.ski_poseptz_dataset_dual_view"
    )
    map_module = importlib.import_module("dual2pose.map_config")
    fusion_module = importlib.import_module("dual2pose.trainer.train_crossview_fusion")
    dual_module = importlib.import_module("dual2pose.trainer.train_dual2pose")
    return (
        dataset_module.LabeledSkiPosePTZDataset,
        map_module.filter_filtered_kpts_to_common,
        map_module.filter_h36m_kpts,
        fusion_module.CrossViewFusionTrainer,
        dual_module.Dual2PoseTrainer,
    )


DEFAULT_CKPT = Path(
    "/home/kaixu_chen/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_ski_poseptz/crossview_fusion/2026-05-25/14-13-23/checkpoints/last.ckpt"
)
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dual2pose.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained dual2pose/crossview_fusion checkpoint on Ski-PosePTZ with masking sweeps."
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=DEFAULT_CKPT,
        help="Checkpoint path to evaluate.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Config file used to build the data module and model.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="crossview_fusion",
        choices=["crossview_fusion", "dual2pose"],
        help="Model backbone to instantiate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Test batch size.")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker count."
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=30,
        help="Temporal window used by the dataset.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index to use when GPU is available.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument(
        "--mask-view-mode",
        type=str,
        default="none",
        choices=["none", "left", "right", "both"],
        help="Which view(s) to corrupt at test time.",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="random",
        choices=["random", "distal", "temporal"],
        help="Masking pattern for joint-frame occlusion.",
    )
    parser.add_argument(
        "--mask-ratio", type=float, default=0.0, help="Mask ratio for joint corruption."
    )
    parser.add_argument(
        "--mask-ratio-sweep",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of mask ratios to compare in one run, e.g. 0.1 0.2 0.3 0.4 0.5.",
    )
    parser.add_argument(
        "--mask-view-modes",
        nargs="*",
        type=str,
        default=None,
        choices=["left", "right", "both"],
        help="Optional list of view modes to compare in one run.",
    )
    parser.add_argument(
        "--mask-patterns",
        nargs="*",
        type=str,
        default=None,
        choices=["random", "distal", "temporal"],
        help="Optional list of masking patterns to compare in one run.",
    )
    parser.add_argument(
        "--mask-corruption",
        type=str,
        default="noise_masking",
        choices=["zero", "hold_last", "noise", "noise_masking"],
        help="How to corrupt masked keypoints.",
    )
    parser.add_argument(
        "--mask-temporal-span",
        type=int,
        default=10,
        help="Length of contiguous temporal masking segments.",
    )
    parser.add_argument(
        "--mask-noise-std",
        type=float,
        default=0.08,
        help="Noise std for masked joints.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="/home/kaixu_chen/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_ski_poseptz_masking",
        help="Optional base output path for this sweep. Overrides config.log_path.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace):
    config = OmegaConf.load(str(args.config_path))
    config.model.backbone = args.backbone
    config.data.left_hip_idx = 4
    config.data.right_hip_idx = 5
    config.data.neck_idx = 12
    config.data.load_frames = False
    config.data.load_2d_kpt = False
    config.data.load_3d_kpt = True
    config.data.time_window = int(args.time_window)
    config.data.batch_size = int(args.batch_size)
    config.data.num_workers = int(args.num_workers)
    ckpt_dir = args.ckpt_path.resolve().parent.parent
    config.log_path = str(ckpt_dir)
    return config


def _build_model(config):
    _, _, _, crossview_fusion_cls, dual2pose_cls = _repo_symbols()
    if config.model.backbone == "crossview_fusion":
        return crossview_fusion_cls(config)
    if config.model.backbone == "dual2pose":
        return dual2pose_cls(config)
    raise ValueError(f"Unsupported backbone: {config.model.backbone}")


def _build_mask_random(
    shape: torch.Size, ratio: float, device: torch.device
) -> torch.Tensor:
    b, t, j, _ = shape
    return (torch.rand((b, t, j), device=device) < ratio).unsqueeze(-1)


def _build_mask_distal(
    shape: torch.Size,
    ratio: float,
    device: torch.device,
    distal_joint_idx: tuple[int, ...],
) -> torch.Tensor:
    b, t, j, _ = shape
    mask = torch.zeros((b, t, j, 1), dtype=torch.bool, device=device)
    valid = [idx for idx in distal_joint_idx if 0 <= idx < j]
    if not valid:
        return _build_mask_random(shape=shape, ratio=ratio, device=device)
    distal_noise = torch.rand((b, t, len(valid)), device=device) < ratio
    mask[:, :, valid, 0] = distal_noise
    return mask


def _build_mask_temporal(
    shape: torch.Size, ratio: float, device: torch.device, temporal_span: int
) -> torch.Tensor:
    b, t, j, _ = shape
    span = max(1, min(int(temporal_span), t))
    joints_per_sample = max(1, int(round(ratio * j)))
    mask = torch.zeros((b, t, j, 1), dtype=torch.bool, device=device)
    for bi in range(b):
        joint_ids = torch.randperm(j, device=device)[:joints_per_sample]
        for jid in joint_ids:
            start = (
                0
                if t == span
                else int(torch.randint(0, t - span + 1, (1,), device=device).item())
            )
            mask[bi, start : start + span, int(jid.item()), 0] = True
    return mask


def _apply_hold_last(pose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = pose.clone()
    mask_btj = mask.squeeze(-1)
    out[:, 0] = torch.where(
        mask_btj[:, 0].unsqueeze(-1), torch.zeros_like(out[:, 0]), out[:, 0]
    )
    for ti in range(1, out.shape[1]):
        out[:, ti] = torch.where(
            mask_btj[:, ti].unsqueeze(-1), out[:, ti - 1], out[:, ti]
        )
    return out


def _apply_corruption(
    pose: torch.Tensor, mask: torch.Tensor, corruption: str, noise_std: float
) -> torch.Tensor:
    if mask.numel() == 0:
        return pose
    if corruption == "zero":
        return torch.where(mask, torch.zeros_like(pose), pose)
    if corruption == "hold_last":
        return _apply_hold_last(pose=pose, mask=mask)
    if corruption == "noise":
        noisy = pose + noise_std * torch.randn_like(pose)
        return torch.where(mask, noisy, pose)
    if corruption == "noise_masking":
        split = torch.rand(mask.shape, device=pose.device) < 0.35
        zero_mask = mask & split
        noise_mask = mask & (~split)
        noisy = pose + noise_std * torch.randn_like(pose)
        tmp = torch.where(noise_mask, noisy, pose)
        return torch.where(zero_mask, torch.zeros_like(tmp), tmp)
    raise ValueError(f"Unsupported corruption type: {corruption}")


def _collate_ski_poseptz_batch(batch, mask_cfg: Dict[str, Any] | None = None):
    _, filter_filtered_kpts_to_common, filter_h36m_kpts, _, _ = _repo_symbols()
    collated = default_collate(batch)
    mask_cfg = mask_cfg or {}
    _masks: dict[str, torch.Tensor] = {}

    def _filter_pose_tensor_batch(pose_batch: torch.Tensor, filter_fn):
        flat_batch = pose_batch.reshape(-1, pose_batch.shape[-2], pose_batch.shape[-1])
        filtered = torch.stack(
            [
                torch.from_numpy(filter_fn(sample.cpu().numpy()))
                for sample in flat_batch
            ],
            dim=0,
        )
        return filtered.view(*pose_batch.shape[:-2], -1, pose_batch.shape[-1])

    def _apply_mask_to_pose_tensor_batch(pose_batch: torch.Tensor, view_key: str):
        # masks collected per view_key
        nonlocal _masks
        mask_view_mode = str(mask_cfg.get("mask_view_mode", "none"))
        mask_ratio = float(mask_cfg.get("mask_ratio", 0.0))
        b, t, j, _ = pose_batch.shape
        if mask_view_mode == "none" or mask_ratio <= 0.0:
            # no mask: create explicit false mask for downstream
            mask_zero = torch.zeros(
                (b, t, j, 1), dtype=torch.bool, device=pose_batch.device
            )
            try:
                _masks[view_key] = mask_zero
            except NameError:
                _masks = {view_key: mask_zero}
            return pose_batch
        if mask_view_mode == "left" and view_key != "cam1":
            mask_zero = torch.zeros(
                (b, t, j, 1), dtype=torch.bool, device=pose_batch.device
            )
            try:
                _masks[view_key] = mask_zero
            except NameError:
                _masks = {view_key: mask_zero}
            return pose_batch
        if mask_view_mode == "right" and view_key != "cam2":
            mask_zero = torch.zeros(
                (b, t, j, 1), dtype=torch.bool, device=pose_batch.device
            )
            try:
                _masks[view_key] = mask_zero
            except NameError:
                _masks = {view_key: mask_zero}
            return pose_batch

        mask_pattern = str(mask_cfg.get("mask_pattern", "random"))
        if mask_pattern == "random":
            mask = _build_mask_random(pose_batch.shape, mask_ratio, pose_batch.device)
        elif mask_pattern == "distal":
            mask = _build_mask_distal(
                pose_batch.shape,
                mask_ratio,
                pose_batch.device,
                (2, 3, 5, 6, 8, 9, 11, 12),
            )
        elif mask_pattern == "temporal":
            mask = _build_mask_temporal(
                pose_batch.shape,
                mask_ratio,
                pose_batch.device,
                int(mask_cfg.get("mask_temporal_span", 10)),
            )
        else:
            raise ValueError(f"Unsupported mask_pattern: {mask_pattern}")

        try:
            _masks[view_key] = mask
        except NameError:
            _masks = {view_key: mask}
        return _apply_corruption(
            pose_batch,
            mask,
            str(mask_cfg.get("mask_corruption", "noise_masking")),
            float(mask_cfg.get("mask_noise_std", 0.08)),
        )

    if "kpt3d" in collated:
        kpt3d_sam = collated.pop("kpt3d")
        collated["kpt3d_sam"] = {
            cam_name: _apply_mask_to_pose_tensor_batch(
                _filter_pose_tensor_batch(cam_pose, filter_filtered_kpts_to_common),
                cam_name,
            )
            for cam_name, cam_pose in kpt3d_sam.items()
        }
        # attach masks collected during masking
        try:
            collated["kpt3d_mask"] = _masks
        except NameError:
            collated["kpt3d_mask"] = {}

    if "gt_kpt3d" in collated:
        gt = collated.pop("gt_kpt3d")
        if gt.ndim == 4 and gt.shape[-2] == 17:
            gt = _filter_pose_tensor_batch(gt, filter_h36m_kpts)
        collated["kpt3d_gt"] = gt
    return collated


def _safe_mpjpe(pred: torch.Tensor | None, label: torch.Tensor | None) -> float:
    if pred is None or label is None or pred.numel() == 0 or label.numel() == 0:
        return float("nan")
    return float(torch.norm(pred - label, dim=-1).mean().item())


def _safe_accel_err(pred: torch.Tensor | None, label: torch.Tensor | None) -> float:
    if pred is None or label is None or pred.shape[1] < 3 or label.shape[1] < 3:
        return float("nan")
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    label_acc = label[:, 2:] - 2.0 * label[:, 1:-1] + label[:, :-2]
    return float(torch.norm(pred_acc - label_acc, dim=-1).mean().item())


def _mean_ignore_nan(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def _summarize_test_outputs(
    test_outputs: list[Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, list[float]]] = {
        "fused": {"mpjpe": [], "accel_err": []},
        "sam3d_left": {"mpjpe": [], "accel_err": []},
        "sam3d_right": {"mpjpe": [], "accel_err": []},
        "raw_avg": {"mpjpe": [], "accel_err": []},
        "left_canonical": {"mpjpe": [], "accel_err": []},
        "right_canonical": {"mpjpe": [], "accel_err": []},
        "canonical_avg": {"mpjpe": [], "accel_err": []},
    }

    for batch_output in test_outputs:
        fused = batch_output.get("fused")
        p_left = batch_output.get("p_left")
        p_right = batch_output.get("p_right")
        left_canonical = batch_output.get("left_canonical")
        right_canonical = batch_output.get("right_canonical")
        ground_truth = batch_output.get("ground_truth")
        ground_truth_canonical = batch_output.get("ground_truth_canonical")

        raw_avg = (
            None if p_left is None or p_right is None else 0.5 * (p_left + p_right)
        )
        canonical_avg = (
            None
            if left_canonical is None or right_canonical is None
            else 0.5 * (left_canonical + right_canonical)
        )

        pairs = {
            "fused": (fused, ground_truth_canonical),
            "sam3d_left": (p_left, ground_truth),
            "sam3d_right": (p_right, ground_truth),
            "raw_avg": (raw_avg, ground_truth),
            "left_canonical": (left_canonical, ground_truth_canonical),
            "right_canonical": (right_canonical, ground_truth_canonical),
            "canonical_avg": (canonical_avg, ground_truth_canonical),
        }

        for name, (pred, label) in pairs.items():
            stats[name]["mpjpe"].append(_safe_mpjpe(pred, label))
            stats[name]["accel_err"].append(_safe_accel_err(pred, label))

    return {
        name: {
            "mpjpe": _mean_ignore_nan(values["mpjpe"]),
            "accel_err": _mean_ignore_nan(values["accel_err"]),
        }
        for name, values in stats.items()
    }


def _format_float(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def _collect_alpha_tensor(
    test_outputs: list[Dict[str, torch.Tensor]],
) -> torch.Tensor | None:
    alpha_chunks: list[torch.Tensor] = []
    for batch_output in test_outputs:
        alpha = batch_output.get("alpha")
        if isinstance(alpha, torch.Tensor) and alpha.numel() > 0:
            alpha_chunks.append(alpha.detach().cpu())
    if not alpha_chunks:
        return None
    return torch.cat(alpha_chunks, dim=0)


def _alpha_joint_names() -> list[str]:
    map_module = importlib.import_module("dual2pose.map_config")
    common_mapping = getattr(map_module, "SKI_PTZ_COMMON_KPTS_MAPPING", None)
    if isinstance(common_mapping, dict) and common_mapping:
        return [common_mapping[idx] for idx in sorted(common_mapping)]
    return []


def _export_alpha_visualization(
    test_outputs: list[Dict[str, torch.Tensor]],
    save_dir: Path,
    setting_name: str,
) -> None:
    alpha = _collect_alpha_tensor(test_outputs)
    if alpha is None or alpha.ndim != 4:
        return

    alpha_mean_joint = alpha.mean(dim=(0, 1, 3)).numpy()
    alpha_std_joint = alpha.std(dim=(0, 1, 3)).numpy()
    alpha_mean_time_joint = alpha.mean(dim=0).squeeze(-1).numpy()
    alpha_right_mean_joint = 1.0 - alpha_mean_joint

    joint_names = _alpha_joint_names()
    if len(joint_names) != alpha_mean_joint.shape[0]:
        joint_names = [f"joint_{idx:02d}" for idx in range(alpha_mean_joint.shape[0])]

    vis_dir = save_dir / "alpha_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = vis_dir / f"alpha_summary_{setting_name}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "joint_index",
                "joint_name",
                "alpha_mean",
                "alpha_std",
                "right_mean",
            ],
        )
        writer.writeheader()
        for idx, joint_name in enumerate(joint_names):
            writer.writerow(
                {
                    "joint_index": idx,
                    "joint_name": joint_name,
                    "alpha_mean": float(alpha_mean_joint[idx]),
                    "alpha_std": float(alpha_std_joint[idx]),
                    "right_mean": float(alpha_right_mean_joint[idx]),
                }
            )

    x = np.arange(len(joint_names))
    fig, ax = plt.subplots(figsize=(max(10, len(joint_names) * 0.75), 4.8))
    ax.bar(
        x,
        alpha_mean_joint,
        yerr=alpha_std_joint,
        color="tab:green",
        alpha=0.9,
        capsize=3,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("alpha (left-view weight)")
    ax.set_title(f"Per-joint fusion ratio: {setting_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    bar_path = vis_dir / f"alpha_joint_bar_{setting_name}.png"
    fig.savefig(bar_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(joint_names) * 0.75), 6.0))
    im = ax.imshow(
        alpha_mean_time_joint, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis"
    )
    ax.set_title(f"Alpha heatmap over time and joints: {setting_name}")
    ax.set_xlabel("joint")
    ax.set_ylabel("time")
    ax.set_xticks(np.arange(len(joint_names)))
    ax.set_xticklabels(joint_names, rotation=35, ha="right")
    y_tick_count = min(alpha_mean_time_joint.shape[0], 8)
    y_ticks = np.linspace(
        0, alpha_mean_time_joint.shape[0] - 1, num=y_tick_count, dtype=int
    )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(v)) for v in y_ticks])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("alpha")
    fig.tight_layout()
    heatmap_path = vis_dir / f"alpha_time_joint_heatmap_{setting_name}.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)


def main() -> None:
    LabeledSkiPosePTZDataset, _, _, _, _ = _repo_symbols()
    args = _parse_args()
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")

    seed_everything(42, workers=True)
    config = _build_config(args)
    base_log_path = (
        Path(args.output_path)
        if getattr(args, "output_path", None)
        else Path(config.log_path)
    )

    test_dataset = LabeledSkiPosePTZDataset(
        index_mapping=Path(config.data.ski_pose_ptz.index_mapping_path),
        transform=None,
        load_frames=False,
        load_2d_kpt=False,
        load_3d_kpt=True,
        target_t=int(args.time_window),
        split=args.split,
    )
    ratio_list = (
        args.mask_ratio_sweep
        if args.mask_ratio_sweep
        else ([args.mask_ratio] if args.mask_ratio > 0 else [0.1, 0.2, 0.3, 0.4, 0.5])
    )
    view_mode_list = (
        args.mask_view_modes
        if args.mask_view_modes
        else ["none", "left", "right", "both"]
    )
    pattern_list = (
        args.mask_patterns if args.mask_patterns else ["random", "distal", "temporal"]
    )

    use_gpu = torch.cuda.is_available() and not args.cpu
    summary_rows = []

    for mask_view_mode in view_mode_list:
        for mask_pattern in pattern_list:
            for mask_ratio in ratio_list:
                setting_name = (
                    f"{mask_view_mode}_{mask_pattern}_r{mask_ratio:.2f}".replace(
                        ".", "p"
                    )
                )
                run_mask_cfg = {
                    "mask_view_mode": mask_view_mode,
                    "mask_pattern": mask_pattern,
                    "mask_ratio": float(mask_ratio),
                    "mask_corruption": args.mask_corruption,
                    "mask_temporal_span": args.mask_temporal_span,
                    "mask_noise_std": args.mask_noise_std,
                }
                config.log_path = str(base_log_path / setting_name)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    drop_last=False,
                    num_workers=int(args.num_workers),
                    pin_memory=not args.cpu,
                    collate_fn=lambda batch, cfg=run_mask_cfg: (
                        _collate_ski_poseptz_batch(batch, mask_cfg=cfg)
                    ),
                )
                model = _build_model(config)
                trainer = Trainer(
                    accelerator="gpu" if use_gpu else "cpu",
                    devices=[int(args.gpu)] if use_gpu else 1,
                    logger=False,
                    enable_checkpointing=False,
                    callbacks=[RichProgressBar(refresh_rate=10, leave=True)],
                )

                metrics = trainer.test(
                    model,
                    dataloaders=test_loader,
                    ckpt_path=str(args.ckpt_path),
                    verbose=True,
                    weights_only=False,
                )

                flat_metrics = metrics[0] if metrics else {}
                fused_mpjpe = flat_metrics.get(
                    "test/mpjpe", flat_metrics.get("mpjpe", float("nan"))
                )
                accel_err = flat_metrics.get(
                    "test/accel_err", flat_metrics.get("accel_err", float("nan"))
                )
                baseline_metrics = _summarize_test_outputs(
                    list(getattr(model, "test_outputs", []))
                )
                alpha_tensor = _collect_alpha_tensor(
                    list(getattr(model, "test_outputs", []))
                )
                alpha_global_mean = (
                    float(alpha_tensor.mean().item())
                    if isinstance(alpha_tensor, torch.Tensor)
                    else float("nan")
                )
                alpha_global_std = (
                    float(alpha_tensor.std().item())
                    if isinstance(alpha_tensor, torch.Tensor)
                    else float("nan")
                )
                _export_alpha_visualization(
                    list(getattr(model, "test_outputs", [])),
                    Path(config.log_path),
                    setting_name,
                )
                # Export alpha statistics restricted to masked joint-frames (if masks were provided)
                all_outputs = list(getattr(model, "test_outputs", []))
                alphas = [
                    o.get("alpha") for o in all_outputs if o.get("alpha") is not None
                ]
                masks_left = [
                    o.get("mask_left")
                    for o in all_outputs
                    if o.get("mask_left") is not None
                ]
                masks_right = [
                    o.get("mask_right")
                    for o in all_outputs
                    if o.get("mask_right") is not None
                ]
                if alphas:
                    alpha_cat = torch.cat([a.detach().cpu() for a in alphas], dim=0)
                    alpha_squeezed = alpha_cat.squeeze(-1)  # shape (N, T, J)
                    joint_names = _alpha_joint_names()
                    if len(joint_names) != alpha_squeezed.shape[-1]:
                        joint_names = [
                            f"joint_{idx:02d}"
                            for idx in range(alpha_squeezed.shape[-1])
                        ]

                    vis_dir = Path(config.log_path) / "alpha_vis"
                    vis_dir.mkdir(parents=True, exist_ok=True)

                    def _compute_masked_stats(alpha_tensor, mask_chunks):
                        if not mask_chunks:
                            return None
                        mask_cat = (
                            torch.cat([m.detach().cpu() for m in mask_chunks], dim=0)
                            .squeeze(-1)
                            .bool()
                        )  # (N, T, J)
                        if mask_cat.numel() == 0:
                            return None
                        A = alpha_tensor.reshape(-1, alpha_tensor.shape[-1])
                        M = mask_cat.reshape(-1, mask_cat.shape[-1])
                        means = []
                        stds = []
                        counts = []
                        for j in range(A.shape[1]):
                            sel = M[:, j]
                            vals = A[sel, j]
                            if vals.numel() == 0:
                                means.append(float("nan"))
                                stds.append(float("nan"))
                                counts.append(0)
                            else:
                                means.append(float(vals.mean().item()))
                                stds.append(float(vals.std().item()))
                                counts.append(int(vals.numel()))
                        return means, stds, counts

                    left_stats = _compute_masked_stats(alpha_squeezed, masks_left)
                    right_stats = _compute_masked_stats(alpha_squeezed, masks_right)

                    masked_csv = vis_dir / f"alpha_masked_stats_{setting_name}.csv"
                    with open(masked_csv, "w", encoding="utf-8", newline="") as mf:
                        mw = csv.writer(mf)
                        hdr = ["joint_index", "joint_name"]
                        hdr += [
                            "left_masked_mean",
                            "left_masked_std",
                            "left_masked_count",
                        ]
                        hdr += [
                            "right_masked_mean",
                            "right_masked_std",
                            "right_masked_count",
                        ]
                        mw.writerow(hdr)
                        J = alpha_squeezed.shape[-1]
                        for j in range(J):
                            row = [
                                j,
                                joint_names[j]
                                if j < len(joint_names)
                                else f"joint_{j:02d}",
                            ]
                            if left_stats is not None:
                                lmean, lstd, lcnt = (
                                    left_stats[0][j],
                                    left_stats[1][j],
                                    left_stats[2][j],
                                )
                            else:
                                lmean, lstd, lcnt = float("nan"), float("nan"), 0
                            if right_stats is not None:
                                rmean, rstd, rcnt = (
                                    right_stats[0][j],
                                    right_stats[1][j],
                                    right_stats[2][j],
                                )
                            else:
                                rmean, rstd, rcnt = float("nan"), float("nan"), 0
                            row += [lmean, lstd, lcnt, rmean, rstd, rcnt]
                            mw.writerow(row)
                canonical_avg_mpjpe = baseline_metrics.get("canonical_avg", {}).get(
                    "mpjpe", float("nan")
                )
                canonical_avg_accel = baseline_metrics.get("canonical_avg", {}).get(
                    "accel_err", float("nan")
                )
                raw_avg_mpjpe = baseline_metrics.get("raw_avg", {}).get(
                    "mpjpe", float("nan")
                )
                raw_avg_accel = baseline_metrics.get("raw_avg", {}).get(
                    "accel_err", float("nan")
                )

                summary_rows.append(
                    {
                        "setting": setting_name,
                        "mask_view_mode": mask_view_mode,
                        "mask_pattern": mask_pattern,
                        "mask_ratio": float(mask_ratio),
                        "alpha_global_mean": alpha_global_mean,
                        "alpha_global_std": alpha_global_std,
                        "mpjpe": fused_mpjpe,
                        "accel_err": accel_err,
                        "canonical_avg_mpjpe": canonical_avg_mpjpe,
                        "raw_avg_mpjpe": raw_avg_mpjpe,
                        "delta_mpjpe_full_minus_canonical_avg": fused_mpjpe
                        - canonical_avg_mpjpe,
                        "delta_mpjpe_full_minus_raw_avg": fused_mpjpe - raw_avg_mpjpe,
                        "canonical_avg_accel_err": canonical_avg_accel,
                        "raw_avg_accel_err": raw_avg_accel,
                        "delta_accel_full_minus_canonical_avg": accel_err
                        - canonical_avg_accel,
                        "delta_accel_full_minus_raw_avg": accel_err - raw_avg_accel,
                    }
                )

                setting_dir = Path(config.log_path)
                setting_dir.mkdir(parents=True, exist_ok=True)
                run_tag = (
                    args.ckpt_path.stem if getattr(args, "ckpt_path", None) else "run"
                )
                report_path = (
                    setting_dir / f"comparison_report_{setting_name}_{run_tag}.txt"
                )
                with open(report_path, "w", encoding="utf-8") as fp:
                    fp.write(f"Setting: {setting_name}\n")
                    fp.write(f"Mask view mode: {mask_view_mode}\n")
                    fp.write(f"Mask pattern: {mask_pattern}\n")
                    fp.write(f"Mask ratio: {mask_ratio:.2f}\n")
                    fp.write(f"alpha_global_mean: {_format_float(alpha_global_mean)}\n")
                    fp.write(f"alpha_global_std: {_format_float(alpha_global_std)}\n")
                    fp.write("\n[Primary]\n")
                    fp.write(f"fused_mpjpe: {_format_float(fused_mpjpe)}\n")
                    fp.write(f"fused_accel_err: {_format_float(accel_err)}\n")
                    fp.write("\n[Baselines]\n")
                    fp.write(
                        f"canonical_avg_mpjpe: {_format_float(canonical_avg_mpjpe)}\n"
                    )
                    fp.write(f"raw_avg_mpjpe: {_format_float(raw_avg_mpjpe)}\n")
                    fp.write(
                        f"delta_mpjpe_full_minus_canonical_avg: {_format_float(fused_mpjpe - canonical_avg_mpjpe)}\n"
                    )
                    fp.write(
                        f"delta_mpjpe_full_minus_raw_avg: {_format_float(fused_mpjpe - raw_avg_mpjpe)}\n"
                    )
                    fp.write(
                        f"canonical_avg_accel_err: {_format_float(canonical_avg_accel)}\n"
                    )
                    fp.write(f"raw_avg_accel_err: {_format_float(raw_avg_accel)}\n")
                    fp.write(
                        f"delta_accel_full_minus_canonical_avg: {_format_float(accel_err - canonical_avg_accel)}\n"
                    )
                    fp.write(
                        f"delta_accel_full_minus_raw_avg: {_format_float(accel_err - raw_avg_accel)}\n"
                    )

                print(
                    f"=== {mask_view_mode} / {mask_pattern} / ratio {mask_ratio:.2f} ==="
                )
                print(f"alpha_global_mean: {_format_float(alpha_global_mean)}")
                print(f"fused_mpjpe: {_format_float(fused_mpjpe)}")
                print(f"canonical_avg_mpjpe: {_format_float(canonical_avg_mpjpe)}")
                print(f"raw_avg_mpjpe: {_format_float(raw_avg_mpjpe)}")
                print(
                    f"delta_mpjpe_full_minus_canonical_avg: {_format_float(fused_mpjpe - canonical_avg_mpjpe)}"
                )
                print(f"fused_accel_err: {_format_float(accel_err)}")
                print(f"canonical_avg_accel_err: {_format_float(canonical_avg_accel)}")
                print(f"raw_avg_accel_err: {_format_float(raw_avg_accel)}")
                print(
                    f"delta_accel_full_minus_canonical_avg: {_format_float(accel_err - canonical_avg_accel)}"
                )

    summary_dir = base_log_path / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.ckpt_path.stem if getattr(args, "ckpt_path", None) else "run"
    summary_csv = summary_dir / f"mask_ratio_sweep_{run_tag}.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "setting",
                "mask_view_mode",
                "mask_pattern",
                "mask_ratio",
                "alpha_global_mean",
                "alpha_global_std",
                "mpjpe",
                "accel_err",
                "canonical_avg_mpjpe",
                "raw_avg_mpjpe",
                "delta_mpjpe_full_minus_canonical_avg",
                "delta_mpjpe_full_minus_raw_avg",
                "canonical_avg_accel_err",
                "raw_avg_accel_err",
                "delta_accel_full_minus_canonical_avg",
                "delta_accel_full_minus_raw_avg",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Sweep summary saved to: {summary_csv}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
