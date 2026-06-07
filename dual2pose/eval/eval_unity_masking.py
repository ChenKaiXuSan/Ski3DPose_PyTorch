#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Occlusion-style robustness evaluation on Unity dual-view 3D keypoints.

This script does not mask RGB frames. Instead, it corrupts input 3D keypoint
streams (kpt3d_sam) at test time to simulate occlusion/missing/low-confidence
observations, then evaluates the fusion model robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import importlib
import logging
import math
import os
from pathlib import Path
import sys
from functools import lru_cache
from typing import Any, Dict, Iterable, List, cast

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from torch.utils.data import DataLoader, default_collate

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@lru_cache(maxsize=1)
def _repo_symbols():
    data_module = importlib.import_module("dataloader.data_loader")
    fusion_module = importlib.import_module("trainer.train_crossview_fusion")
    dual_module = importlib.import_module("trainer.train_dual2pose")
    return (
        data_module.UnityDataModule,
        fusion_module.CrossViewFusionTrainer,
        dual_module.Dual2PoseTrainer,
    )


UnityDataModule, CrossViewFusionTrainer, Dual2PoseTrainer = _repo_symbols()

logger = logging.getLogger(__name__)

DEFAULT_CKPT_PATH = "/home/kaixu_chen/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/crossview_fusion/2026-05-14/02-46-56/checkpoints/fold_0/last.ckpt"


@dataclass(frozen=True)
class OcclusionSetting:
    name: str
    view_mode: str  # none | left | right | both
    pattern: str  # random | distal | temporal
    ratio: float
    corruption: str  # zero | hold_last | noise | noise_masking
    temporal_span: int = 10
    noise_std: float = 0.08


def _clone_batch_for_edit(batch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(batch)
    if isinstance(batch.get("kpt3d_sam"), dict):
        out["kpt3d_sam"] = dict(batch["kpt3d_sam"])
    return out


def _build_mask_random(
    shape: torch.Size,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    b, t, j, _ = shape
    return (torch.rand((b, t, j), device=device) < ratio).unsqueeze(-1)


def _build_mask_distal(
    shape: torch.Size,
    ratio: float,
    device: torch.device,
    distal_joint_idx: Iterable[int],
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
    shape: torch.Size,
    ratio: float,
    device: torch.device,
    temporal_span: int,
) -> torch.Tensor:
    b, t, j, _ = shape
    span = max(1, min(int(temporal_span), t))
    joints_per_sample = max(1, int(round(ratio * j)))
    mask = torch.zeros((b, t, j, 1), dtype=torch.bool, device=device)
    for bi in range(b):
        joint_ids = torch.randperm(j, device=device)[:joints_per_sample]
        for jid in joint_ids:
            if t == span:
                start = 0
            else:
                start = int(torch.randint(0, t - span + 1, (1,), device=device).item())
            mask[bi, start : start + span, int(jid.item()), 0] = True
    return mask


def _apply_hold_last(
    pose: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
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
    pose: torch.Tensor,
    mask: torch.Tensor,
    corruption: str,
    noise_std: float,
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


def _apply_occlusion_to_batch(
    batch: Dict[str, Any],
    setting: OcclusionSetting,
) -> Dict[str, Any]:
    if setting.view_mode == "none" or setting.ratio <= 0.0:
        return batch

    if "kpt3d_sam" not in batch or not isinstance(batch["kpt3d_sam"], dict):
        return batch

    out = _clone_batch_for_edit(batch)
    out.setdefault("_occlusion", {})
    out["_occlusion"]["setting"] = setting.name
    out["_occlusion"]["view_mode"] = setting.view_mode
    out["_occlusion"]["pattern"] = setting.pattern
    out["_occlusion"]["ratio"] = float(setting.ratio)
    out["_occlusion"]["corruption"] = setting.corruption

    def _corrupt_view(view_key: str) -> None:
        pose = out["kpt3d_sam"].get(view_key)
        if pose is None:
            return
        if pose.ndim != 4:
            raise ValueError(f"Expected pose shape (B,T,J,3), got {tuple(pose.shape)}")

        if setting.pattern == "random":
            mask = _build_mask_random(
                shape=pose.shape, ratio=setting.ratio, device=pose.device
            )
        elif setting.pattern == "distal":
            mask = _build_mask_distal(
                shape=pose.shape,
                ratio=setting.ratio,
                device=pose.device,
                distal_joint_idx=[2, 3, 5, 6, 8, 9, 11, 12],
            )
        elif setting.pattern == "temporal":
            mask = _build_mask_temporal(
                shape=pose.shape,
                ratio=setting.ratio,
                device=pose.device,
                temporal_span=setting.temporal_span,
            )
        else:
            raise ValueError(f"Unsupported pattern: {setting.pattern}")

        out["kpt3d_sam"][view_key] = _apply_corruption(
            pose=pose,
            mask=mask,
            corruption=setting.corruption,
            noise_std=setting.noise_std,
        )
        out["_occlusion"][f"{view_key}_mask_ratio_real"] = float(
            mask.float().mean().item()
        )

    if setting.view_mode in ("left", "both"):
        _corrupt_view("cam1")
    if setting.view_mode in ("right", "both"):
        _corrupt_view("cam2")

    return out


class MaskedUnityDataModule(LightningDataModule):
    def __init__(self, base_dm: Any, setting: OcclusionSetting) -> None:
        super().__init__()
        self.base_dm = base_dm
        self.setting = setting

    def prepare_data(self) -> None:
        self.base_dm.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.base_dm.setup(stage)

    def test_dataloader(self) -> DataLoader:
        base_loader = self.base_dm.test_dataloader()

        def _masked_collate(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
            collated = default_collate(batch_items)
            return _apply_occlusion_to_batch(collated, self.setting)

        return DataLoader(
            dataset=base_loader.dataset,
            batch_size=base_loader.batch_size,
            num_workers=base_loader.num_workers,
            pin_memory=base_loader.pin_memory,
            drop_last=base_loader.drop_last,
            shuffle=False,
            collate_fn=_masked_collate,
        )


def _metric_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(torch.norm(pred - gt, dim=-1).mean().item())


def _metric_per_joint_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> List[float]:
    return torch.norm(pred - gt, dim=-1).mean(dim=(0, 1)).tolist()


def _metric_velocity_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if pred.shape[1] < 2:
        return float("nan")
    vel_pred = pred[:, 1:] - pred[:, :-1]
    vel_gt = gt[:, 1:] - gt[:, :-1]
    return float(torch.norm(vel_pred - vel_gt, dim=-1).mean().item())


def _metric_acceleration_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if pred.shape[1] < 3:
        return float("nan")
    acc_pred = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    acc_gt = gt[:, 2:] - 2.0 * gt[:, 1:-1] + gt[:, :-2]
    return float(torch.norm(acc_pred - acc_gt, dim=-1).mean().item())


def _metric_jitter(pred: torch.Tensor) -> float:
    if pred.shape[1] < 3:
        return float("nan")
    acc_pred = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    return float(torch.norm(acc_pred, dim=-1).mean().item())


def _metric_failure_rate(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float
) -> float:
    frame_mpjpe = torch.norm(pred - gt, dim=-1).mean(dim=-1)
    return float((frame_mpjpe > threshold).float().mean().item())


def _flatten_test_outputs(
    test_outputs: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    keys = {
        "fused": [],
        "left_raw": [],
        "right_raw": [],
        "left_canonical": [],
        "right_canonical": [],
        "ground_truth": [],
        "ground_truth_canonical": [],
    }

    for output in test_outputs:
        if (
            output.get("ground_truth") is None
            or output.get("ground_truth_canonical") is None
        ):
            continue
        keys["fused"].append(output["fused"].detach().cpu())
        keys["left_raw"].append(output["p_left"].detach().cpu())
        keys["right_raw"].append(output["p_right"].detach().cpu())
        keys["left_canonical"].append(output["left_canonical"].detach().cpu())
        keys["right_canonical"].append(output["right_canonical"].detach().cpu())
        keys["ground_truth"].append(output["ground_truth"].detach().cpu())
        keys["ground_truth_canonical"].append(
            output["ground_truth_canonical"].detach().cpu()
        )

    return {k: torch.cat(v, dim=0) for k, v in keys.items() if v}


def _summarize_outputs(
    flat: Dict[str, torch.Tensor],
    failure_threshold: float,
) -> Dict[str, Any]:
    if not flat:
        return {"error": "No valid test outputs with ground truth."}

    pred_gt_pairs = {
        "fused": (flat["fused"], flat["ground_truth_canonical"]),
        "canonical_avg": (
            0.5 * (flat["left_canonical"] + flat["right_canonical"]),
            flat["ground_truth_canonical"],
        ),
        "left_canonical": (flat["left_canonical"], flat["ground_truth_canonical"]),
        "right_canonical": (flat["right_canonical"], flat["ground_truth_canonical"]),
        "left_raw": (flat["left_raw"], flat["ground_truth"]),
        "right_raw": (flat["right_raw"], flat["ground_truth"]),
        "raw_avg": (
            0.5 * (flat["left_raw"] + flat["right_raw"]),
            flat["ground_truth"],
        ),
    }

    summary: Dict[str, Any] = {}
    for name, (pred, gt) in pred_gt_pairs.items():
        summary[name] = {
            "mpjpe": _metric_mpjpe(pred, gt),
            "per_joint_mpjpe": _metric_per_joint_mpjpe(pred, gt),
            "velocity_error": _metric_velocity_error(pred, gt),
            "acceleration_error": _metric_acceleration_error(pred, gt),
            "jitter": _metric_jitter(pred),
            "failure_rate": _metric_failure_rate(pred, gt, threshold=failure_threshold),
        }
    return summary


def _build_default_study() -> List[OcclusionSetting]:
    settings: List[OcclusionSetting] = []

    # Full sweep: 4 view_modes (none/left/right/both) x 3 patterns x ratio 0-1 step 0.05
    ratios = [
        i for i in np.arange(0.0, 1.05, 0.05)
    ]  # ~21 points: 0.0 -- 1.0
    view_modes = ["none", "left", "right", "both"]
    patterns = ["random", "distal", "temporal"]
    for view_mode in view_modes:
        ratios_for_view = [0.0] if view_mode == "none" else ratios
        for pattern in patterns:
            for ratio in ratios_for_view:
                r_str = str(ratio).replace(".", "p")
                settings.append(
                    OcclusionSetting(
                        name=f"{view_mode}_{pattern}_r{r_str}",
                        view_mode=view_mode,
                        pattern=pattern,
                        ratio=ratio,
                        corruption="noise_masking",
                        temporal_span=10 if pattern == "temporal" else None,
                    )
                )
    return settings


def _build_model(config: DictConfig):
    if config.model.backbone == "dual2pose":
        return Dual2PoseTrainer(config)
    if config.model.backbone == "crossview_fusion":
        return CrossViewFusionTrainer(config)
    raise ValueError(f"Unsupported backbone: {config.model.backbone}")


def _build_trainer(config: DictConfig, save_dir: Path) -> Trainer:
    progress_bar = RichProgressBar(refresh_rate=10, leave=True)

    use_gpu = torch.cuda.is_available()
    trainer = Trainer(
        devices=[int(config.train.gpu)] if use_gpu else 1,
        accelerator="gpu" if use_gpu else "cpu",
        max_epochs=int(config.train.max_epochs),
        callbacks=[progress_bar],
    )
    return trainer


def _format_float(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


# alpha visualization utilities
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
    map_module = importlib.import_module("map_config")
    common_mapping = getattr(map_module, "UNITY_MALE_MAPPING", None)
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


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="dual2pose.yaml",
)
def init_params(config: DictConfig | None = None) -> None:
    if config is None:
        raise ValueError("Hydra did not provide config")

    seed_everything(42, workers=True)
    config.train.gpu = int(getattr(config.train, "gpu", 0))

    ckpt_path = DEFAULT_CKPT_PATH
    failure_threshold = float(os.environ.get("FAILURE_THRESHOLD", "0.15"))

    results_root = "/home/kaixu_chen/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_unity_masking"
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    settings = _build_default_study()
    summary_rows: List[Dict[str, Any]] = []

    logger.info(
        "Running occlusion study with %d settings",
        len(settings),
    )
    logger.info("Checkpoint: %s", ckpt_path)

    for setting in settings:
        logger.info("[Setting] %s", setting)

        cfg_dict = OmegaConf.to_container(config, resolve=True)
        run_cfg = cast(DictConfig, OmegaConf.create(cfg_dict))
        run_cfg.log_path = str(results_root / setting.name)

        model = _build_model(run_cfg)
        base_dm = UnityDataModule(run_cfg)
        masked_dm = MaskedUnityDataModule(base_dm=base_dm, setting=setting)
        trainer = _build_trainer(run_cfg, save_dir=Path(run_cfg.log_path))

        trainer.test(
            model, datamodule=masked_dm, ckpt_path=ckpt_path, weights_only=False
        )

        test_outputs = list(getattr(model, "test_outputs", []))
        flat = _flatten_test_outputs(test_outputs)
        metrics = _summarize_outputs(flat=flat, failure_threshold=failure_threshold)
        alpha_tensor = _collect_alpha_tensor(test_outputs)
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

        run_tag = Path(ckpt_path).stem if ckpt_path else "run"
        out_json = Path(run_cfg.log_path) / f"occlusion_metrics_{run_tag}.json"
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "setting": setting.__dict__,
                    "failure_threshold": failure_threshold,
                    "metrics": metrics,
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )

        fused_mpjpe = metrics.get("fused", {}).get("mpjpe", math.nan)
        canonical_avg_mpjpe = metrics.get("canonical_avg", {}).get("mpjpe", math.nan)
        raw_avg_mpjpe = metrics.get("raw_avg", {}).get("mpjpe", math.nan)
        fused_acc = metrics.get("fused", {}).get("acceleration_error", math.nan)
        canonical_avg_acc = metrics.get("canonical_avg", {}).get(
            "acceleration_error", math.nan
        )
        raw_avg_acc = metrics.get("raw_avg", {}).get("acceleration_error", math.nan)

        summary_rows.append(
            {
                "setting": setting.name,
                "view_mode": setting.view_mode,
                "pattern": setting.pattern,
                "ratio": setting.ratio,
                "corruption": setting.corruption,
                "temporal_span": setting.temporal_span,
                "alpha_global_mean": alpha_global_mean,
                "alpha_global_std": alpha_global_std,
                "fused_mpjpe": fused_mpjpe,
                "canonical_avg_mpjpe": canonical_avg_mpjpe,
                "raw_avg_mpjpe": raw_avg_mpjpe,
                "delta_mpjpe_full_minus_avg": fused_mpjpe - canonical_avg_mpjpe,
                "delta_mpjpe_full_minus_raw_avg": fused_mpjpe - raw_avg_mpjpe,
                "fused_acceleration_error": fused_acc,
                "canonical_avg_acceleration_error": canonical_avg_acc,
                "raw_avg_acceleration_error": raw_avg_acc,
                "delta_acc_full_minus_avg": fused_acc - canonical_avg_acc,
                "delta_acc_full_minus_raw_avg": fused_acc - raw_avg_acc,
            }
        )

        report_path = (
            Path(run_cfg.log_path) / f"comparison_report_{setting.name}_{run_tag}.txt"
        )
        with open(report_path, "w", encoding="utf-8") as fp:
            fp.write(f"Setting: {setting.name}\n")
            fp.write(f"Mask view mode: {setting.view_mode}\n")
            fp.write(f"Mask pattern: {setting.pattern}\n")
            fp.write(f"Mask ratio: {setting.ratio:.2f}\n")
            fp.write(f"alpha_global_mean: {_format_float(alpha_global_mean)}\n")
            fp.write(f"alpha_global_std: {_format_float(alpha_global_std)}\n")
            fp.write("\n[Primary]\n")
            fp.write(f"fused_mpjpe: {_format_float(fused_mpjpe)}\n")
            fp.write(f"fused_accel_err: {_format_float(fused_acc)}\n")
            fp.write("\n[Baselines]\n")
            fp.write(f"canonical_avg_mpjpe: {_format_float(canonical_avg_mpjpe)}\n")
            fp.write(f"raw_avg_mpjpe: {_format_float(raw_avg_mpjpe)}\n")
            fp.write(
                f"delta_mpjpe_full_minus_canonical_avg: {_format_float(fused_mpjpe - canonical_avg_mpjpe)}\n"
            )
            fp.write(
                f"delta_mpjpe_full_minus_raw_avg: {_format_float(fused_mpjpe - raw_avg_mpjpe)}\n"
            )
            fp.write(f"canonical_avg_accel_err: {_format_float(canonical_avg_acc)}\n")
            fp.write(f"raw_avg_accel_err: {_format_float(raw_avg_acc)}\n")
            fp.write(
                f"delta_accel_full_minus_canonical_avg: {_format_float(fused_acc - canonical_avg_acc)}\n"
            )
            fp.write(
                f"delta_accel_full_minus_raw_avg: {_format_float(fused_acc - raw_avg_acc)}\n"
            )

        logger.info(
            "Completed %s: fused_mpjpe=%.4f, canonical_avg_mpjpe=%.4f",
            setting.name,
            fused_mpjpe,
            canonical_avg_mpjpe,
        )

        _export_alpha_visualization(
            test_outputs=test_outputs,
            save_dir=Path(run_cfg.log_path),
            setting_name=setting.name,
        )

    run_tag = Path(ckpt_path).stem if ckpt_path else "run"
    summary_csv = results_root / f"occlusion_summary_{run_tag}.csv"
    fieldnames = [
        "setting",
        "view_mode",
        "pattern",
        "ratio",
        "corruption",
        "temporal_span",
        "alpha_global_mean",
        "alpha_global_std",
        "fused_mpjpe",
        "canonical_avg_mpjpe",
        "raw_avg_mpjpe",
        "delta_mpjpe_full_minus_avg",
        "delta_mpjpe_full_minus_raw_avg",
        "fused_acceleration_error",
        "canonical_avg_acceleration_error",
        "raw_avg_acceleration_error",
        "delta_acc_full_minus_avg",
        "delta_acc_full_minus_raw_avg",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Saved occlusion study summary to %s", summary_csv)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    entry = cast(Any, init_params)
    entry()  # pyright: ignore[reportCallIssue]
