#!/usr/bin/env python3
"""Evaluate validation-gated pose-only temporal alignment on Unity fold 0."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from dual2pose.eval.eval_unity_temporal_offset import _shift_pose_sequence
from dual2pose.eval.temporal_alignment import (
    correction_accuracy_masks,
    estimate_temporal_correction,
    velocity_descriptor,
)
from dual2pose.trainer.canonicalize import canonicalize_pose_torch
from dual2pose.training_protocol import resolve_fold_index_path, validate_fold_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "logs/ivc_p1/temporal_alignment"
DEFAULT_CHECKPOINT = REPO_ROOT / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
INJECTED_OFFSETS = (-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
CANDIDATE_CORRECTIONS = tuple(value / 2 for value in range(-10, 11))


def fixed_metric_time_slice(time_length: int, max_abs_offset: float) -> slice:
    if time_length <= 0:
        raise ValueError("Temporal metric evaluation requires at least one frame")
    margin = int(math.ceil(abs(float(max_abs_offset))))
    stop = int(time_length) - margin
    if margin >= stop:
        raise ValueError("Temporal metric margin leaves no valid center frames")
    return slice(margin, stop)


def _shift_per_sample(pose: torch.Tensor, corrections: torch.Tensor) -> torch.Tensor:
    if corrections.ndim != 1 or corrections.shape[0] != pose.shape[0]:
        raise ValueError("Temporal alignment requires one correction per sample")
    output = torch.empty_like(pose)
    for correction in torch.unique(corrections.detach()).tolist():
        mask = corrections == float(correction)
        output[mask] = _shift_pose_sequence(pose[mask], float(correction))
    return output


def build_right_stream_variants(
    right: torch.Tensor,
    injected_offset: float,
    estimated_corrections: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if estimated_corrections.ndim != 1 or estimated_corrections.shape[0] != right.shape[0]:
        raise ValueError("Temporal alignment requires one correction per sample")
    injected = _shift_pose_sequence(right, float(injected_offset))
    return {
        "uncorrected": injected,
        "automatic": _shift_per_sample(injected, estimated_corrections),
        "oracle": right.clone(),
    }


def motion_speed(pose: torch.Tensor) -> torch.Tensor:
    canonical, _ = canonicalize_pose_torch(pose, left_hip=6, right_hip=7, neck=14)
    descriptor = velocity_descriptor(canonical)
    return descriptor.norm(dim=-1).mean(dim=(1, 2))


def _model_outputs(
    model: Any,
    left: torch.Tensor,
    right: torch.Tensor,
    gt: torch.Tensor,
    metric_time_slice: slice | None = None,
) -> dict[str, torch.Tensor]:
    left_canonical, _ = canonicalize_pose_torch(
        left, left_hip=model.left_hip_idx, right_hip=model.right_hip_idx, neck=model.neck_idx
    )
    right_canonical, _ = canonicalize_pose_torch(
        right, left_hip=model.left_hip_idx, right_hip=model.right_hip_idx, neck=model.neck_idx
    )
    gt_canonical, _ = canonicalize_pose_torch(
        gt, left_hip=model.left_hip_idx, right_hip=model.right_hip_idx, neck=model.neck_idx
    )
    fused, aux = model.models(left_canonical, right_canonical)
    metric_time_slice = metric_time_slice or slice(None)
    fused_metric = fused[:, metric_time_slice]
    gt_metric = gt_canonical[:, metric_time_slice]
    left_metric = left_canonical[:, metric_time_slice]
    right_metric = right_canonical[:, metric_time_slice]
    alpha_metric = aux["alpha"][:, metric_time_slice]

    mpjpe = torch.norm(fused_metric - gt_metric, dim=-1).mean(dim=(1, 2))
    if fused_metric.shape[1] >= 3:
        pred_acc = fused_metric[:, 2:] - 2.0 * fused_metric[:, 1:-1] + fused_metric[:, :-2]
        gt_acc = gt_metric[:, 2:] - 2.0 * gt_metric[:, 1:-1] + gt_metric[:, :-2]
        accel = torch.norm(pred_acc - gt_acc, dim=-1).mean(dim=(1, 2))
    else:
        accel = torch.full_like(mpjpe, float("nan"))
    return {
        "mpjpe": mpjpe,
        "acceleration_error": accel,
        "alpha": alpha_metric.mean(dim=(1, 2, 3)),
        "left_error": torch.norm(left_metric - gt_metric, dim=-1).mean(dim=(1, 2)),
        "right_error": torch.norm(right_metric - gt_metric, dim=-1).mean(dim=(1, 2)),
    }


@dataclass
class RunningStats:
    count: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        finite = values.detach().double().cpu()
        finite = finite[torch.isfinite(finite)]
        self.count += int(finite.numel())
        self.value_sum += float(finite.sum().item())
        self.value_sq_sum += float(finite.square().sum().item())

    @property
    def mean(self) -> float:
        return self.value_sum / self.count if self.count else float("nan")

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0 if self.count == 1 else float("nan")
        variance = (self.value_sq_sum - self.value_sum**2 / self.count) / (self.count - 1)
        return math.sqrt(max(0.0, variance))


@dataclass
class ConditionStats:
    mpjpe: RunningStats
    acceleration: RunningStats
    preference_correct: int = 0
    preference_count: int = 0
    corr_n: int = 0
    corr_x: float = 0.0
    corr_y: float = 0.0
    corr_x2: float = 0.0
    corr_y2: float = 0.0
    corr_xy: float = 0.0

    @classmethod
    def create(cls) -> "ConditionStats":
        return cls(RunningStats(), RunningStats())

    def update(self, outputs: dict[str, torch.Tensor]) -> None:
        self.mpjpe.update(outputs["mpjpe"])
        self.acceleration.update(outputs["acceleration_error"])
        alpha = outputs["alpha"].detach().double().cpu()
        left_error = outputs["left_error"].detach().double().cpu()
        right_error = outputs["right_error"].detach().double().cpu()
        preferred_left = alpha >= 0.5
        left_is_better = left_error <= right_error
        self.preference_correct += int((preferred_left == left_is_better).sum().item())
        self.preference_count += int(alpha.numel())
        y = right_error - left_error
        finite = torch.isfinite(alpha) & torch.isfinite(y)
        x = alpha[finite]
        y = y[finite]
        self.corr_n += int(x.numel())
        self.corr_x += float(x.sum().item())
        self.corr_y += float(y.sum().item())
        self.corr_x2 += float(x.square().sum().item())
        self.corr_y2 += float(y.square().sum().item())
        self.corr_xy += float((x * y).sum().item())

    def correlation(self) -> float:
        if self.corr_n < 2:
            return float("nan")
        numerator = self.corr_n * self.corr_xy - self.corr_x * self.corr_y
        denominator = math.sqrt(
            max(0.0, self.corr_n * self.corr_x2 - self.corr_x**2)
            * max(0.0, self.corr_n * self.corr_y2 - self.corr_y**2)
        )
        return numerator / denominator if denominator > 0 else float("nan")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(config: DictConfig, checkpoint: Path, device: torch.device) -> Any:
    from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer

    model = CrossViewFusionTrainer(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    return model.to(device).eval()


def _sample_labels(meta: Any, batch_size: int, start: int) -> list[str]:
    if not isinstance(meta, dict):
        return [f"sample_{start + index}" for index in range(batch_size)]
    fields = []
    for key in ("person_id", "action_id", "cam1_id", "cam2_id"):
        value = meta.get(key)
        if isinstance(value, (list, tuple)) and len(value) == batch_size:
            fields.append([str(item) for item in value])
        else:
            fields.append([str(value)] * batch_size)
    return ["/".join(field[index] for field in fields) for index in range(batch_size)]


@hydra.main(version_base=None, config_path="../../configs", config_name="dual2pose.yaml")
def main(config: DictConfig) -> None:
    output_root = Path(os.environ.get("EVAL_OUTPUT_ROOT", str(OUTPUT_ROOT))).resolve()
    threshold_path = output_root / "validation_threshold.json"
    if not threshold_path.is_file():
        raise FileNotFoundError(f"Run validation calibration first: {threshold_path}")
    calibration = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold = float(calibration["selected_threshold"])
    quartiles = [float(value) for value in calibration["motion_speed_quartiles"]]
    metric_margin = int(math.ceil(max(abs(offset) for offset in INJECTED_OFFSETS)))
    data_root = Path(str(config.data.unity.root_path))
    fold_path = resolve_fold_index_path(data_root, 0)
    validate_fold_metadata(fold_path, 0)
    config.train.fold = 0
    config.data.unity.index_mapping_path = str(fold_path)
    from dual2pose.dataloader.data_loader import UnityDataModule

    datamodule = UnityDataModule(config)
    datamodule.prepare_data()
    datamodule.setup("test")
    loader = datamodule.test_dataloader()
    checkpoint = Path(os.environ.get("EVAL_CKPT_PATH", str(DEFAULT_CHECKPOINT))).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(config, checkpoint, device)
    limit_batches = int(os.environ.get("ALIGN_LIMIT_BATCHES", "0"))
    output_root.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_root / "per_sample_offsets.csv"
    fieldnames = [
        "sample_id", "injected_offset", "target_correction", "estimated_correction",
        "confidence", "motion_speed", "motion_quartile",
        "uncorrected_mpjpe", "automatic_mpjpe", "oracle_mpjpe",
        "uncorrected_acceleration_error", "automatic_acceleration_error", "oracle_acceleration_error",
    ]
    condition_stats = {
        (offset, condition): ConditionStats.create()
        for offset in INJECTED_OFFSETS
        for condition in ("uncorrected", "automatic", "oracle")
    }
    zero_stats = ConditionStats.create()
    offset_stats = {
        offset: {
            "absolute": RunningStats(),
            "signed": RunningStats(),
            "exact": 0,
            "within_half": 0,
            "active": 0,
            "count": 0,
        }
        for offset in INJECTED_OFFSETS
    }
    sample_index = 0
    with per_sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if limit_batches and batch_index >= limit_batches:
                    break
                left = batch["kpt3d_sam"]["cam1"].float().to(device)
                right = batch["kpt3d_sam"]["cam2"].float().to(device)
                gt = batch["kpt3d_gt"].float().to(device)
                metric_time_slice = fixed_metric_time_slice(
                    left.shape[1], metric_margin
                )
                speed = motion_speed(left)
                labels = _sample_labels(batch.get("meta"), left.shape[0], sample_index)
                sample_index += left.shape[0]
                zero_stats.update(
                    _model_outputs(model, left, right, gt, metric_time_slice)
                )
                for offset in INJECTED_OFFSETS:
                    injected = _shift_pose_sequence(right, offset)
                    estimate = estimate_temporal_correction(
                        left, injected, CANDIDATE_CORRECTIONS, threshold
                    )
                    variants = build_right_stream_variants(right, offset, estimate.correction_frames)
                    outputs = {
                        condition: _model_outputs(
                            model,
                            left,
                            variant,
                            gt,
                            metric_time_slice,
                        )
                        for condition, variant in variants.items()
                    }
                    for condition, values in outputs.items():
                        condition_stats[(offset, condition)].update(values)
                    target = torch.full_like(estimate.correction_frames, -offset)
                    error = estimate.correction_frames - target
                    offset_stats[offset]["absolute"].update(error.abs())
                    offset_stats[offset]["signed"].update(error)
                    exact, within_half = correction_accuracy_masks(
                        estimate.correction_frames, target
                    )
                    offset_stats[offset]["exact"] += int(exact.sum().item())
                    offset_stats[offset]["within_half"] += int(within_half.sum().item())
                    offset_stats[offset]["active"] += int((estimate.correction_frames != 0).sum().item())
                    offset_stats[offset]["count"] += int(left.shape[0])
                    speed_cpu = speed.detach().cpu().numpy()
                    quartile_ids = np.searchsorted(quartiles, speed_cpu, side="right") + 1
                    correction_cpu = estimate.correction_frames.detach().cpu().tolist()
                    confidence_cpu = estimate.confidence.detach().cpu().tolist()
                    for index, label in enumerate(labels):
                        writer.writerow(
                            {
                                "sample_id": label,
                                "injected_offset": offset,
                                "target_correction": -offset,
                                "estimated_correction": correction_cpu[index],
                                "confidence": confidence_cpu[index],
                                "motion_speed": float(speed_cpu[index]),
                                "motion_quartile": int(quartile_ids[index]),
                                "uncorrected_mpjpe": float(outputs["uncorrected"]["mpjpe"][index].item()),
                                "automatic_mpjpe": float(outputs["automatic"]["mpjpe"][index].item()),
                                "oracle_mpjpe": float(outputs["oracle"]["mpjpe"][index].item()),
                                "uncorrected_acceleration_error": float(outputs["uncorrected"]["acceleration_error"][index].item()),
                                "automatic_acceleration_error": float(outputs["automatic"]["acceleration_error"][index].item()),
                                "oracle_acceleration_error": float(outputs["oracle"]["acceleration_error"][index].item()),
                            }
                        )
    summary_rows: list[dict[str, Any]] = []
    zero_mean = zero_stats.mpjpe.mean
    for offset in INJECTED_OFFSETS:
        uncorrected_mean = condition_stats[(offset, "uncorrected")].mpjpe.mean
        oracle_mean = condition_stats[(offset, "oracle")].mpjpe.mean
        for condition in ("uncorrected", "automatic", "oracle"):
            stats = condition_stats[(offset, condition)]
            gap = uncorrected_mean - oracle_mean
            recovery = (
                (uncorrected_mean - stats.mpjpe.mean) / gap if abs(gap) > 1e-12 else float("nan")
            )
            estimation = offset_stats[offset]
            summary_rows.append(
                {
                    "injected_offset": offset,
                    "condition": condition,
                    "sample_count": stats.mpjpe.count,
                    "mpjpe_mean": stats.mpjpe.mean,
                    "mpjpe_std": stats.mpjpe.std,
                    "acceleration_error_mean": stats.acceleration.mean,
                    "degradation_vs_zero_reference": stats.mpjpe.mean - zero_mean,
                    "recovery_fraction": recovery,
                    "offset_mae": estimation["absolute"].mean,
                    "offset_signed_bias": estimation["signed"].mean,
                    "offset_accuracy_exact": estimation["exact"] / estimation["count"],
                    "offset_accuracy_within_0p5": estimation["within_half"] / estimation["count"],
                    "correction_activation_rate": estimation["active"] / estimation["count"],
                    "gate_error_correlation": stats.correlation(),
                    "view_preference_accuracy": stats.preference_correct / stats.preference_count,
                }
            )
    summary_path = output_root / "temporal_alignment_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_root / "temporal_alignment_provenance.json").write_text(
        json.dumps(
            {
                "experiment": "ivc_p1_temporal_alignment",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "fold_json": str(fold_path),
                "fold": 0,
                "threshold_file": str(threshold_path),
                "threshold_sha256": _sha256(threshold_path),
                "candidate_corrections": list(CANDIDATE_CORRECTIONS),
                "injected_offsets": list(INJECTED_OFFSETS),
                "metric_margin_frames": metric_margin,
                "metric_frame_count": 30 - 2 * metric_margin,
                "offset_accuracy_exact_tolerance": 0.25,
                "offset_accuracy_within_0p5_is_strict": True,
                "limited_smoke_test": limit_batches > 0,
                "evaluated_samples": sample_index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
