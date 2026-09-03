#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Temporal synchronization robustness evaluation for Unity dual-view inputs.

The study perturbs one test-time 3D keypoint stream by sub-frame or integer
frame offsets, then reuses the standard CrossViewFusion test path. It also
summarizes whether the learned gate assigns larger alpha to the lower-error
left canonical view.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import importlib
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, default_collate

from dual2pose.eval.extension_experiment_utils import (
    build_experiment_provenance,
    complete_test_dataloader,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_PATH = (
    REPO_ROOT
    / "logs"
    / "train_unity"
    / "crossview_fusion"
    / "2026-05-14"
    / "04-55-35"
    / "checkpoints"
    / "last.ckpt"
)
DEFAULT_OFFSETS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
DEFAULT_DATA_ROOT_IN_INDEX = "/home/kaixu_chen/data/skiing/skiing_unity_dataset"
LEGACY_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_legacy_import_path() -> None:
    """Expose the package root required by the repository legacy imports."""

    project_root = str(LEGACY_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)



@dataclass(frozen=True)
class TemporalOffsetSetting:
    name: str
    offset_frames: float
    view_mode: str  # left | right | both


def _load_eval_helpers() -> Any:
    return importlib.import_module("dual2pose.eval.eval_unity_masking")


def _rewrite_paths(value: Any, old_root: str, new_root: str) -> Any:
    if old_root == new_root:
        return value
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    if isinstance(value, list):
        return [_rewrite_paths(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_paths(item, old_root, new_root)
            for key, item in value.items()
        }
    return value


def _patch_index_mapping_path_rewrite(old_root: str, new_root: str) -> None:
    _ensure_legacy_import_path()
    data_module = importlib.import_module("dataloader.data_loader")
    original_loader = data_module.load_index_mapping

    if getattr(original_loader, "_temporal_offset_path_rewrite", False):
        return

    def _load_index_mapping_with_rewrite(index_mapping_path: str) -> Dict[str, list]:
        mapping = original_loader(index_mapping_path)
        return cast(Dict[str, list], _rewrite_paths(mapping, old_root, new_root))

    _load_index_mapping_with_rewrite._temporal_offset_path_rewrite = True  # type: ignore[attr-defined]
    data_module.load_index_mapping = _load_index_mapping_with_rewrite


def _clone_batch_for_edit(batch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(batch)
    if isinstance(batch.get("kpt3d_sam"), dict):
        out["kpt3d_sam"] = dict(batch["kpt3d_sam"])
    return out


def _format_float(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _parse_offsets(raw: str | None) -> List[float]:
    if raw is None or raw.strip() == "":
        return list(DEFAULT_OFFSETS)
    offsets: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        offsets.append(float(item))
    if not offsets:
        raise ValueError("TEMPORAL_OFFSETS did not contain any numeric values")
    return offsets


def _format_offset(value: float) -> str:
    prefix = "m" if value < 0 else "p"
    text = f"{abs(value):.2f}".rstrip("0").rstrip(".")
    return prefix + text.replace(".", "p")


def _shift_pose_sequence(pose: torch.Tensor, offset_frames: float) -> torch.Tensor:
    """Shift a B x T x J x C pose sequence with clamped linear interpolation.

    Positive offsets mean the perturbed stream lags the reference stream:
    output[t] samples input[t - offset].
    """

    if pose.ndim != 4:
        raise ValueError(f"Expected pose shape BxTxJxC, got {tuple(pose.shape)}")
    if pose.shape[1] == 0 or float(offset_frames) == 0.0:
        return pose.clone()

    time_len = pose.shape[1]
    positions = (
        torch.arange(time_len, device=pose.device, dtype=pose.dtype)
        - float(offset_frames)
    ).clamp(0, time_len - 1)
    low_idx = torch.floor(positions).long()
    high_idx = torch.ceil(positions).long()
    weight = (positions - low_idx.to(dtype=positions.dtype)).view(1, time_len, 1, 1)

    low = pose.index_select(dim=1, index=low_idx)
    high = pose.index_select(dim=1, index=high_idx)
    return (1.0 - weight) * low + weight * high


def _apply_temporal_offset_to_batch(
    batch: Dict[str, Any],
    setting: TemporalOffsetSetting,
) -> Dict[str, Any]:
    out = _clone_batch_for_edit(batch)
    kpt3d_sam = out.get("kpt3d_sam")
    if not isinstance(kpt3d_sam, dict):
        return out

    view_keys = []
    if setting.view_mode in ("left", "both"):
        view_keys.append("cam1")
    if setting.view_mode in ("right", "both"):
        view_keys.append("cam2")

    for key in view_keys:
        pose = kpt3d_sam.get(key)
        if isinstance(pose, torch.Tensor):
            kpt3d_sam[key] = _shift_pose_sequence(pose, setting.offset_frames)

    out["_temporal_offset"] = {
        "offset_frames": setting.offset_frames,
        "view_mode": setting.view_mode,
    }
    return out


class TemporalOffsetUnityDataModule(LightningDataModule):
    def __init__(
        self,
        base_dm: LightningDataModule,
        setting: TemporalOffsetSetting,
    ) -> None:
        super().__init__()
        self.base_dm = base_dm
        self.setting = setting

    def prepare_data(self) -> None:
        prepare_data = getattr(self.base_dm, "prepare_data", None)
        if callable(prepare_data):
            prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.base_dm.setup(stage)

    def _wrap_collate(self, collate_fn: Any) -> Any:
        def _collate(batch: List[Any]) -> Dict[str, Any]:
            collated = collate_fn(batch)
            return _apply_temporal_offset_to_batch(collated, self.setting)

        return _collate

    def test_dataloader(self) -> DataLoader:
        loader = self.base_dm.test_dataloader()
        collate_fn = loader.collate_fn if loader.collate_fn is not None else default_collate
        return complete_test_dataloader(
            loader,
            collate_fn=self._wrap_collate(collate_fn),
        )


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    valid = torch.isfinite(x) & torch.isfinite(y)
    if int(valid.sum().item()) < 2:
        return float("nan")
    xv = x[valid].float()
    yv = y[valid].float()
    xv = xv - xv.mean()
    yv = yv - yv.mean()
    denom = torch.sqrt((xv.square().sum()) * (yv.square().sum()))
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((xv * yv).sum().div(denom).item())


def _summarize_gate_error_relationship(
    test_outputs: List[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    alpha_chunks: List[torch.Tensor] = []
    advantage_chunks: List[torch.Tensor] = []
    preference_chunks: List[torch.Tensor] = []

    for output in test_outputs:
        alpha = output.get("alpha")
        left = output.get("left_canonical")
        right = output.get("right_canonical")
        gt = output.get("ground_truth_canonical")
        if not all(isinstance(v, torch.Tensor) for v in (alpha, left, right, gt)):
            continue

        alpha_tensor = cast(torch.Tensor, alpha).detach().cpu().squeeze(-1)
        left_err = torch.norm(cast(torch.Tensor, left).detach().cpu() - gt.detach().cpu(), dim=-1)
        right_err = torch.norm(cast(torch.Tensor, right).detach().cpu() - gt.detach().cpu(), dim=-1)
        view_advantage = right_err - left_err
        left_is_better = view_advantage >= 0.0
        gate_prefers_left = alpha_tensor >= 0.5

        alpha_chunks.append(alpha_tensor.flatten())
        advantage_chunks.append(view_advantage.flatten())
        preference_chunks.append((gate_prefers_left == left_is_better).float().flatten())

    if not alpha_chunks:
        return {
            "gate_error_corr": float("nan"),
            "gate_preference_accuracy": float("nan"),
            "alpha_when_left_better": float("nan"),
            "alpha_when_right_better": float("nan"),
        }

    alpha_all = torch.cat(alpha_chunks)
    advantage_all = torch.cat(advantage_chunks)
    left_better = advantage_all >= 0.0
    right_better = advantage_all < 0.0

    return {
        "gate_error_corr": _pearson_corr(alpha_all, advantage_all),
        "gate_preference_accuracy": float(torch.cat(preference_chunks).mean().item()),
        "alpha_when_left_better": (
            float(alpha_all[left_better].mean().item())
            if bool(left_better.any().item())
            else float("nan")
        ),
        "alpha_when_right_better": (
            float(alpha_all[right_better].mean().item())
            if bool(right_better.any().item())
            else float("nan")
        ),
    }


def _build_default_study(view_mode: str, offsets: List[float]) -> List[TemporalOffsetSetting]:
    return [
        TemporalOffsetSetting(
            name=f"{view_mode}_offset_{_format_offset(offset)}",
            offset_frames=offset,
            view_mode=view_mode,
        )
        for offset in offsets
    ]


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

    ckpt_path = os.environ.get("EVAL_CKPT_PATH", str(DEFAULT_CKPT_PATH))
    failure_threshold = float(os.environ.get("FAILURE_THRESHOLD", "0.15"))
    view_mode = os.environ.get("TEMPORAL_OFFSET_VIEW", "right").strip().lower()
    if view_mode not in {"left", "right", "both"}:
        raise ValueError("TEMPORAL_OFFSET_VIEW must be one of: left, right, both")

    offsets = _parse_offsets(os.environ.get("TEMPORAL_OFFSETS"))
    results_root = Path(
        os.environ.get(
            "EVAL_OUTPUT_ROOT",
            str(REPO_ROOT / "logs" / "eval_unity_temporal_offset"),
        )
    )
    results_root.mkdir(parents=True, exist_ok=True)

    settings = _build_default_study(view_mode=view_mode, offsets=offsets)
    summary_rows: List[Dict[str, Any]] = []
    run_tag = Path(ckpt_path).stem if ckpt_path else "run"
    eval_helpers = _load_eval_helpers()
    _patch_index_mapping_path_rewrite(
        old_root=os.environ.get("DATA_PATH_REWRITE_FROM", DEFAULT_DATA_ROOT_IN_INDEX),
        new_root=str(config.data.unity.root_path),
    )

    logger.info("Running temporal offset study with %d settings", len(settings))
    logger.info("Checkpoint: %s", ckpt_path)

    for setting in settings:
        logger.info("[Setting] %s", setting)

        cfg_dict = OmegaConf.to_container(config, resolve=True)
        run_cfg = cast(DictConfig, OmegaConf.create(cfg_dict))
        run_cfg.log_path = str(results_root / setting.name)

        model = eval_helpers._build_model(run_cfg)
        base_dm = eval_helpers.UnityDataModule(run_cfg)
        offset_dm = TemporalOffsetUnityDataModule(base_dm=base_dm, setting=setting)
        trainer = eval_helpers._build_trainer(run_cfg, save_dir=Path(run_cfg.log_path))

        trainer.test(
            model,
            datamodule=offset_dm,
            ckpt_path=ckpt_path,
            weights_only=False,
        )

        test_outputs = list(getattr(model, "test_outputs", []))
        flat = eval_helpers._flatten_test_outputs(test_outputs)
        sample_count = int(flat["fused"].shape[0])
        provenance = build_experiment_provenance(
            ckpt_path,
            sample_count=sample_count,
            fold=int(config.train.fold),
            seed=42,
            joint_subset="all15",
            units="dataset_coordinate_units",
        )
        metrics = eval_helpers._summarize_outputs(flat=flat, failure_threshold=failure_threshold)
        gate_stats = _summarize_gate_error_relationship(test_outputs)

        alpha_tensor = eval_helpers._collect_alpha_tensor(test_outputs)
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

        fused_mpjpe = metrics.get("fused", {}).get("mpjpe", math.nan)
        canonical_avg_mpjpe = metrics.get("canonical_avg", {}).get("mpjpe", math.nan)
        raw_avg_mpjpe = metrics.get("raw_avg", {}).get("mpjpe", math.nan)
        fused_acc = metrics.get("fused", {}).get("acceleration_error", math.nan)
        canonical_avg_acc = metrics.get("canonical_avg", {}).get(
            "acceleration_error", math.nan
        )
        raw_avg_acc = metrics.get("raw_avg", {}).get("acceleration_error", math.nan)

        row = {
            "setting": setting.name,
            "view_mode": setting.view_mode,
            "offset_frames": setting.offset_frames,
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
            **gate_stats,
            **provenance,
        }
        summary_rows.append(row)

        out_json = Path(run_cfg.log_path) / f"temporal_offset_metrics_{run_tag}.json"
        with open(out_json, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "setting": setting.__dict__,
                    "failure_threshold": failure_threshold,
                    "provenance": provenance,
                    "metrics": metrics,
                    "gate_error_relationship": gate_stats,
                },
                fp,
                ensure_ascii=False,
                indent=2,
            )

        report_path = (
            Path(run_cfg.log_path) / f"comparison_report_{setting.name}_{run_tag}.txt"
        )
        with open(report_path, "w", encoding="utf-8") as fp:
            fp.write(f"Setting: {setting.name}\n")
            fp.write(f"Offset view mode: {setting.view_mode}\n")
            fp.write(f"Offset frames: {setting.offset_frames:.2f}\n")
            fp.write(f"alpha_global_mean: {_format_float(alpha_global_mean)}\n")
            fp.write(f"alpha_global_std: {_format_float(alpha_global_std)}\n")
            fp.write("\n[Primary]\n")
            fp.write(f"fused_mpjpe: {_format_float(fused_mpjpe)}\n")
            fp.write(f"fused_accel_err: {_format_float(fused_acc)}\n")
            fp.write("\n[Baselines]\n")
            fp.write(f"canonical_avg_mpjpe: {_format_float(canonical_avg_mpjpe)}\n")
            fp.write(f"raw_avg_mpjpe: {_format_float(raw_avg_mpjpe)}\n")
            fp.write(f"canonical_avg_accel_err: {_format_float(canonical_avg_acc)}\n")
            fp.write(f"raw_avg_accel_err: {_format_float(raw_avg_acc)}\n")
            fp.write("\n[Gate vs view error]\n")
            for key, value in gate_stats.items():
                fp.write(f"{key}: {_format_float(value)}\n")

        eval_helpers._export_alpha_visualization(
            test_outputs=test_outputs,
            save_dir=Path(run_cfg.log_path),
            setting_name=setting.name,
        )

    summary_csv = results_root / f"temporal_offset_summary_{run_tag}.csv"
    fieldnames = [
        "setting",
        "view_mode",
        "offset_frames",
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
        "gate_error_corr",
        "gate_preference_accuracy",
        "alpha_when_left_better",
        "alpha_when_right_better",
        "checkpoint",
        "checkpoint_sha256",
        "sample_count",
        "fold",
        "seed",
        "joint_subset",
        "units",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Saved temporal offset study summary to %s", summary_csv)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    entry = cast(Any, init_params)
    entry()  # pyright: ignore[reportCallIssue]
