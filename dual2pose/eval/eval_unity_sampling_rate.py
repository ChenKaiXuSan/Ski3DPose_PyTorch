#!/usr/bin/env python3
"""Evaluate robustness to a fractional camera sampling-rate mismatch."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, seed_everything
import torch
from torch.utils.data import DataLoader, default_collate

from dual2pose.eval.eval_unity_temporal_offset import (
    DEFAULT_DATA_ROOT_IN_INDEX,
    _patch_index_mapping_path_rewrite,
    _summarize_gate_error_relationship,
)
from dual2pose.eval.extension_experiment_utils import (
    build_experiment_provenance,
    complete_test_dataloader,
    resample_pose_rate,
)


logger = logging.getLogger(__name__)


def _load_eval_helpers() -> Any:
    return importlib.import_module("dual2pose.eval.eval_unity_masking")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_PATH = (
    REPO_ROOT
    / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
)
DEFAULT_RATE_ERRORS = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]


@dataclass(frozen=True)
class SamplingRateSetting:
    name: str
    rate_error: float
    view_mode: str
    anchor: str = "center"


def _format_rate(rate_error: float) -> str:
    sign = "m" if rate_error < 0.0 else "p"
    percent = f"{abs(rate_error) * 100.0:.3f}".rstrip("0").rstrip(".")
    return f"{sign}{percent.replace('.', 'p')}pct"


def _parse_rate_errors(raw: str | None) -> List[float]:
    if raw is None or not raw.strip():
        return list(DEFAULT_RATE_ERRORS)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("SAMPLING_RATE_ERRORS did not contain numeric values")
    if any(not math.isfinite(value) or value <= -1.0 for value in values):
        raise ValueError("Sampling-rate errors must be finite fractions greater than -1")
    return values


def _apply_sampling_rate_to_batch(
    batch: Dict[str, Any], setting: SamplingRateSetting
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(batch)
    streams = batch.get("kpt3d_sam")
    if not isinstance(streams, dict):
        return out
    out["kpt3d_sam"] = dict(streams)
    keys: List[str] = []
    if setting.view_mode in {"left", "both"}:
        keys.append("cam1")
    if setting.view_mode in {"right", "both"}:
        keys.append("cam2")
    for key in keys:
        pose = out["kpt3d_sam"].get(key)
        if isinstance(pose, torch.Tensor):
            out["kpt3d_sam"][key] = resample_pose_rate(
                pose, rate_error=setting.rate_error, anchor=setting.anchor
            )
    out["_sampling_rate"] = {
        "rate_error": setting.rate_error,
        "view_mode": setting.view_mode,
        "anchor": setting.anchor,
    }
    return out


class SamplingRateUnityDataModule(LightningDataModule):
    def __init__(
        self, base_dm: LightningDataModule, setting: SamplingRateSetting
    ) -> None:
        super().__init__()
        self.base_dm = base_dm
        self.setting = setting

    def prepare_data(self) -> None:
        self.base_dm.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.base_dm.setup(stage)

    def test_dataloader(self) -> DataLoader:
        loader = self.base_dm.test_dataloader()
        collate_fn = loader.collate_fn or default_collate

        def _collate(items: List[Any]) -> Dict[str, Any]:
            return _apply_sampling_rate_to_batch(collate_fn(items), self.setting)

        return complete_test_dataloader(loader, collate_fn=_collate)


def _metric_row(
    setting: SamplingRateSetting,
    metrics: Dict[str, Any],
    gate_stats: Dict[str, float],
) -> Dict[str, Any]:
    fused = metrics.get("fused", {})
    canonical_avg = metrics.get("canonical_avg", {})
    return {
        "setting": setting.name,
        "view_mode": setting.view_mode,
        "rate_error_fraction": setting.rate_error,
        "rate_error_percent": setting.rate_error * 100.0,
        "anchor": setting.anchor,
        "fused_mpjpe": fused.get("mpjpe", math.nan),
        "canonical_avg_mpjpe": canonical_avg.get("mpjpe", math.nan),
        "fused_velocity_error": fused.get("velocity_error", math.nan),
        "fused_acceleration_error": fused.get("acceleration_error", math.nan),
        "failure_rate": fused.get("failure_rate", math.nan),
        **gate_stats,
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="dual2pose.yaml")
def init_params(config: DictConfig | None = None) -> None:
    if config is None:
        raise ValueError("Hydra did not provide config")
    seed = int(os.environ.get("EVAL_SEED", "42"))
    seed_everything(seed, workers=True)
    config.train.gpu = int(getattr(config.train, "gpu", 0))
    view_mode = os.environ.get("SAMPLING_RATE_VIEW", "right").strip().lower()
    if view_mode not in {"left", "right", "both"}:
        raise ValueError("SAMPLING_RATE_VIEW must be left, right, or both")
    anchor = os.environ.get("SAMPLING_RATE_ANCHOR", "center").strip().lower()
    if anchor not in {"center", "start"}:
        raise ValueError("SAMPLING_RATE_ANCHOR must be center or start")

    ckpt_path = Path(os.environ.get("EVAL_CKPT_PATH", str(DEFAULT_CKPT_PATH)))
    output_root = Path(
        os.environ.get(
            "EVAL_OUTPUT_ROOT", str(REPO_ROOT / "logs/eval_unity_sampling_rate")
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    failure_threshold = float(os.environ.get("FAILURE_THRESHOLD", "0.15"))
    rate_errors = _parse_rate_errors(os.environ.get("SAMPLING_RATE_ERRORS"))
    settings = [
        SamplingRateSetting(
            name=f"{view_mode}_rate_{_format_rate(rate_error)}",
            rate_error=rate_error,
            view_mode=view_mode,
            anchor=anchor,
        )
        for rate_error in rate_errors
    ]

    _patch_index_mapping_path_rewrite(
        old_root=os.environ.get("DATA_PATH_REWRITE_FROM", DEFAULT_DATA_ROOT_IN_INDEX),
        new_root=str(config.data.unity.root_path),
    )
    eval_helpers = _load_eval_helpers()
    rows: List[Dict[str, Any]] = []
    run_tag = ckpt_path.stem
    for setting in settings:
        run_cfg = cast(
            DictConfig,
            OmegaConf.create(OmegaConf.to_container(config, resolve=True)),
        )
        run_cfg.log_path = str(output_root / setting.name)
        model = eval_helpers._build_model(run_cfg)
        datamodule = SamplingRateUnityDataModule(
            eval_helpers.UnityDataModule(run_cfg), setting
        )
        trainer = eval_helpers._build_trainer(run_cfg, save_dir=Path(run_cfg.log_path))
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=str(ckpt_path),
            weights_only=False,
        )
        test_outputs = list(getattr(model, "test_outputs", []))
        flat = eval_helpers._flatten_test_outputs(test_outputs)
        provenance = build_experiment_provenance(
            ckpt_path,
            sample_count=int(flat["fused"].shape[0]),
            fold=int(config.train.fold),
            seed=seed,
            joint_subset="all15",
            units="dataset_coordinate_units",
        )
        metrics = eval_helpers._summarize_outputs(
            flat, failure_threshold=failure_threshold
        )
        gate_stats = _summarize_gate_error_relationship(test_outputs)
        row = _metric_row(setting, metrics, gate_stats)
        row.update(provenance)
        rows.append(row)
        setting_path = Path(run_cfg.log_path) / f"sampling_rate_metrics_{run_tag}.json"
        setting_path.parent.mkdir(parents=True, exist_ok=True)
        setting_path.write_text(
            json.dumps(
                {"setting": setting.__dict__, "provenance": provenance, "metrics": metrics, "gate": gate_stats},
                indent=2,
            ),
            encoding="utf-8",
        )

    zero_row = next(
        (row for row in rows if float(row["rate_error_fraction"]) == 0.0), None
    )
    if zero_row is None:
        raise ValueError("SAMPLING_RATE_ERRORS must include 0 for relative degradation")
    for row in rows:
        row["delta_mpjpe_vs_zero"] = row["fused_mpjpe"] - zero_row["fused_mpjpe"]
        row["relative_mpjpe_degradation_percent"] = (
            100.0 * row["delta_mpjpe_vs_zero"] / zero_row["fused_mpjpe"]
        )
        row["delta_acceleration_vs_zero"] = (
            row["fused_acceleration_error"] - zero_row["fused_acceleration_error"]
        )

    csv_path = output_root / f"sampling_rate_summary_{run_tag}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / f"sampling_rate_summary_{run_tag}.json").write_text(
        json.dumps(
            {
                "experiment": "unity_sampling_rate_drift",
                "checkpoint": str(ckpt_path.resolve()),
                "data_root": str(config.data.unity.root_path),
                "fold": int(config.train.fold),
                "seed": seed,
                "rate_error_definition": "perturbed_rate/reference_rate - 1",
                "anchor": anchor,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved sampling-rate summary to %s", csv_path)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cast(Any, init_params)()
