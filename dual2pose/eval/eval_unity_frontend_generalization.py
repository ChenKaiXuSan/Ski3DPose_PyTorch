#!/usr/bin/env python3
"""Evaluate CanonFuse3D with a baseline or manifest-provided pose front end."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, cast

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, seed_everything
import torch
from torch.utils.data import DataLoader

from dual2pose.eval.eval_unity_temporal_offset import (
    DEFAULT_DATA_ROOT_IN_INDEX,
    _patch_index_mapping_path_rewrite,
    _summarize_gate_error_relationship,
)
from dual2pose.eval.frontend_manifest import FrontEndManifest, FrontEndPoseDataset
from dual2pose.map_config import FILTERED_15_TO_COMMON_13_INDICES


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_PATH = (
    REPO_ROOT
    / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
)


def _load_eval_helpers() -> Any:
    return importlib.import_module("dual2pose.eval.eval_unity_masking")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrontEndUnityDataModule(LightningDataModule):
    def __init__(
        self, base_dm: LightningDataModule, manifest: FrontEndManifest | None
    ) -> None:
        super().__init__()
        self.base_dm = base_dm
        self.manifest = manifest

    def prepare_data(self) -> None:
        self.base_dm.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.base_dm.setup(stage)
        base_dataset = getattr(self.base_dm, "test_gait_dataset", None)
        if base_dataset is None:
            raise RuntimeError("UnityDataModule did not create a test dataset")
        if self.manifest is not None:
            self.base_dm.test_gait_dataset = FrontEndPoseDataset(
                base_dataset=base_dataset,
                manifest=self.manifest,
            )

    def test_dataloader(self) -> DataLoader:
        base_loader = self.base_dm.test_dataloader()
        kwargs: Dict[str, Any] = {
            "dataset": base_loader.dataset,
            "batch_size": base_loader.batch_size,
            "num_workers": base_loader.num_workers,
            "pin_memory": base_loader.pin_memory,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": base_loader.collate_fn,
            "worker_init_fn": base_loader.worker_init_fn,
        }
        if base_loader.num_workers > 0:
            kwargs["persistent_workers"] = base_loader.persistent_workers
            kwargs["prefetch_factor"] = base_loader.prefetch_factor
        return DataLoader(**kwargs)


def _summary_row(
    frontend_name: str,
    manifest_path: Path | None,
    metrics_all15: Dict[str, Any],
    metrics_common13: Dict[str, Any],
    gate_stats: Dict[str, float],
    sample_count: int,
    checkpoint: Path,
    fold: int,
    seed: int,
    manifest_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    fused = metrics_all15.get("fused", {})
    canonical_avg = metrics_all15.get("canonical_avg", {})
    raw_avg = metrics_all15.get("raw_avg", {})
    common_fused = metrics_common13.get("fused", {})
    common_canonical_avg = metrics_common13.get("canonical_avg", {})
    common_raw_avg = metrics_common13.get("raw_avg", {})
    fused_mpjpe = fused.get("mpjpe", math.nan)
    canonical_avg_mpjpe = canonical_avg.get("mpjpe", math.nan)
    common_fused_mpjpe = common_fused.get("mpjpe", math.nan)
    common_canonical_avg_mpjpe = common_canonical_avg.get("mpjpe", math.nan)
    provenance = manifest_metadata or {}
    return {
        "frontend_name": frontend_name,
        "manifest_path": str(manifest_path.resolve()) if manifest_path else "",
        "manifest_sha256": _sha256(manifest_path) if manifest_path else "",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "fold": int(fold),
        "seed": int(seed),
        "sample_count": sample_count,
        "joint_subset": "common13",
        "units": "dataset_coordinate_units",
        "input_2d_source": provenance.get("input_2d_source", "sam3d_native"),
        "estimator_checkpoint": provenance.get("estimator_checkpoint", ""),
        "estimator_checkpoint_sha256": provenance.get(
            "estimator_checkpoint_sha256", ""
        ),
        "official_repo_commit": provenance.get("official_repo_commit", ""),
        "all15_raw_avg_mpjpe": raw_avg.get("mpjpe", math.nan),
        "fused_mpjpe": fused_mpjpe,
        "canonical_avg_mpjpe": canonical_avg_mpjpe,
        "fusion_gain_mpjpe": canonical_avg_mpjpe - fused_mpjpe,
        "fused_velocity_error": fused.get("velocity_error", math.nan),
        "fused_acceleration_error": fused.get("acceleration_error", math.nan),
        "failure_rate": fused.get("failure_rate", math.nan),
        "common13_raw_avg_mpjpe": common_raw_avg.get("mpjpe", math.nan),
        "common13_canonical_avg_mpjpe": common_canonical_avg_mpjpe,
        "common13_fused_mpjpe": common_fused_mpjpe,
        "common13_fusion_gain_mpjpe": (
            common_canonical_avg_mpjpe - common_fused_mpjpe
        ),
        "common13_fused_velocity_error": common_fused.get(
            "velocity_error", math.nan
        ),
        "common13_fused_acceleration_error": common_fused.get(
            "acceleration_error", math.nan
        ),
        "common13_failure_rate": common_fused.get("failure_rate", math.nan),
        **gate_stats,
    }


def _merge_summary(path: Path, row: Dict[str, Any]) -> None:
    existing: List[Dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    key = (
        str(row["frontend_name"]),
        str(row["checkpoint"]),
        str(row.get("fold", "")),
    )
    kept = [
        item
        for item in existing
        if (
            item.get("frontend_name", ""),
            item.get("checkpoint", ""),
            item.get("fold", ""),
        )
        != key
    ]
    kept.append(row)
    fieldnames = list(row)
    for item in kept:
        for field in item:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)



def _joint_subset(
    flat: Dict[str, torch.Tensor], indices: List[int]
) -> Dict[str, torch.Tensor]:
    subset: Dict[str, torch.Tensor] = {}
    for name, tensor in flat.items():
        if tensor.ndim < 2 or tensor.shape[-2] <= max(indices):
            raise ValueError(
                f"Cannot select common joints from {name} with shape {tuple(tensor.shape)}"
            )
        subset[name] = tensor.index_select(
            -2, torch.tensor(indices, dtype=torch.long, device=tensor.device)
        )
    return subset


@hydra.main(version_base=None, config_path="../../configs", config_name="dual2pose.yaml")
def init_params(config: DictConfig | None = None) -> None:
    if config is None:
        raise ValueError("Hydra did not provide config")
    seed = int(os.environ.get("EVAL_SEED", "42"))
    seed_everything(seed, workers=True)
    config.train.gpu = int(getattr(config.train, "gpu", 0))
    ckpt_path = Path(os.environ.get("EVAL_CKPT_PATH", str(DEFAULT_CKPT_PATH)))
    failure_threshold = float(os.environ.get("FAILURE_THRESHOLD", "0.15"))

    raw_manifest = os.environ.get("FRONTEND_MANIFEST", "").strip()
    manifest_path = Path(raw_manifest) if raw_manifest else None
    manifest = FrontEndManifest.load(manifest_path) if manifest_path else None
    frontend_name = manifest.frontend_name if manifest else "sam3d"
    output_root = Path(
        os.environ.get(
            "EVAL_OUTPUT_ROOT",
            str(REPO_ROOT / "logs/eval_unity_frontend_generalization"),
        )
    )
    run_dir = output_root / frontend_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _patch_index_mapping_path_rewrite(
        old_root=os.environ.get("DATA_PATH_REWRITE_FROM", DEFAULT_DATA_ROOT_IN_INDEX),
        new_root=str(config.data.unity.root_path),
    )
    eval_helpers = _load_eval_helpers()
    base_dm = eval_helpers.UnityDataModule(config)
    # Always use the evaluation wrapper so the final partial batch is retained.
    datamodule: LightningDataModule = FrontEndUnityDataModule(base_dm, manifest)
    model = eval_helpers._build_model(config)
    trainer = eval_helpers._build_trainer(config, save_dir=run_dir)
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path=str(ckpt_path),
        weights_only=False,
    )
    test_outputs = list(getattr(model, "test_outputs", []))
    flat = eval_helpers._flatten_test_outputs(test_outputs)
    metrics_all15 = eval_helpers._summarize_outputs(
        flat, failure_threshold=failure_threshold
    )
    common13_flat = _joint_subset(
        flat, indices=list(FILTERED_15_TO_COMMON_13_INDICES)
    )
    metrics_common13 = eval_helpers._summarize_outputs(
        common13_flat, failure_threshold=failure_threshold
    )
    gate_stats = _summarize_gate_error_relationship(test_outputs)
    sample_count = int(flat["fused"].shape[0]) if "fused" in flat else 0
    row = _summary_row(
        frontend_name=frontend_name,
        manifest_path=manifest_path,
        metrics_all15=metrics_all15,
        metrics_common13=metrics_common13,
        gate_stats=gate_stats,
        sample_count=sample_count,
        checkpoint=ckpt_path,
        fold=int(config.train.fold),
        seed=seed,
        manifest_metadata=manifest.metadata if manifest else None,
    )
    run_tag = ckpt_path.stem
    (run_dir / f"frontend_metrics_{run_tag}.json").write_text(
        json.dumps(
            {
                "experiment": "unity_frontend_generalization",
                "frontend_name": frontend_name,
                "manifest": str(manifest_path.resolve()) if manifest_path else None,
                "checkpoint": str(ckpt_path.resolve()),
                "data_root": str(config.data.unity.root_path),
                "fold": int(config.train.fold),
                "seed": seed,
                "sample_count": sample_count,
                "manifest_metadata": manifest.metadata if manifest else {},
                "metrics": metrics_all15,
                "metrics_all15": metrics_all15,
                "metrics_common13": metrics_common13,
                "gate": gate_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path = output_root / f"frontend_generalization_summary_{run_tag}.csv"
    _merge_summary(summary_path, row)
    logger.info("Saved front-end generalization summary to %s", summary_path)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cast(Any, init_params)()
