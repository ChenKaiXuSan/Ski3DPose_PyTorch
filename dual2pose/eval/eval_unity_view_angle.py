#!/usr/bin/env python3
"""Evaluate CrossViewFusion accuracy by camera azimuth separation."""

from __future__ import annotations

import csv
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from dual2pose.eval.extension_experiment_utils import (
    build_experiment_provenance,
    complete_test_dataloader,
    summarize_outputs_by_angle,
)
from dual2pose.eval.eval_unity_temporal_offset import (
    DEFAULT_DATA_ROOT_IN_INDEX,
    _patch_index_mapping_path_rewrite,
)


logger = logging.getLogger(__name__)


def _load_eval_helpers() -> Any:
    return importlib.import_module("dual2pose.eval.eval_unity_masking")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_PATH = (
    REPO_ROOT
    / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
)
DEFAULT_BIN_EDGES = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]


def _parse_bin_edges(raw: str | None) -> List[float]:
    if raw is None or not raw.strip():
        return list(DEFAULT_BIN_EDGES)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("VIEW_ANGLE_BIN_EDGES must contain at least two values")
    return values


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No angle-bin results were produced")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _attach_provenance(
    rows: list[dict[str, Any]], provenance: dict[str, Any]
) -> None:
    """Attach shared provenance without overwriting per-bin sample counts."""

    total_sample_count = int(provenance["sample_count"])
    shared = {
        key: value for key, value in provenance.items() if key != "sample_count"
    }
    for row in rows:
        row.update(shared)
        row["total_sample_count"] = total_sample_count


@hydra.main(version_base=None, config_path="../../configs", config_name="dual2pose.yaml")
def init_params(config: DictConfig | None = None) -> None:
    if config is None:
        raise ValueError("Hydra did not provide config")
    seed = int(os.environ.get("EVAL_SEED", "42"))
    seed_everything(seed, workers=True)
    config.train.gpu = int(getattr(config.train, "gpu", 0))

    ckpt_path = Path(os.environ.get("EVAL_CKPT_PATH", str(DEFAULT_CKPT_PATH)))
    output_root = Path(
        os.environ.get(
            "EVAL_OUTPUT_ROOT", str(REPO_ROOT / "logs/eval_unity_view_angle")
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    failure_threshold = float(os.environ.get("FAILURE_THRESHOLD", "0.15"))
    bin_edges = _parse_bin_edges(os.environ.get("VIEW_ANGLE_BIN_EDGES"))

    _patch_index_mapping_path_rewrite(
        old_root=os.environ.get("DATA_PATH_REWRITE_FROM", DEFAULT_DATA_ROOT_IN_INDEX),
        new_root=str(config.data.unity.root_path),
    )
    eval_helpers = _load_eval_helpers()
    model = eval_helpers._build_model(config)
    datamodule = eval_helpers.UnityDataModule(config)
    datamodule.prepare_data()
    datamodule.setup("test")
    test_loader = complete_test_dataloader(datamodule.test_dataloader())
    trainer = eval_helpers._build_trainer(config, save_dir=output_root)
    trainer.test(
        model,
        dataloaders=test_loader,
        ckpt_path=str(ckpt_path),
        weights_only=False,
    )

    rows = summarize_outputs_by_angle(
        list(getattr(model, "test_outputs", [])),
        failure_threshold=failure_threshold,
        bin_edges=bin_edges,
    )
    sample_count = sum(int(row["sample_count"]) for row in rows)
    provenance = build_experiment_provenance(
        ckpt_path,
        sample_count=sample_count,
        fold=int(config.train.fold),
        seed=seed,
        joint_subset="all15",
        units="dataset_coordinate_units",
    )
    _attach_provenance(rows, provenance)
    run_tag = ckpt_path.stem
    csv_path = output_root / f"view_angle_summary_{run_tag}.csv"
    _write_rows(csv_path, rows)
    json_path = output_root / f"view_angle_summary_{run_tag}.json"
    json_path.write_text(
        json.dumps(
            {
                "experiment": "unity_view_angle",
                "provenance": provenance,
                "checkpoint": str(ckpt_path.resolve()),
                "data_root": str(config.data.unity.root_path),
                "fold": int(config.train.fold),
                "seed": seed,
                "units": {
                    "angle": "degrees",
                    "pose_error": "dataset_coordinate_units",
                },
                "bin_edges": bin_edges,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved view-angle summary to %s", csv_path)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cast(Any, init_params)()
