#!/usr/bin/env python3
"""Run the full cluster-aware Unity view-angle significance analysis."""

from __future__ import annotations

import csv
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import hydra
import numpy as np
import scipy
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from dual2pose.eval.extension_experiment_utils import (
    DEFAULT_DATA_ROOT_IN_INDEX,
    build_experiment_provenance,
    complete_test_dataloader,
    patch_index_mapping_path_rewrite,
)
from dual2pose.eval.view_angle_significance import (
    DEFAULT_BIN_EDGES,
    analyze_angle_rows,
    extract_angle_pair_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_PATH = (
    REPO_ROOT
    / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs/ivc_mmsports_extension/view_angle"
PAIRWISE_FIELDS = (
    "angle_bin_a",
    "angle_bin_b",
    "n_a",
    "n_b",
    "median_gain_percent_a",
    "median_gain_percent_b",
    "test",
    "statistic",
    "rank_biserial",
    "p_raw",
    "p_holm",
    "significant_holm_0_05",
)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields=None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV fields for empty artifact {path.name}")
        fields = tuple(rows[0].keys())
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_angle_artifacts(
    output_root: Path,
    pair_rows: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    *,
    expected_action_rows: int = 64_440,
    expected_clusters: int = 16_110,
) -> list[Path]:
    """Validate and atomically write the four declared E4 artifacts."""

    if len(pair_rows) != expected_action_rows:
        raise ValueError(
            f"Expected {expected_action_rows} action-pair rows (64440 in production), "
            f"received {len(pair_rows)}"
        )
    if int(statistics.get("action_pair_row_count", -1)) != expected_action_rows:
        raise ValueError("Statistics action-pair row count does not match exported rows")
    if int(statistics.get("cluster_count", -1)) != expected_clusters:
        raise ValueError(
            f"Expected {expected_clusters} camera-pair clusters, "
            f"received {statistics.get('cluster_count')!r}"
        )
    within_bin = statistics.get("within_bin")
    if not isinstance(within_bin, list) or len(within_bin) != 6:
        raise ValueError("Angle significance artifact requires six within-bin rows")
    pairwise = statistics.get("pairwise_contrasts")
    if not isinstance(pairwise, list) or len(pairwise) not in {0, 15}:
        raise ValueError("Angle pairwise artifact requires zero or 15 contrasts")
    omnibus = statistics.get("omnibus")
    if not isinstance(omnibus, Mapping):
        raise ValueError("Angle statistics are missing the omnibus result")
    if bool(omnibus.get("significant_0_05")) != (len(pairwise) == 15):
        raise ValueError("Post-hoc contrast count disagrees with omnibus significance")

    output_root.mkdir(parents=True, exist_ok=True)
    per_pair_path = output_root / "view_angle_per_pair_last.csv"
    significance_path = output_root / "view_angle_significance_last.csv"
    pairwise_path = output_root / "view_angle_pairwise_contrasts_last.csv"
    statistics_path = output_root / "view_angle_statistics_last.json"
    _atomic_csv(per_pair_path, pair_rows)
    _atomic_csv(significance_path, within_bin)
    _atomic_csv(pairwise_path, pairwise, fields=PAIRWISE_FIELDS)
    _atomic_json(
        statistics_path,
        {
            "experiment": "unity_view_angle_significance",
            "provenance": dict(provenance or {}),
            **dict(statistics),
        },
    )
    return [per_pair_path, significance_path, pairwise_path, statistics_path]


def _load_eval_helpers() -> Any:
    return importlib.import_module("dual2pose.eval.eval_unity_masking")


def _parse_bin_edges(raw: str | None) -> list[float]:
    if raw is None or not raw.strip():
        return list(DEFAULT_BIN_EDGES)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("VIEW_ANGLE_BIN_EDGES must contain at least two values")
    return values


def _software_provenance() -> dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_name,
        "command": " ".join(sys.argv),
    }


def validate_protocol_batch_size(batch_size: int) -> None:
    """Lock the batch-referenced canonical transform to the archived protocol."""

    if int(batch_size) != 256:
        raise ValueError(
            "The archived code uses a batch-referenced first-frame transform; "
            "E4 therefore requires data.batch_size=256 for numerical comparability"
        )


@hydra.main(version_base=None, config_path="../../configs", config_name="dual2pose.yaml")
def init_params(config: DictConfig | None = None) -> None:
    if config is None:
        raise ValueError("Hydra did not provide config")
    seed = int(os.environ.get("EVAL_SEED", "42"))
    seed_everything(seed, workers=True)
    config.train.gpu = int(getattr(config.train, "gpu", 0))
    validate_protocol_batch_size(int(config.data.batch_size))
    checkpoint = Path(os.environ.get("EVAL_CKPT_PATH", str(DEFAULT_CKPT_PATH)))
    output_root = Path(os.environ.get("EVAL_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))
    bootstrap_resamples = int(os.environ.get("BOOTSTRAP_RESAMPLES", "10000"))
    bin_edges = _parse_bin_edges(os.environ.get("VIEW_ANGLE_BIN_EDGES"))

    patch_index_mapping_path_rewrite(
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
        ckpt_path=str(checkpoint),
        weights_only=False,
    )

    pair_rows = extract_angle_pair_rows(
        list(getattr(model, "test_outputs", [])),
        bin_edges,
    )
    statistics = analyze_angle_rows(
        pair_rows,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    provenance = build_experiment_provenance(
        checkpoint,
        sample_count=len(pair_rows),
        fold=int(config.train.fold),
        seed=seed,
        joint_subset="all15",
        units="dataset_coordinate_units",
    )
    provenance.update(
        {
            "data_root": str(config.data.unity.root_path),
            "bin_edges": bin_edges,
            "multiplicity_correction": "Holm family-wise error rate",
            "software": _software_provenance(),
        }
    )
    paths = write_angle_artifacts(
        output_root,
        pair_rows,
        statistics,
        provenance,
    )
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cast(Any, init_params)()
