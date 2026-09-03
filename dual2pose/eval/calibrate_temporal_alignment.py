#!/usr/bin/env python3
"""Select the temporal-alignment confidence gate on Unity fold-0 validation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from dual2pose.eval.eval_unity_temporal_alignment import (
    CANDIDATE_CORRECTIONS,
    INJECTED_OFFSETS,
    OUTPUT_ROOT,
    motion_speed,
)
from dual2pose.eval.eval_unity_temporal_offset import _shift_pose_sequence
from dual2pose.eval.temporal_alignment import (
    CalibrationRow,
    choose_confidence_threshold,
    estimate_temporal_correction,
)
from dual2pose.training_protocol import resolve_fold_index_path, validate_fold_metadata


def confidence_threshold_candidates(
    confidence_values: Sequence[float],
    quantiles: int = 101,
) -> list[float]:
    values = np.asarray(confidence_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [0.0]
    if quantiles < 2:
        raise ValueError("quantiles must be at least two")
    candidates = [0.0]
    candidates.extend(
        float(value)
        for value in np.quantile(values, np.linspace(0.0, 1.0, int(quantiles)))
    )
    candidates.append(math.nextafter(float(values.max()), math.inf))
    return sorted(set(candidates))


@hydra.main(version_base=None, config_path="../../configs", config_name="dual2pose.yaml")
def main(config: DictConfig) -> None:
    data_root = Path(str(config.data.unity.root_path))
    fold_path = resolve_fold_index_path(data_root, 0)
    validate_fold_metadata(fold_path, 0)
    config.train.fold = 0
    config.data.unity.index_mapping_path = str(fold_path)
    from dual2pose.dataloader.data_loader import UnityDataModule

    datamodule = UnityDataModule(config)
    datamodule.prepare_data()
    datamodule.setup("fit")
    loader = datamodule.val_dataloader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    limit_batches = int(os.environ.get("ALIGN_LIMIT_BATCHES", "0"))
    rows: list[CalibrationRow] = []
    speeds: list[float] = []
    batch_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if limit_batches and batch_index >= limit_batches:
                break
            left = batch["kpt3d_sam"]["cam1"].float().to(device)
            right = batch["kpt3d_sam"]["cam2"].float().to(device)
            speeds.extend(float(value) for value in motion_speed(left).detach().cpu().tolist())
            for injected_offset in INJECTED_OFFSETS:
                injected = _shift_pose_sequence(right, injected_offset)
                estimate = estimate_temporal_correction(
                    left,
                    injected,
                    CANDIDATE_CORRECTIONS,
                    confidence_threshold=-1.0,
                )
                for confidence, correction in zip(
                    estimate.confidence.detach().cpu().tolist(),
                    estimate.correction_frames.detach().cpu().tolist(),
                ):
                    rows.append(
                        CalibrationRow(
                            split="val",
                            confidence=float(confidence),
                            estimated_correction=float(correction),
                            target_correction=-float(injected_offset),
                        )
                    )
            batch_count += 1
    if not rows or not speeds:
        raise RuntimeError("Validation calibration produced no rows")
    threshold_candidates = confidence_threshold_candidates(
        [row.confidence for row in rows]
    )
    result = choose_confidence_threshold(rows, threshold_candidates)
    speed_array = np.asarray(speeds, dtype=np.float64)
    result.update(
        {
            "experiment": "ivc_p1_temporal_alignment_calibration",
            "split": "val",
            "fold": 0,
            "fold_json": str(fold_path),
            "validation_sample_count": len(speeds),
            "calibration_row_count": len(rows),
            "evaluated_batch_count": batch_count,
            "injected_offsets": list(INJECTED_OFFSETS),
            "candidate_corrections": list(CANDIDATE_CORRECTIONS),
            "motion_speed_quartiles": [
                float(value) for value in np.quantile(speed_array, [0.25, 0.5, 0.75])
            ],
            "limited_smoke_test": limit_batches > 0,
        }
    )
    output_root = Path(os.environ.get("EVAL_OUTPUT_ROOT", str(OUTPUT_ROOT))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "validation_threshold.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
