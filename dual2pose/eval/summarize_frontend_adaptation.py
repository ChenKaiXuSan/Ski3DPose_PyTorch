#!/usr/bin/env python3
"""Validate and summarize the complete 8-model x 4-front-end transfer matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dual2pose.experiments.run_frontend_adaptation_matrix import (
    ROOT,
    build_evaluation_cells,
    evaluation_directory,
)


def validate_and_summarize_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = {(cell.model_name, cell.test_frontend) for cell in build_evaluation_cells()}
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        key = (str(row["model_name"]), str(row["test_frontend"]))
        if key in indexed:
            raise ValueError(f"Duplicate front-end adaptation cell: {key}")
        indexed[key] = row
    actual = set(indexed)
    if actual != expected:
        raise ValueError(
            f"Front-end adaptation matrix mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    baselines = {
        frontend: float(indexed[("mmsports", frontend)]["common13_mpjpe"])
        for frontend in {frontend for _, frontend in expected}
    }
    sam_baseline = baselines["sam3d"]
    enriched: list[dict[str, Any]] = []
    for key in sorted(indexed):
        row = dict(indexed[key])
        model_name, frontend = key
        value = float(row["common13_mpjpe"])
        row["common13_mpjpe"] = value
        row["recovery_vs_mmsports"] = baselines[frontend] - value
        row["relative_recovery_percent"] = (
            100.0 * (baselines[frontend] - value) / baselines[frontend]
            if baselines[frontend] != 0.0
            else float("nan")
        )
        row["sam3d_retention_delta"] = (
            float(indexed[(model_name, "sam3d")]["common13_mpjpe"]) - sam_baseline
        )
        enriched.append(row)
    return enriched


def payload_sample_count(payload: Mapping[str, Any]) -> int:
    if "sample_count" not in payload:
        raise ValueError("Front-end evaluation JSON is missing sample_count")
    count = int(payload["sample_count"])
    if count <= 0:
        raise ValueError(f"Front-end evaluation sample_count must be positive: {count}")
    return count


def collect_matrix_rows(root: Path = ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in build_evaluation_cells():
        directory = evaluation_directory(cell) / "evaluation" / cell.test_frontend
        files = sorted(directory.glob("frontend_metrics_*.json"))
        if len(files) != 1:
            raise ValueError(
                f"Expected one metrics JSON for {cell.name}, found {len(files)} in {directory}"
            )
        path = files[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics_common13", {})
        fused = metrics.get("fused", {}) if isinstance(metrics, dict) else {}
        if "mpjpe" not in fused:
            raise ValueError(f"Missing common-13 fused MPJPE: {path}")
        rows.append(
            {
                "model_name": cell.model_name,
                "test_frontend": cell.test_frontend,
                "common13_mpjpe": float(fused["mpjpe"]),
                "common13_acceleration_error": float(fused.get("acceleration_error", float("nan"))),
                "sample_count": payload_sample_count(payload),
                "checkpoint": str(payload.get("checkpoint", "")),
                "manifest": payload.get("manifest"),
                "metrics_json": str(path),
            }
        )
    return rows


def main() -> None:
    rows = validate_and_summarize_rows(collect_matrix_rows())
    output_csv = ROOT / "frontend_adaptation_matrix.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    output_csv.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(output_csv)


if __name__ == "__main__":
    main()
