#!/usr/bin/env python3
"""Validate and summarize the six repeated CanonFuse3D training runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


EXPECTED_CELLS = {(fold, seed) for fold in (0, 1) for seed in (13, 42, 73)}
T_CRITICAL_DF5_95 = 2.5705818366147395


@dataclass(frozen=True)
class RunMetrics:
    fold: int
    seed: int
    mpjpe: float
    sample_count: int
    best_checkpoint: str
    checkpoint_sha256: str
    best_epoch: int
    per_action: Mapping[str, Mapping[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize empty values")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def summarize_training_runs(rows: Sequence[RunMetrics]) -> dict[str, Any]:
    indexed: dict[tuple[int, int], RunMetrics] = {}
    for row in rows:
        key = (int(row.fold), int(row.seed))
        if key in indexed:
            raise ValueError(f"Repeated-training matrix has duplicate cell {key}")
        indexed[key] = row
    if set(indexed) != EXPECTED_CELLS:
        raise ValueError(
            f"Repeated-training matrix mismatch; missing={sorted(EXPECTED_CELLS - set(indexed))}, "
            f"extra={sorted(set(indexed) - EXPECTED_CELLS)}"
        )
    values = [float(indexed[key].mpjpe) for key in sorted(indexed)]
    mean, std = _mean_std(values)
    margin = T_CRITICAL_DF5_95 * std / math.sqrt(len(values))
    per_fold: dict[str, dict[str, float | int]] = {}
    for fold in (0, 1):
        fold_values = [indexed[(fold, seed)].mpjpe for seed in (13, 42, 73)]
        fold_mean, fold_std = _mean_std(fold_values)
        per_fold[str(fold)] = {
            "run_count": len(fold_values),
            "mpjpe_mean": fold_mean,
            "mpjpe_std": fold_std,
        }
    action_values: dict[str, list[tuple[float, int]]] = {}
    for row in rows:
        for action, metrics in row.per_action.items():
            action_values.setdefault(str(action), []).append(
                (float(metrics["mpjpe"]), int(metrics["sample_count"]))
            )
    per_action: dict[str, dict[str, float | int]] = {}
    for action, pairs in sorted(action_values.items()):
        action_mean, action_std = _mean_std([value for value, _ in pairs])
        per_action[action] = {
            "run_count": len(pairs),
            "sample_count_per_run": pairs[0][1],
            "mpjpe_mean": action_mean,
            "mpjpe_std": action_std,
        }
    return {
        "aggregate": {
            "run_count": len(values),
            "mpjpe_mean": mean,
            "mpjpe_std": std,
            "mpjpe_ci95_low": mean - margin,
            "mpjpe_ci95_high": mean + margin,
            "interval": "two-sided t interval over six training runs, df=5",
        },
        "per_fold": per_fold,
        "per_action": per_action,
    }


def collect_run_metrics(root: Path) -> list[RunMetrics]:
    root = Path(root).resolve()
    rows: list[RunMetrics] = []
    for fold, seed in sorted(EXPECTED_CELLS):
        run_dir = root / f"fold_{fold}" / f"seed_{seed}"
        metrics_path = run_dir / "test_metrics_by_action.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing run metrics: {metrics_path}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if int(payload.get("fold", -1)) != fold or int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Run metadata mismatch: {metrics_path}")
        if int(payload.get("sample_count", -1)) != 64_440:
            raise ValueError(f"Run must contain all 64,440 fold test samples: {metrics_path}")
        checkpoint = Path(str(payload.get("best_model_path", ""))).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint}")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = int(checkpoint_payload.get("epoch", -1))
        rows.append(
            RunMetrics(
                fold=fold,
                seed=seed,
                mpjpe=float(payload["mpjpe"]),
                sample_count=int(payload["sample_count"]),
                best_checkpoint=str(checkpoint),
                checkpoint_sha256=_sha256(checkpoint),
                best_epoch=epoch,
                per_action=dict(payload["per_action"]),
            )
        )
    return rows


def _write_outputs(root: Path, rows: Sequence[RunMetrics], summary: Mapping[str, Any]) -> None:
    run_rows = [
        {
            "fold": row.fold,
            "seed": row.seed,
            "mpjpe": row.mpjpe,
            "sample_count": row.sample_count,
            "best_epoch": row.best_epoch,
            "best_checkpoint": row.best_checkpoint,
            "checkpoint_sha256": row.checkpoint_sha256,
        }
        for row in rows
    ]
    with (root / "per_run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(run_rows)
    action_rows = [
        {"action_id": action, **metrics}
        for action, metrics in summary["per_action"].items()
    ]
    with (root / "per_action_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(action_rows[0].keys()))
        writer.writeheader()
        writer.writerows(action_rows)
    (root / "multiseed_summary.json").write_text(
        json.dumps(
            {"runs": [asdict(row) for row in rows], "summary": summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    aggregate = summary["aggregate"]
    (root / "multiseed_summary.tex").write_text(
        "\\begin{tabular}{lrrrr}\n"
        "Method & Runs & MPJPE & SD & 95\\% CI \\\\\n"
        "\\hline\n"
        f"CanonFuse3D & {aggregate['run_count']} & {aggregate['mpjpe_mean']:.4f} & "
        f"{aggregate['mpjpe_std']:.4f} & [{aggregate['mpjpe_ci95_low']:.4f}, "
        f"{aggregate['mpjpe_ci95_high']:.4f}] \\\\\n"
        "\\end{tabular}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/ivc_p1/multiseed"))
    args = parser.parse_args()
    root = args.root.resolve()
    rows = collect_run_metrics(root)
    summary = summarize_training_runs(rows)
    _write_outputs(root, rows, summary)
    print(root / "multiseed_summary.json")


if __name__ == "__main__":
    main()
