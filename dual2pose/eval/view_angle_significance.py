"""Cluster-aware statistical analysis for the Unity view-angle experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

from dual2pose.eval.extension_experiment_utils import (
    assign_angle_bin,
    circular_angle_distance,
    parse_unity_camera_id,
)


DEFAULT_BIN_EDGES = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)


def _metadata_values(meta: Mapping[str, Any], key: str, count: int) -> list[str]:
    raw = meta.get(key)
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Iterable):
        values = [str(value) for value in raw]
    else:
        raise ValueError(f"Batch metadata is missing {key!r}")
    if len(values) != count:
        raise ValueError(
            f"Batch metadata {key!r} has {len(values)} values for batch size {count}"
        )
    return values


def _validated_output(output: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    keys = (
        "fused",
        "left_canonical",
        "right_canonical",
        "ground_truth_canonical",
    )
    tensors: list[torch.Tensor] = []
    for key in keys:
        value = output.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Angle evaluation output is missing tensor {key!r}")
        value = value.detach().cpu()
        if not torch.isfinite(value).all():
            raise ValueError(f"Angle evaluation tensor {key!r} must be finite")
        tensors.append(value)
    if not all(tensor.shape == tensors[0].shape for tensor in tensors[1:]):
        raise ValueError("Angle evaluation tensors must share the same shape")
    if tensors[0].ndim != 4 or tensors[0].shape[-1] != 3:
        raise ValueError("Angle evaluation tensors must have shape BxTxJx3")
    return tuple(tensors)


def extract_angle_pair_rows(
    test_outputs: Sequence[Mapping[str, Any]],
    bin_edges: Sequence[float] = DEFAULT_BIN_EDGES,
) -> list[dict[str, Any]]:
    """Export one metric row per held-out action and unordered camera pair."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for output in test_outputs:
        fused, left, right, ground_truth = _validated_output(output)
        batch_size = int(fused.shape[0])
        meta = output.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError("Angle evaluation requires per-sample batch metadata")
        person_ids = _metadata_values(meta, "person_id", batch_size)
        action_ids = _metadata_values(meta, "action_id", batch_size)
        cam1_ids = _metadata_values(meta, "cam1_id", batch_size)
        cam2_ids = _metadata_values(meta, "cam2_id", batch_size)

        for index in range(batch_size):
            canonical_avg = 0.5 * (left[index] + right[index])
            fused_error = float(
                torch.norm(fused[index] - ground_truth[index], dim=-1).mean().item()
            )
            baseline_error = float(
                torch.norm(canonical_avg - ground_truth[index], dim=-1).mean().item()
            )
            if not math.isfinite(fused_error) or not math.isfinite(baseline_error):
                raise ValueError("Per-pair angle metrics must be finite")
            ordered_cameras = sorted((cam1_ids[index], cam2_ids[index]))
            pair_id = "|".join(ordered_cameras)
            unique_key = (action_ids[index], pair_id)
            if unique_key in seen:
                raise ValueError(
                    "Found duplicate action-camera-pair record "
                    f"{action_ids[index]!r}/{pair_id!r}"
                )
            seen.add(unique_key)
            _, azimuth_1 = parse_unity_camera_id(cam1_ids[index])
            _, azimuth_2 = parse_unity_camera_id(cam2_ids[index])
            separation = circular_angle_distance(azimuth_1, azimuth_2)
            gain = baseline_error - fused_error
            rows.append(
                {
                    "person_id": person_ids[index],
                    "action_id": action_ids[index],
                    "camera_pair_id": pair_id,
                    "cam1_id": cam1_ids[index],
                    "cam2_id": cam2_ids[index],
                    "separation_deg": separation,
                    "angle_bin": assign_angle_bin(separation, bin_edges),
                    "sample_count": 1,
                    "frame_count": int(fused.shape[1]),
                    "fused_mpjpe": fused_error,
                    "canonical_avg_mpjpe": baseline_error,
                    "fusion_gain_mpjpe": gain,
                    "fusion_gain_percent": (
                        100.0 * gain / baseline_error
                        if baseline_error != 0.0
                        else float("nan")
                    ),
                }
            )
    if not rows:
        raise ValueError("No valid angle evaluation outputs were provided")
    return rows


def collapse_action_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Average repeated actions for each unordered camera-pair cluster."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        action_id = str(row["action_id"])
        pair_id = str(row["camera_pair_id"])
        key = (action_id, pair_id)
        if key in seen:
            raise ValueError(f"Found duplicate action-camera-pair record {key!r}")
        seen.add(key)
        grouped[pair_id].append(row)

    collapsed: list[dict[str, Any]] = []
    metric_keys = (
        "separation_deg",
        "fused_mpjpe",
        "canonical_avg_mpjpe",
        "fusion_gain_mpjpe",
    )
    for pair_id in sorted(grouped):
        pair_rows = grouped[pair_id]
        bins = {str(row["angle_bin"]) for row in pair_rows}
        if len(bins) != 1:
            raise ValueError(f"Camera pair {pair_id!r} spans inconsistent angle bins")
        metrics: dict[str, float] = {}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in pair_rows], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Metric {key!r} must be finite for {pair_id!r}")
            metrics[key] = float(values.mean())
        baseline = metrics["canonical_avg_mpjpe"]
        relative_gain = (
            100.0 * metrics["fusion_gain_mpjpe"] / baseline
            if baseline != 0.0
            else float("nan")
        )
        if not math.isfinite(relative_gain):
            raise ValueError(f"Relative gain is non-finite for {pair_id!r}")
        collapsed.append(
            {
                "camera_pair_id": pair_id,
                "angle_bin": next(iter(bins)),
                "action_count": len(pair_rows),
                "person_count": len({str(row["person_id"]) for row in pair_rows}),
                "fusion_gain_percent": relative_gain,
                **metrics,
            }
        )
    if not collapsed:
        raise ValueError("No angle rows were provided")
    return collapsed


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in their original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Holm p-values must be a finite one-dimensional sequence")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Holm p-values must lie in [0, 1]")
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, float(values[original_index]) * (values.size - rank))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(values.size, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]


def _paired_rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    return (positive - negative) / (positive + negative)


def _wilcoxon_pvalue(differences: np.ndarray) -> float:
    if np.all(differences == 0.0):
        return 1.0
    return float(
        stats.wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        ).pvalue
    )


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _bin_sort_key(label: str) -> float:
    try:
        return float(str(label).split("-", 1)[0])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid angle-bin label {label!r}") from error


def analyze_angle_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze action rows after collapsing to camera-pair clusters."""

    clusters = collapse_action_rows(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clusters:
        grouped[str(row["angle_bin"])].append(row)
    labels = sorted(grouped, key=_bin_sort_key)
    if len(labels) != 6 or any(not grouped[label] for label in labels):
        raise ValueError("Angle analysis requires exactly six nonempty bins")

    rng = np.random.default_rng(seed)
    within_bin: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for label in labels:
        bin_rows = grouped[label]
        fused = np.asarray([row["fused_mpjpe"] for row in bin_rows], dtype=float)
        baseline = np.asarray(
            [row["canonical_avg_mpjpe"] for row in bin_rows], dtype=float
        )
        differences = baseline - fused
        if not np.isfinite(differences).all():
            raise ValueError(f"Angle bin {label!r} contains non-finite metrics")
        ci_low, ci_high = _bootstrap_mean_interval(
            differences,
            resamples=bootstrap_resamples,
            rng=rng,
        )
        p_value = _wilcoxon_pvalue(differences)
        raw_p_values.append(p_value)
        within_bin.append(
            {
                "angle_bin": label,
                "cluster_count": int(differences.size),
                "fused_mpjpe_mean": float(fused.mean()),
                "canonical_avg_mpjpe_mean": float(baseline.mean()),
                "mean_gain_mpjpe": float(differences.mean()),
                "median_gain_mpjpe": float(np.median(differences)),
                "mean_gain_ci95_low": ci_low,
                "mean_gain_ci95_high": ci_high,
                "rank_biserial": _paired_rank_biserial(differences),
                "test": "wilcoxon_signed_rank",
                "p_raw": p_value,
            }
        )
    for row, adjusted in zip(within_bin, holm_adjust(raw_p_values)):
        row["p_holm"] = adjusted
        row["significant_holm_0_05"] = bool(adjusted < 0.05)

    relative_groups = [
        np.asarray([row["fusion_gain_percent"] for row in grouped[label]], dtype=float)
        for label in labels
    ]
    if any(not np.isfinite(group).all() for group in relative_groups):
        raise ValueError("Relative fusion gains must be finite")
    omnibus_result = stats.kruskal(*relative_groups)
    total_count = sum(group.size for group in relative_groups)
    group_count = len(relative_groups)
    epsilon_squared = max(
        0.0,
        (float(omnibus_result.statistic) - group_count + 1.0)
        / (total_count - group_count),
    )
    omnibus = {
        "test": "kruskal_wallis",
        "metric": "pair_averaged_fusion_gain_percent",
        "statistic": float(omnibus_result.statistic),
        "p_value": float(omnibus_result.pvalue),
        "epsilon_squared": epsilon_squared,
        "significant_0_05": bool(omnibus_result.pvalue < 0.05),
    }

    pairwise: list[dict[str, Any]] = []
    if omnibus["significant_0_05"]:
        pairwise_raw: list[float] = []
        for left_index, left_label in enumerate(labels):
            for right_index in range(left_index + 1, len(labels)):
                right_label = labels[right_index]
                left_values = relative_groups[left_index]
                right_values = relative_groups[right_index]
                result = stats.mannwhitneyu(
                    left_values,
                    right_values,
                    alternative="two-sided",
                    method="auto",
                )
                p_value = float(result.pvalue)
                pairwise_raw.append(p_value)
                pairwise.append(
                    {
                        "angle_bin_a": left_label,
                        "angle_bin_b": right_label,
                        "n_a": int(left_values.size),
                        "n_b": int(right_values.size),
                        "median_gain_percent_a": float(np.median(left_values)),
                        "median_gain_percent_b": float(np.median(right_values)),
                        "test": "mann_whitney_u",
                        "statistic": float(result.statistic),
                        "rank_biserial": (
                            2.0 * float(result.statistic)
                            / (left_values.size * right_values.size)
                            - 1.0
                        ),
                        "p_raw": p_value,
                    }
                )
        for row, adjusted in zip(pairwise, holm_adjust(pairwise_raw)):
            row["p_holm"] = adjusted
            row["significant_holm_0_05"] = bool(adjusted < 0.05)

    return {
        "analysis_unit": "unordered_camera_pair_averaged_across_test_actions",
        "seed": int(seed),
        "bootstrap_resamples": int(bootstrap_resamples),
        "action_pair_row_count": len(rows),
        "cluster_count": len(clusters),
        "angle_bins": labels,
        "within_bin": within_bin,
        "omnibus": omnibus,
        "pairwise_contrasts": pairwise,
    }


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "separation_deg",
        "fused_mpjpe",
        "canonical_avg_mpjpe",
        "fusion_gain_mpjpe",
        "fusion_gain_percent",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            if key in row and row[key] != "":
                row[key] = float(row[key])
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = analyze_angle_rows(
        _read_csv_rows(args.input),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / "view_angle_statistics_last.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
