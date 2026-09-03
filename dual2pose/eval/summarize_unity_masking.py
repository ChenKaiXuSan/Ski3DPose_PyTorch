#!/usr/bin/env python3
"""Convert the completed Unity masking sweep into publication-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


CurveKey = Tuple[str, str, str, str]


def _as_float(row: Dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Masking row has invalid {key!r}: {row.get(key)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Masking row has non-finite {key!r}: {value}")
    return value


def _curve_key(row: Dict[str, Any]) -> CurveKey:
    return (
        str(row.get("view_mode", "")),
        str(row.get("pattern", "")),
        str(row.get("corruption", "")),
        str(row.get("temporal_span", "")),
    )


def _normalized_auc(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    ordered = sorted(points)
    width = ordered[-1][0] - ordered[0][0]
    if width <= 0.0:
        return float("nan")
    area = sum(
        0.5 * (left_y + right_y) * (right_x - left_x)
        for (left_x, left_y), (right_x, right_y) in zip(ordered, ordered[1:])
    )
    return area / width


def summarize_masking_rows(
    rows: Iterable[Dict[str, Any]],
    selected_ratios: Sequence[float],
    max_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return selected-point, normalized-AUC, and long-form trend rows."""

    if max_ratio <= 0.0 or max_ratio > 1.0:
        raise ValueError("max_ratio must be in (0, 1]")
    selected_values = sorted({float(value) for value in selected_ratios})
    if any(value < 0.0 or value > max_ratio for value in selected_values):
        raise ValueError("selected_ratios must lie in [0, max_ratio]")

    grouped: Dict[CurveKey, Dict[float, Dict[str, Any]]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        view_mode = str(row.get("view_mode", ""))
        if view_mode == "none":
            continue
        key = _curve_key(row)
        if not all(key[:3]):
            raise ValueError(f"Masking row is missing curve identifiers: {row}")
        ratio = _as_float(row, "ratio")
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"Masking ratio is outside [0, 1]: {ratio}")
        if ratio > max_ratio + 1e-9:
            continue
        curve = grouped.setdefault(key, {})
        rounded_ratio = round(ratio, 9)
        if rounded_ratio in curve:
            raise ValueError(f"Duplicate masking curve ratio for {key}: {ratio}")
        curve[rounded_ratio] = row

    selected_rows: List[Dict[str, Any]] = []
    auc_rows: List[Dict[str, Any]] = []
    trend_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        curve = grouped[key]
        if 0.0 not in curve:
            raise ValueError(f"Masking curve {key} has no ratio=0 baseline")
        baseline_mpjpe = _as_float(curve[0.0], "fused_mpjpe")
        baseline_acc = _as_float(curve[0.0], "fused_acceleration_error")
        if baseline_mpjpe == 0.0:
            raise ValueError(f"Masking curve {key} has a zero MPJPE baseline")

        curve_trends: List[Dict[str, Any]] = []
        for ratio, raw_row in sorted(curve.items()):
            fused_mpjpe = _as_float(raw_row, "fused_mpjpe")
            fused_acc = _as_float(raw_row, "fused_acceleration_error")
            canonical_avg = _as_float(raw_row, "canonical_avg_mpjpe")
            trend = {
                "setting": str(raw_row.get("setting", "")),
                "view_mode": key[0],
                "pattern": key[1],
                "corruption": key[2],
                "temporal_span": key[3],
                "ratio": ratio,
                "fused_mpjpe": fused_mpjpe,
                "canonical_avg_mpjpe": canonical_avg,
                "fusion_gain_mpjpe": canonical_avg - fused_mpjpe,
                "fused_acceleration_error": fused_acc,
                "normalized_mpjpe": fused_mpjpe / baseline_mpjpe,
                "normalized_acceleration_error": (
                    fused_acc / baseline_acc if baseline_acc != 0.0 else float("nan")
                ),
                "mpjpe_degradation_percent": 100.0
                * (fused_mpjpe - baseline_mpjpe)
                / baseline_mpjpe,
                "acceleration_degradation_percent": (
                    100.0 * (fused_acc - baseline_acc) / baseline_acc
                    if baseline_acc != 0.0
                    else float("nan")
                ),
            }
            curve_trends.append(trend)
            trend_rows.append(trend)

        for ratio in selected_values:
            match = next(
                (row for row in curve_trends if abs(float(row["ratio"]) - ratio) < 1e-8),
                None,
            )
            if match is None:
                raise ValueError(f"Masking curve {key} is missing selected ratio {ratio}")
            selected_rows.append(dict(match))

        auc_rows.append(
            {
                "view_mode": key[0],
                "pattern": key[1],
                "corruption": key[2],
                "temporal_span": key[3],
                "ratio_min": curve_trends[0]["ratio"],
                "ratio_max": curve_trends[-1]["ratio"],
                "point_count": len(curve_trends),
                "normalized_mpjpe_auc": _normalized_auc(
                    [
                        (float(row["ratio"]), float(row["normalized_mpjpe"]))
                        for row in curve_trends
                    ]
                ),
                "normalized_acceleration_auc": _normalized_auc(
                    [
                        (
                            float(row["ratio"]),
                            float(row["normalized_acceleration_error"]),
                        )
                        for row in curve_trends
                    ]
                ),
            }
        )
    if not grouped:
        raise ValueError("No masking curves remained after filtering")
    return selected_rows, auc_rows, trend_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_trends(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    patterns = sorted({str(row["pattern"]) for row in rows})
    fig, axes = plt.subplots(
        1, len(patterns), figsize=(5.2 * len(patterns), 4.4), squeeze=False
    )
    colors = {"left": "tab:blue", "right": "tab:orange", "both": "tab:red"}
    for axis, pattern in zip(axes[0], patterns):
        for view_mode in ("left", "right", "both"):
            curve = sorted(
                (
                    row
                    for row in rows
                    if row["pattern"] == pattern and row["view_mode"] == view_mode
                ),
                key=lambda row: float(row["ratio"]),
            )
            if not curve:
                continue
            axis.plot(
                [100.0 * float(row["ratio"]) for row in curve],
                [float(row["normalized_mpjpe"]) for row in curve],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=colors[view_mode],
                label=view_mode,
            )
        axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_title(pattern.capitalize())
        axis.set_xlabel("Corrupted joints / sequence fraction (%)")
        axis.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("Normalized MPJPE (ratio to 0% masking)")
    axes[0][-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "masking_robustness.png", dpi=220)
    fig.savefig(output_dir / "masking_robustness.pdf")
    plt.close(fig)


def _write_report(
    path: Path,
    input_path: Path,
    selected_ratios: Sequence[float],
    max_ratio: float,
    auc_rows: List[Dict[str, Any]],
) -> None:
    lines = [
        "# Unity Masking Robustness Summary",
        "",
        f"- Source: `{input_path.resolve()}`",
        f"- Selected ratios: {', '.join(f'{value:.0%}' for value in selected_ratios)}",
        f"- AUC interval: 0% to {max_ratio:.0%}",
        "- Uncertainty: not estimated; the source is a single corruption sweep.",
        "- Processing: raw points only; no smoothing or interpolation was applied to reported table values.",
        "",
        "## Normalized robustness AUC",
        "",
        "| View | Pattern | MPJPE AUC | Acceleration AUC | Points |",
        "|---|---|---:|---:|---:|",
    ]
    for row in auc_rows:
        lines.append(
            f"| {row['view_mode']} | {row['pattern']} | "
            f"{float(row['normalized_mpjpe_auc']):.4f} | "
            f"{float(row['normalized_acceleration_auc']):.4f} | "
            f"{int(row['point_count'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_ratios(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "logs/eval_unity_masking/occlusion_summary_last.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "logs/eval_unity_masking/paper_summary",
    )
    parser.add_argument("--selected-ratios", default="0,0.1,0.2,0.3,0.5")
    parser.add_argument("--max-ratio", type=float, default=0.5)
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    selected_ratios = _parse_ratios(args.selected_ratios)
    selected, auc_rows, trend_rows = summarize_masking_rows(
        source_rows,
        selected_ratios=selected_ratios,
        max_ratio=float(args.max_ratio),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "masking_selected_points.csv", selected)
    _write_csv(output_dir / "masking_auc.csv", auc_rows)
    _write_csv(output_dir / "masking_trends.csv", trend_rows)
    _plot_trends(trend_rows, output_dir)
    _write_report(
        output_dir / "masking_summary.md",
        input_path=input_path,
        selected_ratios=selected_ratios,
        max_ratio=float(args.max_ratio),
        auc_rows=auc_rows,
    )
    source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    (output_dir / "masking_summary_metadata.json").write_text(
        json.dumps(
            {
                "experiment": "unity_masking_summary",
                "source": str(input_path),
                "source_sha256": source_hash,
                "selected_ratios": selected_ratios,
                "max_ratio": float(args.max_ratio),
                "uncertainty": "not estimated; single corruption sweep",
                "smoothing": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved masking paper summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
