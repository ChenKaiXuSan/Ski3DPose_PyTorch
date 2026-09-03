#!/usr/bin/env python3
"""Render manuscript tables and a figure from validated E4/E5 source artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt


EXPECTED_CHECKPOINT_SHA256 = (
    "869a2217f8676c0ada75ed3c9a3c82a9b8efbb105749f6ffb8bef71e9172f50f"
)
ANGLE_LABELS = ("0-30", "30-60", "60-90", "90-120", "120-150", "150-180")
PATTERNS = ("random", "distal", "temporal")
VIEWS = ("left", "right", "both")
RATIOS = (0.5, 1.0)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing numeric field {key!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"Field {key!r} must be finite")
    return value


def _format_p_value(value: float) -> str:
    return r"\(<0.001\)" if value < 0.001 else f"{value:.3g}"


def _validate_e4(
    significance_path: Path, statistics_path: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = _read_csv(significance_path)
    if len(rows) != 6 or {row.get("angle_bin") for row in rows} != set(ANGLE_LABELS):
        raise ValueError("E4 significance source must contain exactly six angle bins")
    for row in rows:
        for key in (
            "cluster_count",
            "mean_gain_mpjpe",
            "median_gain_mpjpe",
            "mean_gain_ci95_low",
            "mean_gain_ci95_high",
            "rank_biserial",
            "p_holm",
        ):
            _finite(row, key)
    payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    if payload.get("provenance", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("E4 checkpoint hash does not match the frozen manuscript model")
    omnibus = payload.get("omnibus")
    if not isinstance(omnibus, Mapping):
        raise ValueError("E4 statistics JSON lacks an omnibus result")
    _finite(omnibus, "p_value")
    _finite(omnibus, "epsilon_squared")
    return sorted(rows, key=lambda row: ANGLE_LABELS.index(row["angle_bin"])), payload


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str, float]:
    return str(row.get("view_mode")), str(row.get("pattern")), float(row.get("ratio", "nan"))


def _validate_e5(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    expected = {(view, pattern, ratio) for view in VIEWS for pattern in PATTERNS for ratio in RATIOS}
    if len(rows) != 18 or {_cell_key(row) for row in rows} != expected:
        raise ValueError("E5 summary source must contain the complete 18-cell grid")
    for row in rows:
        if row.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("E5 checkpoint hash does not match the frozen manuscript model")
        if int(row.get("sample_count", -1)) != 64_440 or int(row.get("joint_count", -1)) != 15:
            raise ValueError("Every E5 cell must contain 64440 all-15-joint samples")
        for key in (
            "fused_mpjpe",
            "canonical_avg_mpjpe",
            "fusion_gain_percent",
            "sam3d_detection_failure_rate",
        ):
            _finite(row, key)
    return sorted(
        rows,
        key=lambda row: (
            VIEWS.index(row["view_mode"]),
            PATTERNS.index(row["pattern"]),
            RATIOS.index(float(row["ratio"])),
        ),
    )


def _validate_comparison(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    expected = {(view, pattern, ratio) for view in VIEWS for pattern in PATTERNS for ratio in RATIOS}
    if len(rows) != 18 or {_cell_key(row) for row in rows} != expected:
        raise ValueError("Image-vs-pose source must contain the complete 18-cell grid")
    for row in rows:
        _finite(row, "image_fused_mpjpe")
        _finite(row, "pose_fused_mpjpe")
    return rows


def _render_e4_table(rows: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> str:
    body = []
    for row in rows:
        body.append(
            f"{row['angle_bin'].replace('-', '--')} & {int(float(row['cluster_count'])):,} & "
            f"{float(row['mean_gain_mpjpe']):.4f} & "
            f"[{float(row['mean_gain_ci95_low']):.4f}, {float(row['mean_gain_ci95_high']):.4f}] & "
            f"{float(row['rank_biserial']):.3f} & {_format_p_value(float(row['p_holm']))} \\\\"
        )
    omnibus = payload["omnibus"]
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Cluster-aware inference for fusion gain by camera separation.}",
            r"\label{tab:view_angle_significance}",
            r"\footnotesize",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Angle (deg) & Pairs & Mean gain & 95\% CI & $r_{rb}$ & $p_{\mathrm{Holm}}$ \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.3em}",
            (
                r"\parbox{0.96\textwidth}{\footnotesize \textit{Note.} Positive gain favors "
                r"CanonFuse3D. Confidence intervals use camera-pair bootstrap resampling; "
                r"reported $p$ values are Holm-adjusted across six paired Wilcoxon tests. "
                f"The Kruskal--Wallis angle effect is $p={float(omnibus['p_value']):.3g}$ "
                f"with $\\epsilon^2={float(omnibus['epsilon_squared']):.2g}$.}}"
            ),
            r"\end{table*}",
            "",
        ]
    )


def _render_e5_table(rows: Sequence[Mapping[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            f"{row['view_mode'].capitalize()} & {row['pattern'].capitalize()} & "
            f"{float(row['ratio']):.1f} & {float(row['fused_mpjpe']):.4f} & "
            f"{float(row['canonical_avg_mpjpe']):.4f} & "
            f"{float(row['fusion_gain_percent']):.1f} & "
            f"{100.0 * float(row['sam3d_detection_failure_rate']):.2f} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Image-level occlusion propagated through SAM3D and frozen CanonFuse3D.}",
            r"\label{tab:image_occlusion}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{lllrrrr}",
            r"\toprule",
            r"View & Pattern & Ratio & Fused & Canonical avg. & Gain (\%) & SAM3D fail (\%) \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.3em}",
            r"\parbox{0.96\textwidth}{\footnotesize \textit{Note.} Image-level masks are placed from Unity 2D annotations, which are not passed to SAM3D or fusion. Each row evaluates all 64,440 pairs; negative gains are retained.}",
            r"\end{table*}",
            "",
        ]
    )


def _render_comparison_figure(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    colors = {"random": "#D1495B", "distal": "#00798C", "temporal": "#5B8E7D"}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for axis, view in zip(axes, VIEWS):
        for pattern in PATTERNS:
            selected = sorted(
                [row for row in rows if _cell_key(row)[:2] == (view, pattern)],
                key=lambda row: float(row["ratio"]),
            )
            x = [float(row["ratio"]) for row in selected]
            axis.plot(
                x,
                [float(row["image_fused_mpjpe"]) for row in selected],
                color=colors[pattern],
                marker="o",
                linewidth=1.8,
                label=f"{pattern.capitalize()} / image",
            )
            axis.plot(
                x,
                [float(row["pose_fused_mpjpe"]) for row in selected],
                color=colors[pattern],
                marker="s",
                linestyle="--",
                linewidth=1.4,
                label=f"{pattern.capitalize()} / pose",
            )
        axis.set_title(view.capitalize())
        axis.set_xlabel("Severity ratio")
        axis.set_xticks(RATIOS)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Fused MPJPE")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_all(
    *,
    e4_significance_csv: Path,
    e4_statistics_json: Path,
    e5_summary_csv: Path,
    comparison_csv: Path,
    table_root: Path,
    figure_path: Path,
) -> list[Path]:
    e4_rows, e4_payload = _validate_e4(e4_significance_csv, e4_statistics_json)
    e5_rows = _validate_e5(e5_summary_csv)
    comparison_rows = _validate_comparison(comparison_csv)
    table_root.mkdir(parents=True, exist_ok=True)
    e4_path = table_root / "view_angle_significance.tex"
    e5_path = table_root / "image_occlusion_summary.tex"
    e4_path.write_text(_render_e4_table(e4_rows, e4_payload), encoding="utf-8")
    e5_path.write_text(_render_e5_table(e5_rows), encoding="utf-8")
    _render_comparison_figure(comparison_rows, figure_path)
    return [e4_path, e5_path, figure_path]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e4-significance", type=Path, required=True)
    parser.add_argument("--e4-statistics", type=Path, required=True)
    parser.add_argument("--e5-summary", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in render_all(
        e4_significance_csv=args.e4_significance,
        e4_statistics_json=args.e4_statistics,
        e5_summary_csv=args.e5_summary,
        comparison_csv=args.comparison,
        table_root=args.table_root,
        figure_path=args.figure,
    ):
        print(path)


if __name__ == "__main__":
    main()
