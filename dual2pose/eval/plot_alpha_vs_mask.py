"""Plot alpha (fusion weight) and MPJPE vs occlusion mask ratio.

Averages across all sub-patterns (distal / random / temporal) at each ratio.
Smoothed with Savitzky-Golay filter for readability.

Shows Ski-PosePTZ and Unity side-by-side:
  Row 1: alpha vs mask_ratio  (four view modes)
  Row 2: MPJPE vs mask_ratio  (absolute, smoothed)

Usage:
    python scripts/plot_alpha_vs_mask.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SKI = REPO_ROOT / "logs" / "eval_ski_poseptz_masking"
BASE_UNI = REPO_ROOT / "logs" / "eval_unity_masking"
OUT_DIR = REPO_ROOT / "logs" / "alpha_vs_mask_plots"
PAPER_FIG_DIR = (
    REPO_ROOT / "6a33ba78cba883c26fb3823f" / "figure" / "experiment2"
)
OUT_DIR.mkdir(exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────

DIR_PATTERN = re.compile(
    r"^(?P<view>none|left|right|both)_(?P<pattern>distal|random|temporal)"
    r"_r(?P<prefix>\d+(?:p\d+)*)$"
)

VIEW_COLORS: Dict[str, str] = {
    "left": "#E63946",
    "right": "#219EBB",
    "none": "#6C757D",
    "both": "#F4A261",
}

VIEW_LABELS: Dict[str, str] = {
    "left": "Left masked",
    "right": "Right masked",
    "none": "None (baseline)",
    "both": "Both masked",
}

PATTERN_LABELS: Dict[str, str] = {
    "distal": "Distal",
    "random": "Random",
    "temporal": "Temporal",
}

VIEW_ORDER = ("both", "left", "right", "none")


# ── helpers ──────────────────────────────────────────────────────────────────


def _parse_dirname(name: str) -> dict | None:
    """Extract view, pattern, and ratio from a directory name.

    Expects format like ``left_distal_r10p25`` → {"view": "left", "pattern": "distal", "ratio": 0.25}
    Returns ``None`` if the name doesn't match.
    """
    m = DIR_PATTERN.match(name)
    if not m:
        return None
    ratio = float(m.group("prefix").replace("p", "."))
    return {"view": m.group("view"), "pattern": m.group("pattern"), "ratio": ratio}


def _read_report(path: Path) -> dict | None:
    """Parse a single line from a comparison report file.

    Looks for ``Mask ratio:``, ``alpha_global_mean:``, and ``fused_mpjpe:`` lines.
    Returns ``None`` if the file can't be read or has no mask ratio.
    """
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return None

    info: dict = {}
    for line in lines:
        if "Mask ratio:" in line:
            info["ratio"] = float(line.split(":")[1].strip())
        elif "alpha_global_mean:" in line:
            info["alpha"] = float(line.split(":")[1].strip())
        elif "fused_mpjpe:" in line:
            info["mpjpe"] = float(line.split(":")[1].strip())

    return info if "ratio" in info else None


def _collect_data(base: Path) -> Dict[Tuple[str, str], List[Tuple[float, float, float]]]:
    """Walk *base* and collect ``(ratio, alpha, mpjpe)`` per (view, pattern).

    Searches for ``comparison_report_<dirname>_last.txt`` (falling back to
    ``comparison_report_<dirname>.txt``) inside each sub-directory.
    """
    result: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue

        info = _parse_dirname(d.name)
        if info is None:
            continue

        view_key = info["view"]  # already validated by regex
        key = (view_key, info["pattern"])

        report_path_last = Path(d) / f"comparison_report_{d.name}_last.txt"
        report_path_txt = Path(d) / f"comparison_report_{d.name}.txt"
        report = _read_report(report_path_last) or _read_report(report_path_txt)
        if report is None:
            continue

        result.setdefault(key, []).append(
            (info["ratio"], report["alpha"], report["mpjpe"]),
        )
    return result


# ── aggregation ──────────────────────────────────────────────────────────────


def _agg(data: Dict[str, List]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group by (view, pattern) → ``(ratios, alphas, mpjpes)`` arrays.

    Within each (view, pattern) bucket, samples sharing the same ratio (rounded
    to 2 decimal places) are averaged together.
    """
    buckets: Dict[str, Dict[float, List[Tuple[float, float]]]] = {}
    for key in data:
        buckets[key] = {}
        for ratio, alpha, mpjpe in data[key]:
            r = round(ratio, 2)
            buckets[key].setdefault(r, []).append((alpha, mpjpe))

    result = {}
    for key in sorted(buckets):
        if not buckets[key]:
            continue
        ratios = sorted(buckets[key].keys())
        alphas = [np.mean([v[0] for v in vals]) for vals in buckets[key].values()]
        mpjpes = [np.mean([v[1] for v in vals]) for vals in buckets[key].values()]
        result[key] = (np.array(ratios), np.array(alphas), np.array(mpjpes))
    return result


def _sgolay_smooth(
    y: np.ndarray, window: int = 5, polyorder: int = 2
) -> np.ndarray:
    """Savitzky-Golay filter (from scratch).

    For each point fits a polynomial of *polyorder* over the window and
    evaluates at the point's position. Edges are handled by clamping to
    available data; points without enough neighbors are left unchanged.
    """
    n = len(y)
    half = window // 2
    y_out = y.copy()

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < polyorder + 1:
            continue
        coeffs = np.polyfit(np.arange(lo, hi), y[lo:hi], polyorder)
        y_out[i] = np.polyval(coeffs, i)

    return y_out


def _expand_none_baseline(agg: Dict[str, Tuple]) -> Dict[str, Tuple]:
    """Broadcast the 'none' baseline to all ratios seen in other views.

    If 'none' has only a single ratio point but other views span multiple
    ratios, tile its alpha / MPJPE across all those ratios so every subplot
    shares a common x-axis.
    """
    if "none" not in agg:
        return agg

    n_ratios = agg["none"][0]
    n_alphas = agg["none"][1]
    n_mpjpes = agg["none"][2]

    all_ratios: set = set()
    for key in agg:
        if key != "none":
            all_ratios.update(agg[key][0])
    all_ratios = sorted(all_ratios)

    if len(n_ratios) == 1 and len(all_ratios) > 1:
        base_alpha = n_alphas[0]
        base_mpjpe = n_mpjpes[0]
        agg["none"] = (
            np.array(all_ratios),
            np.full(len(all_ratios), base_alpha),
            np.full(len(all_ratios), base_mpjpe),
        )
    return agg


# ── plot ─────────────────────────────────────────────────────────────────────


def _baseline_mpjpe(agg_data: Dict[Tuple[str, str], Tuple]) -> float | None:
    baseline_values = []
    for (view, _pattern), vals in agg_data.items():
        if view != "none":
            continue
        _ratios, _alphas, mpjpes = vals
        if len(mpjpes):
            baseline_values.append(float(mpjpes[0]))
    if not baseline_values:
        return None
    return float(np.mean(baseline_values))


def _build_both_view_paper_trends(
    agg_data: Dict[Tuple[str, str], Tuple],
) -> Dict[str, Dict[str, List[float]]]:
    """Return both-view MPJPE trends with one shared no-mask baseline at x=0."""
    baseline = _baseline_mpjpe(agg_data)
    trends: Dict[str, Dict[str, List[float]]] = {}

    patterns = sorted(pat for view, pat in agg_data if view == "both")
    for pattern in patterns:
        ratios, _alphas, mpjpes = agg_data[("both", pattern)]
        ratio_vals = [round(float(r), 2) for r in ratios]
        mpjpe_vals = [float(m) for m in mpjpes]

        if baseline is not None:
            if ratio_vals and ratio_vals[0] == 0.0:
                mpjpe_vals[0] = baseline
            else:
                ratio_vals.insert(0, 0.0)
                mpjpe_vals.insert(0, baseline)

        trends[pattern] = {
            "ratio": ratio_vals,
            "mpjpe": [round(v, 4) for v in mpjpe_vals],
        }
    return trends


def _make_pattern_fig(agg: Dict[str, Tuple], pattern: str, dataset: str) -> plt.Figure:
    """Create the two-row figure for one (dataset, pattern)."""
    agg = _expand_none_baseline(agg)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    dataset_label = dataset.title().replace("_", " ")
    fig.suptitle(
        f"{PATTERN_LABELS.get(pattern, pattern)} -- {dataset_label}",
        fontsize=14, fontweight="bold", y=0.985,
    )

    # --- alpha subplot ---------------------------------------------------------
    ax_alpha = axes[0]
    ax_alpha.axhline(0.5, ls="--", color="#333", lw=0.8, zorder=1)

    for key in VIEW_ORDER:
        if key not in agg:
            continue
        ratios, alphas, _mpjpes = agg[key]
        color = VIEW_COLORS.get(key, "#888")
        label = VIEW_LABELS[key]
        n_pts = len(alphas)

        # Choose smoothing window自适应数据量
        if n_pts > 3:
            win = min(5, n_pts - 1)
        elif n_pts >= 2:
            win = max(2, n_pts)
        else:
            win = n_pts

        smooth_a = _sgolay_smooth(alphas, window=win, polyorder=2)
        ax_alpha.plot(
            ratios, smooth_a, "o-", color=color, label=label,
            linewidth=2.0, markersize=6, zorder=3,
        )

    ax_alpha.set_ylabel("alpha (left weight)")
    ax_alpha.set_title("(a) alpha vs occlusion", fontweight="bold", fontsize=12)
    ax_alpha.legend(fontsize=9, loc="lower left")
    ax_alpha.set_ylim(0.35, 0.70)
    ax_alpha.grid(alpha=0.3)

    # --- MPJPE subplot (absolute) ----------------------------------------------
    ax_mpjpe = axes[1]

    for key in VIEW_ORDER:
        if key not in agg:
            continue
        ratios, _alphas, mpjpes = agg[key]
        color = VIEW_COLORS.get(key, "#888")
        label = VIEW_LABELS[key]
        n_pts = len(mpjpes)

        if n_pts > 3:
            win = min(5, n_pts - 1)
        elif n_pts >= 2:
            win = max(2, n_pts)
        else:
            win = n_pts

        smooth_m = _sgolay_smooth(mpjpes, window=win, polyorder=2)
        ax_mpjpe.plot(
            ratios, smooth_m, "s-", color=color, label=label,
            linewidth=2.0, markersize=6, zorder=3,
        )

    ax_mpjpe.set_ylabel("MPJPE")
    ax_mpjpe.set_xlabel("Mask ratio")
    ax_mpjpe.set_title("(c) fused MPJPE (absolute)", fontweight="bold", fontsize=12)
    ax_mpjpe.legend(fontsize=9)
    ax_mpjpe.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig



def _make_both_view_paper_fig(paper_data: Dict[str, Dict[str, Dict[str, List[float]]]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    dataset_titles = {
        "ski_poseptz": "Ski-PTZ-Pose",
        "unity": "Unity",
    }
    pattern_colors = {
        "random": "#E63946",
        "distal": "#457B9D",
        "temporal": "#2A9D8F",
    }

    for ax, dataset_name in zip(axes, ["ski_poseptz", "unity"]):
        trends = paper_data.get(dataset_name, {})
        for pattern in ["random", "distal", "temporal"]:
            if pattern not in trends:
                continue
            ratios = trends[pattern]["ratio"]
            mpjpes = trends[pattern]["mpjpe"]
            label = PATTERN_LABELS.get(pattern, pattern)
            ax.plot(
                ratios,
                mpjpes,
                "o-",
                color=pattern_colors.get(pattern, "#555555"),
                label=label,
                linewidth=2.0,
                markersize=4.5,
            )
        ax.set_title(dataset_titles.get(dataset_name, dataset_name), fontweight="bold")
        ax.set_xlabel("Mask ratio")
        ax.set_ylabel("MPJPE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> List[str]:
    """Collect data, aggregate, plot, and save. Returns paths of saved figures."""
    # Collect raw data keyed by (view, pattern)
    print("Collecting Ski-PosePTZ data ...")
    raw_ski = _collect_data(BASE_SKI)
    for k in sorted(raw_ski):
        print(f"  {k}: {len(raw_ski[k])} samples")

    print("Collecting Unity data ...")
    raw_uni = _collect_data(BASE_UNI)
    for k in sorted(raw_uni):
        print(f"  {k}: {len(raw_uni[k])} samples")

    # Aggregate per (view, pattern) pair
    agg_ski = _agg(raw_ski)
    agg_uni = _agg(raw_uni)

    datasets: List[Tuple[str, Dict]] = [
        ("ski_poseptz", agg_ski),
        ("unity", agg_uni),
    ]

    saved_paths: List[str] = []
    for dataset_name, agg_data in datasets:
        patterns = sorted(set(k[1] for k in agg_data))
        for pattern in patterns:
            # Select only the views belonging to this pattern
            agg_single: Dict[str, Tuple] = {}
            for (view, pat), vals in agg_data.items():
                if pat == pattern:
                    agg_single[view] = vals
            if not agg_single:
                continue

            fig = _make_pattern_fig(agg_single, pattern, dataset_name)
            out_path = OUT_DIR / f"alpha_vs_mask_{pattern}_{dataset_name}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            saved_paths.append(str(out_path))
            print(f"Saved -> {out_path}")

    # Save raw aggregated data per pattern as JSON
    json_data: Dict = {}
    for dataset_name, agg_data in datasets:
        json_data[dataset_name] = {}
        for (view, pat), vals in sorted(agg_data.items()):
            ratios, alphas, mpjpes = vals
            if view not in json_data[dataset_name]:
                json_data[dataset_name][view] = {}
            json_data[dataset_name][view][pat] = {
                "ratio": [round(float(r), 2) for r in ratios],
                "alpha_mean": [round(float(a), 4) for a in alphas],
                "mpjpe": [round(float(m), 4) for m in mpjpes],
            }

    json_path = OUT_DIR / "alpha_vs_mask_ratio_data.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Raw data -> {json_path}")

    paper_data = {
        dataset_name: _build_both_view_paper_trends(agg_data)
        for dataset_name, agg_data in datasets
    }
    paper_json_path = OUT_DIR / "masking_both_view_trends_data.json"
    with open(paper_json_path, "w") as f:
        json.dump(paper_data, f, indent=2)
    saved_paths.append(str(paper_json_path))
    print(f"Paper trend data -> {paper_json_path}")

    paper_fig = _make_both_view_paper_fig(paper_data)
    paper_fig_path = PAPER_FIG_DIR / "masking_both_view_trends.png"
    paper_fig.savefig(paper_fig_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(paper_fig)
    saved_paths.append(str(paper_fig_path))
    print(f"Paper figure -> {paper_fig_path}")

    paper_full_json_path = PAPER_FIG_DIR / "alpha_vs_mask_ratio_data.json"
    with open(paper_full_json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    saved_paths.append(str(paper_full_json_path))
    print(f"Paper raw data -> {paper_full_json_path}")

    return saved_paths


if __name__ == "__main__":
    main()
