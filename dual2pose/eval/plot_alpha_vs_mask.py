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

BASE_SKI = Path(
    "/home/kaixu_chen/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_ski_poseptz_masking"
)
BASE_UNI = Path(
    "/home/kaixu_chen/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/eval_unity_masking"
)
OUT_DIR = Path("/home/kaixu_chen/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/alpha_vs_mask_plots")
OUT_DIR.mkdir(exist_ok=True)

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
    parts = [int(x) for x in re.findall(r"\d+", m.group("prefix"))]
    ratio = float(parts[-1]) / 100.0
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

    return saved_paths


if __name__ == "__main__":
    main()
