#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import numpy as np
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_outputs(summary_dir: Path):
    p = summary_dir / 'outputs.pt'
    if not p.exists():
        raise FileNotFoundError(p)
    return torch.load(p)


def compute_stats_from_arrays(A, M):
    # A: numpy (N,T,J), M: bool numpy (N,T,J)
    A2 = A.reshape(-1, A.shape[-1])
    M2 = M.reshape(-1, M.shape[-1])
    J = A2.shape[1]
    results = []
    for j in range(J):
        sel = M2[:, j]
        vals_m = A2[sel, j]
        vals_u = A2[~sel, j]
        n_m = vals_m.size
        n_u = vals_u.size
        mean_m = float(np.nan) if n_m==0 else float(vals_m.mean())
        std_m = float(np.nan) if n_m==0 else float(vals_m.std(ddof=1))
        mean_u = float(np.nan) if n_u==0 else float(vals_u.mean())
        std_u = float(np.nan) if n_u==0 else float(vals_u.std(ddof=1))
        delta = mean_m - mean_u if (not math.isnan(mean_m) and not math.isnan(mean_u)) else float('nan')
        # cohen's d
        cohen_d = float('nan')
        if n_m>1 and n_u>1:
            pooled = math.sqrt(((n_m-1)*(std_m**2) + (n_u-1)*(std_u**2)) / (n_m + n_u - 2)) if (n_m + n_u -2)>0 else float('nan')
            if pooled > 0 and not math.isnan(pooled):
                cohen_d = delta / pooled
        # p-value approx: if both n>=30 use normal approx of difference
        pval = float('nan')
        if n_m >= 30 and n_u >= 30 and (not math.isnan(std_m)) and (not math.isnan(std_u)):
            se = math.sqrt((std_m**2)/n_m + (std_u**2)/n_u)
            if se > 0:
                z = delta / se
                # two-sided p using normal CDF
                pval = 2.0 * (1.0 - 0.5*(1.0 + math.erf(abs(z)/math.sqrt(2.0))))
        results.append((mean_m, std_m, n_m, mean_u, std_u, n_u, delta, cohen_d, pval))
    return results


def process_setting(setting_dir: Path, out_dir: Path):
    summary_dir = setting_dir / 'summary'
    try:
        outputs = load_outputs(summary_dir)
    except FileNotFoundError:
        print('no outputs.pt for', setting_dir)
        return
    alphas = []
    masks_left = []
    masks_right = []
    for out in outputs:
        a = out.get('alpha')
        if a is None or not isinstance(a, torch.Tensor):
            continue
        a = a.detach().cpu()
        if a.ndim == 4 and a.shape[-1] == 1:
            alphas.append(a.squeeze(-1).numpy())
        elif a.ndim == 3:
            alphas.append(a.numpy())
        else:
            raise RuntimeError(f'Unexpected alpha shape: {a.shape}')
        ml = out.get('mask_left')
        mr = out.get('mask_right')
        if ml is not None:
            masks_left.append(ml.detach().cpu().squeeze(-1).numpy())
        if mr is not None:
            masks_right.append(mr.detach().cpu().squeeze(-1).numpy())
    if not alphas:
        print('no alpha tensors in', setting_dir)
        return
    A_cat = np.concatenate(alphas, axis=0)  # (N,T,J)
    if masks_left:
        ML_cat = np.concatenate(masks_left, axis=0).astype(bool)
    else:
        ML_cat = np.zeros_like(A_cat, dtype=bool)
    if masks_right:
        MR_cat = np.concatenate(masks_right, axis=0).astype(bool)
    else:
        MR_cat = np.zeros_like(A_cat, dtype=bool)

    # compute stats for left and right masks
    left_stats = compute_stats_from_arrays(A_cat, ML_cat)
    right_stats = compute_stats_from_arrays(A_cat, MR_cat)

    # write CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'stats_masked_vs_unmasked_{setting_dir.name}.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['joint_index','mean_mask_left','std_mask_left','n_mask_left','mean_unmask_left','std_unmask_left','n_unmask_left','delta_left','cohen_d_left','pval_left',
                         'mean_mask_right','std_mask_right','n_mask_right','mean_unmask_right','std_unmask_right','n_unmask_right','delta_right','cohen_d_right','pval_right'])
        for j in range(len(left_stats)):
            l = left_stats[j]
            r = right_stats[j]
            writer.writerow([j, *l, *r])
    print('wrote', csv_path)

    # plot left bar with significance markers
    J = A_cat.shape[-1]
    x = np.arange(J)
    left_means = np.array([s[0] for s in left_stats])
    left_stds = np.array([s[1] for s in left_stats])
    left_ns = np.array([s[2] for s in left_stats])
    left_p = np.array([s[8] for s in left_stats])

    fig, ax = plt.subplots(figsize=(max(10, J*0.6), 4.8))
    width = 0.35
    # unmasked means
    left_un_means = np.array([s[3] for s in left_stats])
    left_un_stds = np.array([s[4] for s in left_stats])
    ax.bar(x - width/2, left_means, width, yerr=left_stds, label='left_masked', color='tab:blue', capsize=3)
    ax.bar(x + width/2, left_un_means, width, yerr=left_un_stds, label='left_unmasked', color='tab:orange', capsize=3)
    ax.set_ylim(0,1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'j{j:02d}' for j in range(J)], rotation=35, ha='right')
    ax.set_ylabel('alpha (left weight)')
    ax.set_title(f'Left-masked vs unmasked alpha: {setting_dir.name}')
    # mark significant (p<0.05)
    for j, p in enumerate(left_p):
        if not math.isnan(p) and p < 0.05:
            ax.text(j, max(left_means[j], left_un_means[j]) + 0.05, '*', ha='center', va='bottom', color='red', fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f'signif_left_{setting_dir.name}.png', dpi=180)
    plt.close(fig)

    # plot right
    right_means = np.array([s[0] for s in right_stats])
    right_stds = np.array([s[1] for s in right_stats])
    right_un_means = np.array([s[3] for s in right_stats])
    right_un_stds = np.array([s[4] for s in right_stats])
    right_p = np.array([s[8] for s in right_stats])
    fig, ax = plt.subplots(figsize=(max(10, J*0.6), 4.8))
    ax.bar(x - width/2, right_means, width, yerr=right_stds, label='right_masked', color='tab:green', capsize=3)
    ax.bar(x + width/2, right_un_means, width, yerr=right_un_stds, label='right_unmasked', color='tab:red', capsize=3)
    ax.set_ylim(0,1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'j{j:02d}' for j in range(J)], rotation=35, ha='right')
    ax.set_ylabel('alpha (left weight)')
    ax.set_title(f'Right-masked vs unmasked alpha: {setting_dir.name}')
    for j, p in enumerate(right_p):
        if not math.isnan(p) and p < 0.05:
            ax.text(j, max(right_means[j], right_un_means[j]) + 0.05, '*', ha='center', va='bottom', color='red', fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f'signif_right_{setting_dir.name}.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=Path, help='parent dir with multiple setting subdirs')
    parser.add_argument('--out-root', type=Path, default=None)
    args = parser.parse_args()
    root = args.root_dir
    out_root = args.out_root or (root / 'stats')
    out_root.mkdir(parents=True, exist_ok=True)
    # find setting dirs (assume immediate children)
    for setting in sorted([p for p in root.iterdir() if p.is_dir()] ):
        try:
            od = out_root / setting.name
            process_setting(setting, od)
        except Exception as e:
            print('failed for', setting, e)

if __name__ == '__main__':
    main()
