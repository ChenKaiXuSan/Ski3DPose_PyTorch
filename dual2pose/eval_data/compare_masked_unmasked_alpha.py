#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_outputs_pt(summary_dir: Path):
    p = summary_dir / 'outputs.pt'
    if not p.exists():
        raise FileNotFoundError(f'outputs.pt not found: {p}')
    return torch.load(p)


def concat_tensors(chunks):
    return torch.cat([c.detach().cpu() for c in chunks], dim=0)


def compute_stats(alpha_cat, mask_cat):
    # alpha_cat: (N,T,J), mask_cat: (N,T,J) bool
    A = alpha_cat.reshape(-1, alpha_cat.shape[-1])
    M = mask_cat.reshape(-1, mask_cat.shape[-1]).astype(bool)
    J = A.shape[1]
    means_mask = []
    stds_mask = []
    counts_mask = []
    means_unm = []
    stds_unm = []
    counts_unm = []
    for j in range(J):
        sel = M[:, j]
        vals = A[sel, j]
        if vals.size == 0:
            means_mask.append(float('nan'))
            stds_mask.append(float('nan'))
            counts_mask.append(0)
        else:
            means_mask.append(float(vals.mean()))
            stds_mask.append(float(vals.std()))
            counts_mask.append(int(vals.size))
        sel_un = ~sel
        vals_un = A[sel_un, j]
        if vals_un.size == 0:
            means_unm.append(float('nan'))
            stds_unm.append(float('nan'))
            counts_unm.append(0)
        else:
            means_unm.append(float(vals_un.mean()))
            stds_unm.append(float(vals_un.std()))
            counts_unm.append(int(vals_un.size))
    return (means_mask, stds_mask, counts_mask), (means_unm, stds_unm, counts_unm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('setting_dir', type=Path, help='setting directory (parent of alpha_vis)')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    setting_dir = args.setting_dir
    summary_dir = setting_dir / 'summary'
    alpha_vis_dir = setting_dir / 'alpha_vis'
    alpha_vis_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.output_dir or alpha_vis_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_outputs_pt(summary_dir)
    alphas = []
    masks_left = []
    masks_right = []
    for out in outputs:
        a = out.get('alpha')
        if a is None or not isinstance(a, torch.Tensor):
            continue
        # ensure shape (B,T,J,1) -> squeeze last dim
        a = a.detach().cpu()
        if a.ndim == 4 and a.shape[-1] == 1:
            alphas.append(a.squeeze(-1))
        elif a.ndim == 3:
            alphas.append(a)
        else:
            raise RuntimeError(f'Unexpected alpha shape: {a.shape}')
        ml = out.get('mask_left')
        mr = out.get('mask_right')
        if ml is not None:
            masks_left.append(ml.detach().cpu())
        if mr is not None:
            masks_right.append(mr.detach().cpu())

    if not alphas:
        raise RuntimeError('No alpha tensors found in outputs.pt')
    alpha_cat = torch.cat(alphas, dim=0).numpy()  # (N,T,J)

    joint_count = alpha_cat.shape[-1]
    joint_names = [f'joint_{i:02d}' for i in range(joint_count)]
    # if mask lists empty, create all-false masks
    if not masks_left:
        masks_left_cat = np.zeros_like(alpha_cat, dtype=bool)
    else:
        masks_left_cat = np.concatenate([m.squeeze(-1).numpy() for m in masks_left], axis=0)
    if not masks_right:
        masks_right_cat = np.zeros_like(alpha_cat, dtype=bool)
    else:
        masks_right_cat = np.concatenate([m.squeeze(-1).numpy() for m in masks_right], axis=0)

    left_stats, left_un = compute_stats(alpha_cat, masks_left_cat)
    right_stats, right_un = compute_stats(alpha_cat, masks_right_cat)

    # write CSV
    csv_path = out_dir / f'masked_vs_unmasked_alpha_{setting_dir.name}.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['joint_index', 'joint_name',
                         'left_masked_mean','left_masked_std','left_masked_count',
                         'left_unmasked_mean','left_unmasked_std','left_unmasked_count',
                         'right_masked_mean','right_masked_std','right_masked_count',
                         'right_unmasked_mean','right_unmasked_std','right_unmasked_count'])
        for j in range(joint_count):
            row = [j, joint_names[j]]
            row += [left_stats[0][j], left_stats[1][j], left_stats[2][j]]
            row += [left_un[0][j], left_un[1][j], left_un[2][j]]
            row += [right_stats[0][j], right_stats[1][j], right_stats[2][j]]
            row += [right_un[0][j], right_un[1][j], right_un[2][j]]
            writer.writerow(row)

    # plot left comparison
    x = np.arange(joint_count)
    lmean_mask = np.array(left_stats[0])
    lstd_mask = np.array(left_stats[1])
    lmean_un = np.array(left_un[0])
    lstd_un = np.array(left_un[1])

    fig, ax = plt.subplots(figsize=(max(10, joint_count * 0.6), 4.8))
    width = 0.35
    ax.bar(x - width/2, lmean_mask, width, yerr=lstd_mask, label='left_masked', color='tab:blue', capsize=3)
    ax.bar(x + width/2, lmean_un, width, yerr=lstd_un, label='left_unmasked', color='tab:orange', capsize=3)
    ax.set_ylim(0,1)
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=35, ha='right')
    ax.set_ylabel('alpha (left weight)')
    ax.set_title(f'Left-masked vs unmasked alpha: {setting_dir.name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f'masked_vs_unmasked_alpha_left_{setting_dir.name}.png', dpi=180)
    plt.close(fig)

    # plot right comparison (note: alpha is left weight; comparing when right masked)
    rmean_mask = np.array(right_stats[0])
    rstd_mask = np.array(right_stats[1])
    rmean_un = np.array(right_un[0])
    rstd_un = np.array(right_un[1])

    fig, ax = plt.subplots(figsize=(max(10, joint_count * 0.6), 4.8))
    ax.bar(x - width/2, rmean_mask, width, yerr=rstd_mask, label='right_masked', color='tab:green', capsize=3)
    ax.bar(x + width/2, rmean_un, width, yerr=rstd_un, label='right_unmasked', color='tab:red', capsize=3)
    ax.set_ylim(0,1)
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=35, ha='right')
    ax.set_ylabel('alpha (left weight)')
    ax.set_title(f'Right-masked vs unmasked alpha: {setting_dir.name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f'masked_vs_unmasked_alpha_right_{setting_dir.name}.png', dpi=180)
    plt.close(fig)

    print('Wrote CSV:', csv_path)
    print('Wrote plots to:', out_dir)


if __name__ == '__main__':
    main()
