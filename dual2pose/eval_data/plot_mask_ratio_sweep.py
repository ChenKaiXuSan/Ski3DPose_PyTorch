#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_dir', type=Path)
    parser.add_argument('--summary_csv', type=Path, default=None)
    args = parser.parse_args()
    sweep_dir = args.sweep_dir
    summary_dir = sweep_dir / 'summary'
    summary_csv = args.summary_csv or (summary_dir / 'mask_ratio_sweep_last.csv')
    if not summary_csv.exists():
        raise FileNotFoundError(f'summary csv not found: {summary_csv}')
    df = pd.read_csv(summary_csv)
    # keep only random pattern rows
    df = df[df.mask_pattern == 'random']
    # convert mask_ratio to numeric
    df['mask_ratio'] = df['mask_ratio'].astype(float)

    # pivot for alpha
    modes = df['mask_view_mode'].unique()
    fig, ax1 = plt.subplots(figsize=(8,5))
    colors = {'left':'tab:blue','right':'tab:orange','both':'tab:green','none':'tab:gray'}
    for mode in modes:
        sub = df[df.mask_view_mode==mode].sort_values('mask_ratio')
        ax1.plot(sub['mask_ratio']*100, sub['alpha_global_mean'], marker='o', label=f'{mode} alpha_mean', color=colors.get(mode,'k'))
    ax1.set_xlabel('mask ratio (%)')
    ax1.set_ylabel('alpha_global_mean (left weight)')
    ax1.set_ylim(0,1)
    ax1.grid(True, axis='y', alpha=0.3)

    # secondary axis for fused mpjpe
    ax2 = ax1.twinx()
    for mode in modes:
        sub = df[df.mask_view_mode==mode].sort_values('mask_ratio')
        ax2.plot(sub['mask_ratio']*100, sub['mpjpe'], marker='x', linestyle='--', label=f'{mode} mpjpe', color=colors.get(mode,'k'))
    ax2.set_ylabel('fused mpjpe')

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize='small')

    out = sweep_dir / 'mask_ratio_sweep_summary_plot.png'
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print('Saved plot to', out)

if __name__ == '__main__':
    main()
