import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def find_stats_csvs(root):
    pattern = os.path.join(root, '*/stats_masked_vs_unmasked_*.csv')
    return sorted(glob.glob(pattern))


def load_and_tag(csv_path):
    df = pd.read_csv(csv_path)
    setting = os.path.basename(os.path.dirname(csv_path))
    df['setting'] = setting
    return df


def aggregate(csv_paths):
    dfs = [load_and_tag(p) for p in csv_paths]
    if not dfs:
        return pd.DataFrame()
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def plot_bar_with_significance(df, out_dir, side='left'):
    joints = df['joint_index'].values
    mean_mask = df[f'mean_mask_{side}'].values
    std_mask = df.get(f'std_mask_{side}', pd.Series(np.zeros_like(mean_mask))).values
    mean_unmask = df.get(f'mean_unmask_{side}', pd.Series(np.nan * np.ones_like(mean_mask))).values
    std_unmask = df.get(f'std_unmask_{side}', pd.Series(np.zeros_like(mean_mask))).values
    pvals = df.get(f'pval_{side}', pd.Series(np.nan * np.ones_like(mean_mask))).values
    cohen = df.get(f'cohen_d_{side}', pd.Series(np.nan * np.ones_like(mean_mask))).values

    x = np.arange(len(joints))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(joints)*0.2), 4))
    ax.bar(x - width/2, mean_mask, width, yerr=std_mask, label='masked', color='tab:blue')
    ax.bar(x + width/2, mean_unmask, width, yerr=std_unmask, label='unmasked', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels(joints, rotation=90)
    ax.set_ylabel('alpha mean')
    ax.set_title(f'{os.path.basename(out_dir)} {side} masked vs unmasked')
    ax.legend()

    # significance markers
    for i, p in enumerate(pvals):
        if np.isfinite(p) and p < 0.001:
            sig = '***'
        elif np.isfinite(p) and p < 0.01:
            sig = '**'
        elif np.isfinite(p) and p < 0.05:
            sig = '*'
        else:
            sig = ''
        if sig:
            y = max(np.nanmax([mean_mask[i] + (std_mask[i] if i < len(std_mask) else 0),
                               mean_unmask[i] + (std_unmask[i] if i < len(std_unmask) else 0)]), 0)
            ax.text(i, y + 0.01, sig, ha='center', va='bottom', color='red', fontsize=8)

    fpath = os.path.join(out_dir, f'bar_{side}.png')
    fig.tight_layout()
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def plot_cohen_heatmap(df, out_dir):
    # create heatmap of cohen's d for left/right across joints
    joints = df['joint_index'].astype(int).values
    cohen_left = df.get('cohen_d_left', pd.Series(np.nan * np.ones(len(joints)))).values
    cohen_right = df.get('cohen_d_right', pd.Series(np.nan * np.ones(len(joints)))).values
    mat = np.vstack([cohen_left, cohen_right])
    fig, ax = plt.subplots(figsize=(max(6, len(joints)*0.2), 3))
    sns.heatmap(mat, annot=True, fmt='.2f', xticklabels=joints, yticklabels=['left','right'], cmap='vlag', center=0, ax=ax)
    ax.set_title(f'{os.path.basename(out_dir)} Cohen\'s d heatmap')
    fpath = os.path.join(out_dir, 'cohen_heatmap.png')
    fig.tight_layout()
    fig.savefig(fpath)
    plt.close(fig)
    return fpath


def main(root='logs/extreme_tests/stats', out_root=None):
    root = os.path.abspath(root)
    if out_root is None:
        out_root = root
    csvs = find_stats_csvs(root)
    if not csvs:
        print('no stats CSVs found under', root)
        return 1
    merged = aggregate(csvs)
    merged_path = os.path.join(out_root, 'summary_stats_merged.csv')
    merged.to_csv(merged_path, index=False)
    print('wrote', merged_path)

    # generate per-setting plots
    by_setting = merged.groupby('setting')
    for setting, g in by_setting:
        out_dir = os.path.join(out_root, setting, 'plots')
        os.makedirs(out_dir, exist_ok=True)
        print('plotting', setting, '->', out_dir)
        plot_bar_with_significance(g.reset_index(drop=True), out_dir, side='left')
        plot_bar_with_significance(g.reset_index(drop=True), out_dir, side='right')
        plot_cohen_heatmap(g.reset_index(drop=True), out_dir)

    print('done')
    return 0


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('root', nargs='?', default='logs/extreme_tests/stats')
    p.add_argument('--out-root', default=None)
    args = p.parse_args()
    raise SystemExit(main(args.root, args.out_root))
