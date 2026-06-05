import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_merged(path):
    return pd.read_csv(path)


def top_k_joints(df, k=5, side='left'):
    key = f'cohen_d_{side}'
    if key not in df.columns:
        return []
    tmp = df[['setting','joint_index',key]].dropna()
    tmp['absd'] = tmp[key].abs()
    by_setting = tmp.groupby('setting')
    res = {}
    for s, g in by_setting:
        top = g.sort_values('absd', ascending=False).head(k)
        res[s] = top['joint_index'].tolist()
    return res


def parametric_bootstrap(df, n_boot=5000):
    # df contains per-joint mean/std/n for masked/unmasked
    rows = []
    for _, r in df.iterrows():
        ji = int(r['joint_index'])
        for side in ['left','right']:
            mean_m = r.get(f'mean_mask_{side}', np.nan)
            std_m = r.get(f'std_mask_{side}', np.nan)
            n_m = int(r.get(f'n_mask_{side}', 0) or 0)
            mean_u = r.get(f'mean_unmask_{side}', np.nan)
            std_u = r.get(f'std_unmask_{side}', np.nan)
            n_u = int(r.get(f'n_unmask_{side}', 0) or 0)
            out = dict(joint_index=ji, side=side)
            if n_m > 1 and n_u > 1 and np.isfinite(std_m) and np.isfinite(std_u):
                m_samps = np.random.normal(loc=mean_m, scale=std_m, size=(n_boot, n_m))
                u_samps = np.random.normal(loc=mean_u, scale=std_u, size=(n_boot, n_u))
                m_means = m_samps.mean(axis=1)
                u_means = u_samps.mean(axis=1)
                diffs = m_means - u_means
                pval = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
                ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
                out.update(dict(bootstrap_p=pval, diff_ci_lo=ci_lo, diff_ci_hi=ci_hi))
            else:
                out.update(dict(bootstrap_p=np.nan, diff_ci_lo=np.nan, diff_ci_hi=np.nan))
            rows.append(out)
    return pd.DataFrame(rows)


def plot_setting_panel(merged, out_dir, joints=None):
    settings = sorted(merged['setting'].unique())
    if joints is None:
        joints = sorted(merged['joint_index'].unique())[:8]
    joints = list(map(int, joints))
    os.makedirs(out_dir, exist_ok=True)
    # one figure per side
    for side in ['left','right']:
        fig, axes = plt.subplots(len(joints), 1, figsize=(6, 1.6*len(joints)), sharex=True)
        if len(joints) == 1:
            axes = [axes]
        for ax, j in zip(axes, joints):
            rows = merged[merged['joint_index']==j]
            means = []
            errs = []
            for s in settings:
                r = rows[rows['setting']==s]
                if r.empty:
                    means.append(np.nan); errs.append(0)
                else:
                    means.append(float(r[f'mean_mask_{side}'].values[0]))
                    errs.append(float(r.get(f'std_mask_{side}', pd.Series([0])).values[0] or 0))
            ax.errorbar(range(len(settings)), means, yerr=errs, fmt='-o')
            ax.set_ylabel(f'j{j} mean alpha (masked)')
            ax.set_xticks(range(len(settings)))
            ax.set_xticklabels(settings, rotation=45, ha='right')
        fig.tight_layout()
        fpath = os.path.join(out_dir, f'panel_masked_{side}.png')
        fig.savefig(fpath)
        plt.close(fig)


def main(merged_path, out_root, topk=5):
    merged = load_merged(merged_path)
    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)
    # top-k
    tk_left = top_k_joints(merged, k=topk, side='left')
    tk_right = top_k_joints(merged, k=topk, side='right')
    pd.DataFrame([tk_left]).T.to_csv(os.path.join(out_root, 'topk_left_per_setting.csv'))
    pd.DataFrame([tk_right]).T.to_csv(os.path.join(out_root, 'topk_right_per_setting.csv'))

    # parametric bootstrap
    boot = parametric_bootstrap(merged)
    boot.to_csv(os.path.join(out_root, 'parametric_bootstrap_results.csv'), index=False)

    # combined panel plots for top joints across settings (use union of top lists)
    union_j = set()
    for v in tk_left.values(): union_j.update(v)
    for v in tk_right.values(): union_j.update(v)
    union_j = sorted(list(union_j))[:12]
    plot_setting_panel(merged, os.path.join(out_root, 'panels'), joints=union_j)

    print('wrote topk and bootstrap results to', out_root)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('merged', help='path to summary_stats_merged.csv')
    p.add_argument('--out-root', default='logs/extreme_tests/stats/summary_panels')
    p.add_argument('--topk', type=int, default=5)
    args = p.parse_args()
    main(args.merged, args.out_root, args.topk)
