#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Evaluate trained dual-view fusion checkpoints on real-world SAM3D-body data."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[0]
DUAL2POSE_ROOT = REPO_ROOT / "dual2pose"
for p in (str(REPO_ROOT), str(DUAL2POSE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


@lru_cache(maxsize=1)
def _repo_symbols():
    fusion_module = importlib.import_module("trainer.train_crossview_fusion")
    dual_module = importlib.import_module("trainer.train_dual2pose")
    map_module = importlib.import_module("map_config")
    return (
        fusion_module.CrossViewFusionTrainer,
        dual_module.Dual2PoseTrainer,
        map_module.FILTERED_KPTS_MAPPING,
        map_module.filter_sam3d_body_kpts,
    )


def _get_joint_names() -> List[str]:
    _, _, FKM, _ = _repo_symbols()
    try:
        return [FKM[j] for j in sorted(FKM)]
    except Exception:
        pass
    return [f"joint_{j}" for j in range(15)]


def _filter_to_15(kp3d: np.ndarray) -> np.ndarray:
    _, _, _, fsk = _repo_symbols()
    try:
        return fsk(kp3d)
    except Exception as e:
        print(f"[WARN] filter_sam3d_body_kpts failed ({e}), using first 15")
        n = min(kp3d.shape[0], 15)
        return kp3d[:n] if n < kp3d.shape[0] else kp3d


def _load_npz_kpt3d(npz_path: str) -> np.ndarray:
    """Load pred_keypoints_3d from SAM3D-body NPZ, handling both formats."""
    data = np.load(npz_path, allow_pickle=True)
    for key in ("output", "outputs"):
        if key not in data:
            continue
        obj = data[key]
        if isinstance(obj, np.ndarray):
            # Scalar object array → .item() gives the inner dict
            if obj.ndim == 0:
                val = obj.item()
                if isinstance(val, dict) and "pred_keypoints_3d" in val:
                    return np.asarray(val["pred_keypoints_3d"], dtype=np.float32)
            else:
                # Multi-element object array → iterate looking for dicts with kpt3d
                for i in range(len(obj)):
                    inner = obj[i]
                    if isinstance(inner, dict) and "pred_keypoints_3d" in inner:
                        return np.asarray(inner["pred_keypoints_3d"], dtype=np.float32)
    raise KeyError(f"NPZ {npz_path} missing pred_keypoints_3d under 'output'/'outputs'")


class RealWorldSam3dDataset(Dataset):
    def __init__(self, base_dir: str, target_t: int = 30, max_windows: Optional[int] = None) -> None:
        self.base_dir = Path(base_dir)
        self.target_t = target_t

        left_dir = self.base_dir / "left"
        right_dir = self.base_dir / "right"
        left_files = sorted(left_dir.glob("*.npz")) if left_dir.exists() else []
        right_files = sorted(right_dir.glob("*.npz")) if right_dir.exists() else []

        def _extract_nums(file_list):
            nums = {}
            for f in file_list:
                stem = f.stem
                try:
                    num = int(stem.split("_")[1])
                    nums[num] = str(f)
                except (IndexError, ValueError):
                    pass
            return nums

        left_nums = _extract_nums(left_files)
        right_nums = _extract_nums(right_files)
        shared_nums = sorted(set(left_nums.keys()) & set(right_nums.keys()))
        print(
            f"  [{Path(base_dir).name}] "
            f"Left={len(left_files)}, Right={len(right_files)}, Aligned={len(shared_nums)}"
        )

        if len(shared_nums) < target_t:
            raise ValueError(
                f"[{Path(base_dir).name}] Only {len(shared_nums)} aligned frames, "
                f"need >= {target_t} (time_window)"
            )

        # Use a larger stride to limit windows (every ~4x the window length)
        step = max(1, target_t * 3)
        self.windows = []
        for i in range(0, len(shared_nums) - target_t + 1, step):
            if max_windows is not None and len(self.windows) >= max_windows:
                break
            self.windows.append(i)

        # If stride is too large and we get very few windows, fall back to smaller stride
        if len(self.windows) < 3:
            step = target_t
            self.windows = []
            for i in range(0, len(shared_nums) - target_t + 1, step):
                if max_windows is not None and len(self.windows) >= max_windows:
                    break
                self.windows.append(i)

        print(f"         → {len(self.windows)} sliding windows (stride={step})")

        self.shared_nums = shared_nums
        self._left_path_by_num = left_nums
        self._right_path_by_num = right_nums
        self._left_cache: Dict[int, np.ndarray] = {}
        self._right_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window_start = self.windows[idx]
        frame_indices = [self.shared_nums[window_start + t] for t in range(self.target_t)]

        cam1_kpts: List[np.ndarray] = []
        cam2_kpts: List[np.ndarray] = []
        for fi in frame_indices:
            lp = self._left_path_by_num.get(fi)
            rp = self._right_path_by_num.get(fi)
            if lp is not None:
                if fi not in self._left_cache:
                    kpt = _load_npz_kpt3d(lp)
                    self._left_cache[fi] = _filter_to_15(kpt)
                cam1_kpts.append(self._left_cache[fi])
            if rp is not None:
                if fi not in self._right_cache:
                    kpt = _load_npz_kpt3d(rp)
                    self._right_cache[fi] = _filter_to_15(kpt)
                cam2_kpts.append(self._right_cache[fi])

        assert len(cam1_kpts) == self.target_t, f"left {len(cam1_kpts)} != {self.target_t}"
        assert len(cam2_kpts) == self.target_t, f"right {len(cam2_kpts)} != {self.target_t}"

        return {
            "kpt3d_sam": {
                "cam1": torch.from_numpy(np.stack(cam1_kpts, axis=0)),
                "cam2": torch.from_numpy(np.stack(cam2_kpts, axis=0)),
            },
            "run_name": Path(self.base_dir).name,
            "frame_start": frame_indices[0],
            "frame_end": frame_indices[-1],
        }


class RealWorldDataModule(LightningDataModule):
    def __init__(self, dataset: RealWorldSam3dDataset, batch_size: int = 1) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, collate_fn=_realworld_collate,
        )


def _realworld_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    kpt3d = {}
    for cam in ("cam1", "cam2"):
        kpt3d[cam] = torch.stack([b["kpt3d_sam"][cam] for b in batch], dim=0)
    return {
        "kpt3d_sam": kpt3d,
        "run_name": [b["run_name"] for b in batch],
        "frame_start": [b["frame_start"] for b in batch],
        "frame_end": [b["frame_end"] for b in batch],
    }


def _load_model_and_config(ckpt_path: Path, backbone: str) -> Tuple[Any, Any]:
    crossview_fusion_cls, dual2pose_cls, _, _ = _repo_symbols()
    config_path = REPO_ROOT / "configs" / "dual2pose.yaml"
    config = OmegaConf.load(str(config_path))
    config.model.backbone = backbone
    config.data.load_frames = False
    config.data.load_2d_kpt = False
    config.data.load_3d_kpt = True
    config.log_path = str(ckpt_path.parent)

    if backbone == "crossview_fusion":
        model = crossview_fusion_cls(config)
    elif backbone == "dual2pose":
        model = dual2pose_cls(config)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if hasattr(model, "test_outputs"):
        model.test_outputs = []
    return model, config


def _squeeze_last(x):
    return x.squeeze(-1) if x.ndim == 4 and x.shape[-1] == 1 else x


def _format_float(value: float) -> str:
    return "nan" if (math.isnan(value) or math.isinf(value)) else f"{value:.4f}"


DEFAULT_CKPT_UNITY = Path("logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt")
DEFAULT_CKPT_SKI = Path("logs/train_ski_poseptz/crossview_fusion/2026-05-25/14-13-23/checkpoints/last.ckpt")
ALL_RUNS = ["pro_1", "pro_2", "run_3", "run_4", "run_5", "run_6"]


def evaluate_checkpoint(ckpt_path: Path, runs: List[str], args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path.name} (backbone={args.backbone}, tw={args.time_window})")
    print(f"Runs:      {runs}")
    print(f"{'='*60}\n")

    model, config = _load_model_and_config(ckpt_path, backbone=args.backbone)
    use_gpu = torch.cuda.is_available() and not args.cpu
    trainer = Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=[int(args.gpu)] if use_gpu else 1,
        logger=False, enable_checkpointing=False,
        callbacks=[RichProgressBar(refresh_rate=5, leave=True)],
    )

    all_results: Dict[str, Dict[str, Any]] = {}

    for run_name in runs:
        base_dir = Path(args.data_root) / run_name
        if not base_dir.exists():
            print(f"  [SKIP] {base_dir}")
            continue

        try:
            dataset = RealWorldSam3dDataset(str(base_dir), target_t=args.time_window)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        datamodule = RealWorldDataModule(dataset, batch_size=args.batch_size)
        trainer_metrics = trainer.test(
            model, datamodule=datamodule, ckpt_path=str(ckpt_path),
            verbose=False, weights_only=False,
        )

        outputs = list(getattr(model, "test_outputs", []))
        if not outputs:
            print(f"  [WARN] No test outputs for {run_name}")
            all_results[run_name] = {"status": "no_outputs"}
            continue

        fused_chunks, left_chunks, right_chunks, alpha_chunks = [], [], [], []
        canon_left_chunks, canon_right_chunks = [], []

        for batch_out in outputs:
            def _cat(name):
                t = batch_out.get(name)
                if isinstance(t, torch.Tensor) and t.numel() > 0:
                    return t.detach().cpu()
                return None

            ft = _cat("fused")
            lt = _cat("p_left")
            rt = _cat("p_right")
            at = _cat("alpha")
            cl = _cat("left_canonical")
            cr = _cat("right_canonical")
            if ft: fused_chunks.append(ft)
            if lt: left_chunks.append(lt)
            if rt: right_chunks.append(rt)
            if at: alpha_chunks.append(_squeeze_last(at))
            if cl: canon_left_chunks.append(cl)
            if cr: canon_right_chunks.append(cr)

        fused_all = torch.cat(fused_chunks, dim=0) if fused_chunks else None
        left_all = torch.cat(left_chunks, dim=0) if left_chunks else None
        right_all = torch.cat(right_chunks, dim=0) if right_chunks else None
        alpha_all = torch.cat(alpha_chunks, dim=0) if alpha_chunks else None

        stats = {}
        if alpha_all is not None:
            stats["alpha_mean"] = float(_squeeze_last(alpha_all).mean().item())
            stats["alpha_std"] = float(_squeeze_last(alpha_all).std().item())
        if left_all is not None:
            stats["left_norm_mm"] = float(torch.norm(left_all, dim=-1).mean().item()) * 1000
        if right_all is not None:
            stats["right_norm_mm"] = float(torch.norm(right_all, dim=-1).mean().item()) * 1000
        if fused_all is not None:
            stats["fused_norm_mm"] = float(torch.norm(fused_all, dim=-1).mean().item()) * 1000

        if left_all is not None and right_all is not None:
            diff = torch.norm(left_all - right_all, dim=-1)
            stats["view_disagree_mean_mm"] = float(diff.mean().item()) * 1000
            stats["view_disagree_median_mm"] = float(torch.median(diff).item()) * 1000
            stats["view_disagree_max_mm"] = float(diff.max().item()) * 1000

        if fused_all is not None and left_all is not None and right_all is not None:
            avg_base = 0.5 * (left_all + right_all)
            stats["fused_vs_avg_dist_mm"] = float(torch.norm(fused_all - avg_base, dim=-1).mean().item()) * 1000

        per_joint_alpha: Dict[int, float] = {}
        if alpha_all is not None and alpha_all.numel() > 0:
            jc = alpha_all.shape[-1]
            am = alpha_all.mean(dim=(0, 1)).cpu().numpy()
            for j in range(jc):
                per_joint_alpha[j] = float(am[j])

        all_results[run_name] = {
            "alpha_mean": stats.get("alpha_mean", 0),
            "alpha_std": stats.get("alpha_std", 0),
            "left_norm_mm": stats.get("left_norm_mm", 0),
            "right_norm_mm": stats.get("right_norm_mm", 0),
            "fused_norm_mm": stats.get("fused_norm_mm", 0),
            "view_disagree_mean_mm": stats.get("view_disagree_mean_mm", 0),
            "view_disagree_median_mm": stats.get("view_disagree_median_mm", 0),
            "view_disagree_max_mm": stats.get("view_disagree_max_mm", 0),
            "fused_vs_avg_dist_mm": stats.get("fused_vs_avg_dist_mm", 0),
            "per_joint_alpha": per_joint_alpha,
            "num_samples": int(fused_all.shape[0]) if fused_all is not None else 0,
        }

        print(
            f"  [{run_name}] alpha={stats['alpha_mean']:.3f}+/-{stats['alpha_std']:.3f}  "
            f"disagree={stats.get('view_disagree_mean_mm',0):.1f}mm  norm={stats.get('fused_norm_mm',0):.1f}mm  windows={all_results[run_name]['num_samples']}"
        )

    return all_results


def plot_all(results: Dict[str, Dict[str, Any]], ckpt_name: str, output_dir: Path) -> None:
    joint_names = _get_joint_names()
    jc = len(joint_names)
    x = np.arange(jc)
    run_names = list(results.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    vis_dir = output_dir / "alpha_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-joint alpha comparison
    fig, ax = plt.subplots(figsize=(max(12, jc * 0.8), 5))
    for i, (rn, res) in enumerate(results.items()):
        ad = res.get("per_joint_alpha", {})
        if not ad: continue
        means = [ad.get(j, 0) for j in range(jc)]
        ax.plot(x, means, "o-", label=rn, color=colors[i % len(colors)], linewidth=1.5, markersize=4)
    ax.set_xlabel("Joint"); ax.set_ylabel("alpha (left weight)")
    ax.set_ylim(0, 1); ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
    ax.legend(); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(vis_dir / "per_joint_alpha.png", dpi=180); plt.close(fig)

    # 2) View disagreement grouped bar
    fig, ax = plt.subplots(figsize=(max(8, len(run_names) * 1.5), 5))
    xr = np.arange(len(run_names)); w = 0.25
    means_d = [results[r].get("view_disagree_mean_mm", 0) for r in run_names]
    medians_d = [results[r].get("view_disagree_median_mm", 0) for r in run_names]
    maxs_d = [results[r].get("view_disagree_max_mm", 0) for r in run_names]
    ax.bar(xr - w, means_d, w, label="Mean (mm)", color="tab:red")
    ax.bar(xr, medians_d, w, label="Median (mm)", color="tab:purple")
    ax.bar(xr + w, maxs_d, w, label="Max (mm)", color="tab:brown")
    ax.set_xticks(xr); ax.set_xticklabels(run_names, rotation=30, ha="right")
    ax.set_ylabel("disagreement (mm)"); ax.legend(); ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"View Disagreement -- {ckpt_name}"); fig.tight_layout()
    fig.savefig(vis_dir / f"disagreement_{ckpt_name}.png", dpi=180); plt.close(fig)

    # 3) Norm comparison grouped bar
    fig, ax = plt.subplots(figsize=(max(8, len(run_names) * 1.5), 5))
    lr = [results[r].get("left_norm_mm", 0) for r in run_names]
    rr = [results[r].get("right_norm_mm", 0) for r in run_names]
    fr = [results[r].get("fused_norm_mm", 0) for r in run_names]
    ax.bar(xr - w, lr, w, label="Left norm", color="tab:blue")
    ax.bar(xr, rr, w, label="Right norm", color="tab:orange")
    ax.bar(xr + w, fr, w, label="Fused norm", color="tab:green")
    ax.set_xticks(xr); ax.set_xticklabels(run_names, rotation=30, ha="right")
    ax.set_ylabel("norm (mm)"); ax.legend(); ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Pose Norms -- {ckpt_name}"); fig.tight_layout()
    fig.savefig(vis_dir / f"norms_{ckpt_name}.png", dpi=180); plt.close(fig)

    # 4) Alpha overall bar per run
    fig, ax = plt.subplots(figsize=(max(6, len(run_names) * 1.2), 5))
    am_vals = [results[r]["alpha_mean"] for r in run_names]
    as_vals = [results[r]["alpha_std"] for r in run_names]
    ax.bar(xr, am_vals, yerr=as_vals, color="tab:green", capsize=3)
    ax.set_ylim(0, 1); ax.set_ylabel("alpha (left weight)")
    ax.set_xticks(xr); ax.set_xticklabels(run_names, rotation=30, ha="right")
    ax.set_title(f"Overall Alpha -- {ckpt_name}"); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(vis_dir / f"alpha_overall_{ckpt_name}.png", dpi=180); plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate dual-view fusion on real-world SAM3D-body data")
    p.add_argument("--data-root", type=Path, default=Path("/home/kaixu_chen/data/skiing/sam3d_body_results/person"))
    p.add_argument("--ckpt-path", type=Path, default=None)
    p.add_argument("--backbone", type=str, default="crossview_fusion", choices=["crossview_fusion", "dual2pose"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--time-window", "--target-t", type=int, default=30)
    p.add_argument("--runs", nargs="+", default=ALL_RUNS, choices=ALL_RUNS)
    p.add_argument("--output-dir", type=Path, default=Path("logs/eval_realworld"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    seed_everything(42, workers=True)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = []
    if args.ckpt_path:
        if not args.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
        ckpts.append(("custom", args.ckpt_path))
    else:
        if DEFAULT_CKPT_UNITY.exists():
            ckpts.append(("unity", DEFAULT_CKPT_UNITY))
        else:
            print(f"[WARN] Unity checkpoint not found: {DEFAULT_CKPT_UNITY}")
        if DEFAULT_CKPT_SKI.exists():
            ckpts.append(("ski_poseptz", DEFAULT_CKPT_SKI))
        else:
            print(f"[WARN] SkiPosePTZ checkpoint not found: {DEFAULT_CKPT_SKI}")

    if not ckpts:
        raise RuntimeError("No checkpoints available.")

    all_summary = {}

    for ckpt_label, ckpt_path in ckpts:
        results = evaluate_checkpoint(ckpt_path, args.runs, args)
        if not results:
            print(f"[SKIP] No valid runs for {ckpt_path.name}")
            continue
        all_summary[ckpt_label] = results

        # CSV report
        csv_path = output_dir / f"realworld_report_{ckpt_label}.csv"
        fields = ["run","alpha_mean","alpha_std","left_norm_mm","right_norm_mm","fused_norm_mm",
                  "view_disagree_mean_mm","view_disagree_median_mm","view_disagree_max_mm",
                  "fused_vs_avg_dist_mm","num_samples"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for rn, res in results.items():
                row = {"run": rn}
                for k in fields[1:]:
                    v = res.get(k)
                    if isinstance(v, (int, float)): row[k] = round(v, 4)
                    elif isinstance(v, dict): row[k] = "0" if not v else str(len(v))
                    else: row[k] = v
                w.writerow(row)
        print(f"\nCSV report saved to: {csv_path}")

        # Text report
        txt_path = output_dir / f"realworld_report_{ckpt_label}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== Checkpoint: {ckpt_path.name} ===\n\n")
            for rn, res in results.items():
                f.write(f"[{rn}]\n")
                for k, v in res.items():
                    if k == "per_joint_alpha":
                        f.write("  per_joint_alpha:\n")
                        for ji in sorted(v.keys()):
                            f.write(f"    joint_{ji}: {v[ji]:.4f}\n")
                    elif isinstance(v, float):
                        f.write(f"  {k}: {_format_float(v)}\n")
                    else:
                        f.write(f"  {k}: {v}\n")
                f.write("\n")
        print(f"Text report saved to: {txt_path}")

        # Visualizations
        plot_all(results, ckpt_label, output_dir)
        vis_dir = output_dir / "alpha_vis"
        print(f"Visualizations saved to: {vis_dir}")

    # Cross-checkpoint comparison
    if len(all_summary) >= 2:
        print("\n" + "="*60)
        print("Cross-Checkpoint Comparison")
        print("="*60)
        common = set(all_summary["unity"]).intersection(set(all_summary.get("ski_poseptz", {})))
        for rn in sorted(common):
            u = all_summary["unity"][rn]
            s = all_summary["ski_poseptz"][rn]
            print(f"\n[{rn}]")
            print(f"  Unity      alpha={u['alpha_mean']:.3f}+/-{u['alpha_std']:.3f}  "
                  f"disagree={u.get('view_disagree_mean_mm',0):.1f}mm  norm={u['fused_norm_mm']:.1f}mm")
            print(f"  SkiPosePTZ alpha={s['alpha_mean']:.3f}+/-{s['alpha_std']:.3f}  "
                  f"disagree={s.get('view_disagree_mean_mm',0):.1f}mm  norm={s['fused_norm_mm']:.1f}mm")

    print("\n[Done]")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
