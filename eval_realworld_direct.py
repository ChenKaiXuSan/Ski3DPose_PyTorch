#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Direct inference on real-world SAM3D-body data without Lightning Trainer.

Bypasses trainer's character-specific GT loading by directly calling the model forward pass.
Uses our dataset to load real-world frames, then feeds them directly to the model.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[0]
for p in (str(REPO_ROOT), str(REPO_ROOT / "dual2pose")):
    if p not in sys.path:
        sys.path.insert(0, p)

@lru_cache(maxsize=1)
def _symbols():
    fusion_mod = importlib.import_module("trainer.train_crossview_fusion")
    map_mod = importlib.import_module("map_config")
    return (fusion_mod.CrossViewFusionTrainer, map_mod.FILTERED_KPTS_MAPPING, map_mod.filter_sam3d_body_kpts)


def _get_joint_names():
    _, FKM, _ = _symbols()
    try:
        return [FKM[j] for j in sorted(FKM)]
    except Exception:
        return [f"joint_{j}" for j in range(15)]


def _filter_15(kp3d):
    _, _, fsk = _symbols()
    try:
        return fsk(kp3d)
    except:
        n = min(kp3d.shape[0], 15)
        return kp3d[:n] if n < kp3d.shape[0] else kp3d


def _load_kpt3d(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    # Try outputs/output dict format
    for key in ("output", "outputs"):
        if key not in data: continue
        obj = data[key]
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:
                val = obj.item()
                if isinstance(val, dict) and "pred_keypoints_3d" in val:
                    return np.asarray(val["pred_keypoints_3d"], dtype=np.float32)
            else:
                for i in range(len(obj)):
                    inner = obj[i]
                    if isinstance(inner, dict) and "pred_keypoints_3d" in inner:
                        return np.asarray(inner["pred_keypoints_3d"], dtype=np.float32)
    # Try flat format (run_X style)
    for k in data.keys():
        v = data[k]
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3:
            return v.astype(np.float32)
    raise KeyError(f"No 3D keypoints in {npz_path}")


class RWDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, target_t=30):
        self.base_dir = Path(base_dir)
        self.target_t = target_t
        
        left_dir = self.base_dir / "left"
        right_dir = self.base_dir / "right"
        left_files = sorted(left_dir.glob("*.npz")) if left_dir.exists() else []
        right_files = sorted(right_dir.glob("*.npz")) if right_dir.exists() else []
        
        def _nums(fl):
            d = {}
            for f in fl:
                try: d[int(f.stem.split("_")[1])] = str(f)
                except: pass
            return d
        
        ln = _nums(left_files)
        rn = _nums(right_files)
        shared = sorted(set(ln.keys()) & set(rn.keys()))
        
        print(f"  [{Path(base_dir).name}] Left={len(left_files)}, Right={len(right_files)}, Aligned={len(shared)}")
        if len(shared) < target_t:
            raise ValueError(f"[{Path(base_dir).name}] Only {len(shared)} frames, need >= {target_t}")
        
        step = max(1, target_t * 3)
        self.windows = []
        for i in range(0, len(shared) - target_t + 1, step):
            self.windows.append(i)
        if len(self.windows) < 3:
            step = target_t
            self.windows = []
            for i in range(0, len(shared) - target_t + 1, step):
                self.windows.append(i)
        
        print(f"         -> {len(self.windows)} windows (stride={step})")
        self.shared = shared
        self._ln, self._rn = ln, rn
        self._lc, self._rc = {}, {}

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        ws = self.windows[idx]
        fis = [self.shared[ws + t] for t in range(self.target_t)]
        
        c1, c2 = [], []
        for fi in fis:
            lp = self._ln.get(fi)
            rp = self._rn.get(fi)
            if lp is not None:
                if fi not in self._lc:
                    kpt = _load_kpt3d(lp)
                    self._lc[fi] = _filter_15(kpt)
                c1.append(self._lc[fi])
            if rp is not None:
                if fi not in self._rc:
                    kpt = _load_kpt3d(rp)
                    self._rc[fi] = _filter_15(kpt)
                c2.append(self._rc[fi])
        
        assert len(c1) == self.target_t and len(c2) == self.target_t
        return {
            "cam1": torch.from_numpy(np.stack(c1, axis=0)),
            "cam2": torch.from_numpy(np.stack(c2, axis=0)),
            "run_name": Path(self.base_dir).name,
        }


def load_model(ckpt_path, backbone="crossview_fusion"):
    CrossViewFusionTrainer, _, _ = _symbols()
    # Use a minimal dict-based config to avoid Hydra OmegaConf interpolation issues
    class AttrDict(dict):
        def __getattr__(self, k): return self.get(k)
        def __setattr__(self, k, v): self[k] = v
        def get(self, k, default=None):
            # Handle nested access like config.data.time_window
            if "." in str(k):
                parts = k.split(".")
                d = self
                for p in parts:
                    if isinstance(d, dict) and p in d:
                        d = d[p]
                    else: return default
                return d
            return super().get(k, default)
    
    raw = OmegaConf.load(str(REPO_ROOT / "configs" / "dual2pose.yaml"))
    c = AttrDict(OmegaConf.to_container(raw))
    c["model"] = {"backbone": backbone}
    c["data"] = {"load_frames": False, "load_2d_kpt": False, "load_3d_kpt": True,
                 "time_window": 30, "batch_size": 1, "num_workers": 0}
    c["log_path"] = str(ckpt_path.parent) + "/test"
    config = AttrDict(c)
    
    model = CrossViewFusionTrainer(config)
    # Load weights directly
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Handle Lightning checkpoints: might have 'state_dict' key or be direct
        if "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        elif "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            # Assume it's the state dict itself (or try)
            try:
                model.load_state_dict(state_dict)
            except Exception:
                print(f"[WARN] Could not load weights from {ckpt_path}")
    model.eval()
    return model


def direct_inference(model, dataset, device=0):
    """Run inference using canonicalize_pose_torch + model.forward directly."""
    import sys; sys.path.insert(0, '.')
    sys.path.insert(0, 'dual2pose')
    from trainer.canonicalize import canonicalize_pose_torch
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    fused_chunks, left_chunks, right_chunks, alpha_chunks = [], [], [], []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            cam1 = sample["cam1"].unsqueeze(0).to(device)  # (1, T, J, 3)
            cam2 = sample["cam2"].unsqueeze(0).to(device)
            
            # Canonicalize
            left_canon, _ = canonicalize_pose_torch(cam1.squeeze(0), 
                left_hip=model.left_hip_idx, right_hip=model.right_hip_idx, neck=model.neck_idx)
            right_canon, _ = canonicalize_pose_torch(cam2.squeeze(0),
                left_hip=model.left_hip_idx, right_hip=model.right_hip_idx, neck=model.neck_idx)
            
            # Move canonicalized outputs to model device (canonicalize may produce CPU tensors)
            left_canon = left_canon.to(device)
            right_canon = right_canon.to(device)
            
            fused, aux = model.models(left_canon, right_canon)
            alpha = aux["alpha"]
            
            fused_chunks.append(fused.detach().cpu())
            left_chunks.append(cam1.detach().cpu())
            right_chunks.append(cam2.detach().cpu())
            alpha_chunks.append(alpha.squeeze(-1).detach().cpu())
    
    return (torch.cat(fused_chunks, dim=0) if fused_chunks else None,
            torch.cat(left_chunks, dim=0) if left_chunks else None,
            torch.cat(right_chunks, dim=0) if right_chunks else None,
            torch.cat(alpha_chunks, dim=0) if alpha_chunks else None, None, None)
def compute_stats(fused, left, right, alpha):
    s = {}
    alpha_s = alpha.squeeze(-1) if alpha.ndim == 4 else alpha
    s["alpha_mean"] = float(alpha_s.mean().item())
    s["alpha_std"] = float(alpha_s.std().item())
    s["left_norm_mm"] = float(torch.norm(left, dim=-1).mean().item()) * 1000
    s["right_norm_mm"] = float(torch.norm(right, dim=-1).mean().item()) * 1000
    s["fused_norm_mm"] = float(torch.norm(fused, dim=-1).mean().item()) * 1000
    diff = torch.norm(left - right, dim=-1)
    s["disagree_mean_mm"] = float(diff.mean().item()) * 1000
    s["disagree_median_mm"] = float(torch.median(diff).item()) * 1000
    s["disagree_max_mm"] = float(diff.max().item()) * 1000
    avg_base = 0.5 * (left + right)
    s["fused_vs_avg_mm"] = float(torch.norm(fused - avg_base, dim=-1).mean().item()) * 1000
    return s


def format_f(v):
    return "nan" if (math.isnan(v) or math.isinf(v)) else f"{v:.4f}"


DEFAULT_CKPT_UNITY = Path("logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt")
DEFAULT_CKPT_SKI = Path("logs/train_ski_poseptz/crossview_fusion/2026-05-25/14-13-23/checkpoints/last.ckpt")
ALL_RUNS = ["pro_1", "pro_2", "run_3", "run_4", "run_5", "run_6"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/kaixu_chen/data/skiing/sam3d_body_results/person"))
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--backbone", default="crossview_fusion")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--time-window", type=int, default=30)
    parser.add_argument("--runs", nargs="+", default=ALL_RUNS)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/eval_realworld_direct"))
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = []
    if args.ckpt_path:
        if not args.ckpt_path.exists(): raise FileNotFoundError(str(args.ckpt_path))
        ckpts.append(("custom", args.ckpt_path))
    else:
        if DEFAULT_CKPT_UNITY.exists(): ckpts.append(("unity", DEFAULT_CKPT_UNITY))
        if DEFAULT_CKPT_SKI.exists(): ckpts.append(("ski_poseptz", DEFAULT_CKPT_SKI))
    if not ckpts: raise RuntimeError("No checkpoints available.")

    joint_names = _get_joint_names()
    jc = len(joint_names)
    all_results = {}

    for ckpt_label, ckpt_path in ckpts:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_path.name} | Backbone: {args.backbone}")
        print(f"Runs: {args.runs}")
        print(f"{'='*60}\n")

        model = load_model(ckpt_path, args.backbone)
        results = {}

        for run_name in args.runs:
            base_dir = Path(args.data_root) / run_name
            if not base_dir.exists():
                print(f"  [SKIP] {base_dir}")
                continue

            try:
                dataset = RWDataset(str(base_dir), target_t=args.time_window)
            except ValueError as e:
                print(f"  [SKIP] {e}")
                continue

            fused, left, right, alpha, _, _ = direct_inference(model, dataset, device=device)

            stats = compute_stats(fused, left, right, alpha)

            per_joint_alpha = {}
            am = alpha.mean(dim=(0, 1)).cpu().numpy()
            for j in range(am.shape[-1]):
                per_joint_alpha[j] = float(am[j])

            results[run_name] = {**stats, "per_joint_alpha": per_joint_alpha, "num_samples": fused.shape[0]}
            print(f"  [{run_name}] alpha={stats['alpha_mean']:.3f}+/-{stats['alpha_std']:.3f}  "
                  f"disagree={stats['disagree_mean_mm']:.1f}mm  norm={stats['fused_norm_mm']:.1f}mm  windows={results[run_name]['num_samples']}")

        if not results:
            continue

        all_results[ckpt_label] = results

        # CSV
        csv_path = output_dir / f"realworld_report_{ckpt_label}.csv"
        fields = ["run","alpha_mean","alpha_std","left_norm_mm","right_norm_mm","fused_norm_mm",
                  "disagree_mean_mm","disagree_median_mm","disagree_max_mm","fused_vs_avg_mm","num_samples"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for rn, res in results.items():
                row = {"run": rn}
                for k in fields[1:]:
                    v = res.get(k)
                    row[k] = round(v, 4) if isinstance(v, (int, float)) else ("0" if isinstance(v, dict) and not v else str(len(v)))
                w.writerow(row)
        print(f"\nCSV saved to: {csv_path}")

        # Text
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
                        f.write(f"  {k}: {format_f(v)}\n")
                    else:
                        f.write(f"  {k}: {v}\n")
                f.write("\n")
        print(f"Text report saved to: {txt_path}")

    # Cross-checkpoint comparison
    if len(all_results) >= 2:
        print("\n" + "="*60)
        print("Cross-Checkpoint Comparison")
        print("="*60)
        common = set(all_results["unity"]).intersection(set(all_results.get("ski_poseptz", {})))
        for rn in sorted(common):
            u = all_results["unity"][rn]
            s = all_results["ski_poseptz"][rn]
            print(f"\n[{rn}]")
            print(f"  Unity      alpha={u['alpha_mean']:.3f}+/-{u['alpha_std']:.3f}  "
                  f"disagree={u.get('disagree_mean_mm',0):.1f}mm  norm={u['fused_norm_mm']:.1f}mm")
            print(f"  SkiPosePTZ alpha={s['alpha_mean']:.3f}+/-{s['alpha_std']:.3f}  "
                  f"disagree={s.get('disagree_mean_mm',0):.1f}mm  norm={s['fused_norm_mm']:.1f}mm")

    print("\n[Done]")


if __name__ == "__main__":
    main()
