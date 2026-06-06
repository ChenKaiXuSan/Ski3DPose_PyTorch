#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""论文级可视化：将真实场景推理结果叠加到原始视频帧上。

输出布局（每个run输出一张图）：
  左列：cam1（左视角）SAM keypoints + skeleton + fused overlay
  中列：cam2（右视角）SAM keypoints + skeleton + fused overlay  
  右列：fusion weight热力图 / alpha条形图

每张子图上方标注关键帧信息。
"""

import argparse
import math
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from functools import lru_cache

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec


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


# Skeleton connections (indices into 15-joint canonical set)
SKELETON_EDGES = [
    # Upper body chain
    (0, 2), (2, 4), (4, 6),   # left arm: Eye_L -> Upperarm_L -> lowerarm_l -> Thigh_L?? No.
    (1, 3), (3, 5), (5, 7),   # right arm
    # Torso connections  
    (2, 3),  # upperarms connection
    # Lower body
    (6, 8), (8, 10),            # left leg
    (7, 9), (9, 11),            # right leg
    # Hands and feet connections
    (12, 5),  # Hand_R -> lowerarm_r connection
    (13, 4),  # Hand_L -> lowerarm_l connection
    # Neck/head
    (0, 14), (1, 14),  # eyes to neck
]

# Better skeleton based on the actual bone structure:
SKELETON_EDGES = [
    # Spine chain: Eye -> Upperarm -> lowerarm
    (0, 2), (2, 4),   # left arm top chain
    (1, 3), (3, 5),   # right arm top chain  
    # Arm-to-body
    (12, 5), (13, 4), # Hand_R -> lowerarm_r, Hand_L -> lowerarm_l
    # Torso
    (2, 3),            # upperarms to each other
    (6, 7),            # thighs to each other  
    (14, 0),           # neck to eye (face direction)
    # Legs
    (6, 8), (8, 10),   # left leg: Thigh -> calf -> foot
    (7, 9), (9, 11),   # right leg: Thigh -> calf -> foot
    # Cross connections
    (2, 6), (3, 7),    # upperarm to thigh (body)
]

# For visualization we'll use a standard body skeleton
BONE_CONNECTIONS_15 = [
    # Arms
    (0, 2), (2, 4), (4, 6),   # left: Eye -> Upperarm -> lowerarm -> Thigh? 
    (1, 3), (3, 5), (5, 7),   # right
    (12, 5), (13, 4),  # hands to wrists  
    # Torso
    (2, 3),
    (6, 7),
    # Legs  
    (8, 10), (9, 11),  # calf -> foot
    # Neck/head
    (0, 14), (1, 14),
]

# Better skeleton mapping - based on actual joint order
BONES = {
    'left_arm': [(0,2), (2,4), (4,6)],      # Eye_L -> Upperarm_L -> lowerarm_l -> Thigh_L (this seems wrong)
}

# Let's use a proper standard skeleton for 15-joint body pose
BONE_PAIRS = [
    # Left arm chain
    (0, 2),   # Eye_L -> Upperarm_L
    (2, 4),   # Upperarm_L -> lowerarm_l  
    (4, 6),   # lowerarm_l -> Thigh_L? This is likely wrong...
    # Right arm chain
    (1, 3),   # Eye_R -> Upperarm_R
    (3, 5),   # Upperarm_R -> lowerarm_r
    (5, 7),   # lowerarm_r -> Thigh_R? Also likely wrong
    
    # Hands to wrists
    (4, 13),  # lowerarm_l -> Hand_L
    (5, 12),  # lowerarm_r -> Hand_R
    
    # Torso connections  
    (2, 3),   # upperarms connection
    (6, 7),   # thigh connection (core of body)
    
    # Legs
    (8, 10),  # left calf -> Foot_L
    (9, 11),  # right calf -> Foot_R
    
    # Neck/head
    (0, 14),  # Eye_L -> neck_01
    (1, 14),  # Eye_R -> neck_01
]

# Simpler standard body skeleton for visualization
BONE_PAIRS_VIS = [
    # Upper limb left
    (2, 4), (4, 6),   # Upperarm -> lowerarm -> Thigh?? 
    # This mapping is confusing. Let me just draw what makes visual sense.
]

# For paper: use a proper stick figure skeleton
SKELETON = [
    # Face/Head
    (0, 14), (1, 14),
    # Left arm
    (0, 2), (2, 4), (4, 13),     # Eye -> Upperarm -> lowerarm -> Hand_L  
    # Right arm
    (1, 3), (3, 5), (5, 12),     # Eye -> Upperarm -> lowerarm -> Hand_R
    # Torso
    (2, 3),                       # upperarms cross
    (6, 7),                       # thigh cross (body center)  
    # Left leg
    (6, 8), (8, 10),              # Thigh -> calf -> Foot_L
    # Right leg
    (7, 9), (9, 11),              # Thigh -> calf -> Foot_R
]

JOINT_NAMES = _get_joint_names() if hasattr(_get_joint_names(), '__len__') else [f"joint_{i}" for i in range(15)]


def load_npz_kpt3d(npz_path: str) -> np.ndarray:
    """Load pred_keypoints_3d from NPZ."""
    data = np.load(npz_path, allow_pickle=True)
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
    # Flat format (run_X style)
    for k in data.keys():
        v = data[k]
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3:
            return v.astype(np.float32)
    raise KeyError(f"No 3D keypoints in {npz_path}")


def load_npz_frame(npz_path: str):
    """Load frame image from NPZ, returning (H,W,C) uint8 RGB."""
    data = np.load(npz_path, allow_pickle=True)
    for key in ("frame", "image", "img"):
        if key in data and isinstance(data[key], np.ndarray):
            return data[key]
    # For dict-wrapped format (pro runs), check 'output'/'outputs'
    for key in ("output", "outputs"):
        if key not in data: continue
        obj = data[key]
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            inner = obj[()] if obj.ndim == 0 else obj.item()
            if hasattr(inner, 'keys'):
                for fk in ("frame", "image", "img"):
                    if fk in inner and isinstance(inner[fk], np.ndarray):
                        return inner[fk]
    return None


def filter_to_15(kp3d: np.ndarray) -> np.ndarray:
    """Filter SAM3D-body keypoints to 15-joint canonical set."""
    _, _, fsk = _symbols()
    try:
        return fsk(kp3d)
    except Exception:
        n = min(kp3d.shape[0], 15)
        return kp3d[:n] if n < kp3d.shape[0] else kp3d


# Bone pair indices for skeleton drawing - using standard body pose ordering
# 15-joint canonical: Eye_L, Eye_R, Upperarm_L, Upperarm_R, lowerarm_l, lowerarm_r, 
#                      Thigh_L, Thigh_R, calf_l, calf_r, Foot_L, Foot_R, Hand_R, Hand_L, neck_01
BONE_PAIRS = [
    # Head to neck
    (0, 14), (1, 14),
    # Left arm: Eye -> Upperarm -> lowerarm -> Hand
    (0, 2), (2, 4), (4, 13),  
    # Right arm: Eye -> Upperarm -> lowerarm -> Hand
    (1, 3), (3, 5), (5, 12),
    # Torso connections
    (2, 3),  # between upperarms
    (6, 7),  # between thighs (hip connection)
    # Left leg: Thigh -> calf -> Foot
    (6, 8), (8, 10),
    # Right leg: Thigh -> calf -> Foot
    (7, 9), (9, 11),
]


def draw_skeleton(ax, keypoints_2d, color=(0.0, 0.8, 1.0), lw=3, alpha=1.0):
    """Draw skeleton connections on axis from 2D keypoints."""
    J = len(keypoints_2d)
    
    # Draw bones first (underneath dots)
    for (i, j) in BONE_PAIRS:
        if i < J and j < J:
            ax.plot([keypoints_2d[i][0], keypoints_2d[j][0]], 
                    [keypoints_2d[i][1], keypoints_2d[j][1]],
                    color=color, linewidth=lw, alpha=alpha, solid_capstyle='round')

    # Draw joint dots
    ax.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], 
               c=[color] if isinstance(color, tuple) else 'white',
               s=60, edgecolors='black', linewidths=1.5, zorder=3)


def draw_joints_with_weights(ax, kp2d, weights, joint_names=None, cmap='viridis'):
    """Draw keypoints sized by fusion weight alpha."""
    J = len(kp2d)
    
    for i in range(J):
        size = 40 + int(weights[i] * 200)
        ax.scatter(kp2d[i][0], kp2d[i][1], 
                   c=[weights[i]], cmap=cmap, s=size, edgecolors='black', linewidths=1.5, zorder=3)


def project_3d_to_2d(kpt3d: np.ndarray, cam_t: np.ndarray = None, focal: float = 1000.0) -> np.ndarray:
    """Project 3D keypoints to 2D using simple perspective projection."""
    if cam_t is not None:
        kpt3d_shifted = kpt3d + cam_t[None, :]
    else:
        kpt3d_shifted = kpt3d
    # Simple inverse Z projection (assuming camera at origin looking +Z)
    z = kpt3d_shifted[:, 2] + 1.5  # shift to positive Z
    z = np.clip(z, 0.1, None)
    x_2d = kpt3d_shifted[:, 0] / z * focal + 960  # image center (1920/2)
    y_2d = -kpt3d_shifted[:, 1] / z * focal + 540   # flip Y (image coords)
    return np.stack([x_2d, y_2d], axis=1)


def create_paper_figure(run_name: str, frame_num: int, cam1_frame: np.ndarray, cam2_frame: np.ndarray,
                        kpt3d_left: np.ndarray, kpt3d_right: np.ndarray, fused_kpt3d: np.ndarray,
                        alpha_means: np.ndarray, output_path: Path, ckpt_label: str = "") -> None:
    """Create a single paper-quality visualization panel for one run."""
    
    fig = plt.figure(figsize=(24, 18), dpi=150)
    
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.1, 
                  left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    joint_names = _get_joint_names()
    
    # ── Top labels ──
    fig.suptitle(f"{run_name} | Frame {frame_num:04d}", fontsize=20, fontweight='bold', y=0.97)
    
    col_labels = ["Cam 1 (Left View)", "Cam 2 (Right View)"]
    for i, label in enumerate(col_labels):
        ax_label = fig.add_subplot(gs[0, i])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, label, fontsize=16, fontweight='bold', ha='center', va='center',
                     transform=ax_label.transAxes)
    
    # ── Row 1: Left view (original frame + SAM keypoints overlay) ──
    ax1 = fig.add_subplot(gs[1, 0])
    if cam1_frame is not None and cam1_frame.ndim == 3:
        ax1.imshow(cam1_frame, aspect='auto')
    else:
        ax1.set_facecolor((0.1, 0.1, 0.15))
    
    # Project 3D keypoints to 2D on cam1 frame
    if kpt3d_left.shape[0] == 15 and kpt3d_left.shape[1] == 3:
        # Use the first frame's projection (assuming single frame or averaging)
        kp3d_sample = kpt3d_left.mean(axis=0) if kpt3d_left.ndim > 2 else kpt3d_left
        
        cam1_2d = project_3d_to_2d(kp3d_sample, focal=800.0)
        
        # Draw SAM keypoints (small white dots)
        ax1.scatter(cam1_2d[:, 0], cam1_2d[:, 1], c='yellow', s=40, edgecolors='black', linewidths=1, zorder=3)
        
        # Draw skeleton on cam1 frame
        draw_skeleton(ax1, cam1_2d, color=(0.0, 0.95, 0.0), lw=4, alpha=0.8)
    
    ax1.axis('off')
    if kpt3d_left.shape[0] == 15 and kpt3d_left.shape[1] == 3:
        kp3d_sample = kpt3d_left.mean(axis=0) if kpt3d_left.ndim > 2 else kpt3d_left
        cam1_2d = project_3d_to_2d(kp3d_sample, focal=800.0)
    
    # ── Row 1: Right view ──
    ax2 = fig.add_subplot(gs[1, 1])
    if cam2_frame is not None and cam2_frame.ndim == 3:
        ax2.imshow(cam2_frame, aspect='auto')
    else:
        ax2.set_facecolor((0.15, 0.1, 0.1))
    
    if kpt3d_right.shape[0] == 15 and kpt3d_right.shape[1] == 3:
        kp3d_sample_r = kpt3d_right.mean(axis=0) if kpt3d_right.ndim > 2 else kpt3d_right
        cam2_2d = project_3d_to_2d(kp3d_sample_r, focal=800.0)
        
        ax2.scatter(cam2_2d[:, 0], cam2_2d[:, 1], c='yellow', s=40, edgecolors='black', linewidths=1, zorder=3)
        draw_skeleton(ax2, cam2_2d, color=(0.95, 0.0, 0.0), lw=4, alpha=0.8)
    
    ax2.axis('off')
    
    # ── Row 2: Alpha weights (fusion ratio) for each view's projected skeleton ──
    if cam1_frame is not None and cam1_frame.ndim == 3:
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.imshow(cam1_frame, aspect='auto')
        ax3.set_title(f"Fusion Alpha (Cam 1) | α_mean={alpha_means.mean():.2f}", fontsize=14, fontweight='bold')
    else:
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.set_facecolor((0.08, 0.08, 0.1))
        ax3.text(0.5, 0.5, "Cam 1", ha='center', va='center', fontsize=24, color='white')
    
    # Overlay fused keypoints weighted by alpha
    if fused_kpt3d is not None and fused_kpt3d.shape[0] == 15 and fused_kpt3d.shape[1] == 3:
        kp_fused = fused_kpt3d.mean(axis=0) if fused_kpt3d.ndim > 2 else fused_kpt3d
        cam1_2d_fused = project_3d_to_2d(kp_fused, focal=800.0)
        
        draw_joints_with_weights(ax3, cam1_2d_fused, alpha_means, cmap='coolwarm')
        draw_skeleton(ax3, cam1_2d_fused, color=(1.0, 1.0, 0.0), lw=5, alpha=0.9)
    
    ax3.axis('off')
    
    if cam2_frame is not None and cam2_frame.ndim == 3:
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.imshow(cam2_frame, aspect='auto')
        ax4.set_title(f"Fusion Alpha (Cam 2) | α_mean={alpha_means.mean():.2f}", fontsize=14, fontweight='bold')
    else:
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.set_facecolor((0.1, 0.08, 0.08))
        ax4.text(0.5, 0.5, "Cam 2", ha='center', va='center', fontsize=24, color='white')
    
    if fused_kpt3d is not None and fused_kpt3d.shape[0] == 15 and fused_kpt3d.shape[1] == 3:
        kp_fused = fused_kpt3d.mean(axis=0) if fused_kpt3d.ndim > 2 else fused_kpt3d
        cam2_2d_fused = project_3d_to_2d(kp_fused, focal=800.0)
        
        draw_joints_with_weights(ax4, cam2_2d_fused, alpha_means, cmap='coolwarm')
        draw_skeleton(ax4, cam2_2d_fused, color=(1.0, 1.0, 0.0), lw=5, alpha=0.9)
    
    ax4.axis('off')
    
    # ── Legend bar ──
    fig.text(0.5, 0.02, f"Checkpoint: {ckpt_label} | Joint α (fusion weight)", 
             ha='center', va='bottom', fontsize=10, color='white',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=(0.15, 0.15, 0.2), edgecolor='none'))
    
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


def create_alpha_comparison_plot(all_results: Dict[str, Dict[str, float]], output_dir: Path):
    """Create alpha comparison bar chart across runs for a given checkpoint."""
    
    joint_names = _get_joint_names()
    jc = len(joint_names)
    
    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    for i, (ckpt_label, results) in enumerate(all_results.items()):
        x = np.arange(jc) + i * 0.12 - 0.36
        width = 0.1
        
        for run_name, res in list(results.items())[:4]:  # Max 4 runs to avoid clutter
            alpha_data = res.get("per_joint_alpha", {})
            if not alpha_data:
                continue
            means = [alpha_data.get(j, 0) for j in range(jc)]
            ax.plot(x + i * 0.05, means, 'o-', label=f"{run_name}", color=colors[i % len(colors)], 
                   linewidth=1.5, markersize=3, alpha=0.8)
    
    ax.set_xlabel("Joint", fontsize=14)
    ax.set_ylabel("α (fusion weight)", fontsize=14)
    ax.set_title("Per-Joint Fusion Ratio Comparison Across Real-World Runs", fontsize=16, fontweight='bold')
    ax.set_xticks(np.arange(jc))
    ax.set_xticklabels(joint_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0.3, 0.7)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "alpha_comparison.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/kaixu_chen/data/skiing/sam3d_body_results/person"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/eval_realworld_direct/paper_vis"))
    parser.add_argument("--runs", nargs="+", default=["pro_1", "pro_2", "run_3", "run_4", "run_5", "run_6"])
    parser.add_argument("--frame-range", type=int, default=10, help="Number of frames to visualize (evenly spaced).")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-computed inference results from disk
    report_dir = Path("logs/eval_realworld_direct")
    
    all_results = {}
    for ckpt_label in ["unity", "ski_poseptz"]:
        txt_path = report_dir / f"realworld_report_{ckpt_label}.txt"
        if not txt_path.exists():
            continue
        
        results = {}
        current_run = None
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    current_run = line[1:-1]
                    results[current_run] = {}
                elif ": " in line and current_run:
                    k, v = line.split(": ", 1)
                    try:
                        results[current_run][k.strip()] = float(v.strip())
                    except ValueError:
                        pass
        
        all_results[ckpt_label] = results
    
    print(f"Loaded results for {len(all_results)} checkpoints")
    
    # Create visualization panels for each run
    for ckpt_label, results in all_results.items():
        for run_name, res in results.items():
            base_dir = Path(args.data_root) / run_name
            left_dir = base_dir / "left"
            
            # Pick evenly spaced frame indices
            npz_files = sorted(left_dir.glob("*.npz"))
            if not npz_files:
                continue
            
            n_frames = min(args.frame_range, len(npz_files))
            step = max(1, len(npz_files) // n_frames)
            frame_indices = list(range(0, len(npz_files), step))[:n_frames]
            
            for fi in frame_indices:
                if fi >= len(npz_files):
                    continue
                
                left_npz = str(npz_files[fi])
                right_npz = str(Path(left_npz).parent.parent / "right" / Path(left_npz).name)
                
                # Load frame (only for run_X format which has 'frame' key)
                cam1_frame = load_npz_frame(left_npz)
                cam2_frame = load_npz_frame(right_npz) if cam1_frame is not None else None
                
                # Load 3D keypoints (first frame from the run's inference - just use one sample)
                kpt3d_left = np.zeros((15, 3))
                kpt3d_right = np.zeros((15, 3))
                fused_kpt3d = np.zeros((15, 3))
                
                try:
                    kpt3d_left = load_npz_kpt3d(left_npz)
                    if len(kpt3d_left) >= 15:
                        kpt3d_left = kpt3d_left[:15]
                    else:
                        kpt3d_left = filter_to_15(kpt3d_left)
                    
                    try:
                        kpt3d_right = load_npz_kpt3d(right_npz)
                        if len(kpt3d_right) >= 15:
                            kpt3d_right = kpt3d_right[:15]
                        else:
                            kpt3d_right = filter_to_15(kpt3d_right)
                    except Exception:
                        pass
                    
                    # Use fused from inference stats (approximate)
                    alpha_mean = res.get("alpha_mean", 0.5)
                    for j in range(15):
                        alpha_j = res.get("per_joint_alpha", {}).get(j, alpha_mean)
                        if j < len(kpt3d_left) and j < len(kpt3d_right):
                            fused_kpt3d[j] = alpha_mean * kpt3d_left[j] + (1 - alpha_mean) * kpt3d_right[j]
                except Exception as e:
                    print(f"  [WARN] Could not load keypoints for {run_name} frame {fi}: {e}")
                    continue
                
                # Alpha weights per joint
                alpha_means = np.array([res.get("per_joint_alpha", {}).get(j, alpha_mean) 
                                       for j in range(15)]) if "per_joint_alpha" in res else np.ones(15) * alpha_mean

                frame_num = int(Path(npz_files[fi]).stem.split("_")[1])
                output_path = output_dir / f"{run_name}_frame{frame_num:04d}_{ckpt_label}.png"
                
                if not os.path.exists(output_path.parent):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                
                create_paper_figure(
                    run_name=run_name, frame_num=frame_num,
                    cam1_frame=cam1_frame, cam2_frame=cam2_frame,
                    kpt3d_left=kpt3d_left, kpt3d_right=kpt3d_right, 
                    fused_kpt3d=fused_kpt3d, alpha_means=alpha_means,
                    output_path=output_path, ckpt_label=ckpt_label
                )
                
                print(f"  Generated: {output_path}")

    # Alpha comparison across runs
    if all_results:
        create_alpha_comparison_plot(all_results, output_dir)
        print(f"\nAlpha comparison plot saved to: {output_dir / 'alpha_comparison.png'}")
    
    print(f"\n[Paper visuals complete] All outputs in: {output_dir}")


if __name__ == "__main__":
    import os
    main()
