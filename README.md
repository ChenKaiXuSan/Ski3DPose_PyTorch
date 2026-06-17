<div align="center">

# CanonFuse3D

**Geometry-Free (No Camera Head), Canonical-Aligned Dual-View 3D Pose Fusion for Skiing Videos**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

</div>

---

## TL;DR

This repo provides a **two-stage** framework to reconstruct **stable and complete 3D human keypoints** from **synchronized left/right skiing videos** **without camera calibration** and **without a camera head**.

**Key idea:**

1. Convert per-view 3D poses into a **human-centric canonical coordinate system** (pelvis-centered, scale-normalized, orientation-normalized).
2. Perform **frame-wise dual-view fusion** via **joint-wise gating + spatial GCN refinement**.
3. Apply **temporal refinement** with **ST-GCN** to reduce jitter and enforce motion consistency.
4. Train with **strong supervision on Unity** and **uncertainty-aware weak constraints on real videos**.

---

## Motivation

Real-world skiing videos often suffer from:

- unknown / unreliable camera extrinsics,
- fast motion, motion blur, self-occlusion,
- view-dependent pose noise.

Instead of estimating camera poses, we aim to **fuse multi-view 3D keypoints directly** using **canonical alignment** and **learning-based fusion**, which is more robust under uncalibrated settings.

---

## Method Overview

### Stage 0: Per-view 3D lifting (external)

We use an off-the-shelf estimator (e.g., **SAM 3D Body**) to obtain initial 3D keypoints:

- `X_L^0`, `X_R^0`

### Stage 1: Canonical Alignment (human-centric)

For each frame, we normalize poses into a canonical human coordinate system:

1. pelvis-centered translation
2. scale normalization
3. orientation normalization (anatomical axes)

Output:

- `X̃_L`, `X̃_R` (comparable across views)

### Stage 2a: Frame-wise Dual-view Fusion (Gate + Spatial GCN)

Input: `X̃_L`, `X̃_R`  
Output: per-frame fused pose `X_mv`

- **Joint-wise gate** selects the more reliable view for each joint.
- **Spatial GCN** refines the fused pose using skeletal topology.

### Stage 2b: Temporal Refinement (ST-GCN)

Input: `{X_mv(t)}_{t=1..T}`  
Output: final stable pose sequence `X_fused`

---

## Data

### Labeled (Unity)

- synchronized dual-view videos
- **2D GT keypoints**
- **3D GT keypoints**

### Unlabeled (Real skiing)

- synchronized dual-view videos
- 2D keypoints from pose estimator (**may be noisy**)
- no 3D GT, no reliable extrinsics

---

## Training: Mixed Supervision (Unity + Real)

### Unity (strong supervision)

**Step1 (frame-wise):**

- `L_3D` (3D regression)
- `L_2D` (2D reprojection using Unity cameras)
- `L_bone` (bone length constraint)

**Step2 (temporal):**

- `L_3D^seq` (sequence 3D regression)
- `L_v`, `L_a` (velocity / acceleration consistency)
- `L_bone-var` (bone length stability over time)

### Real (weak / uncertainty-aware constraints)

**Step1 (frame-wise):**

- `L_mv` (multi-view consistency w.r.t. inputs, confidence-weighted)
- `L_bone-prior` (bone priors)

**Step2 (temporal):**

- `L_smooth` (temporal smoothness, avoid over-smoothing fast skiing motions)
- `L_bone-seq` (sequence bone consistency)
- `L_EMA` (teacher-student consistency)
- optional `L_2D-weak` (very low weight, treated as noisy observation)

**Total:**
\[
L = L*{Unity} + \lambda(t) L*{Real}, \quad \lambda(t)\ \text{ramp-up}
\]

## Quick Start

> NOTE: This repo currently provides a research codebase template.  
> You need to prepare your dataset folders and configuration.

### 1 Install

```bash
conda create -n canonfuse3d python=3.10 -y
conda activate canonfuse3d
pip install -r requirements.txt
```

### 2. Train (example)

Unity training:

```bash
python dual2pose/train_unity.py
```

Ski-PosePTZ training:

```bash
python dual2pose/train_ski_poseptz.py
```

### 3. Inference (example)

Real-world direct evaluation / inference:

```bash
python eval_realworld_direct.py
```

## Project Structure

```text
Skiing_Canonical_DualView_3D_Pose_PyTorch/
├── dual2pose/                 # Core dual-view 3D pose fusion package
│   ├── dataloader/            # Unity and Ski-PosePTZ dataset loaders
│   ├── eval/                  # Evaluation helpers
│   ├── models/                # Dual2Pose, cross-view fusion, ST-GCN, Sim3 modules
│   ├── trainer/               # PyTorch Lightning trainers
│   └── utils/                 # Utility functions
├── configs/                   # Hydra configs for training and model variants
│   └── model/                 # Model-specific config fragments
├── cross_validation/          # Camera-pair and Ski-PosePTZ split generation
├── pegasus/                   # PBS/Pegasus cluster job scripts
├── SAM3/                      # External SAM3-related utilities
├── SAM3Dbody/                 # External SAM 3D Body utilities
├── unity/                     # Unity dataset analysis / helper files
├── analysis/                  # Analysis notebooks and reports
├── tests/                     # Lightweight repository tests
├── logs/                      # Training/evaluation outputs (ignored by git)
├── ckpt/                      # Local checkpoints (ignored by git)
├── eval_realworld_direct.py   # Real-world direct evaluation entry point
├── eval_realworld_inference.py# Real-world inference entry point
├── requirements.txt           # Python dependencies
└── environment.yaml           # Conda environment definition
```

## Citation

If you find this useful, please cite our work (coming soon):

```
@article{chen2026canonfuse3d,
title = {Canonical-Aligned Dual-View 3D Pose Fusion without Camera Calibration for Skiing Videos},
author = {Chen, Kaixu and ...},
year = {2026}
}
```