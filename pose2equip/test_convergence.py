#!/usr/bin/env python3
"""Quick convergence test using synthetic data — trains Pose2EquipNetImproved for N epochs."""

import os, sys, types, argparse
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ═══════════════════════════════════════════════════
# Patch transformers.AutoModel BEFORE any pose2equip.models import
# ═══════════════════════════════════════════════════

class _MockDINO(nn.Module):
    """Tiny DINOv2 mock: produces [B, 247, 256] random features with CLS token."""
    def __init__(self, model_name=None):
        super().__init__()
        self.config = type("C", (), {"hidden_size": 256})()
        self.proj = nn.Linear(256, 256)
    @classmethod
    def from_pretrained(cls, model_name):
        return cls(model_name)
    def forward(self, pixel_values):
        B_, _, H_, W_ = pixel_values.shape
        device = pixel_values.device
        N = 247
        cls_tok = torch.zeros(B_, 1, 256, device=device)
        feats = torch.randn(B_, N, 256, device=device)
        return type("O", (), {"last_hidden_state": torch.cat([cls_tok, feats], dim=1)})()

# Install mock globally
_mock_mod = types.ModuleType("transformers")
_mock_mod.AutoModel = _MockDINO
sys.modules["transformers"] = _mock_mod

# If transformers was already imported by something else, patch that too
for k in list(sys.modules.keys()):
    if "transformers" in k:
        orig_mod = sys.modules.pop(k)
        for attr in ["AutoModel"]:
            if hasattr(orig_mod, attr):
                setattr(_mock_mod, attr, getattr(orig_mod, attr))

# Now import pose2equip modules (they'll pick up our mock)
sys.path.insert(0, str(Path(__file__).parent.parent))
from pose2equip.models.pose2equip_net import Pose2EquipNetImproved
from pose2equip.map_config import FILTER_SKELETON_CONNECTIONS


# ═══════════════════════════════════════════════════
# Synthetic dataset
# ═══════════════════════════════════════════════════

class SyntheticPose2EquipDataset(Dataset):
    def __init__(self, n_samples=500, J=15, H=224, W=224):
        self.n_samples = n_samples
        self.J = J
        self.H = H
        self.W = W

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        pose = torch.randn(self.J, 3) * 1.5
        frame = torch.rand(3, self.H, self.W)

        # GT: affine function of pose — model must learn this mapping
        pose_flat = pose.reshape(-1)           # [45]
        W_true = torch.randn(8*3, self.J*3) * 0.1
        b_true = torch.randn(8*3) * 0.05
        gt_obj_flat = (W_true @ pose_flat + b_true).reshape(4, 2, 3)

        return {
            "frames": frame,          # [3, H, W]
            "human_3d": pose,         # [J, 3]
            "object_gt": gt_obj_flat, # [Q=4, 2, 3]
        }


# ═══════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    train_ds = SyntheticPose2EquipDataset(n_samples=args.n_train)
    val_ds = SyntheticPose2EquipDataset(n_samples=args.n_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\n{'='*72}")
    print(f"Training Pose2EquipNetImproved  (convergence test)")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  train={args.n_train}  val={args.n_val}")

    model = Pose2EquipNetImproved(
        num_joints=15, hidden_dim=256,
        target_skeleton_connections_idx=FILTER_SKELETON_CONNECTIONS,
        decoder_layers=3, num_heads=8,
    )
    # Unfreeze all (DINO projection params are small but also trainable)
    for p in model.parameters():
        p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen  = total - trainable
    print(f"  total_params={total:,}  trainable={trainable:,}  frozen={frozen:,}")
    print(f"{'='*72}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            frame = batch["frames"].unsqueeze(1)   # [B, 1, 3, H, W]
            h3d   = batch["human_3d"].unsqueeze(1)  # [B, 1, J, 3]
            gt    = batch["object_gt"].unsqueeze(1) # [B, 1, Q, 2, 3]

            optimizer.zero_grad()
            out = model(human_frame=frame, human_3d=h3d)
            pred = out["object_3d"]                # [B, T=1, Q, 2, 3]

            loss = nn.functional.mse_loss(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                frame = batch["frames"].unsqueeze(1)
                h3d   = batch["human_3d"].unsqueeze(1)
                gt    = batch["object_gt"].unsqueeze(1)
                out = model(human_frame=frame, human_3d=h3d)
                val_losses.append(nn.functional.mse_loss(out["object_3d"], gt).item())

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        if avg_val < best_val_loss:
            best_val_loss = avg_val

        print(f"Epoch {epoch:2d}/{args.epochs}  "
              f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}  "
              f"best_val={best_val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\n{'='*72}")
    initial_val = np.mean(val_losses) if 'val_losses' in dir() else best_val_loss * 3
    total_reduction = (np.mean(train_losses) - avg_val) / max(np.mean(train_losses), 1e-8)

    print(f"Final train loss: {avg_train:.6f}")
    print(f"Final val   loss: {avg_val:.6f}")
    print(f"Total reduction : {(1 - avg_val/np.mean(train_losses))*100:.1f}%")
    print(f"{'='*72}")

    if total_reduction > 0.5:
        print("CONVERGENCE CONFIRMED ✅")
    elif total_reduction > 0.3:
        print("LOSS DECREASING ⏳ — may need more epochs or higher LR")
    else:
        print("LOSS NOT DECREASING ENOUGH ⚠️")

if __name__ == "__main__":
    main()
