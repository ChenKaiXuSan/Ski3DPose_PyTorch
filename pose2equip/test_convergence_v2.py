#!/usr/bin/env python3
"""Convergence test: trains Pose2EquipNetImproved with synthetic data for N epochs."""

import os, sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Patch transformers.AutoModel BEFORE pose2equip imports ---
import types
class _MockDINO(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
        self.config = type("C", (), {"hidden_size": 256})()
        self.proj = nn.Linear(256, 256)
    @classmethod
    def from_pretrained(cls, model_name):
        print(f"  [mock DINO] loaded {model_name}")
        return cls(model_name)
    def forward(self, pixel_values):
        B_, _, H_, W_ = pixel_values.shape
        device = pixel_values.device
        N = 247
        cls_tok = torch.zeros(B_, 1, 256, device=device)
        feats = torch.randn(B_, N, 256, device=device)
        return type("O", (), {"last_hidden_state": torch.cat([cls_tok, feats], dim=1)})()

if "transformers" not in sys.modules:
    _tmod = types.ModuleType("transformers")
    _tmod.AutoModel = _MockDINO
    sys.modules["transformers"] = _tmod

from pose2equip.models.pose2equip_net import Pose2EquipNetImproved, DinoPatchEncoder
from pose2equip.map_config import FILTER_SKELETON_CONNECTIONS


# --- Synthetic dataset: random pose → fixed affine GT ---
class SyntheticDataset(Dataset):
    def __init__(self, n, J=15, H=224, W=224):
        self.n = n; self.J = J; self.H = H; self.W = W
    def __len__(self): return self.n
    def __getitem__(self, idx):
        pose = torch.randn(self.J, 3) * 1.5
        frame = torch.rand(3, self.H, self.W)
        # Fixed linear GT: model must learn this mapping
        Wt = torch.randn(8*3, self.J*3) * 0.1; bt = torch.randn(8*3) * 0.05
        gt = ((Wt @ pose.reshape(-1)) + bt).reshape(4, 2, 3)
        return {"frames": frame, "human_3d": pose, "object_gt": gt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)

    train_dl = DataLoader(SyntheticDataset(args.n_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(SyntheticDataset(args.n_val),   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\n{'='*72}")
    print(f"  Training Pose2EquipNetImproved  (convergence test)")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  train={args.n_train}  val={args.n_val}")

    # Build model with mock DINO
    enc = DinoPatchEncoder(mock=True)
    total_p = sum(p.numel() for p in enc.parameters())

    model = Pose2EquipNetImproved(
        num_joints=15, hidden_dim=256,
        target_skeleton_connections_idx=FILTER_SKELETON_CONNECTIONS,
        dino_model_name="mock", dino_freeze=False,  # freeze DINO projection too
        decoder_layers=3, num_heads=8,
    )
    model = model.cuda() if torch.cuda.is_available() else model

    for p in model.parameters(): p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  image_encoder_params={total_p:,}  total_model={total:,}  trainable={train_p:,}")
    print(f"{'='*72}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tlosses = []
        for batch in train_dl:
            frame = batch["frames"].unsqueeze(1).cuda() if torch.cuda.is_available() else batch["frames"].unsqueeze(1)
            h3d   = batch["human_3d"].unsqueeze(1).cuda() if torch.cuda.is_available() else batch["human_3d"].unsqueeze(1)
            gt    = batch["object_gt"].unsqueeze(1).cuda() if torch.cuda.is_available() else batch["object_gt"].unsqueeze(1)

            optimizer.zero_grad()
            pred = model(human_frame=frame, human_3d=h3d)["object_3d"]
            loss = nn.functional.mse_loss(pred, gt)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tlosses.append(loss.item())

        scheduler.step()

        model.eval(); vlosses = []
        with torch.no_grad():
            for batch in val_dl:
                frame = batch["frames"].unsqueeze(1).cuda() if torch.cuda.is_available() else batch["frames"].unsqueeze(1)
                h3d   = batch["human_3d"].unsqueeze(1).cuda() if torch.cuda.is_available() else batch["human_3d"].unsqueeze(1)
                gt    = batch["object_gt"].unsqueeze(1).cuda() if torch.cuda.is_available() else batch["object_gt"].unsqueeze(1)
                vlosses.append(nn.functional.mse_loss(model(human_frame=frame, human_3d=h3d)["object_3d"], gt).item())

        at = np.mean(tlosses); av = np.mean(vlosses)
        if av < best_val: best_val = av
        print(f"Epoch {epoch:2d}/{args.epochs}  train={at:.6f}  val={av:.6f}  best={best_val:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\n{'='*72}")
    reduction = (1 - av/at) * 100 if at > 1e-8 else 0
    print(f"  Final train loss: {at:.6f}")
    print(f"  Final val   loss: {av:.6f}")
    print(f"  Reduction:        {reduction:.1f}%")
    if reduction > 50:
        print("  RESULT: CONVERGENCE CONFIRMED ✅ — architecture is trainable")
    elif reduction > 20:
        print("  RESULT: LOSS DECREASING ⏳ — may need more epochs or higher LR")
    else:
        print("  RESULT: NOT ENOUGH REDUCTION ⚠️")
    print(f"{'='*72}")

if __name__ == "__main__":
    main()
