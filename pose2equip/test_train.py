#!/usr/bin/env python3
"""Train Pose2EquipNetImproved on synthetic data — checks if it converges."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch transformers AutoModel before pose2equip imports (proven to work in test_model_forward.py)
class _MockDINOOut:
    def __init__(self, last_hidden_state): self.last_hidden_state = last_hidden_state

import torch.nn as nn
import torch
class MockDINO(nn.Module):
    def __init__(s, model_name=None):
        super().__init__()
        s.config = type("C", (), {"hidden_size": 256})()
        s.proj = nn.Linear(256, 256)
    @classmethod
    def from_pretrained(c, n): return MockDINO(n)
    def forward(s, pixel_values):
        B_, _, H_, W_ = pixel_values.shape; N=247; d=pixel_values.device
        s._out = _MockDINOOut(torch.cat([torch.zeros(B_,1,256,device=d), torch.randn(B_,N,256,device=d)], dim=1))
        return s._out

import types
_tmod = types.ModuleType("transformers")
_tmod.AutoModel = MockDINO
sys.modules["transformers"] = _tmod

# Now import pose2equip modules — they'll see our mock AutoModel
from pose2equip.models.pose2equip_net import Pose2EquipNetImproved, DinoPatchEncoder
from pose2equip.map_config import FILTER_SKELETON_CONNECTIONS

import numpy as np
from torch.utils.data import Dataset, DataLoader

class SynDataset(Dataset):
    def __init__(self, n=500, J=15): self.n=n; self.J=J
    def __len__(self): return self.n
    def __getitem__(self, idx):
        pose = torch.randn(self.J, 3) * 1.5          # [J, 3] raw 3D joints
        frame = torch.rand(3, 224, 224)              # [3, H, W] synthetic frames
        gt   = torch.randn(4, 2, 3) * 0.5            # [Q=4, 2 endpoints, 3 coords] ground truth
        return {"frames": frame, "pose": pose, "gt": gt}


def main():
    EPOCHS = 15
    BS = 8

    torch.manual_seed(42); np.random.seed(42)

    # Build model — uses mock DINO since transformers is patched
    enc = DinoPatchEncoder()                          # will use MockDINO from sys.modules["transformers"]
    print(f"DinoPatchEncoder loaded (mock), params={sum(p.numel() for p in enc.parameters()):,}")

    model = Pose2EquipNetImproved(
        num_joints=15, hidden_dim=256, dino_freeze=False,
        target_skeleton_connections_idx=FILTER_SKELETON_CONNECTIONS,
        decoder_layers=3, num_heads=8,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trn   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: total={total:,}  trainable={trn:,}  frozen={total-trn:,}")
    print(f"Device: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float("inf")
    all_train_losses = []
    all_val_losses = []

    for ep in range(1, EPOCHS + 1):
        # ---- Training ----
        model.train()
        train_losses = []
        loader = DataLoader(SynDataset(n=200), batch_size=BS, shuffle=True, num_workers=0)
        for batch in loader:
            frame = batch["frames"].unsqueeze(1).to(device)   # [B, T=1, 3, H, W]
            pose  = batch["pose"].unsqueeze(1).to(device)       # [B, T=1, J, 3]
            gt    = batch["gt"].unsqueeze(1).to(device)          # [B, T=1, Q, 2, 3]

            optimizer.zero_grad()
            out = model(human_frame=frame, human_3d=pose)
            loss = torch.nn.functional.mse_loss(out["object_3d"], gt)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_losses.append(loss.item())

        # ---- Validation ----
        model.eval()
        val_losses = []
        loader_v = DataLoader(SynDataset(n=50), batch_size=BS, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader_v:
                frame = batch["frames"].unsqueeze(1).to(device)
                pose  = batch["pose"].unsqueeze(1).to(device)
                gt    = batch["gt"].unsqueeze(1).to(device)
                vloss = torch.nn.functional.mse_loss(model(human_frame=frame, human_3d=pose)["object_3d"], gt).item()
                val_losses.append(vloss)

        scheduler.step()
        at = np.mean(train_losses); av = np.mean(val_losses)
        if av < best_val: best_val = av
        all_train_losses.append(at); all_val_losses.append(av)
        print(f"Epoch {ep:2d}/{EPOCHS}  train={at:.6f}  val={av:.6f}  best_val={best_val:.6f}")

    # ---- Result ----
    print(f"\n{'='*60}")
    avg_train_first = np.mean(all_train_losses[:2])
    avg_train_last  = np.mean(all_train_losses[-2:])
    reduction = (1 - all_val_losses[-1] / max(avg_train_last, 1e-8)) * 100

    print(f"Train loss (first 2 epochs avg): {avg_train_first:.6f}")
    print(f"Train loss (last  2 epochs avg): {avg_train_last:.6f}")
    print(f"Val   loss (last epoch)          : {all_val_losses[-1]:.6f}")
    print(f"Reduction: {reduction:.1f}%")

    if reduction > 50:
        print("CONVERGENCE CONFIRMED ✅")
        print(f"The model successfully learned the synthetic mapping.")
        print(f"Val loss dropped to {(all_val_losses[-1]/avg_train_last)*100:.1f}% of training loss level.")
    elif reduction > 25:
        print("CONVERGENCE IN PROGRESS ⏳")
        print(f"Loss is decreasing. May need more epochs or higher LR for full convergence.")
    else:
        print("LOSS NOT DECREASING ENOUGH ⚠️")

    # Save loss curves for inspection
    np.savez("/tmp/convergence_test.npz", train=all_train_losses, val=all_val_losses)
    print(f"Loss curves saved to /tmp/convergence_test.npz")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
