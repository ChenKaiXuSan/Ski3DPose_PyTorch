#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule

from models.crossview_fusion import CrossViewCanonicalFusion
from .canonicalize import canonicalize_pose_torch

logger = logging.getLogger(__name__)


class CrossViewFusionTrainer(LightningModule):
    """Pose fusion trainer for character variant using CrossViewFusion."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        data_cfg = getattr(hparams, "data", None)
        self.left_hip_idx = int(getattr(data_cfg, "left_hip_idx", 6))
        self.right_hip_idx = int(getattr(data_cfg, "right_hip_idx", 7))
        self.neck_idx = int(getattr(data_cfg, "neck_idx", 14))

        model_cfg = getattr(hparams, "cross_view_fusion", None)
        self.lr = float(getattr(hparams.loss, "lr", 0.1))
        self.weight_decay = float(
            getattr(
                model_cfg, "weight_decay", getattr(hparams.loss, "weight_decay", 0.01)
            )
        )
        self.lambda_view_recon = float(
            getattr(
                model_cfg,
                "lambda_view_recon",
                getattr(hparams.loss, "lambda_view_recon", 0.05),
            )
        )
        self.lambda_alpha_balance = float(
            getattr(model_cfg, "lambda_alpha_balance", 0.02)
        )
        self.lambda_alpha_entropy = float(
            getattr(model_cfg, "lambda_alpha_entropy", 0.005)
        )
        self.lambda_temporal_smooth = float(
            getattr(model_cfg, "lambda_temporal_smooth", 0.01)
        )

        self.models = CrossViewCanonicalFusion(
            num_heads=4,
        )

        self.save_root = str(getattr(hparams, "log_path", "./logs"))

        self.test_save_dir: Path = Path(self.save_root) / "summary"

    @staticmethod
    def _get_character_data(batch: Dict[str, Any]) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Extract character SAM, GT data, and frames.

        Returns:
            Tuple of (p_left, p_right, p_gt) where:
                - p_left, p_right: SAM 3D predictions
                - p_gt: ground truth (or None if not available)
        """

        kpt3d_sam = batch["kpt3d_sam"]
        if "cam1" in kpt3d_sam and "cam2" in kpt3d_sam:
            left_sam_kpt3d = kpt3d_sam["cam1"].float()
            right_sam_kpt3d = kpt3d_sam["cam2"].float()
        else:
            left_sam_kpt3d = kpt3d_sam["character_cam1"].float()
            right_sam_kpt3d = kpt3d_sam["character_cam2"].float()

        # Validate shape
        for p in [left_sam_kpt3d, right_sam_kpt3d]:
            if p.ndim != 4 or p.shape[-1] != 3:
                raise ValueError(
                    f"Expected pose tensor shape (B,T,J,3) for character, got {tuple(p.shape)}"
                )

        # GT data
        unity_gt_kpt3d = None
        if "kpt3d_gt" in batch:
            if isinstance(batch["kpt3d_gt"], dict):
                unity_gt_kpt3d = batch["kpt3d_gt"].get("character")
                if unity_gt_kpt3d is not None:
                    unity_gt_kpt3d = unity_gt_kpt3d.float()
            else:
                unity_gt_kpt3d = batch["kpt3d_gt"].float()

        return (
            left_sam_kpt3d,
            right_sam_kpt3d,
            unity_gt_kpt3d,
        )

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        """Process character variant with dual-view SAM 3D poses.

        Returns:
            Total loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        variant_results: Dict[str, Any] = {}

        # ===== CHARACTER: Dual2PoseNet with SAM =====
        p_left, p_right, p_gt = self._get_character_data(batch)

        # canonicalize
        left_canonical, left_transform = canonicalize_pose_torch(
            p_left,
            left_hip=self.left_hip_idx,
            right_hip=self.right_hip_idx,
            neck=self.neck_idx,
        )

        right_canonical, right_transform = canonicalize_pose_torch(
            p_right,
            left_hip=self.left_hip_idx,
            right_hip=self.right_hip_idx,
            neck=self.neck_idx,
        )

        gt_canonical, _ = (
            canonicalize_pose_torch(
                p_gt,
                left_hip=self.left_hip_idx,
                right_hip=self.right_hip_idx,
                neck=self.neck_idx,
            )
            if p_gt is not None
            else None
        )

        # Forward pass
        fused, aux = self.models(
            left_canonical,
            right_canonical,
        )
        alpha = aux["alpha"]

        # Reconstruct left/right poses from fused pose + reliability weight.
        # With delta = (p_left - p_right):
        #   p_hat = alpha * p_left_recon + (1-alpha) * p_right_recon
        #   p_left_recon - p_right_recon = delta
        delta_lr = left_canonical - right_canonical
        p_left_recon = fused + (1.0 - alpha) * delta_lr
        p_right_recon = fused - alpha * delta_lr

        loss_recon_left = torch.nn.functional.l1_loss(p_left_recon, left_canonical)
        loss_recon_right = torch.nn.functional.l1_loss(p_right_recon, right_canonical)
        loss_view_recon = 0.5 * (loss_recon_left + loss_recon_right)

        # Reliability diagnostics: cross-view discrepancy and temporal stability.
        cv_gap = torch.norm(left_canonical - right_canonical, dim=-1).mean()

        # Compute loss
        mpjpe = fused.new_tensor(float("nan"))
        if p_gt is not None:
            loss_sup = torch.nn.functional.l1_loss(fused, gt_canonical)
            loss_dict = {
                "loss": loss_sup,
            }
            mpjpe = torch.norm(fused - gt_canonical, dim=-1).mean()
            accel_err = self._safe_accel_err(fused, gt_canonical)[1]
            self.log(
                f"{stage}/mpjpe",
                mpjpe,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=fused.shape[0],
            )
            self.log(
                f"{stage}/accel_err",
                accel_err,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=fused.shape[0],
            )
        else:
            pseudo_gt = 0.5 * (left_canonical + right_canonical)
            loss_self = torch.nn.functional.l1_loss(fused, pseudo_gt)
            loss_dict = {
                "loss": loss_self,
            }

        # Prevent gating collapse to one side.
        eps = 1e-6
        alpha_mean = alpha.mean()
        loss_alpha_balance = (alpha_mean - 0.5).abs()
        alpha_entropy = -(
            alpha * torch.log(alpha.clamp(min=eps))
            + (1.0 - alpha) * torch.log((1.0 - alpha).clamp(min=eps))
        ).mean()
        # Minimize negative entropy == maximize entropy.
        loss_alpha_entropy = -alpha_entropy

        # temporla smooth loss
        acc = fused[:, 2:] - 2.0 * fused[:, 1:-1] + fused[:, :-2]
        temporal_smooth_loss = torch.norm(acc, dim=-1).mean()
        loss = (
            loss_dict["loss"]
            + self.lambda_view_recon * loss_view_recon
            + self.lambda_alpha_balance * loss_alpha_balance
            + self.lambda_alpha_entropy * loss_alpha_entropy
            + self.lambda_temporal_smooth * temporal_smooth_loss
        )
        total_loss = total_loss + loss

        # Log loss components
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/temporal_smooth_loss",
            temporal_smooth_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=fused.shape[0],
        )

        # Log alpha statistics
        self.log(
            f"{stage}/alpha_mean",
            alpha.mean(),
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/alpha_std",
            alpha.std(),
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )

        self.log(
            f"{stage}/cross_view_gap",
            cv_gap,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )

        self.log(
            f"{stage}/recon_left",
            loss_recon_left,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/recon_right",
            loss_recon_right,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/recon_view",
            loss_view_recon,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/loss_alpha_balance",
            loss_alpha_balance,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )
        self.log(
            f"{stage}/alpha_entropy",
            alpha_entropy,
            on_step=False,
            on_epoch=True,
            batch_size=fused.shape[0],
        )

        # Store results
        variant_results = {
            "fused": fused,
            "alpha": alpha,
            "p_left": p_left,
            "p_right": p_right,
            "left_canonical": left_canonical,
            "right_canonical": right_canonical,
            "p_left_recon": p_left_recon,
            "p_right_recon": p_right_recon,
            "ground_truth": p_gt if p_gt is not None else None,
            "ground_truth_canonical": (
                gt_canonical if gt_canonical is not None else None
            ),
        }

        # Include input masks (if provided by the dataloader/collate)
        kpt3d_mask = batch.get("kpt3d_mask", {})
        if isinstance(kpt3d_mask, dict):
            variant_results["mask_left"] = kpt3d_mask.get("cam1")
            variant_results["mask_right"] = kpt3d_mask.get("cam2")

        # Store variant results in batch for later use
        batch["_variant_results"] = variant_results

        return total_loss

    def training_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def on_test_start(self) -> None:
        self.test_outputs = []
        self.test_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cross-view Fusion test start")

    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        """Run inference on all variants and collect outputs."""
        self._shared_step(batch, stage="test")

        _results = batch.get("_variant_results", {})

        self.test_outputs.append(_results)

        return torch.tensor(0.0)

    @staticmethod
    def _safe_mpjpe(
        pred: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, float]:

        mpjpe_per_point = torch.norm(pred - label, dim=-1)
        _mean = mpjpe_per_point.mean().item()
        return mpjpe_per_point, _mean

    @staticmethod
    def _safe_accel_err(
        pred: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Acceleration error against GT via second-order temporal differences."""
        if pred.shape[1] < 3 or label.shape[1] < 3:
            empty = pred.new_zeros((*pred.shape[:1], 0, pred.shape[2]))
            return empty, float("nan")

        pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
        label_acc = label[:, 2:] - 2.0 * label[:, 1:-1] + label[:, :-2]
        accel_err_per_point = torch.norm(pred_acc - label_acc, dim=-1)
        _mean = accel_err_per_point.mean().item()
        return accel_err_per_point, _mean

    def on_test_epoch_end(self) -> None:

        save_dir = self.test_save_dir

        # per-joint collectors: each element shape (B, T, J)
        all_fused_per_joint = []
        all_left_raw_per_joint = []
        all_right_raw_per_joint = []
        all_left_canonical_per_joint = []
        all_right_canonical_per_joint = []
        all_left_recon_per_joint = []
        all_right_recon_per_joint = []
        all_pesudo_fuse_per_joint = []
        all_pesudo_canonical_fuse_per_joint = []

        # acceleration-error collectors: each element shape (B, T-2, J)
        all_fused_accel_err = []
        all_left_raw_accel_err = []
        all_right_raw_accel_err = []
        all_left_canonical_accel_err = []
        all_right_canonical_accel_err = []
        all_pesudo_fuse_accel_err = []
        all_pesudo_canonical_fuse_accel_err = []

        for output in self.test_outputs:

            fused = output.get("fused")
            p_left = output.get("p_left")
            p_right = output.get("p_right")
            left_canonical = output.get("left_canonical")
            right_canonical = output.get("right_canonical")
            p_left_recon = output.get("p_left_recon")
            p_right_recon = output.get("p_right_recon")
            ground_truth = output.get("ground_truth")
            ground_truth_canonical = output.get("ground_truth_canonical")

            # calc mpjpe
            fused_mpjpe_per_point, _ = self._safe_mpjpe(fused, ground_truth_canonical)

            left_raw_mpjpe_per_point, _ = self._safe_mpjpe(p_left, ground_truth)
            right_raw_mpjpe_per_point, _ = self._safe_mpjpe(p_right, ground_truth)

            left_canonical_mpjpe_per_point, _ = self._safe_mpjpe(
                left_canonical, ground_truth_canonical
            )
            right_canonical_mpjpe_per_point, _ = self._safe_mpjpe(
                right_canonical, ground_truth_canonical
            )

            # Reconstruction error should be measured against canonical inputs.
            left_recon_mpjpe_per_point, _ = self._safe_mpjpe(
                p_left_recon, left_canonical
            )
            right_recon_mpjpe_per_point, _ = self._safe_mpjpe(
                p_right_recon, right_canonical
            )

            pesudo_fuse = 0.5 * (p_left + p_right)
            pesudo_fuse_mpjpe_per_point, _ = self._safe_mpjpe(pesudo_fuse, ground_truth)

            pesudo_canonical_fuse = 0.5 * (left_canonical + right_canonical)
            pesudo_canonical_fuse_mpjpe_per_point, _ = self._safe_mpjpe(
                pesudo_canonical_fuse, ground_truth_canonical
            )

            fused_accel_per_point, _ = self._safe_accel_err(
                fused, ground_truth_canonical
            )
            left_raw_accel_per_point, _ = self._safe_accel_err(p_left, ground_truth)
            right_raw_accel_per_point, _ = self._safe_accel_err(p_right, ground_truth)
            left_canonical_accel_per_point, _ = self._safe_accel_err(
                left_canonical, ground_truth_canonical
            )
            right_canonical_accel_per_point, _ = self._safe_accel_err(
                right_canonical, ground_truth_canonical
            )
            pesudo_fuse_accel_per_point, _ = self._safe_accel_err(
                pesudo_fuse, ground_truth
            )
            pesudo_canonical_fuse_accel_per_point, _ = self._safe_accel_err(
                pesudo_canonical_fuse, ground_truth_canonical
            )

            # collect per-joint tensors (B, T, J) -> flatten to (N, J)
            all_fused_per_joint.append(
                fused_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_left_raw_per_joint.append(
                left_raw_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_right_raw_per_joint.append(
                right_raw_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_left_canonical_per_joint.append(
                left_canonical_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_right_canonical_per_joint.append(
                right_canonical_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_left_recon_per_joint.append(
                left_recon_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_right_recon_per_joint.append(
                right_recon_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_pesudo_fuse_per_joint.append(
                pesudo_fuse_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )
            all_pesudo_canonical_fuse_per_joint.append(
                pesudo_canonical_fuse_mpjpe_per_point.detach().cpu().flatten(0, -2)
            )

            if fused_accel_per_point.numel() > 0:
                all_fused_accel_err.append(
                    fused_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_left_raw_accel_err.append(
                    left_raw_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_right_raw_accel_err.append(
                    right_raw_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_left_canonical_accel_err.append(
                    left_canonical_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_right_canonical_accel_err.append(
                    right_canonical_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_pesudo_fuse_accel_err.append(
                    pesudo_fuse_accel_per_point.detach().cpu().flatten(0, -2)
                )
                all_pesudo_canonical_fuse_accel_err.append(
                    pesudo_canonical_fuse_accel_per_point.detach().cpu().flatten(0, -2)
                )

        # aggregate metrics across all samples: concatenate (N, J) then reduce.
        def _per_joint_mean(lst):
            if not lst:
                return []
            return torch.cat(lst, dim=0).mean(dim=0).tolist()  # list of length J

        def _global_mean(lst):
            if not lst:
                return float("nan")
            return float(torch.cat(lst, dim=0).mean().item())

        per_joint_stats = {
            "fused": _per_joint_mean(all_fused_per_joint),
            "left_raw": _per_joint_mean(all_left_raw_per_joint),
            "right_raw": _per_joint_mean(all_right_raw_per_joint),
            "left_canonical": _per_joint_mean(all_left_canonical_per_joint),
            "right_canonical": _per_joint_mean(all_right_canonical_per_joint),
            "left_recon": _per_joint_mean(all_left_recon_per_joint),
            "right_recon": _per_joint_mean(all_right_recon_per_joint),
            "pesudo_fuse": _per_joint_mean(all_pesudo_fuse_per_joint),
            "pesudo_canonical_fuse": _per_joint_mean(
                all_pesudo_canonical_fuse_per_joint
            ),
        }

        gt_summary = {
            "fused_mpjpe_mean": _global_mean(all_fused_per_joint),
            "sam3d_left_mpjpe_mean": _global_mean(all_left_raw_per_joint),
            "sam3d_right_mpjpe_mean": _global_mean(all_right_raw_per_joint),
            "left_canonical_mpjpe_mean": _global_mean(all_left_canonical_per_joint),
            "right_canonical_mpjpe_mean": _global_mean(all_right_canonical_per_joint),
            "pesudo_fuse_mpjpe_mean": _global_mean(all_pesudo_fuse_per_joint),
            "pesudo_canonical_fuse_mpjpe_mean": _global_mean(
                all_pesudo_canonical_fuse_per_joint
            ),
        }

        recon_summary = {
            "left_recon_mpjpe_mean": _global_mean(all_left_recon_per_joint),
            "right_recon_mpjpe_mean": _global_mean(all_right_recon_per_joint),
        }

        accel_summary = {
            "fused_accel_err_mean": _global_mean(all_fused_accel_err),
            "sam3d_left_accel_err_mean": _global_mean(all_left_raw_accel_err),
            "sam3d_right_accel_err_mean": _global_mean(all_right_raw_accel_err),
            "left_canonical_accel_err_mean": _global_mean(all_left_canonical_accel_err),
            "right_canonical_accel_err_mean": _global_mean(all_right_canonical_accel_err),
            "pesudo_fuse_accel_err_mean": _global_mean(all_pesudo_fuse_accel_err),
            "pesudo_canonical_fuse_accel_err_mean": _global_mean(
                all_pesudo_canonical_fuse_accel_err
            ),
        }

        # save results (disabled: eval scripts get tensors from memory via _flatten_test_outputs)
        # save_file = save_dir / f"outputs.pt"
        # torch.save(self.test_outputs, save_file)

        # report summary to txt
        txt_file = save_dir / f"report.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("Cross-View Fusion Test Report\n")
            f.write("=" * 40 + "\n")
            f.write("[Canonical MPJPE vs Ground Truth]\n")
            f.write("All values in this section compare canonicalized poses.\n")
            for k, v in gt_summary.items():
                f.write(f"{k}: {v:.4f}\n")

            f.write("\n[Raw SAM3D vs Ground Truth]\n")
            f.write("These values compare the original pseudo GT poses before canonicalization.\n")
            f.write(f"sam3d_cam1_mpjpe_mean: {gt_summary['sam3d_left_mpjpe_mean']:.4f}\n")
            f.write(f"sam3d_cam2_mpjpe_mean: {gt_summary['sam3d_right_mpjpe_mean']:.4f}\n")

            f.write("\n[Reconstruction Error (vs canonical input)]\n")
            f.write("This section is against canonicalized camera inputs.\n")
            for k, v in recon_summary.items():
                f.write(f"{k}: {v:.4f}\n")

            f.write("\n[Acceleration Error vs Ground Truth]\n")
            f.write("All values compare second-order temporal differences against GT.\n")
            for k, v in accel_summary.items():
                f.write(f"{k}: {v:.4f}\n")

            f.write("\n[Per-Joint Canonical MPJPE vs Ground Truth (joint index: mean error)]\n")
            for variant_name in [
                "fused",
                "sam3d_cam1",
                "sam3d_cam2",
                "left_canonical",
                "right_canonical",
                "pesudo_fuse",
                "pesudo_canonical_fuse",
            ]:
                if variant_name == "sam3d_cam1":
                    joint_values = per_joint_stats["left_raw"]
                elif variant_name == "sam3d_cam2":
                    joint_values = per_joint_stats["right_raw"]
                else:
                    joint_values = per_joint_stats[variant_name]
                f.write(f"  {variant_name}:\n")
                for j, val in enumerate(joint_values):
                    f.write(f"    joint_{j:02d}: {val:.4f}\n")

            f.write("\n[Per-Joint Reconstruction Error (vs canonical input)]\n")
            for variant_name in ["left_recon", "right_recon"]:
                joint_values = per_joint_stats[variant_name]
                f.write(f"  {variant_name}:\n")
                for j, val in enumerate(joint_values):
                    f.write(f"    joint_{j:02d}: {val:.4f}\n")

        logger.info("Saved pose predictions/labels to %s", save_file)
        logger.info("Saved test report to %s", txt_file)

    def configure_optimizers(self):
        """Configure optimizer for the character model."""
        optimizer = torch.optim.AdamW(
            self.models.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        tmax = getattr(self.trainer, "estimated_stepping_batches", None)
        if not isinstance(tmax, int) or tmax <= 0:
            tmax = 1000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }
