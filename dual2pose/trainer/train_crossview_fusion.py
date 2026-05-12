#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule

from ..models.crossview_fusion import CrossViewCanonicalFusion
from .canonicalize import canonicalize_pose_numpy, canonicalize_pose_torch

logger = logging.getLogger(__name__)


class CrossViewFusionTrainer(LightningModule):
    """Pose fusion trainer for character variant using CrossViewFusion."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_cfg = getattr(hparams, "crossview_fusion", None)
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
        self.lambda_p0_supervise = float(getattr(model_cfg, "lambda_p0_supervise", 0.2))
        self.rigid_align_right_to_left = bool(
            getattr(model_cfg, "rigid_align_right_to_left", False)
        )
        self.console_print_train_metrics = bool(
            getattr(model_cfg, "console_print_train_metrics", True)
        )
        self.console_print_every_n_steps = int(
            getattr(model_cfg, "console_print_every_n_steps", 20)
        )
        self.console_print_include_val = bool(
            getattr(model_cfg, "console_print_include_val", False)
        )

        # Default to confidence-free reliability modeling.
        use_conf = bool(getattr(model_cfg, "use_conf", False))
        predict_logvar = bool(getattr(model_cfg, "predict_logvar", False))

        num_joints_character = 15  # After filtering with FILTER_SKELETON_CONNECTIONS

        self.models = torch.nn.ModuleDict()
        self.models["character"] = CrossViewCanonicalFusion(
            num_heads=4,
        )

        logger.info(
            "Created model: character=CrossViewCanonicalFusion(%s)",
            num_joints_character,
        )

        self.save_root = str(getattr(hparams, "log_path", "./logs"))
        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir: Path = Path(self.save_root) / "pose_analysis"
        self.test_vis_enabled = bool(getattr(model_cfg, "save_test_vis", True))
        self.test_vis_max_samples = int(getattr(model_cfg, "test_vis_max_samples", 2))
        self.test_vis_dir: Path = self.test_save_dir / "vis"

        logger.info(
            "CrossViewFusionTrainer config: use_conf=%s, predict_logvar=%s,"
            "lambda_view_recon=%.4f, lambda_alpha_balance=%.4f, lambda_alpha_entropy=%.4f, lambda_p0_supervise=%.4f, "
            "rigid_align_right_to_left=%s",
            use_conf,
            predict_logvar,
            self.lambda_view_recon,
            self.lambda_alpha_balance,
            self.lambda_alpha_entropy,
            self.lambda_p0_supervise,
            self.rigid_align_right_to_left,
        )

    @staticmethod
    def _temporal_velocity_norm(x: torch.Tensor) -> torch.Tensor:
        """Mean first-order temporal difference norm for (B,T,J,3)."""
        if x.ndim != 4 or x.shape[1] <= 1:
            return x.new_tensor(0.0)
        vel = x[:, 1:] - x[:, :-1]
        return torch.norm(vel, dim=-1).mean()

    @staticmethod
    def _temporal_acceleration_norm(x: torch.Tensor) -> torch.Tensor:
        """Mean second-order temporal difference norm for (B,T,J,3)."""
        if x.ndim != 4 or x.shape[1] <= 2:
            return x.new_tensor(0.0)
        acc = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
        return torch.norm(acc, dim=-1).mean()

    def _is_global_zero(self) -> bool:
        rank = int(getattr(self, "global_rank", 0))
        return rank == 0

    def _to_float(self, value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float("nan")
            return float(value.detach().float().mean().item())
        if isinstance(value, (int, float)):
            return float(value)
        return float("nan")

    def _maybe_print_metrics_to_console(
        self,
        stage: str,
        metrics: Dict[str, Any],
    ) -> None:
        if not self._is_global_zero():
            return

        if stage == "train":
            if not self.console_print_train_metrics:
                return
            every = max(1, self.console_print_every_n_steps)
            step = int(getattr(self, "global_step", 0))
            if step % every != 0:
                return
        elif stage == "val":
            if not self.console_print_include_val:
                return
        else:
            return

        epoch = int(getattr(self, "current_epoch", 0))
        step = int(getattr(self, "global_step", 0))

        parts = [f"[{stage}] epoch={epoch} step={step}"]
        ordered_keys = [
            "loss",
            "mpjpe",
            "alpha_mean",
            "cross_view_gap",
            "recon_view",
            "vel_norm",
            "acc_norm",
            "p0_drift",
            "rigid_applied_ratio",
            "rigid_rmse_before",
            "rigid_rmse_after",
            "dino_feat_gap",
        ]
        for key in ordered_keys:
            if key in metrics:
                value = self._to_float(metrics[key])
                if value == value:
                    parts.append(f"{key}={value:.6f}")

        print(" | ".join(parts), flush=True)

    @staticmethod
    def _get_character_data(batch: Dict[str, Any]) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Extract character SAM, GT data, and frames.

        Returns:
            Tuple of (p_left, p_right, p_gt, frames_left, frames_right) where:
                - p_left, p_right: SAM 3D predictions
                - p_gt: ground truth (or None if not available)
                - frames_left, frames_right: video frames (or None if not available)
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

        # Frame data (video frames)
        frames_left = None
        frames_right = None
        if "frames" in batch and isinstance(batch["frames"], dict):
            if "cam1" in batch["frames"]:
                frames_left = batch["frames"]["cam1"].float()
            if "cam2" in batch["frames"]:
                frames_right = batch["frames"]["cam2"].float()

        return (
            left_sam_kpt3d,
            right_sam_kpt3d,
            unity_gt_kpt3d,
            frames_left,
            frames_right,
        )

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        """Process character variant with dual-view SAM 3D poses.

        Returns:
            Total loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        variant_results: Dict[str, Any] = {}

        # ===== CHARACTER: Dual2PoseNet with SAM =====
        p_left, p_right, p_gt, frames_left, frames_right = self._get_character_data(
            batch
        )

        # canonicalize
        left_canonical, left_transform = canonicalize_pose_torch(p_left)

        right_canonical, right_transform = canonicalize_pose_torch(p_right)

        gt_canonical, _ = canonicalize_pose_torch(p_gt) if p_gt is not None else None

        # Forward pass
        fused, aux = self.models["character"](
            left_canonical,
            right_canonical,
        )
        p_hat = fused
        alpha = aux["alpha"]
        p0 = 0.5 * (p_left + p_right)
        logvar = None
        img_feat_left = None
        img_feat_right = None
        rigid_stats = {
            "rmse_before": float("nan"),
            "rmse_after": float("nan"),
            "applied_ratio": 0.0,
        }

        # Reconstruct left/right poses from fused pose + reliability weight.
        # With delta = (p_left - p_right):
        #   p_hat = alpha * p_left_recon + (1-alpha) * p_right_recon
        #   p_left_recon - p_right_recon = delta
        delta_lr = p_left - p_right
        p_left_recon = p_hat + (1.0 - alpha) * delta_lr
        p_right_recon = p_hat - alpha * delta_lr

        loss_recon_left = torch.nn.functional.l1_loss(p_left_recon, left_canonical)
        loss_recon_right = torch.nn.functional.l1_loss(p_right_recon, right_canonical)
        loss_view_recon = 0.5 * (loss_recon_left + loss_recon_right)

        # Reliability diagnostics: cross-view discrepancy and temporal stability.
        cv_gap = torch.norm(left_canonical - right_canonical, dim=-1).mean()
        vel_norm = self._temporal_velocity_norm(p_hat)
        acc_norm = self._temporal_acceleration_norm(p_hat)

        # Compute loss
        mpjpe = p_hat.new_tensor(float("nan"))
        if p_gt is not None:
            loss_sup = torch.nn.functional.l1_loss(fused, gt_canonical)
            loss_dict = {
                "loss": loss_sup,
                "l1": loss_sup,
            }
            mpjpe = torch.norm(fused - gt_canonical, dim=-1).mean()
            loss_p0_supervise = torch.nn.functional.l1_loss(p0, gt_canonical)
            self.log(
                f"{stage}/mpjpe",
                mpjpe,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=p_hat.shape[0],
            )
        else:
            pseudo_gt = 0.5 * (left_canonical + right_canonical)
            loss_self = torch.nn.functional.l1_loss(fused, pseudo_gt)
            loss_dict = {
                "loss": loss_self,
                "l1": loss_self,
            }
            loss_p0_supervise = p_hat.new_tensor(0.0)

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

        loss = (
            loss_dict["loss"]
            + self.lambda_view_recon * loss_view_recon
            + self.lambda_alpha_balance * loss_alpha_balance
            + self.lambda_alpha_entropy * loss_alpha_entropy
            + self.lambda_p0_supervise * loss_p0_supervise
        )
        total_loss = total_loss + loss

        # Log loss components
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=p_hat.shape[0],
        )
        for k, v in loss_dict.items():
            if k != "loss":
                self.log(
                    f"{stage}/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    batch_size=p_hat.shape[0],
                )

        # Log alpha statistics
        self.log(
            f"{stage}/alpha_mean",
            alpha.mean(),
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/alpha_std",
            alpha.std(),
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/cross_view_gap",
            cv_gap,
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        if self.rigid_align_right_to_left:
            rmse_before = rigid_stats["rmse_before"]
            rmse_after = rigid_stats["rmse_after"]
            if rmse_before == rmse_before:
                self.log(
                    f"{stage}/rigid_rmse_before",
                    p_hat.new_tensor(rmse_before),
                    on_step=True,
                    on_epoch=True,
                    batch_size=p_hat.shape[0],
                )
            if rmse_after == rmse_after:
                self.log(
                    f"{stage}/rigid_rmse_after",
                    p_hat.new_tensor(rmse_after),
                    on_step=True,
                    on_epoch=True,
                    batch_size=p_hat.shape[0],
                )
            self.log(
                f"{stage}/rigid_applied_ratio",
                p_hat.new_tensor(rigid_stats["applied_ratio"]),
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
        self.log(
            f"{stage}/vel_norm",
            vel_norm,
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/acc_norm",
            acc_norm,
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/recon_left",
            loss_recon_left,
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/recon_right",
            loss_recon_right,
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/recon_view",
            loss_view_recon,
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/loss_alpha_balance",
            loss_alpha_balance,
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/alpha_entropy",
            alpha_entropy,
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/loss_p0_supervise",
            loss_p0_supervise,
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )
        self.log(
            f"{stage}/p0_drift",
            torch.nn.functional.l1_loss(p_hat, p0),
            on_step=True,
            on_epoch=True,
            batch_size=p_hat.shape[0],
        )

        console_metrics: Dict[str, Any] = {
            "loss": loss,
            "alpha_mean": alpha.mean(),
            "cross_view_gap": cv_gap,
            "recon_view": loss_view_recon,
            "vel_norm": vel_norm,
            "acc_norm": acc_norm,
            "p0_drift": torch.nn.functional.l1_loss(p_hat, p0),
        }
        if p_gt is not None:
            console_metrics["mpjpe"] = mpjpe
        if self.rigid_align_right_to_left:
            console_metrics["rigid_applied_ratio"] = p_hat.new_tensor(
                rigid_stats["applied_ratio"]
            )
            if rigid_stats["rmse_before"] == rigid_stats["rmse_before"]:
                console_metrics["rigid_rmse_before"] = p_hat.new_tensor(
                    rigid_stats["rmse_before"]
                )
            if rigid_stats["rmse_after"] == rigid_stats["rmse_after"]:
                console_metrics["rigid_rmse_after"] = p_hat.new_tensor(
                    rigid_stats["rmse_after"]
                )

        self._maybe_print_metrics_to_console(stage=stage, metrics=console_metrics)

        # Store results
        variant_results["character"] = {
            "p_hat": p_hat,
            "alpha": alpha,
            "p_left": p_left,
            "p_right": p_right,
            "p_left_recon": p_left_recon,
            "p_right_recon": p_right_recon,
            "logvar": logvar,
        }
        if img_feat_left is not None:
            variant_results["character"]["img_feat_left"] = img_feat_left
        if img_feat_right is not None:
            variant_results["character"]["img_feat_right"] = img_feat_right
        if p_gt is not None:
            variant_results["character"]["p_gt"] = p_gt
        if frames_left is not None:
            variant_results["character"]["frames_cam1"] = frames_left
        if frames_right is not None:
            variant_results["character"]["frames_cam2"] = frames_right

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
        self.test_vis_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dual2PoseNet test start")

    @staticmethod
    def _set_equal_axes_3d(ax, pts: torch.Tensor) -> None:
        if pts.numel() == 0:
            return
        mins = pts.min(dim=0).values
        maxs = pts.max(dim=0).values
        center = (mins + maxs) * 0.5
        radius = float((maxs - mins).max().item() * 0.6)
        if radius <= 0:
            radius = 1.0
        ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
        ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
        ax.set_zlim(float(center[2] - radius), float(center[2] + radius))

    def _save_test_visualizations(
        self,
        variant_results: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        if not self.test_vis_enabled or not self._is_global_zero():
            return

        for variant, results in variant_results.items():
            p_hat = results.get("p_hat")
            p_left = results.get("p_left")
            p_right = results.get("p_right")
            p_gt = results.get("p_gt")

            if not isinstance(p_hat, torch.Tensor):
                continue
            if not isinstance(p_left, torch.Tensor) or not isinstance(
                p_right, torch.Tensor
            ):
                continue
            if p_hat.ndim != 4 or p_left.ndim != 4 or p_right.ndim != 4:
                continue

            bsz = p_hat.shape[0]
            max_samples = min(bsz, max(1, self.test_vis_max_samples))
            t_idx = p_hat.shape[1] // 2

            for b_idx in range(max_samples):
                left_pose = p_left[b_idx, t_idx].detach().cpu()
                right_pose = p_right[b_idx, t_idx].detach().cpu()
                fused_pose = p_hat[b_idx, t_idx].detach().cpu()
                gt_pose = (
                    p_gt[b_idx, t_idx].detach().cpu()
                    if isinstance(p_gt, torch.Tensor) and p_gt.ndim == 4
                    else None
                )

                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(1, 3, 1, projection="3d")
                ax2 = fig.add_subplot(1, 3, 2, projection="3d")
                ax3 = fig.add_subplot(1, 3, 3, projection="3d")

                ax1.scatter(
                    left_pose[:, 0],
                    left_pose[:, 1],
                    left_pose[:, 2],
                    s=10,
                    c="tab:blue",
                )
                ax1.set_title("left")

                ax2.scatter(
                    right_pose[:, 0],
                    right_pose[:, 1],
                    right_pose[:, 2],
                    s=10,
                    c="tab:orange",
                )
                ax2.set_title("right")

                ax3.scatter(
                    fused_pose[:, 0],
                    fused_pose[:, 1],
                    fused_pose[:, 2],
                    s=12,
                    c="tab:red",
                    label="fused",
                )
                if gt_pose is not None:
                    ax3.scatter(
                        gt_pose[:, 0],
                        gt_pose[:, 1],
                        gt_pose[:, 2],
                        s=10,
                        c="tab:green",
                        alpha=0.75,
                        label="gt",
                    )
                    ax3.legend(loc="upper right")
                    pts_ref = torch.cat(
                        [left_pose, right_pose, fused_pose, gt_pose], dim=0
                    )
                else:
                    pts_ref = torch.cat([left_pose, right_pose, fused_pose], dim=0)
                ax3.set_title("fused_vs_gt")

                self._set_equal_axes_3d(ax1, pts_ref)
                self._set_equal_axes_3d(ax2, pts_ref)
                self._set_equal_axes_3d(ax3, pts_ref)

                for ax in (ax1, ax2, ax3):
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.set_zlabel("Z")

                fig.suptitle(f"{variant} batch={batch_idx} sample={b_idx} t={t_idx}")
                fig.tight_layout()
                out_file = (
                    self.test_vis_dir
                    / f"batch_{batch_idx:05d}_{variant}_b{b_idx:02d}_t{t_idx:03d}.png"
                )
                fig.savefig(out_file, dpi=160)
                plt.close(fig)

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        """Run inference on all variants and collect outputs."""
        self._shared_step(batch, stage="test")

        variant_results = batch.get("_variant_results", {})
        self._save_test_visualizations(
            variant_results=variant_results, batch_idx=_batch_idx
        )

        pack: Dict[str, Any] = {
            "variant_results": {},
        }

        for variant, results in variant_results.items():
            pack_entry: Dict[str, Any] = {
                "p_hat": results["p_hat"].detach().cpu(),
            }
            if "alpha" in results:
                pack_entry["alpha"] = results["alpha"].detach().cpu()
            if "p_left" in results:
                pack_entry["p_left"] = results["p_left"].detach().cpu()
            if "p_right" in results:
                pack_entry["p_right"] = results["p_right"].detach().cpu()
            if "p_left_recon" in results:
                pack_entry["p_left_recon"] = results["p_left_recon"].detach().cpu()
            if "p_right_recon" in results:
                pack_entry["p_right_recon"] = results["p_right_recon"].detach().cpu()
            if "img_feat_left" in results:
                pack_entry["img_feat_left"] = results["img_feat_left"].detach().cpu()
            if "img_feat_right" in results:
                pack_entry["img_feat_right"] = results["img_feat_right"].detach().cpu()
            pack["variant_results"][variant] = pack_entry
            if results.get("logvar") is not None:
                pack["variant_results"][variant]["logvar"] = (
                    results["logvar"].detach().cpu()
                )
            if "p_gt" in results:
                pack["variant_results"][variant]["label"] = (
                    results["p_gt"].detach().cpu()
                )
            if "frames_cam1" in results:
                pack["variant_results"][variant]["frames_cam1"] = (
                    results["frames_cam1"].detach().cpu()
                )
            if "frames_cam2" in results:
                pack["variant_results"][variant]["frames_cam2"] = (
                    results["frames_cam2"].detach().cpu()
                )

        if "meta" in batch:
            pack["meta"] = batch["meta"]

        self.test_outputs.append(pack)

        return torch.tensor(0.0)

    @staticmethod
    def _safe_shape(x: Any) -> str:
        if isinstance(x, torch.Tensor):
            return str(tuple(x.shape))
        return "N/A"

    @staticmethod
    def _safe_mpjpe(pred: Any, label: Any) -> float | None:
        if not isinstance(pred, torch.Tensor) or not isinstance(label, torch.Tensor):
            return None
        if pred.shape != label.shape:
            return None
        if pred.ndim < 2 or pred.shape[-1] != 3:
            return None
        return float(torch.norm(pred - label, dim=-1).mean().item())

    def _save_test_txt_report(
        self, payload: Dict[str, Any], txt_file: Path, fold: str
    ) -> None:
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("Dual2Pose Test Report")
        lines.append("=" * 72)
        lines.append(f"fold: {fold}")
        lines.append(f"num_test_steps: {len(self.test_outputs)}")
        lines.append("")

        variants = payload.get("variants", {}) if isinstance(payload, dict) else {}
        if not isinstance(variants, dict) or len(variants) == 0:
            lines.append("No variant outputs available.")
        else:
            for variant in sorted(variants.keys()):
                data = variants[variant]
                if not isinstance(data, dict):
                    continue

                p_hat = data.get("p_hat")
                label = data.get("label")
                alpha = data.get("alpha")
                p_left = data.get("p_left")
                p_right = data.get("p_right")
                p_left_recon = data.get("p_left_recon")
                p_right_recon = data.get("p_right_recon")

                lines.append(f"[{variant}]")
                lines.append(f"  p_hat_shape: {self._safe_shape(p_hat)}")
                lines.append(f"  label_shape: {self._safe_shape(label)}")

                mpjpe = self._safe_mpjpe(p_hat, label)
                if mpjpe is not None:
                    lines.append(f"  mpjpe: {mpjpe:.6f}")

                if isinstance(alpha, torch.Tensor):
                    lines.append(f"  alpha_mean: {float(alpha.mean().item()):.6f}")
                    lines.append(f"  alpha_std: {float(alpha.std().item()):.6f}")

                if isinstance(p_left, torch.Tensor) and isinstance(
                    p_right, torch.Tensor
                ):
                    cross_view_gap = float(
                        torch.norm(p_left - p_right, dim=-1).mean().item()
                    )
                    lines.append(f"  cross_view_gap: {cross_view_gap:.6f}")

                if isinstance(p_left_recon, torch.Tensor) and isinstance(
                    p_left, torch.Tensor
                ):
                    recon_left = float(
                        torch.nn.functional.l1_loss(p_left_recon, p_left).item()
                    )
                    lines.append(f"  recon_left_l1: {recon_left:.6f}")

                if isinstance(p_right_recon, torch.Tensor) and isinstance(
                    p_right, torch.Tensor
                ):
                    recon_right = float(
                        torch.nn.functional.l1_loss(p_right_recon, p_right).item()
                    )
                    lines.append(f"  recon_right_l1: {recon_right:.6f}")

                lines.append("")

        txt_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def on_test_epoch_end(self) -> None:
        if not hasattr(self, "test_outputs") or len(self.test_outputs) == 0:
            logger.warning("No test outputs to save.")
            return

        fold = (
            getattr(self.logger, "root_dir", "fold").split("/")[-1]
            if self.logger is not None
            else "fold"
        )
        save_dir = self.test_save_dir

        # Collect all outputs by variant
        all_variants = set()
        for output in self.test_outputs:
            all_variants.update(output.get("variant_results", {}).keys())

        payload: Dict[str, Any] = {"variants": {}}

        for variant in sorted(all_variants):
            variant_outputs = []
            for output in self.test_outputs:
                if variant in output.get("variant_results", {}):
                    variant_outputs.append(output["variant_results"][variant])

            if variant_outputs:
                variant_data = {
                    "p_hat": torch.cat([x["p_hat"] for x in variant_outputs], dim=0),
                }

                if all("alpha" in x for x in variant_outputs):
                    variant_data["alpha"] = torch.cat(
                        [x["alpha"] for x in variant_outputs], dim=0
                    )
                if all("p_left" in x for x in variant_outputs):
                    variant_data["p_left"] = torch.cat(
                        [x["p_left"] for x in variant_outputs], dim=0
                    )
                if all("p_right" in x for x in variant_outputs):
                    variant_data["p_right"] = torch.cat(
                        [x["p_right"] for x in variant_outputs], dim=0
                    )
                if all("p_left_recon" in x for x in variant_outputs):
                    variant_data["p_left_recon"] = torch.cat(
                        [x["p_left_recon"] for x in variant_outputs], dim=0
                    )
                if all("p_right_recon" in x for x in variant_outputs):
                    variant_data["p_right_recon"] = torch.cat(
                        [x["p_right_recon"] for x in variant_outputs], dim=0
                    )
                if all("img_feat_left" in x for x in variant_outputs):
                    variant_data["img_feat_left"] = torch.cat(
                        [x["img_feat_left"] for x in variant_outputs], dim=0
                    )
                if all("img_feat_right" in x for x in variant_outputs):
                    variant_data["img_feat_right"] = torch.cat(
                        [x["img_feat_right"] for x in variant_outputs], dim=0
                    )

                if all("label" in x for x in variant_outputs):
                    variant_data["label"] = torch.cat(
                        [x["label"] for x in variant_outputs], dim=0
                    )
                if all("logvar" in x for x in variant_outputs):
                    variant_data["logvar"] = torch.cat(
                        [x["logvar"] for x in variant_outputs], dim=0
                    )
                if all("frames_cam1" in x for x in variant_outputs):
                    variant_data["frames_cam1"] = torch.cat(
                        [x["frames_cam1"] for x in variant_outputs], dim=0
                    )
                if all("frames_cam2" in x for x in variant_outputs):
                    variant_data["frames_cam2"] = torch.cat(
                        [x["frames_cam2"] for x in variant_outputs], dim=0
                    )

                payload["variants"][variant] = variant_data

        # Add metadata if available
        if any("meta" in x for x in self.test_outputs):
            payload["meta"] = [x.get("meta", None) for x in self.test_outputs]

        save_file = save_dir / f"{fold}_pose_outputs.pt"
        torch.save(payload, save_file)

        txt_file = save_dir / f"{fold}_pose_report.txt"
        self._save_test_txt_report(payload=payload, txt_file=txt_file, fold=fold)

        logger.info("Saved pose predictions/labels to %s", save_file)
        logger.info("Saved test report to %s", txt_file)
        logger.info("Variants saved: %s", list(payload["variants"].keys()))

    def configure_optimizers(self):
        """Configure optimizer for the character model."""
        optimizer = torch.optim.AdamW(
            self.models["character"].parameters(),
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
                "monitor": "train/character/loss",
            },
        }
