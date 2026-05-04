#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule

from project.map_config import FILTER_SKELETON_CONNECTIONS
from project.models.dual2pose_net import Dual2PoseNet, PoseLossWeights, PoseRefineLoss

logger = logging.getLogger(__name__)


class Dual2PoseTrainer(LightningModule):
    """Pose fusion trainer for character variant using Dual2PoseNet."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_cfg = getattr(hparams, "dual2pose", None)
        self.lr = float(getattr(hparams.loss, "lr", 0.1))
        self.weight_decay = float(
            getattr(model_cfg, "weight_decay", getattr(hparams.loss, "weight_decay", 0.01))
        )
        self.lambda_view_recon = float(
            getattr(model_cfg, "lambda_view_recon", getattr(hparams.loss, "lambda_view_recon", 0.05))
        )
        self.lambda_alpha_balance = float(getattr(model_cfg, "lambda_alpha_balance", 0.02))
        self.lambda_alpha_entropy = float(getattr(model_cfg, "lambda_alpha_entropy", 0.005))
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

        d_model = int(getattr(model_cfg, "d_model", 256))
        n_layers = int(getattr(model_cfg, "n_layers", 4))
        # Default to confidence-free reliability modeling.
        use_conf = bool(getattr(model_cfg, "use_conf", False))
        predict_logvar = bool(getattr(model_cfg, "predict_logvar", False))
        self.use_dino_features = bool(getattr(model_cfg, "use_dino_features", False))
        self.dino_model_name = str(
            getattr(
                model_cfg,
                "dino_model_name",
                "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            )
        )
        self.dino_freeze = bool(getattr(model_cfg, "dino_freeze", True))
        self.dino_image_size = int(getattr(model_cfg, "dino_image_size", 224))
        self.dino_feature_dim = int(getattr(model_cfg, "dino_feature_dim", 768))

        num_joints_character = 15  # After filtering with FILTER_SKELETON_CONNECTIONS

        self.models = torch.nn.ModuleDict()
        self.models["character"] = Dual2PoseNet(
            num_joints=num_joints_character,
            d_model=d_model,
            n_layers=n_layers,
            use_conf=use_conf,
            predict_logvar=predict_logvar,
            bone_edges=FILTER_SKELETON_CONNECTIONS,
            use_dino_features=self.use_dino_features,
            dino_model_name=self.dino_model_name,
            dino_freeze=self.dino_freeze,
            dino_image_size=self.dino_image_size,
            dino_feature_dim=self.dino_feature_dim,
        )

        logger.info(f"Created model: character=Dual2PoseNet({num_joints_character})")

        weights = PoseLossWeights(
            mpjpe=float(getattr(hparams.loss, "lambda_mpjpe", 1.0)),
            bone=float(getattr(hparams.loss, "lambda_bone", 0.2)),
            vel=float(getattr(hparams.loss, "lambda_vel", 0.05)),
            acc=float(getattr(hparams.loss, "lambda_acc", 0.02)),
            agree=float(getattr(hparams.loss, "lambda_agree", 0.1)),
            bone_stab=float(getattr(hparams.loss, "lambda_bone_stab", 0.05)),
        )

        # Create loss function only for character (has skeleton)
        self.loss_fns = torch.nn.ModuleDict()
        self.loss_fns["character"] = PoseRefineLoss(
            bone_edges=FILTER_SKELETON_CONNECTIONS,
            weights=weights,
        )

        self.save_root = str(getattr(hparams, "log_path", "./logs"))
        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir: Path = Path(self.save_root) / "pose_analysis"

        logger.info(
            "Dual2PoseTrainer config: use_conf=%s, predict_logvar=%s, use_dino_features=%s, "
            "lambda_view_recon=%.4f, lambda_alpha_balance=%.4f, lambda_alpha_entropy=%.4f, lambda_p0_supervise=%.4f, "
            "rigid_align_right_to_left=%s",
            use_conf,
            predict_logvar,
            self.use_dino_features,
            self.lambda_view_recon,
            self.lambda_alpha_balance,
            self.lambda_alpha_entropy,
            self.lambda_p0_supervise,
            self.rigid_align_right_to_left,
        )

    @staticmethod
    def _rigid_align_right_pose_to_left_batch(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Align right pose to left pose for each sample in a batch using Kabsch.

        Args:
            left: (B,T,J,3)
            right: (B,T,J,3)
        Returns:
            (right_aligned, stats)
        """
        if left.shape != right.shape:
            raise ValueError(
                f"Rigid alignment expects same shape, got {tuple(left.shape)} vs {tuple(right.shape)}"
            )
        if left.ndim != 4 or left.shape[-1] != 3:
            raise ValueError(
                f"Rigid alignment expects (B,T,J,3), got {tuple(left.shape)}"
            )

        bsz = int(left.shape[0])
        aligned = right.clone()
        rmse_before_vals: List[float] = []
        rmse_after_vals: List[float] = []
        valid_points_vals: List[float] = []
        applied = 0

        for b in range(bsz):
            x_full = right[b].reshape(-1, 3)
            y_full = left[b].reshape(-1, 3)
            valid = torch.isfinite(x_full).all(dim=-1) & torch.isfinite(y_full).all(dim=-1)
            n_valid = int(valid.sum().item())
            valid_points_vals.append(float(n_valid))
            if n_valid < 3:
                continue

            x = x_full[valid]
            y = y_full[valid]
            x_mean = x.mean(dim=0)
            y_mean = y.mean(dim=0)
            x0 = x - x_mean
            y0 = y - y_mean

            h = x0.transpose(0, 1) @ y0
            u, _, vh = torch.linalg.svd(h)
            r = vh.transpose(0, 1) @ u.transpose(0, 1)

            # Enforce proper rotation (det(R)=+1) to avoid reflection.
            if torch.det(r) < 0:
                vh = vh.clone()
                vh[-1, :] *= -1
                r = vh.transpose(0, 1) @ u.transpose(0, 1)

            t = y_mean - x_mean @ r.transpose(0, 1)

            aligned_b = right[b].reshape(-1, 3) @ r.transpose(0, 1) + t
            aligned[b] = aligned_b.reshape_as(right[b])

            diff_before = x_full[valid] - y_full[valid]
            diff_after = aligned[b].reshape(-1, 3)[valid] - y_full[valid]
            rmse_before_vals.append(
                float(torch.sqrt((diff_before.pow(2).sum(dim=-1)).mean()).item())
            )
            rmse_after_vals.append(
                float(torch.sqrt((diff_after.pow(2).sum(dim=-1)).mean()).item())
            )
            applied += 1

        def _safe_mean(vals: List[float]) -> float:
            return float(sum(vals) / len(vals)) if vals else float("nan")

        return aligned, {
            "enabled": 1.0,
            "applied": float(applied),
            "applied_ratio": float(applied / max(1, bsz)),
            "valid_points": _safe_mean(valid_points_vals),
            "rmse_before": _safe_mean(rmse_before_vals),
            "rmse_after": _safe_mean(rmse_after_vals),
        }

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
        if (
            "kpt3d_sam" not in batch
            or "character_cam1" not in batch["kpt3d_sam"]
            or "character_cam2" not in batch["kpt3d_sam"]
        ):
            kpt3d_sam = batch.get("kpt3d_sam", {})
            available = list(kpt3d_sam.keys()) if isinstance(kpt3d_sam, dict) else []
            raise KeyError(f"SAM data missing for character: {available}")
        p_left = batch["kpt3d_sam"]["character_cam1"].float()
        p_right = batch["kpt3d_sam"]["character_cam2"].float()

        # Validate shape
        for p in [p_left, p_right]:
            if p.ndim != 4 or p.shape[-1] != 3:
                raise ValueError(
                    f"Expected pose tensor shape (B,T,J,3) for character, got {tuple(p.shape)}"
                )

        # GT data
        p_gt = None
        if "kpt3d_gt" in batch and isinstance(batch["kpt3d_gt"], dict):
            if "character" in batch["kpt3d_gt"]:
                p_gt = batch["kpt3d_gt"]["character"].float()
                if p_gt.ndim != 4 or p_gt.shape[-1] != 3:
                    raise ValueError(
                        f"Expected GT shape (B,T,J,3) for character, got {tuple(p_gt.shape)}"
                    )

        # Frame data (video frames)
        frames_left = None
        frames_right = None
        if "frames" in batch and isinstance(batch["frames"], dict):
            if "cam1" in batch["frames"]:
                frames_left = batch["frames"]["cam1"].float()
            if "cam2" in batch["frames"]:
                frames_right = batch["frames"]["cam2"].float()

        return p_left, p_right, p_gt, frames_left, frames_right

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        """Process character variant with dual-view SAM 3D poses.

        Returns:
            Total loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        variant_results: Dict[str, Any] = {}

        # ===== CHARACTER: Dual2PoseNet with SAM =====
        try:
            p_left, p_right, p_gt, frames_left, frames_right = self._get_character_data(
                batch
            )
            rigid_stats: Dict[str, float] = {
                "enabled": 1.0 if self.rigid_align_right_to_left else 0.0,
                "applied": 0.0,
                "applied_ratio": 0.0,
                "valid_points": 0.0,
                "rmse_before": float("nan"),
                "rmse_after": float("nan"),
            }
            if self.rigid_align_right_to_left:
                p_right, rigid_stats = self._rigid_align_right_pose_to_left_batch(
                    left=p_left,
                    right=p_right,
                )

            model = self.models["character"]
            loss_fn = self.loss_fns["character"]

            img_feat_left = None
            img_feat_right = None
            if self.use_dino_features:
                if frames_left is None or frames_right is None:
                    raise ValueError(
                        "use_dino_features=True requires frames.cam1 and frames.cam2 in batch"
                    )

            # Forward pass
            out = model(
                img_l=frames_left,
                img_r=frames_right,
                p_left=p_left,
                p_right=p_right,
            )
            p_hat = out["p_hat"]
            p0 = out["p0"]
            alpha = out["alpha"]
            logvar = out.get("logvar", None)
            img_feat_left = out.get("img_feat_left", None)
            img_feat_right = out.get("img_feat_right", None)

            # Reconstruct left/right poses from fused pose + reliability weight.
            # With delta = (p_left - p_right):
            #   p_hat = alpha * p_left_recon + (1-alpha) * p_right_recon
            #   p_left_recon - p_right_recon = delta
            delta_lr = p_left - p_right
            p_left_recon = p_hat + (1.0 - alpha) * delta_lr
            p_right_recon = p_hat - alpha * delta_lr

            loss_recon_left = torch.nn.functional.l1_loss(p_left_recon, p_left)
            loss_recon_right = torch.nn.functional.l1_loss(p_right_recon, p_right)
            loss_view_recon = 0.5 * (loss_recon_left + loss_recon_right)

            # Reliability diagnostics: cross-view discrepancy and temporal stability.
            cv_gap = torch.norm(p_left - p_right, dim=-1).mean()
            vel_norm = self._temporal_velocity_norm(p_hat)
            acc_norm = self._temporal_acceleration_norm(p_hat)

            # Compute loss
            if p_gt is not None:
                loss_dict = loss_fn(p_hat=p_hat, p_gt=p_gt, logvar=logvar)
                mpjpe = torch.norm(p_hat - p_gt, dim=-1).mean()
                loss_p0_supervise = torch.nn.functional.l1_loss(p0, p_gt)
                self.log(
                    f"{stage}/character/mpjpe",
                    mpjpe,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=p_hat.shape[0],
                )
            else:
                loss_dict = loss_fn(
                    p_hat=p_hat,
                    p_left=p_left,
                    p_right=p_right,
                    alpha=alpha,
                )
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
                f"{stage}/character/loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=p_hat.shape[0],
            )
            for k, v in loss_dict.items():
                if k != "loss":
                    self.log(
                        f"{stage}/character/{k}",
                        v,
                        on_step=True,
                        on_epoch=True,
                        batch_size=p_hat.shape[0],
                    )

            # Log alpha statistics
            self.log(
                f"{stage}/character/alpha_mean",
                alpha.mean(),
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/alpha_std",
                alpha.std(),
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/cross_view_gap",
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
                        f"{stage}/character/rigid_rmse_before",
                        p_hat.new_tensor(rmse_before),
                        on_step=True,
                        on_epoch=True,
                        batch_size=p_hat.shape[0],
                    )
                if rmse_after == rmse_after:
                    self.log(
                        f"{stage}/character/rigid_rmse_after",
                        p_hat.new_tensor(rmse_after),
                        on_step=True,
                        on_epoch=True,
                        batch_size=p_hat.shape[0],
                    )
                self.log(
                    f"{stage}/character/rigid_applied_ratio",
                    p_hat.new_tensor(rigid_stats["applied_ratio"]),
                    on_step=True,
                    on_epoch=True,
                    batch_size=p_hat.shape[0],
                )
            self.log(
                f"{stage}/character/vel_norm",
                vel_norm,
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/acc_norm",
                acc_norm,
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/recon_left",
                loss_recon_left,
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/recon_right",
                loss_recon_right,
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/recon_view",
                loss_view_recon,
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/loss_alpha_balance",
                loss_alpha_balance,
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/alpha_entropy",
                alpha_entropy,
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/loss_p0_supervise",
                loss_p0_supervise,
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            self.log(
                f"{stage}/character/p0_drift",
                torch.nn.functional.l1_loss(p_hat, p0),
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )
            if (
                self.use_dino_features
                and img_feat_left is not None
                and img_feat_right is not None
            ):
                self.log(
                    f"{stage}/character/dino_feat_gap",
                    torch.norm(img_feat_left - img_feat_right, dim=-1).mean(),
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
            if (
                self.use_dino_features
                and img_feat_left is not None
                and img_feat_right is not None
            ):
                console_metrics["dino_feat_gap"] = torch.norm(
                    img_feat_left - img_feat_right, dim=-1
                ).mean()

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

        except KeyError as e:
            logger.warning(f"Skipping character: {e}")

        # Store variant results in batch for later use
        batch["_variant_results"] = variant_results

        return total_loss

    def training_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def on_test_start(self) -> None:
        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dual2PoseNet test start")

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        """Run inference on all variants and collect outputs."""
        self._shared_step(batch, stage="test")

        variant_results = batch.get("_variant_results", {})

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
