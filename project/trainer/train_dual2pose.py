#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule

from project.map_config import ID_TO_INDEX, TARGET_SKELETON_CONNECTIONS_INDEX
from project.models import Dual2PoseNet, PoseLossWeights, PoseRefineLoss

logger = logging.getLogger(__name__)


class Dual2PoseTrainer(LightningModule):
    """Pose fusion trainer for character variant using Dual2PoseNet."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(getattr(hparams.loss, "lr", 1e-4))
        self.weight_decay = float(getattr(hparams.loss, "weight_decay", 1e-4))
        self.lambda_view_recon = float(getattr(hparams.loss, "lambda_view_recon", 0.05))

        model_cfg = getattr(hparams, "model", None)
        d_model = int(getattr(model_cfg, "d_model", 256))
        n_layers = int(getattr(model_cfg, "n_layers", 4))
        # Default to confidence-free reliability modeling.
        use_conf = bool(getattr(model_cfg, "use_conf", False))
        predict_logvar = bool(getattr(model_cfg, "predict_logvar", False))

        num_joints_character = len(ID_TO_INDEX)  # 15

        self.models = torch.nn.ModuleDict()
        self.models["character"] = Dual2PoseNet(
            num_joints=num_joints_character,
            d_model=d_model,
            n_layers=n_layers,
            use_conf=use_conf,
            predict_logvar=predict_logvar,
            bone_edges=TARGET_SKELETON_CONNECTIONS_INDEX,
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
            bone_edges=TARGET_SKELETON_CONNECTIONS_INDEX,
            weights=weights,
        )

        self.save_root = str(getattr(hparams, "log_path", "./logs"))
        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir: Path = Path(self.save_root) / "pose_analysis"

        logger.info(
            "Dual2PoseTrainer config: use_conf=%s, predict_logvar=%s",
            use_conf,
            predict_logvar,
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
            p_left, p_right, p_gt, frames_left, frames_right = self._get_character_data(batch)

            model = self.models["character"]
            loss_fn = self.loss_fns["character"]

            # Forward pass
            out = model(p_left=p_left, p_right=p_right)
            p_hat = out["p_hat"]
            p0 = out["p0"]
            alpha = out["alpha"]
            logvar = out.get("logvar", None)

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

            loss = loss_dict["loss"] + self.lambda_view_recon * loss_view_recon
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
                f"{stage}/character/p0_drift",
                torch.nn.functional.l1_loss(p_hat, p0),
                on_step=True,
                on_epoch=True,
                batch_size=p_hat.shape[0],
            )

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
        logger.info("Saved pose predictions/labels to %s", save_file)
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
