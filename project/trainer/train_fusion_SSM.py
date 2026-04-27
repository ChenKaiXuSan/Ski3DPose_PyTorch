#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from pytorch_lightning import LightningModule

from project.map_config import ID_TO_INDEX, SKELETON_CONNECTIONS
from project.models import FusionSSM, PoseLossWeights, PoseRefineLoss
from project.models.video_to_pose import SimpleVideo2Pose

logger = logging.getLogger(__name__)


def _build_target_bone_edges() -> List[Tuple[int, int]]:
    """Convert global skeleton ids to contiguous target-joint indices."""
    edges: List[Tuple[int, int]] = []
    for src_id, dst_id in SKELETON_CONNECTIONS:
        if src_id in ID_TO_INDEX and dst_id in ID_TO_INDEX:
            edges.append((ID_TO_INDEX[src_id], ID_TO_INDEX[dst_id]))
    return edges


class FusionSSMTrainer(LightningModule):
    """Pose fusion trainer with separate models for character, pole, and ski variants."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = float(getattr(hparams.loss, "lr", 1e-4))
        self.weight_decay = float(getattr(hparams.loss, "weight_decay", 1e-4))

        model_cfg = getattr(hparams, "model", None)
        d_model = int(getattr(model_cfg, "d_model", 256))
        n_layers = int(getattr(model_cfg, "n_layers", 4))
        use_conf = bool(getattr(model_cfg, "use_conf", True))
        predict_logvar = bool(getattr(model_cfg, "predict_logvar", False))

        # Number of joints for each variant (from data)
        # character: 14 filtered joints, pole: 4 joints, ski: 6 joints
        num_joints_by_variant = {
            "character": len(ID_TO_INDEX),  # 14
            "pole": 4,
            "ski": 6,
        }

        # Create separate models ONLY for character
        # (pole and ski have no SAM predictions, so no fusion model needed)
        self.models = torch.nn.ModuleDict()
        self.models["character"] = FusionSSM(
            num_joints=num_joints_by_variant["character"],
            d_model=d_model,
            n_layers=n_layers,
            use_conf=use_conf,
            predict_logvar=predict_logvar,
        )
        
        # Create Video2Pose models for pole and ski
        self.models["pole"] = SimpleVideo2Pose(
            num_joints=num_joints_by_variant["pole"],
            hidden_dim=max(64, d_model // 4),
        )
        self.models["ski"] = SimpleVideo2Pose(
            num_joints=num_joints_by_variant["ski"],
            hidden_dim=max(64, d_model // 4),
        )
        
        logger.info(
            f"Created models: "
            f"character=FusionSSM({num_joints_by_variant['character']}), "
            f"pole=Video2Pose({num_joints_by_variant['pole']}), "
            f"ski=Video2Pose({num_joints_by_variant['ski']})"
        )

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
            bone_edges=_build_target_bone_edges(),
            weights=weights,
        )
        # pole and ski GT data is stored but not used for model training
        
        self.save_root = str(getattr(hparams, "log_path", "./logs"))
        self.test_outputs: List[Dict[str, Any]] = []
        self.test_save_dir: Path = Path(self.save_root) / "pose_analysis"

    @staticmethod
    def _require_pose(batch: Dict[str, Any], path: Sequence[str]) -> torch.Tensor:
        cur: Any = batch
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                raise KeyError(f"Missing batch key path: {'/'.join(path)}")
            cur = cur[key]

        if not isinstance(cur, torch.Tensor):
            raise TypeError(f"Expected tensor at {'/'.join(path)}, got {type(cur)}")
        if cur.ndim != 4 or cur.shape[-1] != 3:
            raise ValueError(
                f"Expected pose tensor shape (B,T,J,3) at {'/'.join(path)}, got {tuple(cur.shape)}"
            )
        return cur.float()

    @staticmethod
    def _get_variant_data(batch: Dict[str, Any], variant: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Extract SAM, GT data, and frames for a specific variant.
        
        Returns:
            Tuple of (p_left, p_right, p_gt, frames_left, frames_right) where:
                - p_left, p_right: SAM 3D predictions
                - p_gt: ground truth (or None if not available)
                - frames_left, frames_right: video frames (or None if not available)
        """
        # SAM data is only available for character variant
        if variant == "character":
            if "character_cam1" not in batch["kpt3d_sam"] or "character_cam2" not in batch["kpt3d_sam"]:
                raise KeyError(f"SAM data missing for character: {list(batch['kpt3d_sam'].keys())}")
            p_left = batch["kpt3d_sam"]["character_cam1"].float()
            p_right = batch["kpt3d_sam"]["character_cam2"].float()
        else:
            # For pole and ski, no SAM data, use character SAM as fallback
            if "character_cam1" not in batch["kpt3d_sam"] or "character_cam2" not in batch["kpt3d_sam"]:
                raise KeyError(f"Character SAM data missing as fallback for {variant}")
            p_left = batch["kpt3d_sam"]["character_cam1"].float()
            p_right = batch["kpt3d_sam"]["character_cam2"].float()
        
        # Validate shape
        for p in [p_left, p_right]:
            if p.ndim != 4 or p.shape[-1] != 3:
                raise ValueError(
                    f"Expected pose tensor shape (B,T,J,3) for {variant}, got {tuple(p.shape)}"
                )
        
        # GT data
        p_gt = None
        if "kpt3d_gt" in batch and isinstance(batch["kpt3d_gt"], dict):
            if variant in batch["kpt3d_gt"]:
                p_gt = batch["kpt3d_gt"][variant].float()
                if p_gt.ndim != 4 or p_gt.shape[-1] != 3:
                    raise ValueError(
                        f"Expected GT shape (B,T,J,3) for {variant}, got {tuple(p_gt.shape)}"
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
        """Process all variants:
        - character: FusionSSM with SAM predictions (2 views of 3D poses)
        - pole, ski: Video2Pose with frame inputs
        
        Returns:
            Total loss.
        """
        total_loss = 0.0
        variant_results: Dict[str, Any] = {}
        
        # ===== CHARACTER: FusionSSM with SAM =====
        try:
            p_left, p_right, p_gt, frames_left, frames_right = self._get_variant_data(batch, "character")
            
            model = self.models["character"]
            loss_fn = self.loss_fns["character"]
            
            # Forward pass
            out = model(p_left=p_left, p_right=p_right)
            p_hat = out["p_hat"]
            alpha = out["alpha"]
            logvar = out.get("logvar", None)
            
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
            
            loss = loss_dict["loss"]
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
            
            # Store results
            variant_results["character"] = {
                "p_hat": p_hat,
                "alpha": alpha,
                "p_left": p_left,
                "p_right": p_right,
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
        
        # ===== POLE & SKI: Video2Pose with frames =====
        for variant in ["pole", "ski"]:
            try:
                _, _, p_gt, frames_left, frames_right = self._get_variant_data(batch, variant)
                
                if frames_left is None or frames_right is None:
                    logger.warning(f"Skipping {variant}: no frame data")
                    continue
                
                model = self.models[variant]
                
                # Forward pass on frames (average both views or use them separately)
                # For simplicity, use left view frames
                # Convert frames format if needed: (B*T, H, W, 3) -> (B*T, 3, H, W)
                if frames_left.shape[-1] == 3 and frames_left.ndim == 4:
                    frames_input = frames_left.permute(0, 3, 1, 2)  # (B*T, 3, H, W)
                else:
                    frames_input = frames_left
                
                p_hat = model(frames_input)  # (B*T, 1, J, 3)
                
                # Compute loss if GT available
                if p_gt is not None:
                    # Simple L1 loss for now (no bone/temporal losses for pole/ski)
                    loss = torch.nn.functional.l1_loss(p_hat, p_gt)
                    total_loss = total_loss + loss
                    
                    self.log(
                        f"{stage}/{variant}/loss",
                        loss,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        batch_size=p_hat.shape[0],
                    )
                    
                    mpjpe = torch.norm(p_hat - p_gt, dim=-1).mean()
                    self.log(
                        f"{stage}/{variant}/mpjpe",
                        mpjpe,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        batch_size=p_hat.shape[0],
                    )
                
                # Store results
                variant_results[variant] = {
                    "p_hat": p_hat,
                }
                if p_gt is not None:
                    variant_results[variant]["p_gt"] = p_gt
                if frames_left is not None:
                    variant_results[variant]["frames_cam1"] = frames_left
                if frames_right is not None:
                    variant_results[variant]["frames_cam2"] = frames_right
                    
            except KeyError as e:
                logger.warning(f"Skipping {variant}: {e}")
        
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
        logger.info("FusionSSM test start")

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], _batch_idx: int) -> torch.Tensor:
        """Run inference on all variants and collect outputs."""
        self._shared_step(batch, stage="test")
        
        variant_results = batch.get("_variant_results", {})
        
        pack: Dict[str, Any] = {
            "variant_results": {},
        }
        
        for variant, results in variant_results.items():
            pack["variant_results"][variant] = {
                "p_hat": results["p_hat"].detach().cpu(),
                "alpha": results["alpha"].detach().cpu(),
                "p_left": results["p_left"].detach().cpu(),
                "p_right": results["p_right"].detach().cpu(),
            }
            if results.get("logvar") is not None:
                pack["variant_results"][variant]["logvar"] = results["logvar"].detach().cpu()
            if "p_gt" in results:
                pack["variant_results"][variant]["label"] = results["p_gt"].detach().cpu()
            if "frames_cam1" in results:
                pack["variant_results"][variant]["frames_cam1"] = results["frames_cam1"].detach().cpu()
            if "frames_cam2" in results:
                pack["variant_results"][variant]["frames_cam2"] = results["frames_cam2"].detach().cpu()
        
        if "meta" in batch:
            pack["meta"] = batch["meta"]
        
        self.test_outputs.append(pack)
        
        # Return first variant's loss for summary
        if variant_results:
            first_variant = list(variant_results.keys())[0]
            return torch.tensor(0.0)
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
        
        payload: Dict[str, Any] = {
            "variants": {}
        }
        
        for variant in sorted(all_variants):
            variant_outputs = []
            for output in self.test_outputs:
                if variant in output.get("variant_results", {}):
                    variant_outputs.append(output["variant_results"][variant])
            
            if variant_outputs:
                variant_data = {
                    "p_hat": torch.cat([x["p_hat"] for x in variant_outputs], dim=0),
                    "alpha": torch.cat([x["alpha"] for x in variant_outputs], dim=0),
                    "p_left": torch.cat([x["p_left"] for x in variant_outputs], dim=0),
                    "p_right": torch.cat([x["p_right"] for x in variant_outputs], dim=0),
                }
                
                if all("label" in x for x in variant_outputs):
                    variant_data["label"] = torch.cat([x["label"] for x in variant_outputs], dim=0)
                if all("logvar" in x for x in variant_outputs):
                    variant_data["logvar"] = torch.cat([x["logvar"] for x in variant_outputs], dim=0)
                if all("frames_cam1" in x for x in variant_outputs):
                    variant_data["frames_cam1"] = torch.cat([x["frames_cam1"] for x in variant_outputs], dim=0)
                if all("frames_cam2" in x for x in variant_outputs):
                    variant_data["frames_cam2"] = torch.cat([x["frames_cam2"] for x in variant_outputs], dim=0)
                
                payload["variants"][variant] = variant_data
        
        # Add metadata if available
        if any("meta" in x for x in self.test_outputs):
            payload["meta"] = [x.get("meta", None) for x in self.test_outputs]

        save_file = save_dir / f"{fold}_pose_outputs.pt"
        torch.save(payload, save_file)
        logger.info("Saved pose predictions/labels to %s", save_file)
        logger.info("Variants saved: %s", list(payload["variants"].keys()))

    def configure_optimizers(self):
        """Configure optimizer for all models (character, pole, ski)."""
        # Collect parameters from all models
        all_params = []
        all_params.extend(self.models["character"].parameters())
        all_params.extend(self.models["pole"].parameters())
        all_params.extend(self.models["ski"].parameters())
        
        optimizer = torch.optim.AdamW(
            all_params,
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
