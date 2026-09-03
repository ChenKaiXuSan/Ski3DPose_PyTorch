#!/usr/bin/env python3
"""Fine-tune CanonFuse3D on one or a balanced mixture of pose front ends."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DUAL2POSE_ROOT = REPO_ROOT / "dual2pose"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DUAL2POSE_ROOT) not in sys.path:
    sys.path.insert(0, str(DUAL2POSE_ROOT))

from dataloader.data_loader import UnityDataModule
from trainer.train_crossview_fusion import CrossViewFusionTrainer

from dual2pose.dataloader.frontend_pose_data import FrontEndDataModule
from dual2pose.frontend_adaptation import (
    common13_mpjpe,
    configure_trainable_scope,
    load_model_weights_only,
)
from dual2pose.training_protocol import resolve_fold_index_path, validate_fold_metadata


class FrontendAdaptationTrainer(CrossViewFusionTrainer):
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = super().validation_step(batch, batch_idx)
        results = batch.get("_variant_results", {})
        fused = results.get("fused")
        target = results.get("ground_truth_canonical")
        if isinstance(fused, torch.Tensor) and isinstance(target, torch.Tensor):
            metric = common13_mpjpe(fused, target)
            self.log(
                "val/common13_mpjpe",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=fused.shape[0],
            )
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        parameters = [parameter for parameter in self.models.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("Adaptation has no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        steps = getattr(self.trainer, "estimated_stepping_batches", None)
        t_max = int(steps) if isinstance(steps, int) and steps > 0 else 1000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def _manifest_path(config: DictConfig, frontend: str, split: str) -> Path:
    raw = config.adaptation.manifests[frontend][split]
    if raw is None or str(raw).strip() in {"", "null", "None"}:
        raise ValueError(f"Missing adaptation manifest for {frontend}/{split}")
    path = Path(str(raw)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Adaptation manifest does not exist: {path}")
    return path


@hydra.main(version_base=None, config_path="../configs", config_name="frontend_adaptation.yaml")
def main(config: DictConfig) -> None:
    seed_everything(int(config.train.seed), workers=True)
    data_root = Path(str(config.data.unity.root_path))
    fold_path = resolve_fold_index_path(data_root, int(config.train.fold))
    validate_fold_metadata(fold_path, int(config.train.fold))
    config.data.unity.index_mapping_path = str(fold_path)
    config.loss.lr = float(config.adaptation.learning_rate)
    output_root = Path(str(config.log_path)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model = FrontendAdaptationTrainer(config)
    load_report = load_model_weights_only(
        model, Path(str(config.adaptation.source_checkpoint))
    )
    scope_report = configure_trainable_scope(model, str(config.adaptation.scope))
    frontend = str(config.adaptation.frontend).lower()
    base_dm = UnityDataModule(config)
    if frontend == "mixed":
        train_sources = [None] + [
            _manifest_path(config, name, "train")
            for name in ("videopose3d", "poseformer", "motionbert")
        ]
        val_sources = [None] + [
            _manifest_path(config, name, "val")
            for name in ("videopose3d", "poseformer", "motionbert")
        ]
        datamodule = FrontEndDataModule(
            base_dm,
            train_manifest=None,
            val_manifest=None,
            test_manifest=None,
            mixed_train_sources=train_sources,
            mixed_val_sources=val_sources,
        )
    elif frontend in {"videopose3d", "poseformer", "motionbert"}:
        datamodule = FrontEndDataModule(
            base_dm,
            train_manifest=_manifest_path(config, frontend, "train"),
            val_manifest=_manifest_path(config, frontend, "val"),
            test_manifest=None,
        )
    else:
        raise ValueError(f"Unsupported adaptation front end: {frontend}")

    checkpoint = ModelCheckpoint(
        dirpath=output_root / "checkpoints",
        filename="{epoch:02d}-{val/common13_mpjpe:.6f}",
        monitor="val/common13_mpjpe",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=[int(config.train.gpu)],
        max_epochs=int(config.adaptation.epochs),
        logger=[
            TensorBoardLogger(save_dir=output_root / "tb_logs", name="adaptation"),
            CSVLogger(save_dir=output_root / "csv_logs", name="adaptation"),
        ],
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
    )
    (output_root / "adaptation_init.json").write_text(
        json.dumps(
            {
                "frontend": frontend,
                "scope": str(config.adaptation.scope),
                "seed": int(config.train.seed),
                "fold": int(config.train.fold),
                "epochs": int(config.adaptation.epochs),
                "learning_rate": float(config.adaptation.learning_rate),
                "weights_only_load": asdict(load_report),
                "trainable_scope": scope_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    trainer.fit(model, datamodule=datamodule)
    (output_root / "adaptation_complete.json").write_text(
        json.dumps(
            {
                "best_model_path": checkpoint.best_model_path,
                "best_model_score": (
                    float(checkpoint.best_model_score.item())
                    if checkpoint.best_model_score is not None
                    else None
                ),
                "last_model_path": checkpoint.last_model_path,
                "trainer_epoch": int(trainer.current_epoch),
                "trainer_global_step": int(trainer.global_step),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
