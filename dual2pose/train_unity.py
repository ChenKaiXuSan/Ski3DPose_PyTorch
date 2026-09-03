#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/dual2pose/main.py
Project: /workspace/code/dual2pose
Created Date: Tuesday April 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday May 1st 2025 8:34:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import json
import logging
import os
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.data_loader import UnityDataModule, SkiPosePTZDataModule

#####################################
# select different experiment trainer
#####################################
from trainer.train_dual2pose import Dual2PoseTrainer
from trainer.train_crossview_fusion import CrossViewFusionTrainer
from training_protocol import resolve_fold_index_path, validate_fold_metadata
from dual2pose.eval.multiseed_metrics import summarize_test_outputs_by_action

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="dual2pose.yaml",
)
def init_params(config: DictConfig):

    fold_index_path = resolve_fold_index_path(
        Path(str(config.data.unity.root_path)), int(config.train.fold)
    )
    validate_fold_metadata(fold_index_path, int(config.train.fold))
    config.data.unity.index_mapping_path = str(fold_index_path)
    seed_everything(int(config.train.seed), workers=True)

    # * select experiment
    monitor_metric = "val/video_acc"
    monitor_mode = "max"
    ckpt_filename = "{epoch}-{val/loss:.2f}-{val/video_acc:.4f}"

    if config.model.backbone == "dual2pose":
        classification_module = Dual2PoseTrainer(config)
        # dual2pose 当前验证阶段记录的是 val/character/mpjpe 与 val/loss。
        monitor_metric = "val/character/mpjpe"
        monitor_mode = "min"
        ckpt_filename = "{epoch}-{val/loss:.4f}-{val/character/mpjpe:.4f}"
    elif config.model.backbone == "crossview_fusion":
        classification_module = CrossViewFusionTrainer(config)
        # crossview_fusion 当前验证阶段记录的是 val/mpjpe 与 val/loss。
        monitor_metric = "val/mpjpe"
        monitor_mode = "min"
        ckpt_filename = "{epoch}-{val/loss:.4f}-{val/mpjpe:.4f}"

    # * prepare data module
    # ski_pose_ptz_data_module = SkiPosePTZDataModule(hparams)
    unity_data_module = UnityDataModule(config)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(config.log_path, "tb_logs"),
        name="train",
    )

    # 初始化 CSVLogger
    cvs_logger = CSVLogger(
        save_dir=os.path.join(config.log_path, "csv_logs"),
        name="train",
        flush_logs_every_n_steps=100,
    )

    # some callbacks
    progress_bar = RichProgressBar(refresh_rate=10, leave=True)
    rich_model_summary = RichModelSummary(max_depth=3)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        dirpath=os.path.join(
            config.log_path,
            "checkpoints",
        ),
        filename=ckpt_filename,
        auto_insert_metric_name=False,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
        save_top_k=2,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=[
            int(config.train.gpu),
        ],
        accelerator="gpu",
        max_epochs=config.train.max_epochs,
        logger=[tb_logger, cvs_logger],
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            lr_monitor,
        ],
        # limit_train_batches=1,
        # limit_val_batches=1,
        # limit_test_batches=10,
    )

    trainer.fit(classification_module, unity_data_module)

    # save the metrics to file
    test_metrics = trainer.test(
        classification_module,
        unity_data_module,
        ckpt_path=str(config.train.test_ckpt_path),
    )
    if config.model.backbone == "crossview_fusion":
        action_metrics = summarize_test_outputs_by_action(
            list(getattr(classification_module, "test_outputs", []))
        )
        result_path = Path(str(config.log_path)) / "test_metrics_by_action.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "fold": int(config.train.fold),
                    "seed": int(config.train.seed),
                    "index_mapping_path": str(config.data.unity.index_mapping_path),
                    "test_checkpoint_policy": str(config.train.test_ckpt_path),
                    "best_model_path": model_check_point.best_model_path,
                    "best_model_score": (
                        float(model_check_point.best_model_score.item())
                        if model_check_point.best_model_score is not None
                        else None
                    ),
                    "last_model_path": model_check_point.last_model_path,
                    "trainer_test_metrics": test_metrics,
                    **action_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
