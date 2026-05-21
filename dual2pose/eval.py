#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/main.py
Project: /workspace/code/project
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

import logging
import os

import hydra

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from dataloader.data_loader import UnityDataModule, SkiPosePTZDataModule

#####################################
# select different experiment trainer
#####################################
from trainer.train_dual2pose import Dual2PoseTrainer
from trainer.train_crossview_fusion import CrossViewFusionTrainer

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="dual2pose.yaml",
)
def init_params(config):

    config.train.gpu = 1

    seed_everything(42, workers=True)

    if config.model.backbone == "dual2pose":
        classification_module = Dual2PoseTrainer(config)

    elif config.model.backbone == "crossview_fusion":
        classification_module = CrossViewFusionTrainer(config)

    # * prepare data module
    # ski_pose_ptz_data_module = SkiPosePTZDataModule(hparams)
    unity_data_module = UnityDataModule(config)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(config.log_path, "tb_logs"), name="test"
    )

    # 初始化 CSVLogger
    cvs_logger = CSVLogger(
        save_dir=os.path.join(config.log_path, "csv_logs"),
        name="test",
    )

    # some callbacks
    # progress_bar = TQDMProgressBar(refresh_rate=10)
    progress_bar = RichProgressBar(refresh_rate=10, leave=True)

    trainer = Trainer(
        devices=[
            int(config.train.gpu),
        ],
        accelerator="gpu",
        max_epochs=config.train.max_epochs,
        logger=[tb_logger, cvs_logger],
        callbacks=[
            progress_bar,
        ],
    )

    # save the metrics to file
    metrics = trainer.test(
        classification_module,
        unity_data_module,
        ckpt_path="/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt",
    )

    logger.info(f"Test metrics: {metrics}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
