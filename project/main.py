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

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from project.dataloader.data_loader import UnityDataModule
from project.map_config import UnityDataConfig

#####################################
# select different experiment trainer
#####################################
# baseline
from project.trainer.train_dual2pose import Dual2PoseTrainer
from project.trainer.train_pose2equip import Pose2EquipTrainer

logger = logging.getLogger(__name__)


def load_fold_dataset_idx_from_fold_json(
    config: DictConfig, fold: int
) -> Dict[str, List[UnityDataConfig]]:
    """加载指定fold的JSON文件

    Args:
        config: Hydra配置对象
        fold: fold号 (0-4 for 5-fold, etc.)

    Returns:
        Dict[str, List[UnityDataConfig]]: {"train": [...], "val": [...], "test": [...]}
    """

    index_file_path = Path(str(config.data.index_mapping_path))

    fold_file = index_file_path / f"fold_{fold:02d}.json"

    with open(fold_file, "r", encoding="utf-8") as f:
        fold_data = json.load(f)

    fold_data.pop("_metadata", None)

    dataset_idx: Dict[str, List[UnityDataConfig]] = {"train": [], "val": [], "test": []}

    # 处理三种split
    for split in ["train", "val", "test"]:
        src_list = fold_data.get(split, [])
        for item in src_list:
            dataset_idx[split].append(UnityDataConfig.from_dict(item))

    logger.info(
        f"✓ Loaded fold {fold}: train={len(dataset_idx['train'])}, val={len(dataset_idx['val'])}, test={len(dataset_idx['test'])}"
    )
    return dataset_idx


def train(hparams: DictConfig, dataset_idx, fold: int):
    """the train process for the one fold.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (int): the dataset index for the one fold.
        fold (int): the fold index.

    Returns:
        list: best trained model, data loader
    """

    seed_everything(42, workers=True)

    # * select experiment
    monitor_metric = "val/video_acc"
    monitor_mode = "max"
    ckpt_filename = "{epoch}-{val/loss:.2f}-{val/video_acc:.4f}"

    if hparams.train.view == "dual":
        if hparams.model.backbone == "dual2pose":
            classification_module = Dual2PoseTrainer(hparams)
            monitor_metric = "val/loss"
            monitor_mode = "min"
            ckpt_filename = "{epoch}-{val/loss:.4f}"
        else:
            raise ValueError("the experiment backbone is not supported.")
    elif hparams.train.view == "single":
        if hparams.model.backbone == "pose2equip":
            classification_module = Pose2EquipTrainer(hparams)
            # pose2equip 当前验证阶段记录的是 val/mpjpe 与 val/loss。
            monitor_metric = "val/mpjpe"
            monitor_mode = "min"
            ckpt_filename = "{epoch}-{val/loss:.4f}-{val/mpjpe:.4f}"
    else:
        raise ValueError("the experiment view is not supported.")

    # * prepare data module
    data_module = UnityDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.log_path, "tb_logs"),
        name="fold_" + str(fold),  # here should be str type.
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=10)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        dirpath=os.path.join(hparams.log_path, "checkpoints", "fold_" + str(fold)),
        filename=ckpt_filename,
        auto_insert_metric_name=False,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=5,
        mode=monitor_mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=[
            int(hparams.train.gpu),
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        logger=[tb_logger],
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            # early_stopping,
            lr_monitor,
        ],
        # limit_train_batches=10,
        # limit_val_batches=10,
        # limit_test_batches=10,
    )

    trainer.fit(classification_module, data_module)

    # save the metrics to file
    trainer.test(
        classification_module,
        data_module,
        ckpt_path="best",
    )


@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="train.yaml",
)
def init_params(config):

    # Load precomputed fold mapping only; do not prepare CV splits here.
    # 使用预生成的单fold JSON文件（每个fold文件必须存在）

    requested_fold = int(config.train.fold)

    # 检测可用的fold数量
    available_folds = _detect_available_folds(config)

    # train.fold >= 0: run only the specified fold (recommended for multi-node jobs)
    # train.fold < 0: run all folds sequentially (backward compatible mode)
    if requested_fold >= 0:
        if requested_fold not in available_folds:
            raise KeyError(
                f"Requested fold {requested_fold} is not available. "
                f"Available folds: {available_folds}"
            )
        target_folds = [requested_fold]
    else:
        target_folds = available_folds

    logger.info("#" * 50)
    logger.info(
        "Start training folds: %s (requested train.fold=%s)",
        target_folds,
        requested_fold,
    )
    logger.info("#" * 50)

    #########
    # K fold
    #########
    # * for one fold, we first train/val model, then save the best ckpt preds/label into .pt file.

    for fold in target_folds:
        # 加载单个fold的JSON文件
        dataset_value = load_fold_dataset_idx_from_fold_json(config, fold)
        logger.info("#" * 50)
        logger.info(f"Start train fold: {fold}")
        logger.info("#" * 50)

        train(config, dataset_value, fold)

        logger.info("#" * 50)
        logger.info(f"finish train fold: {fold}")
        logger.info("#" * 50)

    logger.info("#" * 50)
    logger.info("finish train folds: %s", target_folds)
    logger.info("#" * 50)


def _detect_available_folds(config: DictConfig) -> List[int]:
    """检测可用的fold文件数量"""
    index_file_path = Path(str(config.data.index_mapping_path))

    # 查找所有fold_XX.json文件
    fold_files = sorted(index_file_path.glob("fold_*.json"))

    available_folds = []
    for fold_file in fold_files:
        # 从fold_00.json提取00并转为int
        match = fold_file.stem.replace("fold_", "")
        fold_num = int(match)
        available_folds.append(fold_num)

    return sorted(available_folds)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
