#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/trainer/train_3dcnn.py
Project: /workspace/code/project/trainer
Created Date: Monday November 10th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday November 10th 2025 7:22:14 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from typing import Any, List, Optional, Union
import logging

import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)


from project.models.res_3dcnn import Res3DCNN

from project.utils.helper import save_helper
from project.utils.save_CAM import dump_all_feature_maps

logger = logging.getLogger(__name__)


class Res3DCNNTrainer(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = getattr(hparams.loss, "lr", 1e-3)  # default lr

        self.num_classes = hparams.model.model_class_num

        # define model
        self.model = Res3DCNN(hparams)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

        self.save_root = getattr(hparams.train, "log_path", "./logs")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        # prepare the input and label
        video = batch["video"].detach()  # b, c, t, h, w
        attn_map = batch["attn_map"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float()  # b

        b, c, t, h, w = video.shape

        video_preds = self.model(video, attn_map)
        video_preds_softmax = torch.softmax(video_preds, dim=1)

        assert label.shape[0] == video_preds.shape[0]

        loss = F.cross_entropy(video_preds, label.long())

        self.log("train/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)
        video_confusion_matrix = self._confusion_matrix(video_preds_softmax, label)

        self.log_dict(
            {
                "train/video_acc": video_acc,
                "train/video_precision": video_precision,
                "train/video_recall": video_recall,
                "train/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=b,
        )
        logger.info(f"train loss: {loss.item()}")

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        attn_map = batch["attn_map"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float()  # b

        b, c, t, h, w = video.shape

        video_preds = self.model(video, attn_map)
        video_preds_softmax = torch.softmax(video_preds, dim=1)

        loss = F.cross_entropy(video_preds, label.long())

        self.log("val/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)
        video_confusion_matrix = self._confusion_matrix(video_preds_softmax, label)

        self.log_dict(
            {
                "val/video_acc": video_acc,
                "val/video_precision": video_precision,
                "val/video_recall": video_recall,
                "val/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=b,
        )

        logger.info(f"val loss: {loss.item()}")

    ##############
    # test step
    ##############
    # the order of the hook function is:
    # on_test_start -> test_step -> on_test_batch_end -> on_test_epoch_end -> on_test_end

    def on_test_start(self) -> None:
        """hook function for test start"""

        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []

        logger.info("test start")

    def on_test_end(self) -> None:
        """hook function for test end"""
        logger.info("test end")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        attn_map = batch["attn_map"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float()  # b

        b, c, t, h, w = video.shape

        video_preds = self.model(video, attn_map)
        video_preds_softmax = torch.softmax(video_preds, dim=1)

        loss = F.cross_entropy(video_preds, label.long())

        self.log("test/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)
        video_confusion_matrix = self._confusion_matrix(video_preds_softmax, label)

        metric_dict = {
            "test/video_acc": video_acc,
            "test/video_precision": video_precision,
            "test/video_recall": video_recall,
            "test/video_f1_score": video_f1_score,
        }
        self.log_dict(metric_dict, on_epoch=True, on_step=True, batch_size=b)

        self.test_pred_list.append(video_preds_softmax.detach().cpu())
        self.test_label_list.append(label.detach().cpu())

        fold = (
            getattr(self.logger, "root_dir", "fold").split("/")[-1]
            if self.logger
            else "fold"
        )
        if batch_idx < 10:
            dump_all_feature_maps(
                model=self.model,
                video=video,
                video_info=batch.get("info", None),
                attn_map=attn_map,
                save_root=f"{self.save_root}/test_all_feature_maps/{fold}/batch_{batch_idx}",
                include_types=(torch.nn.Conv3d, torch.nn.Linear),
                include_name_contains=["conv_c"],
                resize_to=(256, 256),  # 指定输出大小
                resize_mode="bilinear",  # 放大更平滑
            )

        return video_preds_softmax, video_preds

    def on_test_epoch_end(self) -> None:
        """hook function for test epoch end"""

        # save the metrics to file
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=self.logger.root_dir.split("/")[-1],
            save_path=self.save_root,
            num_class=self.num_classes,
        )

        logger.info("test epoch end")

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.estimated_stepping_batches,
                    # verbose=True,
                ),
                "monitor": "train/loss",
            },
        }
