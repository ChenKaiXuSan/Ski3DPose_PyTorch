#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/helper.py
Project: /workspace/skeleton/project
Created Date: Tuesday May 14th 2024
Author: Kaixu Chen
-----
Comment:
This is a helper script to save the results of the training.
The saved items include:
1. the prediction and label for the further analysis.
2. the metrics for the model evaluation.
3. the confusion matrix for the model evaluation.

This script is executed at the end of each training in main.py file.

Have a good code time :)
-----
Last Modified: Tuesday May 14th 2024 3:23:52 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

04-12-2024	Kaixu Chen	refactor the code, add the save_inference method.

14-05-2024	Kaixu Chen	add save_CAM method, now it can save the CAM for the model evaluation.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

logger = logging.getLogger(__name__)


def save_helper(
    all_pred: list[torch.Tensor],
    all_label: list[torch.Tensor],
    fold: str,
    save_path: str,
    num_class: int,
):
    """save the inference results and metrics.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        fold (str): fold number.
        save_path (str): save path.
        num_class (int): number of class.
    """

    # check device 
    if all_pred[0].is_cuda:
        all_pred = [pred.cpu() for pred in all_pred]
    if all_label[0].is_cuda:
        all_label = [label.cpu() for label in all_label]
        
    all_pred: torch.Tensor = torch.cat(all_pred, dim=0)
    all_label: torch.Tensor = torch.cat(all_label, dim=0)

    save_inference(all_pred, all_label, fold, save_path)
    save_metrics(all_pred, all_label, fold, save_path, num_class)
    save_CM(all_pred, all_label, save_path, num_class, fold)


def save_inference(
    all_pred: torch.Tensor, all_label: torch.Tensor, fold: str, save_path: str
):
    """save the inference results to .pt file.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        fold (str): fold number.
        save_path (str): save path.
    """

    # save the results
    save_path = Path(save_path) / "best_preds"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    torch.save(
        all_pred,
        save_path / f"{fold}_pred.pt",
    )
    torch.save(
        all_label,
        save_path / f"{fold}_label.pt",
    )

    logger.info(f"save the pred and label into {save_path} / {fold}")


def save_metrics(
    all_pred: torch.Tensor,
    all_label: torch.Tensor,
    fold: str,
    save_path: str,
    num_class: int,
):
    """save the metrics to .txt file.

    Args:
        all_pred (list): all the predict result.
        all_label (list): all the label result.
        fold (str): the fold number.
        save_path (str): the path to save the metrics.
        num_class (int): number of class.
    """

    save_path = Path(save_path) / "metrics.txt"

    _accuracy = MulticlassAccuracy(num_class)
    _precision = MulticlassPrecision(num_class)
    _recall = MulticlassRecall(num_class)
    _f1_score = MulticlassF1Score(num_class)
    _auroc = MulticlassAUROC(num_class)
    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    logger.info("*" * 100)
    logger.info("accuracy: %s" % _accuracy(all_pred, all_label))
    logger.info("precision: %s" % _precision(all_pred, all_label))
    logger.info("recall: %s" % _recall(all_pred, all_label))
    logger.info("f1_score: %s" % _f1_score(all_pred, all_label))
    logger.info("aurroc: %s" % _auroc(all_pred, all_label.long()))
    logger.info("confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))
    logger.info("#" * 100)

    with open(save_path, "a") as f:
        f.writelines(f"Fold {fold}\n")
        f.writelines(f"accuracy: {_accuracy(all_pred, all_label)}\n")
        f.writelines(f"precision: {_precision(all_pred, all_label)}\n")
        f.writelines(f"recall: {_recall(all_pred, all_label)}\n")
        f.writelines(f"f1_score: {_f1_score(all_pred, all_label)}\n")
        f.writelines(f"aurroc: {_auroc(all_pred, all_label.long())}\n")
        f.writelines(f"confusion_matrix: {_confusion_matrix(all_pred, all_label)}\n")
        f.writelines("#" * 100)
        f.writelines("\n")


def save_CM(
    all_pred: torch.Tensor,
    all_label: torch.Tensor,
    save_path: str,
    num_class: int,
    fold: str,
):
    """save the confusion matrix to file.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        save_path (Path): the path to save the confusion matrix.
        num_class (int): the number of class.
        fold (str): the fold number.
    """

    save_path = Path(save_path) / "CM"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    logger.info("_confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))

    # set the font and title
    plt.rcParams.update({"font.size": 30, "font.family": "sans-serif"})

    confusion_matrix_data = _confusion_matrix(all_pred, all_label).cpu().numpy() * 100

    axis_labels = ["ASD", "DHS", "LCS_HipOA"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        vmin=0,
        vmax=100,
    )
    plt.title(f"Fold {fold} (%)", fontsize=30)
    plt.ylabel("Actual Label", fontsize=30)
    plt.xlabel("Predicted Label", fontsize=30)

    plt.savefig(
        save_path / f"fold{fold}_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )

    logger.info(
        f"save the confusion matrix into {save_path}/fold{fold}_confusion_matrix.png"
    )

