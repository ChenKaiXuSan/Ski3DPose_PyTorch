#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/SAM3Dbody/main copy.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/SAM3Dbody
Created Date: Tuesday May 12th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday May 12th 2026 8:11:57 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

from .infer import process_frame_list
from .load import load_capture_frames

logger = logging.getLogger(__name__)


def process_single_action(
    flag: str,
    source_root: Path,
    vis_root: Path,
    infer_root: Path,
    action_log_root: Path,
    cfg: DictConfig,
) -> None:
    """Process all captures in one action directory.

    Args:
        camera_layers: Optional list of layer indices (0-4) to filter captures.
    """
    log_dir = action_log_root / "action_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_dir / f"{flag}.log", mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    action_logger = logging.getLogger(f"{flag}_action_logger")
    action_logger.handlers.clear()
    action_logger.setLevel(logging.INFO)
    action_logger.addHandler(handler)
    action_logger.propagate = False

    seq_list = list(source_root.iterdir())

    for seq_dir in seq_list:

        if not seq_dir.is_dir():
            continue

        cam_list = list(seq_dir.iterdir())

        for cam_dir in cam_list:
            frame_list = load_capture_frames(cam_dir)

            out_dir = vis_root / seq_dir.name / cam_dir.name
            infer_dir = infer_root / seq_dir.name / cam_dir.name

            process_frame_list(
                frame_list=frame_list,
                out_dir=out_dir,
                inference_output_path=infer_dir,
                cfg=cfg,
            )


def processer(
    source_root: Path,
    vis_root: Path,
    infer_root: Path,
    action_log_root: Path,
    cfg_dict: DictConfig,
    flag: str = "train",
) -> None:
    """Worker entrypoint: pin device and process assigned actions.

    Args:
        camera_layers: Optional list of layer indices to filter captures.
    """

    process_single_action(
        flag,
        source_root / flag,
        vis_root / flag,
        infer_root / flag,
        action_log_root,
        cfg_dict,
    )


@hydra.main(config_path="../configs", config_name="sam3d_body", version_base=None)
def main(cfg: DictConfig) -> None:
    source_root = Path(cfg.paths.ski_poseptz.ski_poseptz_camera_path).resolve()
    result_root = Path(cfg.paths.ski_poseptz.ski_poseptz_result_root).resolve()

    vis_root = result_root / "visualization"
    infer_root = result_root / "inference"
    action_log_root = Path(cfg.paths.log_path).resolve()

    vis_root.mkdir(parents=True, exist_ok=True)
    infer_root.mkdir(parents=True, exist_ok=True)

    logger.info("Source data root: %s", source_root)
    logger.info("Result root: %s", result_root)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    processes: List[mp.Process] = []
    for flag in ["train", "test"]:

        logger.info("Processing flag: %s", flag)

        process = mp.Process(
            target=processer,
            args=(
                source_root,
                vis_root,
                infer_root,
                action_log_root,
                cfg,
                flag,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    logger.info("[SUCCESS] All action workers completed")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
