#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/SAM3Dbody/load.py
Project: /workspace/code/SAM3Dbody
Created Date: Friday January 23rd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 23rd 2026 4:50:57 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def infer_person_side(video_stem: str) -> Optional[str]:
    """Infer person camera side from a video stem.

    Mapping rules:
    - Names containing "left"  -> left
    - Names containing "right" -> right
    - run_* convention: osmo2 -> left, osmo1 -> right
    """
    name = video_stem.strip().lower()
    compact = name.replace("_", "").replace("-", "")

    if "left" in name:
        return "left"
    if "right" in name:
        return "right"

    if "osmo2" in compact:
        return "left"
    if "osmo1" in compact:
        return "right"

    return None


@dataclass
class DataConfig:
    subject_name: str
    camera_name: str
    frame_path: Path  # path to frames/ folder or video file
    sam_3d_body_result_path: Path  # path to save SAM-3D-Body inference results
    sam3_output_path: Path  # path to save SAM-3 inference results


def build_unity_data_configs(
    unity_root: Path,
    sam3d_root: Path,
    sam3_root: Path,
    camera_layers=None,
    person_filter: Optional[str] = None,
    action_filter: Optional[str] = None,
) -> List[DataConfig]:
    """Build DataConfig list for unity layout.

    Expected unity layout:
    - unity_root/person/action/(frames|viz)/capture_xxx/*.png
    """
    data_configs: List[DataConfig] = []

    for subject_dir in sorted([x for x in unity_root.iterdir() if x.is_dir()]):
        if person_filter and subject_dir.name != person_filter:
            continue

        for action_dir in sorted([x for x in subject_dir.iterdir() if x.is_dir()]):
            if action_filter and action_dir.name != action_filter:
                continue

            capture_dirs = collect_capture_dirs(action_dir, camera_layers)
            for capture_dir in capture_dirs:
                rel_capture = capture_dir.relative_to(unity_root)
                data_configs.append(
                    DataConfig(
                        subject_name=subject_dir.name,
                        camera_name=capture_dir.name,
                        frame_path=capture_dir,
                        sam_3d_body_result_path=sam3d_root / rel_capture,
                        sam3_output_path=sam3_root / rel_capture,
                    )
                )

    return data_configs


def build_person_data_configs(
    person_root: Path,
    sam3d_root: Path,
    sam3_root: Path,
    person_filter: Optional[str] = None,
    action_filter: Optional[str] = None,
) -> List[DataConfig]:
    """Build DataConfig list for person layout.

    Expected person layout:
    - person_root/run_x/*.mp4
    """
    data_configs: List[DataConfig] = []

    for run_dir in sorted([x for x in person_root.iterdir() if x.is_dir()]):
        run_name = run_dir.name
        if person_filter and run_name != person_filter:
            continue
        if action_filter and run_name != action_filter:
            continue

        for video_path in list_person_video_files(run_dir):
            side = infer_person_side(video_path.stem)
            if side is None:
                logger.warning(
                    "[Skip] Cannot infer person side for video: %s", video_path
                )
                continue

            data_configs.append(
                DataConfig(
                    subject_name=run_name,
                    camera_name=video_path.stem,
                    frame_path=video_path,
                    sam_3d_body_result_path=sam3d_root / run_name / side,
                    sam3_output_path=sam3_root / run_name / video_path.stem,
                )
            )

    return data_configs


def load_capture_frames(capture_dir: Path) -> List[np.ndarray]:
    """Load one capture folder into an RGB frame list."""
    frame_files = sorted(
        [
            p
            for p in capture_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
    )

    frames: List[np.ndarray] = []
    for frame_file in frame_files:
        frame_bgr = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            logger.warning("[Skip] Failed to read image: %s", frame_file)
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    return frames


def load_video_frames(video_path: Path) -> List[np.ndarray]:
    """Load one video file into an RGB frame list."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("[Skip] Failed to open video: %s", video_path)
        return []

    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    return frames


def _has_image_files(folder: Path) -> bool:
    """Check whether a folder directly contains image files."""
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            return True
    return False


def _has_video_files(folder: Path) -> bool:
    """Check whether a folder directly contains video files."""
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            return True
    return False


def list_person_video_files(run_dir: Path) -> List[Path]:
    """List person-view videos (e.g., osmo_1.mp4/osmo_2.mp4) under one run directory."""
    if not run_dir.is_dir():
        return []
    return sorted(
        [
            p
            for p in run_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        ]
    )


def load_person_run_frames(run_dir: Path) -> Dict[str, List[np.ndarray]]:
    """Load all videos under one person run directory into RGB frame lists."""
    outputs: Dict[str, List[np.ndarray]] = {}
    for video_path in list_person_video_files(run_dir):
        outputs[video_path.stem] = load_video_frames(video_path)
    return outputs


def collect_capture_dirs(action_dir: Path, camera_layers=None) -> List[Path]:
    """Collect camera folders under one action directory.

    Supported layout:
    - person/action/frames/camera

    Args:
        action_dir: Action directory containing frames/
        camera_layers: Optional list of layer indices (0-4) to filter.
                      E.g., [0, 1] means only L0 and L1.
                      None means all layers.
    """
    # Unity canonical dataset may store captures under action/frames or action/viz.
    frames_dir = action_dir / "frames"
    viz_dir = action_dir / "viz"

    base_dir = None
    if frames_dir.is_dir():
        base_dir = frames_dir
    elif viz_dir.is_dir():
        base_dir = viz_dir
    else:
        # Person dataset fallback: run directory may directly contain images/videos.
        return collect_person_capture_dirs(action_dir)

    capture_dirs = sorted(
        [x for x in base_dir.iterdir() if x.is_dir() and _has_image_files(x)]
    )

    # Apply camera layer filter if specified
    if camera_layers is not None and len(camera_layers) > 0:
        filtered_dirs = []
        for capture_dir in capture_dirs:
            # Expected format: capture_L{layer}_A{angle}
            name = capture_dir.name
            if name.startswith("capture_L"):
                try:
                    layer_str = name.split("_")[1]  # "L0", "L1", etc.
                    layer_num = int(layer_str[1:])  # Extract 0, 1, 2, 3, 4
                    if layer_num in camera_layers:
                        filtered_dirs.append(capture_dir)
                except (IndexError, ValueError):
                    logger.warning("[Skip] Unexpected capture dir name: %s", name)
            else:
                # If not matching expected pattern, keep it (backward compatibility)
                filtered_dirs.append(capture_dir)
        return filtered_dirs

    return capture_dirs


def collect_person_capture_dirs(action_dir: Path) -> List[Path]:
    """Collect capture candidates for person dataset.

    person data layout (current):
    - side_raw/run_x/*.mp4
    For SAM3 image inference, videos should be decoded to frames first.
    """
    if not action_dir.is_dir():
        return []

    if _has_image_files(action_dir):
        return [action_dir]

    if _has_video_files(action_dir):
        return [action_dir]

    return []


def collect_action_dirs(
    source_root: Path,
    camera_layers=None,
    person_filter=None,
    action_filter=None,
    data_type: Optional[str] = None,
) -> List[Path]:
    """Collect action directories for camera-based inference.

    Supported layout:
    - person/action/frames/camera

    An action directory is identified if it has a ``frames`` folder containing
    camera subdirectories with image files.

    Args:
        source_root: Root directory containing person folders
        camera_layers: Optional list of layer indices to filter captures
        person_filter: Optional person name to filter (e.g., "male", "female")
        action_filter: Optional action name to filter (exact match)
    """
    action_dirs_set = set()

    layout_type = (data_type or "").strip().lower()
    if not layout_type:
        # Auto-detect from folder signature.
        if any(
            (source_root / x).is_dir() for x in ["run_3", "run_4", "run_5", "run_6"]
        ):
            layout_type = "person"
        else:
            layout_type = "unity"

    if layout_type == "person":
        for run_dir in sorted([x for x in source_root.iterdir() if x.is_dir()]):
            if person_filter is not None and run_dir.name != person_filter:
                continue
            if action_filter is not None and run_dir.name != action_filter:
                continue
            if collect_person_capture_dirs(run_dir):
                action_dirs_set.add(run_dir)
        return sorted(action_dirs_set)

    # Strict scan: only one level below each person directory.
    for person_dir in sorted([x for x in source_root.iterdir() if x.is_dir()]):
        # Apply person filter
        if person_filter is not None and person_dir.name != person_filter:
            continue

        for candidate in sorted([x for x in person_dir.iterdir() if x.is_dir()]):
            # Apply action filter
            if action_filter is not None and candidate.name != action_filter:
                continue

            if collect_capture_dirs(candidate, camera_layers):
                action_dirs_set.add(candidate)

    # Keep sorted deterministic ordering for sharding.
    action_dirs = sorted(action_dirs_set)
    return action_dirs
