#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/sam3d_body/save.py
Project: /workspace/code/sam3d_body
Created Date: Friday December 5th 2025
Author: Kaixu Chen
-----
Comment:
因为数据太多了，所以按照一帧一个 npz 文件来存储

Save utilities for SAM-3D-Body
- ONLY supports per-frame saving
- One frame -> one .npz

Have a good code time :)
-----
Last Modified: Friday December 5th 2025 11:52:16 am
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
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_object_array(x: Any) -> np.ndarray:
    """
    Wrap arbitrary python object (dict with numpy arrays etc.)
    into numpy object array for np.savez.
    """
    return np.array(x, dtype=object)


def _verify_npz_output(path: Path) -> None:
    """Raise ValueError if the saved npz is unreadable or missing required key."""
    with np.load(path, allow_pickle=True) as data:
        if "output" not in data:
            raise ValueError(f"missing 'output' key in {path}")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def save_frame(
    output: Dict[str, Any],
    save_dir: Path,
    frame_idx: int,
    verify_after_write: bool = True,
    max_retries: int = 2,
) -> Path:
    """
    Save ONE frame result.

    Args:
        output:
            dict for one frame (can include numpy arrays, lists, etc.)
        save_dir:
            directory to save into
        frame_idx:
            frame index (used in filename)
        verify_after_write:
            if True, re-open npz after save and verify key "output" exists
        max_retries:
            retry count when write/verify fails

    Returns:
        Path to saved .npz
    """
    if not isinstance(output, dict):
        raise TypeError(f"output must be Dict[str, Any], got {type(output)}")

    _ensure_dir(save_dir)

    save_path = save_dir / f"{frame_idx:06d}_sam3d_body.npz"

    # Write to temp then atomically replace to avoid half-written target files.
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        tmp_path = save_dir / f".{frame_idx:06d}_sam3d_body.{os.getpid()}.{attempt}.tmp.npz"
        try:
            with open(tmp_path, "wb") as f:
                np.savez_compressed(
                    f,
                    output=_to_object_array(output),
                )
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, save_path)

            if verify_after_write:
                _verify_npz_output(save_path)

            logger.info(f"[SAVE] frame {frame_idx} -> {save_path} (attempt={attempt + 1})")
            return save_path
        except Exception as e:
            last_error = e
            logger.warning(
                f"[SAVE-RETRY] frame {frame_idx} failed on attempt {attempt + 1}/{max_retries + 1}: {e}"
            )
            if save_path.exists():
                save_path.unlink()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    assert last_error is not None
    raise RuntimeError(
        f"Failed to save frame {frame_idx} after {max_retries + 1} attempts: {last_error}"
    ) from last_error
