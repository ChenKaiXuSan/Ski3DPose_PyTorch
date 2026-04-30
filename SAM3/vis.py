#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/sam3d_body/utils.py
Project: /workspace/code/sam3d_body
Created Date: Thursday December 4th 2025
Author: Kaixu Chen
-----
Comment:
Visualization utilities for SAM3Dbody results.

Have a good code time :)
-----
Last Modified: Thursday December 4th 2025 4:24:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def overlay_masks(
    image_rgb: np.ndarray, masks: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Overlay N binary masks onto an RGB image."""
    if masks.size == 0:
        return image_rgb

    out = image_rgb.astype(np.float32).copy()
    rng = np.random.default_rng(seed=42)
    colors = rng.integers(0, 256, size=(masks.shape[0], 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        if not np.any(mask):
            continue
        color = colors[i].astype(np.float32)
        m = mask.astype(bool)
        out[m] = out[m] * (1.0 - alpha) + color * alpha

    return np.clip(out, 0, 255).astype(np.uint8)
