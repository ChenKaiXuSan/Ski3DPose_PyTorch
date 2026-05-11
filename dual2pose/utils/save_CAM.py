#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/utils/save_CAM.py
Project: /workspace/code/project/utils
Created Date: Monday November 10th 2025
Author: Kaixu Chen
-----
Comment:
This is a utility script to save the Class Activation Maps (CAM) of the model.
The saved CAMs can be used for model evaluation and visualization.

Have a good code time :)
-----
Last Modified: Monday November 10th 2025 8:39:34 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Iterable

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------- 基础工具 ----------


def _resize_u8(
    img: np.ndarray,
    size: Optional[Tuple[int, int]] = None,  # (H_out, W_out)
    mode: str = "nearest",
) -> np.ndarray:
    """
    将 uint8 图像放缩到指定尺寸
    """
    if size is None:
        return img
    h, w = img.shape[:2]
    if (h, w) == size:
        return img
    pil = Image.fromarray(img)
    resample = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
    }.get(mode, Image.NEAREST)
    pil = pil.resize((size[1], size[0]), resample=resample)
    return np.asarray(pil)


def _to_uint8_gray(x: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    """(H,W) -> uint8 灰度"""
    x = x.detach().float().cpu()
    x_min, x_max = x.min(), x.max()
    if (x_max - x_min) < eps:
        x = torch.zeros_like(x)
    else:
        x = (x - x_min) / (x_max - x_min + eps)
    return (x * 255.0).clamp(0, 255).byte().numpy()


def _save_grid(
    images: List[np.ndarray], save_path: str, ncols: int = 8, pad: int = 2
) -> None:
    """保存灰度或RGB网格拼图。
    支持:
        - 单通道 (H, W)
        - 三通道 (H, W, 3)
    """
    import math
    from PIL import Image
    import numpy as np

    if len(images) == 0:
        return

    # 检查通道数
    first = images[0]
    if first.ndim == 2:
        channels = 1
        mode = "L"
    elif first.ndim == 3 and first.shape[2] == 3:
        channels = 3
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported image shape: {first.shape}")

    h, w = first.shape[:2]
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    grid_h = nrows * h + (nrows - 1) * pad
    grid_w = ncols * w + (ncols - 1) * pad

    if channels == 1:
        canvas = np.full((grid_h, grid_w), 0, dtype=np.uint8)
    else:
        canvas = np.full((grid_h, grid_w, 3), 0, dtype=np.uint8)

    for idx, img in enumerate(images):
        if channels == 1 and img.ndim == 3:
            img = img[..., 0]  # 如果误传三维灰度图
        r, c = divmod(idx, ncols)
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0 : y0 + h, x0 : x0 + w] = img

    Image.fromarray(canvas, mode=mode).save(save_path)


def _colormap_jet(gray_u8: np.ndarray) -> np.ndarray:
    """简易 JET（无额外依赖） -> (H,W,3) uint8"""
    g = gray_u8.astype(np.float32) / 255.0
    c = np.zeros((g.shape[0], g.shape[1], 3), dtype=np.float32)
    c[..., 0] = np.clip(1.5 - np.abs(4 * g - 3), 0, 1)  # R
    c[..., 1] = np.clip(1.5 - np.abs(4 * g - 2), 0, 1)  # G
    c[..., 2] = np.clip(1.5 - np.abs(4 * g - 1), 0, 1)  # B

    return (c * 255.0 + 0.5).astype(np.uint8)


# ---------- 主功能：保存每层特征图 ----------
class _FeatureHook:
    def __init__(self, name: str, module: nn.Module, feats: Dict[str, torch.Tensor]):
        self.name = name
        self.module = module
        self.feats = feats
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        # 只保留 tensor 输出；多值输出时取第一个 tensor
        if isinstance(out, torch.Tensor):
            self.feats[self.name] = out
        elif isinstance(out, (list, tuple)) and len(out) > 0:
            for v in out:
                if isinstance(v, torch.Tensor):
                    self.feats[self.name] = v
                    break

    def remove(self):
        try:
            self.handle.remove()
        except Exception:
            pass


def _list_match(names: Iterable[str], patterns: Iterable[str]) -> bool:
    """任意名字包含任意 pattern 即匹配"""
    ps = list(patterns) if patterns else []
    if not ps:
        return True
    for n in names:
        s = str(n)
        for p in ps:
            if p == s:
                return True
    return False


@torch.no_grad()
def dump_all_feature_maps(
    model: nn.Module,
    video: torch.Tensor,  # (B,3,T,H,W)
    video_info: Optional[List[str]] = None,
    attn_map: Optional[torch.Tensor] = None,  # (B,1,T,H,W) 若你的前向需要
    save_root: str = "fusion_vis/all_features",
    include_types: Tuple[type, ...] = (
        nn.Conv3d,
        nn.Conv2d,
        nn.ReLU,
        nn.BatchNorm3d,
        nn.MaxPool3d,
        nn.AvgPool3d,
    ),
    include_name_contains: Tuple[str, ...] = (),  # 名称包含关键字才抓取；空则不过滤
    exclude_name_contains: Tuple[str, ...] = (
        "proj",
    ),  # 名称包含关键字则跳过（默认跳过分类 head 如 "proj"/"head" 可按需改）
    resize_to: Optional[Tuple[int, int]] = None,  # 统一放大到指定尺寸 (H, W)
    resize_mode: str = "nearest",  # 最近邻/双线性/双三次
) -> None:
    """
    对一个前向过程抓取并保存**每一层**输出特征图。
    - 仅抓取 4D/5D 张量（(B,C,H,W) 或 (B,C,T,H,W)）。
    - 每层按 batch 分别保存到目录：save_root/<layer_name>/b{k}/...
    返回：{ layer_name: [保存的文件路径...] }
    """
    os.makedirs(save_root, exist_ok=True)

    # prepare video info
    video_info_list = []
    for one_video in video_info:
        for i in range(one_video["video"].shape[0]):
            video_info_list.append(one_video["video_name"])

    assert len(video_info_list) == video.size(0), "video_info length must match batch size"

    # 1) 注册 hooks
    feats: Dict[str, torch.Tensor] = {}
    hooks: List[_FeatureHook] = []
    # FIXME: 这里的过滤条件还需要修改
    for name, mod in model.named_modules():
        if not isinstance(mod, include_types):
            continue
        parts = name.split(".")
        if not _list_match(parts, include_name_contains):
            continue
        if _list_match(parts, exclude_name_contains):
            continue
        hooks.append(_FeatureHook(name, mod, feats))

    if not hooks:
        logger.warning(
            "没有匹配到任何模块，检查 include_types/include_name_contains 过滤条件。"
        )

    # 2) 前向一次抓取
    model_was_training = model.training
    model.eval()
    model = model.to(video.device)

    try:
        _ = model(video, attn_map)
    finally:
        if model_was_training:
            model.train()
        for h in hooks:
            h.remove()

    # 3) 保存每层
    B, C, T, H, W = video.shape

    for lname, tensor in feats.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim not in (4, 5):
            # 跳过非 (B,C,H,W)/(B,C,T,H,W)
            continue

        layer_dir = os.path.join(save_root, lname.replace(".", "_"))
        os.makedirs(layer_dir, exist_ok=True)

        # 按时间维度分别保存
        for b in range(B):
            # 收集前 K 个通道的图像（灰度或伪彩）
            imgs_gray: List[np.ndarray] = []
            imgs_color: List[np.ndarray] = []

            subdir = os.path.join(layer_dir, f"sample{b}")
            os.makedirs(subdir, exist_ok=True)

            for t in range(T):
                #  通道平均响应图
                g = _to_uint8_gray(tensor.mean(dim=1)[b, t])
                if resize_to is not None:
                    g = _resize_u8(g, size=resize_to, mode=resize_mode)

                imgs_gray.append(g)
                imgs_color.append(_colormap_jet(g))

            # 也逐时间保存彩色
            for i, col in enumerate(imgs_color):
                p = os.path.join(subdir, f"{video_info_list[b]}_time{i:02d}.png")
                Image.fromarray(col).save(p)

            _save_grid(
                imgs_gray,
                os.path.join(subdir, f"{video_info_list[b]}_grid_gray.png"),
                ncols=T,
                pad=2,
            )
            _save_grid(
                imgs_color,
                os.path.join(subdir, f"{video_info_list[b]}_grid_color.png"),
                ncols=T,
                pad=2,
            )