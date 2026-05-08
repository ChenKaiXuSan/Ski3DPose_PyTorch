#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/dataloader/whole_video_dataset_single_view.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/dataloader
Created Date: Tuesday April 28th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday April 28th 2026 4:34:03 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import lru_cache
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from project.map_config import (
    UnityDataConfig,
    filter_sam3d_body_kpts,
    filter_unity_kpts,
)

logger = logging.getLogger(__name__)


class LabeledUnityDataset(Dataset):
    """
    Multi-view labeled video dataset.
    """

    def __init__(
        self,
        experiment: str,
        index_mapping: List[UnityDataConfig],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        load_frames: bool = True,
        load_2d_kpt: bool = True,
        load_3d_kpt: bool = True,
        load_mask: bool = True,
        target_t: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = index_mapping
        self._transform = transform
        self._load_frames = bool(load_frames)
        self._load_2d_kpt = bool(load_2d_kpt)
        self._load_3d_kpt = bool(load_3d_kpt)
        self._load_mask = bool(load_mask)
        self._target_t = int(target_t) if target_t is not None else None
        if self._target_t is not None and self._target_t <= 0:
            raise ValueError("target_t must be > 0 when provided.")
        if (
            not self._load_frames
            and not self._load_2d_kpt
            and not self._load_3d_kpt
            and not self._load_mask
        ):
            raise ValueError(
                "At least one of load_frames/load_2d_kpt/load_3d_kpt/load_mask must be enabled."
            )

    def __len__(self) -> int:
        return len(self._index_mapping)

    @staticmethod
    def _item_get(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    @staticmethod
    def _normalize_item_dict(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(cast(Any, item))
        if hasattr(item, "__dict__"):
            return dict(item.__dict__)
        raise TypeError(f"Unsupported index item type: {type(item)}")

    @staticmethod
    def _load_frames_dir(path: Path) -> torch.Tensor:
        """Load image sequence directory into (T,C,H,W)."""
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Frame directory not found: {path}")

        frame_files = sorted(path.glob("*.png"))
        if len(frame_files) == 0:
            frame_files = sorted(path.glob("*.jpg"))
        if len(frame_files) == 0:
            raise RuntimeError(f"No frame files found in: {path}")

        frames = []
        for p in frame_files:
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to read frame: {p}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            frames.append(
                torch.from_numpy(np.ascontiguousarray(img_rgb)).permute(2, 0, 1)
            )
        return torch.stack(frames, dim=0)

    @staticmethod
    def _extract_last_int(name: str) -> int:
        nums = re.findall(r"(\d+)", name)
        if not nums:
            raise ValueError(f"No frame index found in filename: {name}")

        # Prefer 6-digit frame indices (e.g. frame_000012, kpt2d_000012, 000012_sam3d_body).
        six_digits = [x for x in nums if len(x) >= 6]
        if six_digits:
            return int(six_digits[0])

        # Fallback for uncommon naming.
        return int(nums[-1])

    @classmethod
    @lru_cache(maxsize=4096)
    def _build_idx_file_map_cached(
        cls, root_str: str, pattern: str
    ) -> Tuple[Tuple[int, str], ...]:
        root = Path(root_str)
        out: List[Tuple[int, str]] = []
        for p in sorted(root.glob(pattern)):
            idx = cls._extract_last_int(p.stem)
            out.append((idx, str(p)))
        return tuple(out)

    @classmethod
    def _build_idx_file_map(cls, root: Path, pattern: str) -> Dict[int, Path]:
        if not root.exists() or not root.is_dir():
            return {}
        pairs = cls._build_idx_file_map_cached(str(root.resolve()), pattern)
        return {idx: Path(path_str) for idx, path_str in pairs}

    @staticmethod
    def _load_sam3d_file(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load SAM3D 2D/3D keypoints from one npz file.

        Returns:
            (sam_2d, sam_3d)
        """
        data = np.load(npz_path, allow_pickle=True)
        if "output" not in data.files:
            raise KeyError(f"Missing 'output' in SAM npz: {npz_path}")
        output = data["output"]
        if isinstance(output, np.ndarray) and output.shape == ():
            output = output.item()

        if not isinstance(output, dict):
            raise TypeError(f"Unexpected SAM output type in {npz_path}: {type(output)}")

        if "pred_keypoints_3d" in output:
            arr_3d = output["pred_keypoints_3d"]
        elif "pred_joint_coords" in output:
            arr_3d = output["pred_joint_coords"]
        else:
            raise KeyError(
                f"No 3D keypoint key found in SAM output: {npz_path}, keys={list(output.keys())}"
            )

        if "pred_keypoints_2d" in output:
            arr_2d = output["pred_keypoints_2d"]
        else:
            # fallback: keep first 2 dims from 3d keypoints for compatibility
            arr_2d = np.asarray(arr_3d, dtype=np.float32)[..., :2]

        return np.asarray(arr_2d, dtype=np.float32), np.asarray(
            arr_3d, dtype=np.float32
        )

    @staticmethod
    def _load_mask_file(npz_path: Path) -> np.ndarray:
        """Load one SAM3 mask npz and return a fixed single-channel binary mask.

        SAM may output variable number of instances per frame (V,H,W).
        For stable batching and cross-class consistency, we merge all instances
        into one semantic mask with shape (1,H,W).
        """
        data = np.load(npz_path, allow_pickle=True)
        if "masks" not in data.files:
            raise KeyError(f"Missing 'masks' in SAM mask npz: {npz_path}")

        masks = np.asarray(data["masks"])
        if masks.ndim == 2:
            # Already a single mask (H,W) -> (1,H,W)
            merged = (masks > 0)[None, ...]
            return merged

        if masks.ndim != 3:
            raise ValueError(
                f"Expected SAM mask shape (V,H,W) or (H,W), got {tuple(masks.shape)} in {npz_path}"
            )

        if masks.shape[0] == 0:
            # Keep spatial shape and return empty semantic mask.
            return np.zeros((1, masks.shape[-2], masks.shape[-1]), dtype=np.bool_)

        # Merge all instances into one foreground mask and keep channel dim.
        merged = np.any(masks > 0, axis=0)
        return merged[None, ...]

    @staticmethod
    def _is_empty_keypoint_array(arr: np.ndarray) -> bool:
        """Return True when keypoint array has no valid joint entries."""
        kpt = np.asarray(arr)
        if kpt.size == 0:
            return True
        if kpt.ndim == 3 and kpt.shape[0] == 1:
            kpt = kpt[0]
        if kpt.ndim < 2:
            return True
        return int(kpt.shape[0]) == 0

    @staticmethod
    def _pick_fallback_frame_index(
        target_idx: int, valid_indices: List[int]
    ) -> Optional[int]:
        """Pick fallback index: prefer nearest previous, otherwise nearest next."""
        if not valid_indices:
            return None

        prev = [x for x in valid_indices if x < target_idx]
        if prev:
            return prev[-1]

        nxt = [x for x in valid_indices if x > target_idx]
        if nxt:
            return nxt[0]

        return None

    @staticmethod
    def _read_none_detected_indices(
        output_dir: Path,
    ) -> Tuple[bool, List[int], List[str]]:
        """Read none_detected_frames.txt under one SAM output directory.

        Returns:
            (exists, sorted unique indices, invalid_lines)
        """
        none_file = output_dir / "none_detected_frames.txt"
        if not none_file.exists() or not none_file.is_file():
            return False, [], []

        indices: List[int] = []
        invalid_lines: List[str] = []
        for raw in none_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                idx = int(line)
            except ValueError:
                invalid_lines.append(line)
                continue
            if idx < 0:
                invalid_lines.append(line)
                continue
            indices.append(idx)

        return True, sorted(set(indices)), invalid_lines

    def _resolve_sam_sequence_with_fallback(
        self,
        frame_indices: List[int],
        sam_file_map: Dict[int, Path],
        camera_id: Any,
        none_detected_indices: Optional[set[int]] = None,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Resolve SAM2D/SAM3D per frame index, filling empty entries from nearby frames."""
        loaded: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        valid_indices: List[int] = []
        none_detected = none_detected_indices or set()

        for idx in frame_indices:
            if idx not in sam_file_map:
                if idx in none_detected:
                    continue
                logger.warning(
                    "Missing SAM file at frame idx=%s for camera %s; will try nearby-frame fallback",
                    idx,
                    camera_id,
                )
                continue

            sam2d, sam3d = self._load_sam3d_file(sam_file_map[idx])
            loaded[idx] = (sam2d, sam3d)
            if not self._is_empty_keypoint_array(
                sam2d
            ) and not self._is_empty_keypoint_array(sam3d):
                valid_indices.append(idx)

        if not valid_indices:
            raise RuntimeError(
                f"All SAM keypoint results are empty for camera {camera_id}. "
                "Cannot apply nearby-frame fallback."
            )

        resolved: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for idx in frame_indices:
            if idx in loaded:
                sam2d, sam3d = loaded[idx]
            else:
                sam2d, sam3d = (
                    np.empty((0, 0), dtype=np.float32),
                    np.empty((0, 0), dtype=np.float32),
                )

            if (
                idx in loaded
                and not self._is_empty_keypoint_array(sam2d)
                and not self._is_empty_keypoint_array(sam3d)
            ):
                resolved[idx] = (sam2d, sam3d)
                continue

            fallback_idx = self._pick_fallback_frame_index(idx, valid_indices)
            if fallback_idx is None:
                raise RuntimeError(
                    f"SAM keypoint result at idx={idx} for camera {camera_id} is empty, "
                    "and no fallback frame exists."
                )

            logger.warning(
                "SAM result empty at frame idx=%s for camera %s; using fallback idx=%s",
                idx,
                camera_id,
                fallback_idx,
            )
            resolved[idx] = loaded[fallback_idx]

        return resolved

    @staticmethod
    def _log_missing_sam_paths(
        camera_id: Any,
        sam_dir: Path,
        missing_indices: List[int],
    ) -> None:
        """Log all expected SAM file paths for missing frame indices."""
        if not missing_indices:
            return

        expected_paths = [
            str((sam_dir / f"{idx:06d}_sam3d_body.npz").resolve())
            for idx in missing_indices
        ]
        logger.warning(
            "Missing SAM files for camera %s (count=%s). Full expected paths:\n%s",
            camera_id,
            len(missing_indices),
            "\n".join(expected_paths),
        )

    @staticmethod
    def _temporal_resample_indices(src_len: int, dst_len: int) -> torch.Tensor:
        if src_len <= 0:
            raise ValueError("src_len must be > 0")
        if dst_len <= 0:
            raise ValueError("dst_len must be > 0")
        if src_len == dst_len:
            return torch.arange(src_len, dtype=torch.long)
        # Same strategy as uniform temporal sampling: evenly spaced indices.
        return torch.linspace(0, src_len - 1, steps=dst_len).long()

    @staticmethod
    def _temporal_average_resample(
        tensor: torch.Tensor, target_t: int, time_dim: int
    ) -> torch.Tensor:
        """Resample temporal dimension with averaging semantics.

        - Downsample: average over evenly partitioned temporal bins.
        - Upsample: linear interpolation on temporal axis (weighted average).
        """
        if target_t <= 0:
            raise ValueError("target_t must be > 0")

        src_t = int(tensor.shape[time_dim])
        if src_t == target_t:
            return tensor

        x = tensor.movedim(time_dim, 0)

        if src_t > target_t:
            edges = torch.linspace(0, src_t, steps=target_t + 1, device=x.device)
            out_chunks: List[torch.Tensor] = []
            for i in range(target_t):
                s = int(torch.floor(edges[i]).item())
                e = int(torch.ceil(edges[i + 1]).item())
                if e <= s:
                    e = min(s + 1, src_t)
                out_chunks.append(x[s:e].mean(dim=0))
            y = torch.stack(out_chunks, dim=0)
            return y.movedim(0, time_dim)

        # src_t < target_t
        rest_shape = x.shape[1:]
        x_flat = x.reshape(src_t, -1).transpose(0, 1).unsqueeze(0)  # (1,C_flat,T)
        y_flat = F.interpolate(
            x_flat,
            size=target_t,
            mode="linear",
            align_corners=False,
        )
        y = y_flat.squeeze(0).transpose(0, 1).reshape((target_t,) + rest_shape)
        return y.movedim(0, time_dim)

    @classmethod
    def _resample_frame_indices_avg(
        cls, frame_indices: torch.Tensor, target_t: int
    ) -> torch.Tensor:
        """Resample frame indices to target_t using average semantics and round to int."""
        if frame_indices.numel() == target_t:
            return frame_indices
        idx_f = frame_indices.to(torch.float32)
        idx_f = cls._temporal_average_resample(idx_f, target_t=target_t, time_dim=0)
        return idx_f.round().to(torch.long)

    @staticmethod
    def _extract_cam_ide_token(cam_id: Any) -> str:
        """Extract comparable IDE token from camera id (e.g. L2_A001 -> A001)."""
        cam_str = str(cam_id)
        if "_" in cam_str:
            return cam_str.split("_")[-1]
        return cam_str

    @staticmethod
    def _extract_cam_layer_token(cam_id: Any) -> str:
        """Extract layer/group token from camera id (e.g. L2_A001 -> L2)."""
        cam_str = str(cam_id)
        if "_" in cam_str:
            return cam_str.split("_")[1]
        return ""

    def _load_single_variant_keypoints(
        self,
        variant: str,
        cam1_kpt2d_dir: Path,
        kpt3d_dir: Path,
        sam3d_cam1_kpt2d_dir: Path,
        sam3d_cam1_kpt3d_dir: Path,
        common_idx: List[int],
        sam3d_cam1_kpt2d_map: Optional[Dict[int, Path]] = None,
        sam3d_cam1_kpt3d_map: Optional[Dict[int, Path]] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Load keypoints for a single variant.

        Returns:
            (unity_gt_cam1_kpt2d_t, unity_gt_kpt3d_t, sam3d_cam1_kpt2d_t, sam3d_cam1_kpt3d_t, frame_indices_t)
        """
        cam1_kpt2d_map = (
            self._build_idx_file_map(cam1_kpt2d_dir, "kpt2d_*.npy")
            if self._load_2d_kpt
            else {}
        )
        kpt3d_map = (
            self._build_idx_file_map(kpt3d_dir, "frame_*.npy")
            if self._load_3d_kpt
            else {}
        )
        if self._load_2d_kpt:
            if sam3d_cam1_kpt2d_map is None:
                sam3d_cam1_kpt2d_map = self._build_idx_file_map(
                    sam3d_cam1_kpt2d_dir, "kpt2d_*.npy"
                )
        else:
            sam3d_cam1_kpt2d_map = {}

        if self._load_3d_kpt:
            if sam3d_cam1_kpt3d_map is None:
                sam3d_cam1_kpt3d_map = self._build_idx_file_map(
                    sam3d_cam1_kpt3d_dir, "kpt3d_*.npy"
                )
        else:
            sam3d_cam1_kpt3d_map = {}

        unity_gt_cam1_kpt2d: List[torch.Tensor] = []
        unity_gt_kpt3d: List[torch.Tensor] = []
        sam3d_cam1_kpt2d: List[torch.Tensor] = []
        sam3d_cam1_kpt3d: List[torch.Tensor] = []

        is_character_variant = variant == "character"

        for idx in common_idx:
            if self._load_2d_kpt:
                cam1_2d = np.asarray(np.load(cam1_kpt2d_map[idx]), dtype=np.float32)
                cam1_2d_filtered = (
                    filter_unity_kpts(cam1_2d) if is_character_variant else cam1_2d
                )
                unity_gt_cam1_kpt2d.append(torch.from_numpy(cam1_2d_filtered))

            if self._load_3d_kpt:
                gt_3d = np.asarray(np.load(kpt3d_map[idx]), dtype=np.float32)
                gt_3d_filtered = (
                    filter_unity_kpts(gt_3d) if is_character_variant else gt_3d
                )
                unity_gt_kpt3d.append(torch.from_numpy(gt_3d_filtered))

            if self._load_2d_kpt:
                sam1_2d = np.asarray(
                    np.load(sam3d_cam1_kpt2d_map[idx]), dtype=np.float32
                )

                sam3d_cam1_kpt2d.append(
                    torch.from_numpy(filter_sam3d_body_kpts(sam1_2d))
                )

            if self._load_3d_kpt:
                sam1_3d = np.asarray(
                    np.load(sam3d_cam1_kpt3d_map[idx]), dtype=np.float32
                )

                sam3d_cam1_kpt3d.append(
                    torch.from_numpy(filter_sam3d_body_kpts(sam1_3d))
                )

        unity_gt_cam1_kpt2d_t = (
            torch.stack(unity_gt_cam1_kpt2d, dim=0) if self._load_2d_kpt else None
        )
        unity_gt_kpt3d_t = (
            torch.stack(unity_gt_kpt3d, dim=0) if self._load_3d_kpt else None
        )
        sam3d_cam1_kpt2d_t = (
            torch.stack(sam3d_cam1_kpt2d, dim=0) if self._load_2d_kpt else None
        )
        sam3d_cam1_kpt3d_t = (
            torch.stack(sam3d_cam1_kpt3d, dim=0) if self._load_3d_kpt else None
        )
        frame_indices_t = torch.tensor(common_idx, dtype=torch.long)

        return (
            unity_gt_cam1_kpt2d_t,
            unity_gt_kpt3d_t,
            sam3d_cam1_kpt2d_t,
            sam3d_cam1_kpt3d_t,
            frame_indices_t,
        )

    def _load_single_view_modalities(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Load aligned modalities for one single-view sample.

        Modalities:
          - cam1/cam2 frames
          - cam1/cam2 2D kpt
          - GT 3D kpt
          - SAM3D pred 3D kpt for cam1/cam2

        Supports loading multiple variants (character, pole, ski) if variant dicts are available.
        """
        cam1_frames_dir = Path(item["cam1_frames_dir"])
        cam1_kpt2d_dir = Path(item["cam1_kpt2d_dir"])
        kpt3d_dir = Path(item["kpt3d_dir"])
        sam3d_cam1_kpt2d_dir = Path(item["sam3d_cam1_kpt2d_dir"])
        sam3d_cam1_kpt3d_dir = Path(item["sam3d_cam1_kpt3d_dir"])
        sam3_cam1_mask_ski_dir = Path(item["sam3_cam1_mask_ski_dir"])
        sam3_cam2_mask_ski_pole_dir = Path(item["sam3_cam2_mask_ski_pole_dir"])

        # Detect variants: character, pole, ski
        cam1_kpt2d_dirs_raw = item.get("cam1_kpt2d_dirs")
        kpt3d_dirs_raw = item.get("kpt3d_dirs")
        cam1_kpt2d_dirs: Dict[str, str] = (
            dict(cam1_kpt2d_dirs_raw) if isinstance(cam1_kpt2d_dirs_raw, dict) else {}
        )
        kpt3d_dirs: Dict[str, str] = (
            dict(kpt3d_dirs_raw) if isinstance(kpt3d_dirs_raw, dict) else {}
        )

        has_variants = bool(cam1_kpt2d_dirs) and bool(kpt3d_dirs)
        variants = (
            sorted(set(cam1_kpt2d_dirs.keys()) & set(kpt3d_dirs.keys()))
            if has_variants
            else ["default"]
        )
        if has_variants and not variants:
            raise RuntimeError(
                "Variant dirs mismatch: cam1_kpt2d_dirs and kpt3d_dirs have no common variant keys."
            )
        sam_variant_key = "character" if "character" in variants else variants[0]

        cam1_frames_map = (
            self._build_idx_file_map(cam1_frames_dir, "frame_*.png")
            if self._load_frames
            else {}
        )

        # none_detected_frames.txt is copied to both kpt2d and kpt3d dirs by the export script
        cam1_none_dir = (
            sam3d_cam1_kpt2d_dir if self._load_2d_kpt else sam3d_cam1_kpt3d_dir
        )
        cam1_none_exists, cam1_none_idx, cam1_none_invalid = (
            self._read_none_detected_indices(cam1_none_dir)
        )
        if cam1_none_exists and cam1_none_invalid:
            logger.warning(
                "Invalid lines in none_detected_frames.txt for camera %s: %s",
                item.get("cam1_id", "unknown"),
                cam1_none_invalid[:5],
            )

        # 单目：跳过 cam1 没有 SAM 检测结果的帧
        cam1_none_set = set(cam1_none_idx)
        all_common_set: Optional[set[int]] = None
        if self._load_frames:
            all_common_set = set(cam1_frames_map)
        if self._load_2d_kpt:
            # Use default (character) variant for frame discovery
            cam1_kpt2d_map = self._build_idx_file_map(
                Path(cam1_kpt2d_dirs[variants[0]]) if has_variants else cam1_kpt2d_dir,
                "kpt2d_*.npy",
            )
            cur = set(cam1_kpt2d_map)
            all_common_set = cur if all_common_set is None else all_common_set & cur
        if self._load_3d_kpt:
            kpt3d_map = self._build_idx_file_map(
                Path(kpt3d_dirs[variants[0]]) if has_variants else kpt3d_dir,
                "frame_*.npy",
            )
            cur = set(kpt3d_map)
            all_common_set = cur if all_common_set is None else all_common_set & cur
        if all_common_set is None:
            raise RuntimeError("No modality selected for aligned frame discovery.")
        all_common = sorted(all_common_set)
        sam_valid_set: Optional[set[int]] = None
        sam3d_cam1_kpt2d_map: Dict[int, Path] = {}
        sam3d_cam1_kpt3d_map: Dict[int, Path] = {}
        if self._load_2d_kpt:
            sam3d_cam1_kpt2d_map = self._build_idx_file_map(
                sam3d_cam1_kpt2d_dir, "kpt2d_*.npy"
            )
            sam_valid_set = (
                set(sam3d_cam1_kpt2d_map)
                if sam_valid_set is None
                else sam_valid_set & set(sam3d_cam1_kpt2d_map)
            )
        if self._load_3d_kpt:
            sam3d_cam1_kpt3d_map = self._build_idx_file_map(
                sam3d_cam1_kpt3d_dir, "kpt3d_*.npy"
            )
            sam_valid_set = (
                set(sam3d_cam1_kpt3d_map)
                if sam_valid_set is None
                else sam_valid_set & set(sam3d_cam1_kpt3d_map)
            )
        sam_valid_set = sam_valid_set or set()
        mask_ski_map: Dict[int, Path] = {}
        mask_ski_pole_map: Dict[int, Path] = {}
        mask_valid_set: Optional[set[int]] = None
        if self._load_mask:
            mask_ski_map = self._build_idx_file_map(sam3_cam1_mask_ski_dir, "*.npz")
            mask_ski_pole_map = self._build_idx_file_map(
                sam3_cam2_mask_ski_pole_dir, "*.npz"
            )
            mask_valid_set = set(mask_ski_map) & set(mask_ski_pole_map)

        common_idx = [
            idx
            for idx in all_common
            if idx in sam_valid_set
            and idx not in cam1_none_set
            and (
                not self._load_mask
                or (mask_valid_set is not None and idx in mask_valid_set)
            )
        ]
        skipped = len(all_common) - len(common_idx)
        if skipped:
            logger.debug(
                "Skipped %d/%d frames with missing SAM data for %s/%s/%s-%s",
                skipped,
                len(all_common),
                item.get("person_id", "?"),
                item.get("action_id", "?"),
                item.get("cam1_id", "?"),
                "single_view",
            )
        if not common_idx:
            missing_sam_idx = [idx for idx in all_common if idx not in sam_valid_set]
            cam1_none_blocked = [idx for idx in all_common if idx in cam1_none_set]

            logger.error(
                "No valid frames after filtering for sample %s/%s/%s. "
                "Counts: all_common=%d, sam_valid=%d, cam1_none=%d, missing_sam=%d. "
                "Sample blocked idx (first 20): missing_sam=%s, cam1_none=%s",
                item.get("person_id", "unknown"),
                item.get("action_id", "unknown"),
                item.get("cam1_id", "unknown"),
                len(all_common),
                len(sam_valid_set),
                len(cam1_none_set),
                len(missing_sam_idx),
                missing_sam_idx[:20],
                cam1_none_blocked[:20],
            )

            logger.error(
                "none_detected path: cam1=%s",
                str((cam1_none_dir / "none_detected_frames.txt").resolve()),
            )

            if self._load_mask:
                missing_mask_ski = [
                    idx for idx in all_common if idx not in mask_ski_map
                ]
                missing_mask_ski_pole = [
                    idx for idx in all_common if idx not in mask_ski_pole_map
                ]
                logger.error(
                    "Missing mask frames for %s/%s/%s. ski(first 20)=%s, ski_pole(first 20)=%s",
                    item.get("person_id", "unknown"),
                    item.get("action_id", "unknown"),
                    item.get("cam1_id", "unknown"),
                    missing_mask_ski[:20],
                    missing_mask_ski_pole[:20],
                )

            if self._load_2d_kpt:
                cam1_missing_2d = [
                    idx for idx in all_common if idx not in sam3d_cam1_kpt2d_map
                ]
                self._log_missing_sam_paths(
                    camera_id=item.get("cam1_id", "unknown"),
                    sam_dir=sam3d_cam1_kpt2d_dir,
                    missing_indices=cam1_missing_2d,
                )

            if self._load_3d_kpt:
                cam1_missing_3d = [
                    idx for idx in all_common if idx not in sam3d_cam1_kpt3d_map
                ]
                self._log_missing_sam_paths(
                    camera_id=item.get("cam1_id", "unknown"),
                    sam_dir=sam3d_cam1_kpt3d_dir,
                    missing_indices=cam1_missing_3d,
                )

            raise RuntimeError(
                "No valid frames with SAM data for sample: "
                f"{item.get('person_id', 'unknown')} / {item.get('action_id', 'unknown')} / "
                f"{item.get('cam1_id', 'unknown')}"
            )

        source_t = len(common_idx)
        target_t = self._target_t if self._target_t is not None else source_t
        if target_t <= 0:
            raise RuntimeError("Resolved target_t must be > 0")

        # Build one shared temporal selection first, then only read selected files.
        temporal_sel = self._temporal_resample_indices(source_t, target_t)
        selected_common_idx = [common_idx[int(i)] for i in temporal_sel.tolist()]

        cam1_frames: List[torch.Tensor] = []
        for idx in selected_common_idx:
            if self._load_frames:
                img1 = cv2.imread(str(cam1_frames_map[idx]), cv2.IMREAD_COLOR)
                if img1 is None:
                    raise RuntimeError(f"Failed to read aligned frame at idx={idx}")

                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                cam1_frames.append(
                    torch.from_numpy(np.ascontiguousarray(img1)).permute(2, 0, 1)
                )

        cam1_frames_t: Optional[torch.Tensor] = None
        if self._load_frames:
            cam1_frames_t = torch.stack(cam1_frames, dim=0)
            # apply transform to single view
            cam1_frames_t = self._apply_transform(cam1_frames_t)

            # Single sample frame layout: (C,T,H,W); DataLoader adds batch dim.
            cam1_frames_t = cam1_frames_t.permute(1, 0, 2, 3)

        mask_ski_t: Optional[torch.Tensor] = None
        mask_ski_pole_t: Optional[torch.Tensor] = None
        if self._load_mask:
            mask_ski = [
                torch.from_numpy(self._load_mask_file(mask_ski_map[idx]))
                for idx in selected_common_idx
            ]
            mask_ski_pole = [
                torch.from_numpy(self._load_mask_file(mask_ski_pole_map[idx]))
                for idx in selected_common_idx
            ]

            # Masks may come with mixed source resolutions (e.g. 288x288 and 1080x1920).
            # Unify H/W per sample before stacking.
            if self._load_frames and cam1_frames_t is not None:
                target_h = int(cam1_frames_t.shape[2])
                target_w = int(cam1_frames_t.shape[3])
            elif len(mask_ski) > 0:
                target_h = int(mask_ski[0].shape[-2])
                target_w = int(mask_ski[0].shape[-1])
            elif len(mask_ski_pole) > 0:
                target_h = int(mask_ski_pole[0].shape[-2])
                target_w = int(mask_ski_pole[0].shape[-1])
            else:
                raise RuntimeError("Mask loading enabled but no mask frames found.")

            def _resize_mask_list(mask_list: List[torch.Tensor]) -> List[torch.Tensor]:
                out_masks: List[torch.Tensor] = []
                for m in mask_list:
                    if int(m.shape[-2]) != target_h or int(m.shape[-1]) != target_w:
                        # m: (1,H,W) -> interpolate expects (N,C,H,W)
                        m = (
                            F.interpolate(
                                m.unsqueeze(0).to(torch.float32),
                                size=(target_h, target_w),
                                mode="nearest",
                            )
                            .squeeze(0)
                            .gt(0.5)
                        )
                    out_masks.append(m)
                return out_masks

            mask_ski = _resize_mask_list(mask_ski)
            mask_ski_pole = _resize_mask_list(mask_ski_pole)

            mask_ski_t = torch.stack(mask_ski, dim=0)  # t, v, h, w
            mask_ski_pole_t = torch.stack(mask_ski_pole, dim=0)  #

            mask_ski_t = mask_ski_t.to(torch.float32).permute(1, 0, 2, 3)
            mask_ski_pole_t = mask_ski_pole_t.to(torch.float32).permute(1, 0, 2, 3)

        # Load keypoints for all variants
        variant_kpts: Dict[str, Dict[str, Any]] = {}
        frame_indices_t = torch.empty(0, dtype=torch.long)
        for variant in variants:
            if has_variants:
                cam1_kpt2d_dir_variant = Path(cam1_kpt2d_dirs[variant])
                kpt3d_dir_variant = Path(kpt3d_dirs[variant])
            else:
                cam1_kpt2d_dir_variant = cam1_kpt2d_dir
                kpt3d_dir_variant = kpt3d_dir

            (
                unity_gt_cam1_kpt2d_t,
                unity_gt_kpt3d_t,
                sam3d_cam1_kpt2d_t,
                sam3d_cam1_kpt3d_t,
                frame_indices_t,
            ) = self._load_single_variant_keypoints(
                variant=variant,
                cam1_kpt2d_dir=cam1_kpt2d_dir_variant,
                kpt3d_dir=kpt3d_dir_variant,
                sam3d_cam1_kpt2d_dir=sam3d_cam1_kpt2d_dir,
                sam3d_cam1_kpt3d_dir=sam3d_cam1_kpt3d_dir,
                common_idx=selected_common_idx,
                sam3d_cam1_kpt2d_map=sam3d_cam1_kpt2d_map,
                sam3d_cam1_kpt3d_map=sam3d_cam1_kpt3d_map,
            )

            variant_kpts[variant] = {
                "unity_gt_cam1_kpt2d": unity_gt_cam1_kpt2d_t,
                "unity_gt_kpt3d": unity_gt_kpt3d_t,
                "sam3d_cam1_kpt2d": sam3d_cam1_kpt2d_t,
                "sam3d_cam1_kpt3d": sam3d_cam1_kpt3d_t,
            }

        out: Dict[str, Any] = {
            "frame_indices": frame_indices_t,
            "meta": {
                "experiment": self._experiment,
                "person_id": item.get("person_id", "unknown"),
                "action_id": item.get("action_id", "unknown"),
                "cam1_id": item.get("cam1_id", "unknown"),
                "num_aligned_frames": int(frame_indices_t.numel()),
            },
        }

        if self._load_frames and cam1_frames_t is not None:
            out["frames"] = {
                "cam1": cam1_frames_t,
            }

        if self._load_mask and mask_ski_t is not None and mask_ski_pole_t is not None:
            out["masks"] = {
                "ski": mask_ski_t,
                "ski_pole": mask_ski_pole_t,
            }

        # Add keypoint data organized by variant
        if self._load_2d_kpt:
            out["kpt2d_gt"] = {}
            out["kpt2d_sam"] = {}

            # GT: all variants (character, pole, ski)
            for v in variants:
                if variant_kpts[v]["unity_gt_cam1_kpt2d"] is not None:
                    out["kpt2d_gt"][f"{v}_cam1"] = variant_kpts[v][
                        "unity_gt_cam1_kpt2d"
                    ]

            # SAM: character only (fallback to first variant when character is unavailable)
            if variant_kpts[sam_variant_key]["sam3d_cam1_kpt2d"] is not None:
                out["kpt2d_sam"][f"{sam_variant_key}_cam1"] = variant_kpts[
                    sam_variant_key
                ]["sam3d_cam1_kpt2d"]

        if self._load_3d_kpt:
            out["kpt3d_gt"] = {}
            out["kpt3d_sam"] = {}

            # GT: all variants (character, pole, ski)
            for v in variants:
                if variant_kpts[v]["unity_gt_kpt3d"] is not None:
                    out["kpt3d_gt"][v] = variant_kpts[v]["unity_gt_kpt3d"]

            # SAM: character only
            if variant_kpts[sam_variant_key]["sam3d_cam1_kpt3d"] is not None:
                out["kpt3d_sam"][f"{sam_variant_key}_cam1"] = variant_kpts[
                    sam_variant_key
                ]["sam3d_cam1_kpt3d"]

        return out

    def _apply_transform(self, video_tchw: torch.Tensor) -> torch.Tensor:
        """
        Apply transform on a segment.

        Expect transform: (T,C,H,W) -> (T,C,H,W) or compatible.
        """
        if self._transform is None:
            return video_tchw
        return self._transform(video_tchw)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw_item = self._index_mapping[index]
        item = self._normalize_item_dict(raw_item)

        # ---------------- single-view frame-dir format ----------------
        if "cam1_frames_dir" in item:
            out = self._load_single_view_modalities(item)
            out["meta"]["index"] = index
            return out

        raise ValueError(
            "This dataset currently expects single-view index items with "
            "cam1_frames_dir and corresponding kpt/sam paths."
        )


def whole_video_dataset(
    experiment: str,
    dataset_idx: List[UnityDataConfig],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    load_frames: bool = True,
    load_2d_kpt: bool = True,
    load_3d_kpt: bool = True,
    load_mask: bool = True,
    target_t: Optional[int] = None,
) -> LabeledUnityDataset:
    return LabeledUnityDataset(
        experiment=experiment,
        transform=transform,
        index_mapping=dataset_idx,
        load_frames=load_frames,
        load_2d_kpt=load_2d_kpt,
        load_3d_kpt=load_3d_kpt,
        load_mask=load_mask,
        target_t=target_t,
    )
