#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/dataloader/whole_video_dataset_dual_view.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/dataloader
Created Date: Tuesday February 10th 2026
Author: Kaixu Chen
-----
Comment:
TODO: 加载时间信息的时候，需要对每个数据段的T进行统一
例如，进来的是C, T, h, w，
因为每个数据段的T是不一样的，没有办法叠起来。
需要上采样/下采样把时间统一起来。
Have a good code time :)
-----
Last Modified: Tuesday February 10th 2026 12:11:11 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

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

from project.map_config import ID_TO_INDEX, TARGET_IDS, UnityDataConfig

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
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = index_mapping
        self._transform = transform
        self._load_frames = bool(load_frames)
        self._load_2d_kpt = bool(load_2d_kpt)
        self._load_3d_kpt = bool(load_3d_kpt)
        self._load_mask = bool(load_mask)
        if (
            not self._load_frames
            and not self._load_2d_kpt
            and not self._load_3d_kpt
            and not self._load_mask
        ):
            raise ValueError(
                "At least one of load_frames/load_2d_kpt/load_3d_kpt/load_mask must be enabled."
            )
        self._source_index_cache: Dict[int, List[int]] = {}

    def __len__(self) -> int:
        return len(self._index_mapping)

    @staticmethod
    def _ordered_target_ids() -> List[int]:
        # Keep target joint order consistent with ID_TO_INDEX.
        return [jid for jid, _ in sorted(ID_TO_INDEX.items(), key=lambda kv: kv[1])]

    @classmethod
    def _select_source_joint_indices(cls, num_joints: int) -> List[int]:
        """Map configured target ids to source array indices.

        Priority:
          1) direct index (jid)
          2) compact target index from ID_TO_INDEX
        """
        selected: List[int] = []
        for jid in cls._ordered_target_ids():
            candidates = (jid, ID_TO_INDEX[jid])
            src_idx = next((c for c in candidates if 0 <= c < num_joints), None)
            if src_idx is None:
                raise IndexError(
                    f"Target joint id {jid} cannot be mapped for source joint count {num_joints}."
                )
            selected.append(int(src_idx))
        return selected

    def _get_or_build_source_joint_indices(self, num_joints: int) -> List[int]:
        cached = self._source_index_cache.get(num_joints)
        if cached is not None:
            return cached
        selected = self._select_source_joint_indices(num_joints)
        self._source_index_cache[num_joints] = selected
        return selected

    @staticmethod
    def _filter_keypoints_with_indices(
        arr: np.ndarray, source_indices: List[int]
    ) -> np.ndarray:
        """Fast-path keypoint filtering using precomputed source indices."""
        kpt = np.asarray(arr, dtype=np.float32)
        if kpt.ndim == 3 and kpt.shape[0] == 1:
            kpt = kpt[0]
        if kpt.ndim != 2:
            raise ValueError(f"Expected keypoints shape (J,C), got {kpt.shape}")
        if max(source_indices) >= kpt.shape[0]:
            raise IndexError(
                f"Source index out of range for shape {kpt.shape} and indices up to {max(source_indices)}"
            )
        return kpt[source_indices]

    @classmethod
    def _filter_keypoints_by_target_ids(cls, arr: np.ndarray) -> np.ndarray:
        """Filter keypoints to configured TARGET_IDS in a fixed order."""
        kpt = np.asarray(arr, dtype=np.float32)
        if kpt.ndim == 3 and kpt.shape[0] == 1:
            kpt = kpt[0]
        if kpt.ndim != 2:
            raise ValueError(f"Expected keypoints shape (J,C), got {kpt.shape}")

        # Ensure mapping uses the configured target ids.
        if len(TARGET_IDS) != len(ID_TO_INDEX):
            raise ValueError("TARGET_IDS and ID_TO_INDEX are inconsistent in size.")

        source_indices = cls._select_source_joint_indices(kpt.shape[0])
        return cls._filter_keypoints_with_indices(kpt, source_indices)

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
                sam2d, sam3d = np.empty((0, 0), dtype=np.float32), np.empty(
                    (0, 0), dtype=np.float32
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

        rest_shape = x.shape[1:]
        x_flat = x.reshape(src_t, -1).transpose(0, 1).unsqueeze(0)
        y_flat = F.interpolate(
            x_flat,
            size=target_t,
            mode="linear",
            align_corners=False,
        )
        y = y_flat.squeeze(0).transpose(0, 1).reshape((target_t,) + rest_shape)
        return y.movedim(0, time_dim)

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

    def _validate_stereo_pair_consistency(
        self,
        item: Dict[str, Any],
        cam1_frames_t: Optional[torch.Tensor],
        cam2_frames_t: Optional[torch.Tensor],
        cam1_kpt2d_t: Optional[torch.Tensor],
        cam2_kpt2d_t: Optional[torch.Tensor],
        sam2d_cam1_t: Optional[torch.Tensor],
        sam2d_cam2_t: Optional[torch.Tensor],
        sam3d_cam1_t: Optional[torch.Tensor],
        sam3d_cam2_t: Optional[torch.Tensor],
        mask_ski_t: Optional[torch.Tensor],
        mask_ski_pole_t: Optional[torch.Tensor],
        frame_indices_t: torch.Tensor,
    ) -> None:
        """Validate left/right camera shape alignment and camera IDE consistency."""
        if (
            cam1_frames_t is not None
            and cam2_frames_t is not None
            and cam1_frames_t.shape != cam2_frames_t.shape
        ):
            raise ValueError(
                f"Left/right frame shape mismatch: {tuple(cam1_frames_t.shape)} vs {tuple(cam2_frames_t.shape)}"
            )

        if (
            cam1_kpt2d_t is not None
            and cam2_kpt2d_t is not None
            and cam1_kpt2d_t.shape != cam2_kpt2d_t.shape
        ):
            raise ValueError(
                f"Left/right GT 2D shape mismatch: {tuple(cam1_kpt2d_t.shape)} vs {tuple(cam2_kpt2d_t.shape)}"
            )

        if (
            sam2d_cam1_t is not None
            and sam2d_cam2_t is not None
            and sam2d_cam1_t.shape != sam2d_cam2_t.shape
        ):
            raise ValueError(
                f"Left/right SAM 2D shape mismatch: {tuple(sam2d_cam1_t.shape)} vs {tuple(sam2d_cam2_t.shape)}"
            )

        if (
            sam3d_cam1_t is not None
            and sam3d_cam2_t is not None
            and sam3d_cam1_t.shape != sam3d_cam2_t.shape
        ):
            raise ValueError(
                f"Left/right SAM 3D shape mismatch: {tuple(sam3d_cam1_t.shape)} vs {tuple(sam3d_cam2_t.shape)}"
            )

        t_ref = int(frame_indices_t.numel())
        if cam1_frames_t is not None:
            t_frames = int(cam1_frames_t.shape[1])
            if frame_indices_t.numel() != t_frames:
                raise ValueError(
                    f"frame_indices length {int(frame_indices_t.numel())} != frame T {t_frames}"
                )
            t_ref = t_frames

        if cam1_kpt2d_t is not None and int(cam1_kpt2d_t.shape[0]) != t_ref:
            raise ValueError(f"GT 2D T {int(cam1_kpt2d_t.shape[0])} != ref T {t_ref}")
        if sam2d_cam1_t is not None and int(sam2d_cam1_t.shape[0]) != t_ref:
            raise ValueError(f"SAM 2D T {int(sam2d_cam1_t.shape[0])} != ref T {t_ref}")
        if sam3d_cam1_t is not None and int(sam3d_cam1_t.shape[0]) != t_ref:
            raise ValueError(f"SAM 3D T {int(sam3d_cam1_t.shape[0])} != ref T {t_ref}")
        if mask_ski_t is not None:
            if mask_ski_t.ndim != 4 or int(mask_ski_t.shape[0]) != 1:
                raise ValueError(
                    f"mask ski shape must be (1,T,H,W), got {tuple(mask_ski_t.shape)}"
                )
            if int(mask_ski_t.shape[1]) != t_ref:
                raise ValueError(
                    f"mask ski T {int(mask_ski_t.shape[1])} != ref T {t_ref}"
                )
        if mask_ski_pole_t is not None:
            if mask_ski_pole_t.ndim != 4 or int(mask_ski_pole_t.shape[0]) != 1:
                raise ValueError(
                    "mask ski_pole shape must be (1,T,H,W), got "
                    f"{tuple(mask_ski_pole_t.shape)}"
                )
            if int(mask_ski_pole_t.shape[1]) != t_ref:
                raise ValueError(
                    f"mask ski_pole T {int(mask_ski_pole_t.shape[1])} != ref T {t_ref}"
                )

        cam1_id = item.get("cam1_id", "")
        cam2_id = item.get("cam2_id", "")
        if not cam1_id or not cam2_id:
            raise ValueError(
                f"Missing camera id(s): cam1_id={cam1_id!r}, cam2_id={cam2_id!r}"
            )
        if str(cam1_id) == str(cam2_id):
            raise ValueError(f"cam1_id and cam2_id must be different, got {cam1_id}")

        cam1_ide = self._extract_cam_ide_token(cam1_id)
        cam2_ide = self._extract_cam_ide_token(cam2_id)
        cam1_layer = self._extract_cam_layer_token(cam1_id)
        cam2_layer = self._extract_cam_layer_token(cam2_id)
        if cam1_ide == cam2_ide and cam1_layer == cam2_layer:
            raise ValueError(
                "Invalid camera pair: same layer and IDE token are not allowed, got "
                f"{cam1_id} ({cam1_layer}, {cam1_ide}) vs {cam2_id} ({cam2_layer}, {cam2_ide})"
            )

    def _load_single_variant_keypoints(
        self,
        cam1_kpt2d_dir: Path,
        cam2_kpt2d_dir: Path,
        kpt3d_dir: Path,
        sam3d_cam1_kpt2d_dir: Path,
        sam3d_cam2_kpt2d_dir: Path,
        sam3d_cam1_kpt3d_dir: Path,
        sam3d_cam2_kpt3d_dir: Path,
        common_idx: List[int],
        apply_filtering: bool = True,
        sam3d_cam1_kpt2d_map: Optional[Dict[int, Path]] = None,
        sam3d_cam2_kpt2d_map: Optional[Dict[int, Path]] = None,
        sam3d_cam1_kpt3d_map: Optional[Dict[int, Path]] = None,
        sam3d_cam2_kpt3d_map: Optional[Dict[int, Path]] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Load keypoints for a single variant.

        Args:
            apply_filtering: If True, filter keypoints using TARGET_IDS. If False, keep all joints.
            sam3d_cam1_kpt2d_map: Pre-built SAM cam1 2D map (reused across variants).
            sam3d_cam2_kpt2d_map: Pre-built SAM cam2 2D map (reused across variants).
            sam3d_cam1_kpt3d_map: Pre-built SAM cam1 3D map (reused across variants).
            sam3d_cam2_kpt3d_map: Pre-built SAM cam2 3D map (reused across variants).

        Returns:
            (cam1_kpt2d_t, cam2_kpt2d_t, gt_kpt3d_t, sam2d_cam1_t, sam2d_cam2_t, sam3d_cam1_t, sam3d_cam2_t, frame_indices_t)
        """
        cam1_kpt2d_map = (
            self._build_idx_file_map(cam1_kpt2d_dir, "kpt2d_*.npy")
            if self._load_2d_kpt
            else {}
        )
        cam2_kpt2d_map = (
            self._build_idx_file_map(cam2_kpt2d_dir, "kpt2d_*.npy")
            if self._load_2d_kpt
            else {}
        )
        kpt3d_map = (
            self._build_idx_file_map(kpt3d_dir, "frame_*.npy")
            if self._load_3d_kpt
            else {}
        )
        if sam3d_cam1_kpt2d_map is None:
            sam3d_cam1_kpt2d_map = (
                self._build_idx_file_map(sam3d_cam1_kpt2d_dir, "kpt2d_*.npy")
                if self._load_2d_kpt
                else {}
            )
        if sam3d_cam2_kpt2d_map is None:
            sam3d_cam2_kpt2d_map = (
                self._build_idx_file_map(sam3d_cam2_kpt2d_dir, "kpt2d_*.npy")
                if self._load_2d_kpt
                else {}
            )
        if sam3d_cam1_kpt3d_map is None:
            sam3d_cam1_kpt3d_map = (
                self._build_idx_file_map(sam3d_cam1_kpt3d_dir, "kpt3d_*.npy")
                if self._load_3d_kpt
                else {}
            )
        if sam3d_cam2_kpt3d_map is None:
            sam3d_cam2_kpt3d_map = (
                self._build_idx_file_map(sam3d_cam2_kpt3d_dir, "kpt3d_*.npy")
                if self._load_3d_kpt
                else {}
            )

        cam1_kpt2d: List[torch.Tensor] = []
        cam2_kpt2d: List[torch.Tensor] = []
        gt_kpt3d: List[torch.Tensor] = []
        sam2d_cam1: List[torch.Tensor] = []
        sam2d_cam2: List[torch.Tensor] = []
        sam3d_cam1: List[torch.Tensor] = []
        sam3d_cam2: List[torch.Tensor] = []

        cam1_2d_sel: Optional[List[int]] = None
        cam2_2d_sel: Optional[List[int]] = None
        gt_3d_sel: Optional[List[int]] = None
        sam1_2d_sel: Optional[List[int]] = None
        sam2_2d_sel: Optional[List[int]] = None
        sam1_3d_sel: Optional[List[int]] = None
        sam2_3d_sel: Optional[List[int]] = None

        for idx in common_idx:
            if self._load_2d_kpt:
                cam1_2d = np.asarray(np.load(cam1_kpt2d_map[idx]), dtype=np.float32)
                cam2_2d = np.asarray(np.load(cam2_kpt2d_map[idx]), dtype=np.float32)

                if apply_filtering:
                    if cam1_2d_sel is None:
                        cam1_2d_sel = self._get_or_build_source_joint_indices(
                            cam1_2d.shape[0]
                        )
                    if cam2_2d_sel is None:
                        cam2_2d_sel = self._get_or_build_source_joint_indices(
                            cam2_2d.shape[0]
                        )
                    cam1_kpt2d.append(
                        torch.from_numpy(
                            self._filter_keypoints_with_indices(cam1_2d, cam1_2d_sel)
                        )
                    )
                    cam2_kpt2d.append(
                        torch.from_numpy(
                            self._filter_keypoints_with_indices(cam2_2d, cam2_2d_sel)
                        )
                    )
                else:
                    # No filtering: keep all joints as-is
                    cam1_kpt2d.append(torch.from_numpy(cam1_2d))
                    cam2_kpt2d.append(torch.from_numpy(cam2_2d))

            if self._load_3d_kpt:
                gt_3d = np.asarray(np.load(kpt3d_map[idx]), dtype=np.float32)

                if apply_filtering:
                    if gt_3d_sel is None:
                        gt_3d_sel = self._get_or_build_source_joint_indices(
                            gt_3d.shape[0]
                        )
                    gt_kpt3d.append(
                        torch.from_numpy(
                            self._filter_keypoints_with_indices(gt_3d, gt_3d_sel)
                        )
                    )
                else:
                    # No filtering: keep all joints as-is
                    gt_kpt3d.append(torch.from_numpy(gt_3d))

            if self._load_2d_kpt:
                sam1_2d = np.asarray(
                    np.load(sam3d_cam1_kpt2d_map[idx]), dtype=np.float32
                )
                sam2_2d = np.asarray(
                    np.load(sam3d_cam2_kpt2d_map[idx]), dtype=np.float32
                )

                if apply_filtering:
                    if sam1_2d_sel is None:
                        sam1_2d_sel = self._get_or_build_source_joint_indices(
                            sam1_2d.shape[0]
                        )
                    if sam2_2d_sel is None:
                        sam2_2d_sel = self._get_or_build_source_joint_indices(
                            sam2_2d.shape[0]
                        )
                    sam2d_cam1.append(
                        torch.from_numpy(
                            self._filter_keypoints_with_indices(sam1_2d, sam1_2d_sel)
                        )
                    )
                    sam2d_cam2.append(
                        torch.from_numpy(
                            self._filter_keypoints_with_indices(sam2_2d, sam2_2d_sel)
                        )
                    )
                else:
                    # No filtering: keep all joints as-is
                    sam2d_cam1.append(torch.from_numpy(sam1_2d))
                    sam2d_cam2.append(torch.from_numpy(sam2_2d))

            if self._load_3d_kpt:
                sam1_3d = np.asarray(
                    np.load(sam3d_cam1_kpt3d_map[idx]), dtype=np.float32
                )
                sam2_3d = np.asarray(
                    np.load(sam3d_cam2_kpt3d_map[idx]), dtype=np.float32
                )

                if apply_filtering:
                    if sam1_3d_sel is None:
                        sam1_3d_sel = self._get_or_build_source_joint_indices(
                            sam1_3d.shape[0]
                        )
                    if sam2_3d_sel is None:
                        sam2_3d_sel = self._get_or_build_source_joint_indices(
                            sam2_3d.shape[0]
                        )
                    sam3d_cam1.append(
                        torch.from_numpy(
                            self._filter_keypoints_with_indices(sam1_3d, sam1_3d_sel)
                        )
                    )
                    sam3d_cam2.append(
                        torch.from_numpy(
                            self._filter_keypoints_with_indices(sam2_3d, sam2_3d_sel)
                        )
                    )
                else:
                    # No filtering: keep all joints as-is
                    sam3d_cam1.append(torch.from_numpy(sam1_3d))
                    sam3d_cam2.append(torch.from_numpy(sam2_3d))

        cam1_kpt2d_t = (
            torch.stack(cam1_kpt2d, dim=0) if self._load_2d_kpt else None
        )
        cam2_kpt2d_t = (
            torch.stack(cam2_kpt2d, dim=0) if self._load_2d_kpt else None
        )
        gt_kpt3d_t = torch.stack(gt_kpt3d, dim=0) if self._load_3d_kpt else None
        sam2d_cam1_t = (
            torch.stack(sam2d_cam1, dim=0) if self._load_2d_kpt else None
        )
        sam2d_cam2_t = (
            torch.stack(sam2d_cam2, dim=0) if self._load_2d_kpt else None
        )
        sam3d_cam1_t = (
            torch.stack(sam3d_cam1, dim=0) if self._load_3d_kpt else None
        )
        sam3d_cam2_t = (
            torch.stack(sam3d_cam2, dim=0) if self._load_3d_kpt else None
        )
        frame_indices_t = torch.tensor(common_idx, dtype=torch.long)

        return (
            cam1_kpt2d_t,
            cam2_kpt2d_t,
            gt_kpt3d_t,
            sam2d_cam1_t,
            sam2d_cam2_t,
            sam3d_cam1_t,
            sam3d_cam2_t,
            frame_indices_t,
        )

    def _load_pair_modalities(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Load aligned modalities for one camera-pair sample.

        Modalities:
          - cam1/cam2 frames
          - cam1/cam2 2D kpt
          - GT 3D kpt
          - SAM3D pred 3D kpt for cam1/cam2

        Supports loading multiple variants (character, pole, ski) if variant dicts are available.
        """
        cam1_frames_dir = Path(item["cam1_frames_dir"])
        cam2_frames_dir = Path(item["cam2_frames_dir"])
        cam1_kpt2d_dir = Path(item["cam1_kpt2d_dir"])
        cam2_kpt2d_dir = Path(item["cam2_kpt2d_dir"])
        kpt3d_dir = Path(item["kpt3d_dir"])
        sam3d_cam1_kpt2d_dir = Path(item["sam3d_cam1_kpt2d_dir"])
        sam3d_cam2_kpt2d_dir = Path(item["sam3d_cam2_kpt2d_dir"])
        sam3d_cam1_kpt3d_dir = Path(item["sam3d_cam1_kpt3d_dir"])
        sam3d_cam2_kpt3d_dir = Path(item["sam3d_cam2_kpt3d_dir"])
        sam3_cam1_mask_ski_dir = Path(item["sam3_cam1_mask_ski_dir"])
        sam3_cam2_mask_ski_pole_dir = Path(item["sam3_cam2_mask_ski_pole_dir"])

        # Detect variants: character, pole, ski
        cam1_kpt2d_dirs = item.get("cam1_kpt2d_dirs")  # Dict[str, str]
        cam2_kpt2d_dirs = item.get("cam2_kpt2d_dirs")  # Dict[str, str]
        kpt3d_dirs = item.get("kpt3d_dirs")  # Dict[str, str]

        has_variants = (
            cam1_kpt2d_dirs is not None
            and cam2_kpt2d_dirs is not None
            and kpt3d_dirs is not None
        )

        variants = list(cam1_kpt2d_dirs.keys()) if has_variants else ["default"]

        cam1_frames_map = (
            self._build_idx_file_map(cam1_frames_dir, "frame_*.png")
            if self._load_frames
            else {}
        )
        cam2_frames_map = (
            self._build_idx_file_map(cam2_frames_dir, "frame_*.png")
            if self._load_frames
            else {}
        )

        # none_detected_frames.txt is copied to both kpt2d and kpt3d dirs by the export script
        cam1_none_dir = (
            sam3d_cam1_kpt2d_dir if self._load_2d_kpt else sam3d_cam1_kpt3d_dir
        )
        cam2_none_dir = (
            sam3d_cam2_kpt2d_dir if self._load_2d_kpt else sam3d_cam2_kpt3d_dir
        )
        cam1_none_exists, cam1_none_idx, cam1_none_invalid = (
            self._read_none_detected_indices(cam1_none_dir)
        )
        cam2_none_exists, cam2_none_idx, cam2_none_invalid = (
            self._read_none_detected_indices(cam2_none_dir)
        )
        if cam1_none_exists and cam1_none_invalid:
            logger.warning(
                "Invalid lines in none_detected_frames.txt for camera %s: %s",
                item.get("cam1_id", "unknown"),
                cam1_none_invalid[:5],
            )
        if cam2_none_exists and cam2_none_invalid:
            logger.warning(
                "Invalid lines in none_detected_frames.txt for camera %s: %s",
                item.get("cam2_id", "unknown"),
                cam2_none_invalid[:5],
            )

        # 跳过两摄像头中任意一个没有 SAM 检测结果的帧
        cam1_none_set = set(cam1_none_idx)
        cam2_none_set = set(cam2_none_idx)
        all_common_set: Optional[set[int]] = None
        if self._load_frames:
            all_common_set = set(cam1_frames_map) & set(cam2_frames_map)
        if self._load_2d_kpt:
            # Use default (character) variant for frame discovery
            cam1_kpt2d_map = self._build_idx_file_map(
                Path(cam1_kpt2d_dirs[variants[0]]) if has_variants else cam1_kpt2d_dir,
                "kpt2d_*.npy",
            )
            cam2_kpt2d_map = self._build_idx_file_map(
                Path(cam2_kpt2d_dirs[variants[0]]) if has_variants else cam2_kpt2d_dir,
                "kpt2d_*.npy",
            )
            cur = set(cam1_kpt2d_map) & set(cam2_kpt2d_map)
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
        sam3d_cam2_kpt2d_map: Dict[int, Path] = {}
        sam3d_cam1_kpt3d_map: Dict[int, Path] = {}
        sam3d_cam2_kpt3d_map: Dict[int, Path] = {}
        if self._load_2d_kpt:
            sam3d_cam1_kpt2d_map = self._build_idx_file_map(
                sam3d_cam1_kpt2d_dir, "kpt2d_*.npy"
            )
            sam3d_cam2_kpt2d_map = self._build_idx_file_map(
                sam3d_cam2_kpt2d_dir, "kpt2d_*.npy"
            )
            both_2d = set(sam3d_cam1_kpt2d_map) & set(sam3d_cam2_kpt2d_map)
            sam_valid_set = (
                both_2d if sam_valid_set is None else sam_valid_set & both_2d
            )
        if self._load_3d_kpt:
            sam3d_cam1_kpt3d_map = self._build_idx_file_map(
                sam3d_cam1_kpt3d_dir, "kpt3d_*.npy"
            )
            sam3d_cam2_kpt3d_map = self._build_idx_file_map(
                sam3d_cam2_kpt3d_dir, "kpt3d_*.npy"
            )
            both_3d = set(sam3d_cam1_kpt3d_map) & set(sam3d_cam2_kpt3d_map)
            sam_valid_set = (
                both_3d if sam_valid_set is None else sam_valid_set & both_3d
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
            and idx not in cam2_none_set
            and (not self._load_mask or (mask_valid_set is not None and idx in mask_valid_set))
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
                item.get("cam2_id", "?"),
            )
        if not common_idx:
            missing_sam_idx = [idx for idx in all_common if idx not in sam_valid_set]
            cam1_none_blocked = [idx for idx in all_common if idx in cam1_none_set]
            cam2_none_blocked = [idx for idx in all_common if idx in cam2_none_set]

            logger.error(
                "No valid frames after filtering for sample %s/%s/%s-%s. "
                "Counts: all_common=%d, sam_valid=%d, cam1_none=%d, cam2_none=%d, missing_sam=%d. "
                "Sample blocked idx (first 20): missing_sam=%s, cam1_none=%s, cam2_none=%s",
                item.get("person_id", "unknown"),
                item.get("action_id", "unknown"),
                item.get("cam1_id", "unknown"),
                item.get("cam2_id", "unknown"),
                len(all_common),
                len(sam_valid_set),
                len(cam1_none_set),
                len(cam2_none_set),
                len(missing_sam_idx),
                missing_sam_idx[:20],
                cam1_none_blocked[:20],
                cam2_none_blocked[:20],
            )

            logger.error(
                "none_detected paths: cam1=%s, cam2=%s",
                str((cam1_none_dir / "none_detected_frames.txt").resolve()),
                str((cam2_none_dir / "none_detected_frames.txt").resolve()),
            )

            if self._load_mask:
                missing_mask_ski = [idx for idx in all_common if idx not in mask_ski_map]
                missing_mask_ski_pole = [
                    idx for idx in all_common if idx not in mask_ski_pole_map
                ]
                logger.error(
                    "Missing dual-view mask frames for %s/%s/%s-%s. ski(first 20)=%s, ski_pole(first 20)=%s",
                    item.get("person_id", "unknown"),
                    item.get("action_id", "unknown"),
                    item.get("cam1_id", "unknown"),
                    item.get("cam2_id", "unknown"),
                    missing_mask_ski[:20],
                    missing_mask_ski_pole[:20],
                )

            if self._load_2d_kpt:
                cam1_missing_2d = [
                    idx for idx in all_common if idx not in sam3d_cam1_kpt2d_map
                ]
                cam2_missing_2d = [
                    idx for idx in all_common if idx not in sam3d_cam2_kpt2d_map
                ]
                self._log_missing_sam_paths(
                    camera_id=item.get("cam1_id", "unknown"),
                    sam_dir=sam3d_cam1_kpt2d_dir,
                    missing_indices=cam1_missing_2d,
                )
                self._log_missing_sam_paths(
                    camera_id=item.get("cam2_id", "unknown"),
                    sam_dir=sam3d_cam2_kpt2d_dir,
                    missing_indices=cam2_missing_2d,
                )

            if self._load_3d_kpt:
                cam1_missing_3d = [
                    idx for idx in all_common if idx not in sam3d_cam1_kpt3d_map
                ]
                cam2_missing_3d = [
                    idx for idx in all_common if idx not in sam3d_cam2_kpt3d_map
                ]
                self._log_missing_sam_paths(
                    camera_id=item.get("cam1_id", "unknown"),
                    sam_dir=sam3d_cam1_kpt3d_dir,
                    missing_indices=cam1_missing_3d,
                )
                self._log_missing_sam_paths(
                    camera_id=item.get("cam2_id", "unknown"),
                    sam_dir=sam3d_cam2_kpt3d_dir,
                    missing_indices=cam2_missing_3d,
                )

            raise RuntimeError(
                "No valid frames with SAM data for sample: "
                f"{item.get('person_id', 'unknown')} / {item.get('action_id', 'unknown')} / "
                f"{item.get('cam1_id', 'unknown')} - {item.get('cam2_id', 'unknown')}"
            )

        cam1_frames: List[torch.Tensor] = []
        cam2_frames: List[torch.Tensor] = []

        for idx in common_idx:
            if self._load_frames:
                img1 = cv2.imread(str(cam1_frames_map[idx]), cv2.IMREAD_COLOR)
                img2 = cv2.imread(str(cam2_frames_map[idx]), cv2.IMREAD_COLOR)
                if img1 is None or img2 is None:
                    raise RuntimeError(f"Failed to read aligned frame at idx={idx}")

                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                cam1_frames.append(
                    torch.from_numpy(np.ascontiguousarray(img1)).permute(2, 0, 1)
                )
                cam2_frames.append(
                    torch.from_numpy(np.ascontiguousarray(img2)).permute(2, 0, 1)
                )

        cam1_frames_t: Optional[torch.Tensor] = None
        cam2_frames_t: Optional[torch.Tensor] = None
        mask_ski_t: Optional[torch.Tensor] = None
        mask_ski_pole_t: Optional[torch.Tensor] = None
        if self._load_frames:
            cam1_frames_t = torch.stack(cam1_frames, dim=0)
            cam2_frames_t = torch.stack(cam2_frames, dim=0)

            # apply same image transform to both views
            cam1_frames_t = self._apply_transform(cam1_frames_t)
            cam2_frames_t = self._apply_transform(cam2_frames_t)

            # Single-sample frame layout: (C,T,H,W); DataLoader adds batch dim.
            cam1_frames_t = cam1_frames_t.permute(1, 0, 2, 3)
            cam2_frames_t = cam2_frames_t.permute(1, 0, 2, 3)
            t_after = int(cam1_frames_t.shape[1])
        else:
            t_after = len(common_idx)

        if self._load_mask:
            mask_ski = [
                torch.from_numpy(self._load_mask_file(mask_ski_map[idx]))
                for idx in common_idx
            ]
            mask_ski_pole = [
                torch.from_numpy(self._load_mask_file(mask_ski_pole_map[idx]))
                for idx in common_idx
            ]
            mask_ski_t = torch.stack(mask_ski, dim=0)
            mask_ski_pole_t = torch.stack(mask_ski_pole, dim=0)
            mask_ski_t = self._temporal_average_resample(
                mask_ski_t, target_t=t_after, time_dim=0
            )
            mask_ski_pole_t = self._temporal_average_resample(
                mask_ski_pole_t, target_t=t_after, time_dim=0
            )
            mask_ski_t = mask_ski_t.gt(0.5).to(torch.float32).permute(1, 0, 2, 3)
            mask_ski_pole_t = mask_ski_pole_t.gt(0.5).to(torch.float32).permute(1, 0, 2, 3)

        src_t = len(common_idx)
        sel = self._temporal_resample_indices(src_t, t_after)

        # Load keypoints for all variants
        variant_kpts: Dict[str, Dict[str, Any]] = {}
        for variant in variants:
            if has_variants:
                cam1_kpt2d_dir_variant = Path(cam1_kpt2d_dirs[variant])
                cam2_kpt2d_dir_variant = Path(cam2_kpt2d_dirs[variant])
                kpt3d_dir_variant = Path(kpt3d_dirs[variant])
            else:
                cam1_kpt2d_dir_variant = cam1_kpt2d_dir
                cam2_kpt2d_dir_variant = cam2_kpt2d_dir
                kpt3d_dir_variant = kpt3d_dir

            # Only apply filtering for 'character' variant; pole and ski keep all joints
            apply_filtering = variant == "character"

            (
                cam1_kpt2d_t,
                cam2_kpt2d_t,
                gt_kpt3d_t,
                sam2d_cam1_t,
                sam2d_cam2_t,
                sam3d_cam1_t,
                sam3d_cam2_t,
                frame_indices_t,
            ) = self._load_single_variant_keypoints(
                cam1_kpt2d_dir=cam1_kpt2d_dir_variant,
                cam2_kpt2d_dir=cam2_kpt2d_dir_variant,
                kpt3d_dir=kpt3d_dir_variant,
                sam3d_cam1_kpt2d_dir=sam3d_cam1_kpt2d_dir,
                sam3d_cam2_kpt2d_dir=sam3d_cam2_kpt2d_dir,
                sam3d_cam1_kpt3d_dir=sam3d_cam1_kpt3d_dir,
                sam3d_cam2_kpt3d_dir=sam3d_cam2_kpt3d_dir,
                common_idx=common_idx,
                apply_filtering=apply_filtering,
                sam3d_cam1_kpt2d_map=sam3d_cam1_kpt2d_map if self._load_2d_kpt else None,
                sam3d_cam2_kpt2d_map=sam3d_cam2_kpt2d_map if self._load_2d_kpt else None,
                sam3d_cam1_kpt3d_map=sam3d_cam1_kpt3d_map if self._load_3d_kpt else None,
                sam3d_cam2_kpt3d_map=sam3d_cam2_kpt3d_map if self._load_3d_kpt else None,
            )

            # Apply temporal resampling
            if cam1_kpt2d_t is not None:
                cam1_kpt2d_t = cam1_kpt2d_t[sel]
            if cam2_kpt2d_t is not None:
                cam2_kpt2d_t = cam2_kpt2d_t[sel]
            if gt_kpt3d_t is not None:
                gt_kpt3d_t = gt_kpt3d_t[sel]
            if sam2d_cam1_t is not None:
                sam2d_cam1_t = sam2d_cam1_t[sel]
            if sam2d_cam2_t is not None:
                sam2d_cam2_t = sam2d_cam2_t[sel]
            if sam3d_cam1_t is not None:
                sam3d_cam1_t = sam3d_cam1_t[sel]
            if sam3d_cam2_t is not None:
                sam3d_cam2_t = sam3d_cam2_t[sel]
            frame_indices_t = frame_indices_t[sel]

            variant_kpts[variant] = {
                "cam1_kpt2d": cam1_kpt2d_t,
                "cam2_kpt2d": cam2_kpt2d_t,
                "gt_kpt3d": gt_kpt3d_t,
                "sam2d_cam1": sam2d_cam1_t,
                "sam2d_cam2": sam2d_cam2_t,
                "sam3d_cam1": sam3d_cam1_t,
                "sam3d_cam2": sam3d_cam2_t,
            }

        self._validate_stereo_pair_consistency(
            item=item,
            cam1_frames_t=cam1_frames_t,
            cam2_frames_t=cam2_frames_t,
            cam1_kpt2d_t=variant_kpts[variants[0]]["cam1_kpt2d"],
            cam2_kpt2d_t=variant_kpts[variants[0]]["cam2_kpt2d"],
            sam2d_cam1_t=variant_kpts[variants[0]]["sam2d_cam1"],
            sam2d_cam2_t=variant_kpts[variants[0]]["sam2d_cam2"],
            sam3d_cam1_t=variant_kpts[variants[0]]["sam3d_cam1"],
            sam3d_cam2_t=variant_kpts[variants[0]]["sam3d_cam2"],
            mask_ski_t=mask_ski_t,
            mask_ski_pole_t=mask_ski_pole_t,
            frame_indices_t=frame_indices_t,
        )

        out: Dict[str, Any] = {
            "frame_indices": frame_indices_t,
            "meta": {
                "experiment": self._experiment,
                "person_id": item.get("person_id", "unknown"),
                "action_id": item.get("action_id", "unknown"),
                "cam1_id": item.get("cam1_id", "unknown"),
                "cam2_id": item.get("cam2_id", "unknown"),
                "num_aligned_frames": int(frame_indices_t.numel()),
            },
        }

        if (
            self._load_frames
            and cam1_frames_t is not None
            and cam2_frames_t is not None
        ):
            out["frames"] = {
                "cam1": cam1_frames_t,
                "cam2": cam2_frames_t,
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
                if variant_kpts[v]["cam1_kpt2d"] is not None:
                    out["kpt2d_gt"][f"{v}_cam1"] = variant_kpts[v]["cam1_kpt2d"]
                if variant_kpts[v]["cam2_kpt2d"] is not None:
                    out["kpt2d_gt"][f"{v}_cam2"] = variant_kpts[v]["cam2_kpt2d"]

            # SAM: character only
            if variant_kpts["character"]["sam2d_cam1"] is not None:
                out["kpt2d_sam"]["character_cam1"] = variant_kpts["character"][
                    "sam2d_cam1"
                ]
            if variant_kpts["character"]["sam2d_cam2"] is not None:
                out["kpt2d_sam"]["character_cam2"] = variant_kpts["character"][
                    "sam2d_cam2"
                ]

        if self._load_3d_kpt:
            out["kpt3d_gt"] = {}
            out["kpt3d_sam"] = {}

            # GT: all variants (character, pole, ski)
            for v in variants:
                if variant_kpts[v]["gt_kpt3d"] is not None:
                    out["kpt3d_gt"][v] = variant_kpts[v]["gt_kpt3d"]

            # SAM: character only
            if variant_kpts["character"]["sam3d_cam1"] is not None:
                out["kpt3d_sam"]["character_cam1"] = variant_kpts["character"][
                    "sam3d_cam1"
                ]
            if variant_kpts["character"]["sam3d_cam2"] is not None:
                out["kpt3d_sam"]["character_cam2"] = variant_kpts["character"][
                    "sam3d_cam2"
                ]

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

        # ---------------- camera-pair frame-dir format ----------------
        if "cam1_frames_dir" in item and "cam2_frames_dir" in item:
            out = self._load_pair_modalities(item)
            out["meta"]["index"] = index
            return out

        raise ValueError(
            "This dataset currently expects camera-pair index items with "
            "cam1_frames_dir/cam2_frames_dir and corresponding kpt/sam paths."
        )


def whole_video_dataset(
    experiment: str,
    dataset_idx: List[UnityDataConfig],
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    load_frames: bool = True,
    load_2d_kpt: bool = True,
    load_3d_kpt: bool = True,
    load_mask: bool = True,
) -> LabeledUnityDataset:
    return LabeledUnityDataset(
        experiment=experiment,
        transform=transform,
        index_mapping=dataset_idx,
        load_frames=load_frames,
        load_2d_kpt=load_2d_kpt,
        load_3d_kpt=load_3d_kpt,
        load_mask=load_mask,
    )
