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
from torch.utils.data import Dataset

from map_config import (
    UnityDataConfig,
    filter_sam3d_body_kpts,
    filter_unity_kpts,
)

from .utils import uniform_subsample_along_dim

logger = logging.getLogger(__name__)


class DualViewUnityDataset(Dataset):
    """
    Multi-view labeled video dataset.
    """

    def __init__(
        self,
        index_mapping: List[UnityDataConfig],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        load_frames: bool = True,
        load_2d_kpt: bool = True,
        load_3d_kpt: bool = True,
        load_mask: bool = True,
        target_t: int = 32,
    ) -> None:
        super().__init__()

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
        self._target_t = target_t

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

    def _load_frames_dir(
        self,
        common_idx: List[int],
        cam1_frames_map: Dict[int, Path],
        cam2_frames_map: Dict[int, Path],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        cam1_frames = []
        cam2_frames = []
        for idx in common_idx:
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

        cam1_frames_t = torch.stack(cam1_frames, dim=0)
        cam2_frames_t = torch.stack(cam2_frames, dim=0)

        # apply same image transform to both views
        cam1_frames_t = self._apply_transform(cam1_frames_t)
        cam2_frames_t = self._apply_transform(cam2_frames_t)

        cam1_frames_t_uniform = uniform_subsample_along_dim(
            cam1_frames_t, self._target_t, dim=0
        )
        cam2_frames_t_uniform = uniform_subsample_along_dim(
            cam2_frames_t, self._target_t, dim=0
        )

        return cam1_frames_t_uniform, cam2_frames_t_uniform

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
        cam1_kpt2d_dir: Path,
        cam2_kpt2d_dir: Path,
        kpt3d_dir: Path,
        gender: str,
        sam3d_cam1_kpt2d_dir: Path,
        sam3d_cam2_kpt2d_dir: Path,
        sam3d_cam1_kpt3d_dir: Path,
        sam3d_cam2_kpt3d_dir: Path,
        common_idx: List[int],
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
            sam3d_cam1_kpt2d_map: Pre-built SAM cam1 2D map (reused across variants).
            sam3d_cam2_kpt2d_map: Pre-built SAM cam2 2D map (reused across variants).
            sam3d_cam1_kpt3d_map: Pre-built SAM cam1 3D map (reused across variants).
            sam3d_cam2_kpt3d_map: Pre-built SAM cam2 3D map (reused across variants).

        Returns:
            (unity_gt_cam1_kpt2d_t, unity_gt_cam2_kpt2d_t, unity_gt_kpt3d_t, sam3d_cam1_kpt2d_t, sam3d_cam2_kpt2d_t, sam3d_cam1_kpt3d_t, sam3d_cam2_kpt3d_t, frame_indices_t)
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

        sam3d_cam1_kpt2d_map = (
            self._build_idx_file_map(sam3d_cam1_kpt2d_dir, "kpt2d_*.npy")
            if self._load_2d_kpt
            else {}
        )
        sam3d_cam2_kpt2d_map = (
            self._build_idx_file_map(sam3d_cam2_kpt2d_dir, "kpt2d_*.npy")
            if self._load_2d_kpt
            else {}
        )
        sam3d_cam1_kpt3d_map = (
            self._build_idx_file_map(sam3d_cam1_kpt3d_dir, "kpt3d_*.npy")
            if self._load_3d_kpt
            else {}
        )
        sam3d_cam2_kpt3d_map = (
            self._build_idx_file_map(sam3d_cam2_kpt3d_dir, "kpt3d_*.npy")
            if self._load_3d_kpt
            else {}
        )

        unity_gt_cam1_kpt2d: List[torch.Tensor] = []
        unity_gt_cam2_kpt2d: List[torch.Tensor] = []
        unity_gt_kpt3d: List[torch.Tensor] = []
        sam3d_cam1_kpt2d: List[torch.Tensor] = []
        sam3d_cam2_kpt2d: List[torch.Tensor] = []
        sam3d_cam1_kpt3d: List[torch.Tensor] = []
        sam3d_cam2_kpt3d: List[torch.Tensor] = []

        for idx in common_idx:
            if self._load_2d_kpt:
                cam1_2d = np.asarray(np.load(cam1_kpt2d_map[idx]), dtype=np.float32)
                cam2_2d = np.asarray(np.load(cam2_kpt2d_map[idx]), dtype=np.float32)

                unity_gt_cam1_kpt2d.append(
                    torch.from_numpy(
                        filter_unity_kpts(cam1_2d, flag="2d", gender=gender)
                    )
                )
                unity_gt_cam2_kpt2d.append(
                    torch.from_numpy(
                        filter_unity_kpts(cam2_2d, flag="2d", gender=gender)
                    )
                )

            if self._load_3d_kpt:
                gt_3d = np.asarray(np.load(kpt3d_map[idx]), dtype=np.float32)

                unity_gt_kpt3d.append(
                    torch.from_numpy(filter_unity_kpts(gt_3d, flag="3d", gender=gender))
                )

            if self._load_2d_kpt:
                sam1_2d = np.asarray(
                    np.load(sam3d_cam1_kpt2d_map[idx]), dtype=np.float32
                )
                sam2_2d = np.asarray(
                    np.load(sam3d_cam2_kpt2d_map[idx]), dtype=np.float32
                )

                sam3d_cam1_kpt2d.append(
                    torch.from_numpy(filter_sam3d_body_kpts(sam1_2d))
                )
                sam3d_cam2_kpt2d.append(
                    torch.from_numpy(filter_sam3d_body_kpts(sam2_2d))
                )

            if self._load_3d_kpt:
                sam1_3d = np.asarray(
                    np.load(sam3d_cam1_kpt3d_map[idx]), dtype=np.float32
                )
                sam2_3d = np.asarray(
                    np.load(sam3d_cam2_kpt3d_map[idx]), dtype=np.float32
                )

                sam3d_cam1_kpt3d.append(
                    torch.from_numpy(filter_sam3d_body_kpts(sam1_3d))
                )
                sam3d_cam2_kpt3d.append(
                    torch.from_numpy(filter_sam3d_body_kpts(sam2_3d))
                )

        unity_gt_cam1_kpt2d_t = (
            torch.stack(unity_gt_cam1_kpt2d, dim=0) if self._load_2d_kpt else None
        )
        unity_gt_cam2_kpt2d_t = (
            torch.stack(unity_gt_cam2_kpt2d, dim=0) if self._load_2d_kpt else None
        )
        unity_gt_kpt3d_t = (
            torch.stack(unity_gt_kpt3d, dim=0) if self._load_3d_kpt else None
        )
        sam3d_cam1_kpt2d_t = (
            torch.stack(sam3d_cam1_kpt2d, dim=0) if self._load_2d_kpt else None
        )
        sam3d_cam2_kpt2d_t = (
            torch.stack(sam3d_cam2_kpt2d, dim=0) if self._load_2d_kpt else None
        )
        sam3d_cam1_kpt3d_t = (
            torch.stack(sam3d_cam1_kpt3d, dim=0) if self._load_3d_kpt else None
        )
        sam3d_cam2_kpt3d_t = (
            torch.stack(sam3d_cam2_kpt3d, dim=0) if self._load_3d_kpt else None
        )
        frame_indices_t = torch.tensor(common_idx, dtype=torch.long)

        return (
            unity_gt_cam1_kpt2d_t,
            unity_gt_cam2_kpt2d_t,
            unity_gt_kpt3d_t,
            sam3d_cam1_kpt2d_t,
            sam3d_cam2_kpt2d_t,
            sam3d_cam1_kpt3d_t,
            sam3d_cam2_kpt3d_t,
            frame_indices_t,
        )

    def _load_pair_modalities(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Load aligned modalities for one camera-pair sample.

        Modalities:
          - cam1/cam2 frames
          - cam1/cam2 2D kpt
          - GT 3D kpt
          - SAM3D pred 3D kpt for cam1/cam2

        This optimized path only loads the character variant.
        """
        gender = item.get("person_id", "male")
        cam1_frames_dir = Path(item["cam1_frames_dir"])
        cam2_frames_dir = Path(item["cam2_frames_dir"])

        cam1_kpt2d_dir = Path(item["cam1_kpt2d_dir"])
        cam2_kpt2d_dir = Path(item["cam2_kpt2d_dir"])

        kpt3d_dir = Path(item["kpt3d_dir"])

        sam3d_cam1_kpt2d_dir = Path(item["sam3d_cam1_kpt2d_dir"])
        sam3d_cam2_kpt2d_dir = Path(item["sam3d_cam2_kpt2d_dir"])
        sam3d_cam1_kpt3d_dir = Path(item["sam3d_cam1_kpt3d_dir"])
        sam3d_cam2_kpt3d_dir = Path(item["sam3d_cam2_kpt3d_dir"])

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
            cam1_kpt2d_map = self._build_idx_file_map(cam1_kpt2d_dir, "kpt2d_*.npy")
            cam2_kpt2d_map = self._build_idx_file_map(cam2_kpt2d_dir, "kpt2d_*.npy")
            cur = set(cam1_kpt2d_map) & set(cam2_kpt2d_map)
            all_common_set = cur if all_common_set is None else all_common_set & cur
        if self._load_3d_kpt:
            kpt3d_map = self._build_idx_file_map(kpt3d_dir, "frame_*.npy")
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

        common_idx = [
            idx
            for idx in all_common
            if idx in sam_valid_set
            and idx not in cam1_none_set
            and idx not in cam2_none_set
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

        if self._load_frames:
            cam1_frames_t_uniform, cam2_frames_t_uniform = self._load_frames_dir(
                common_idx=common_idx,
                cam1_frames_map=cam1_frames_map,
                cam2_frames_map=cam2_frames_map,
            )
        else:
            cam1_frames_t_uniform, cam2_frames_t_uniform = None, None

        (
            unity_gt_cam1_kpt2d_t,
            unity_gt_cam2_kpt2d_t,
            unity_gt_kpt3d_t,
            sam3d_cam1_kpt2d_t,
            sam3d_cam2_kpt2d_t,
            sam3d_cam1_kpt3d_t,
            sam3d_cam2_kpt3d_t,
            frame_indices_t,
        ) = self._load_single_variant_keypoints(
            cam1_kpt2d_dir=cam1_kpt2d_dir,
            cam2_kpt2d_dir=cam2_kpt2d_dir,
            kpt3d_dir=kpt3d_dir,
            gender=gender,
            sam3d_cam1_kpt2d_dir=sam3d_cam1_kpt2d_dir,
            sam3d_cam2_kpt2d_dir=sam3d_cam2_kpt2d_dir,
            sam3d_cam1_kpt3d_dir=sam3d_cam1_kpt3d_dir,
            sam3d_cam2_kpt3d_dir=sam3d_cam2_kpt3d_dir,
            common_idx=common_idx,
        )

        # Apply temporal resampling
        if unity_gt_cam1_kpt2d_t is not None:
            unity_gt_cam1_kpt2d_t = uniform_subsample_along_dim(
                unity_gt_cam1_kpt2d_t, self._target_t, dim=0
            )
        if unity_gt_cam2_kpt2d_t is not None:
            unity_gt_cam2_kpt2d_t = uniform_subsample_along_dim(
                unity_gt_cam2_kpt2d_t, self._target_t, dim=0
            )
        if unity_gt_kpt3d_t is not None:
            unity_gt_kpt3d_t = uniform_subsample_along_dim(
                unity_gt_kpt3d_t, self._target_t, dim=0
            )
        if sam3d_cam1_kpt2d_t is not None:
            sam3d_cam1_kpt2d_t = uniform_subsample_along_dim(
                sam3d_cam1_kpt2d_t, self._target_t, dim=0
            )
        if sam3d_cam2_kpt2d_t is not None:
            sam3d_cam2_kpt2d_t = uniform_subsample_along_dim(
                sam3d_cam2_kpt2d_t, self._target_t, dim=0
            )
        if sam3d_cam1_kpt3d_t is not None:
            sam3d_cam1_kpt3d_t = uniform_subsample_along_dim(
                sam3d_cam1_kpt3d_t, self._target_t, dim=0
            )
        if sam3d_cam2_kpt3d_t is not None:
            sam3d_cam2_kpt3d_t = uniform_subsample_along_dim(
                sam3d_cam2_kpt3d_t, self._target_t, dim=0
            )
        frame_indices_t = uniform_subsample_along_dim(
            frame_indices_t, self._target_t, dim=0
        )

        out: Dict[str, Any] = {
            "frame_indices": frame_indices_t,
            "meta": {
                "person_id": item.get("person_id", "unknown"),
                "action_id": item.get("action_id", "unknown"),
                "cam1_id": item.get("cam1_id", "unknown"),
                "cam2_id": item.get("cam2_id", "unknown"),
                "num_aligned_frames": int(frame_indices_t.numel()),
            },
        }

        if (
            self._load_frames
            and cam1_frames_t_uniform is not None
            and cam2_frames_t_uniform is not None
        ):
            out["frames"] = {
                "cam1": cam1_frames_t_uniform,
                "cam2": cam2_frames_t_uniform,
            }

        # Character-only keypoint outputs.
        if self._load_2d_kpt:
            out["kpt2d_gt"] = {}
            out["kpt2d_sam"] = {}

            if unity_gt_cam1_kpt2d_t is not None:
                out["kpt2d_gt"] = unity_gt_cam1_kpt2d_t
            if unity_gt_cam2_kpt2d_t is not None:
                out["kpt2d_gt"] = unity_gt_cam2_kpt2d_t

            if sam3d_cam1_kpt2d_t is not None:
                out["kpt2d_sam"]["cam1"] = sam3d_cam1_kpt2d_t
            if sam3d_cam2_kpt2d_t is not None:
                out["kpt2d_sam"]["cam2"] = sam3d_cam2_kpt2d_t

        if self._load_3d_kpt:
            out["kpt3d_gt"] = {}
            out["kpt3d_gt_canonical"] = {}
            out["kpt3d_sam"] = {}
            out["kpt3d_sam_canonical"] = {}

            if unity_gt_kpt3d_t is not None:
                out["kpt3d_gt"] = unity_gt_kpt3d_t

            if sam3d_cam1_kpt3d_t is not None:
                out["kpt3d_sam"]["cam1"] = sam3d_cam1_kpt3d_t

            if sam3d_cam2_kpt3d_t is not None:
                out["kpt3d_sam"]["cam2"] = sam3d_cam2_kpt3d_t

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
        out = self._load_pair_modalities(item)

        return out
