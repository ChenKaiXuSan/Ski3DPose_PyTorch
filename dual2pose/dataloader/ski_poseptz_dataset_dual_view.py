#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import h5py  # type: ignore[import-untyped]
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dual2pose.map_config import filter_sam3d_body_kpts


class LabeledSkiPosePTZDataset(Dataset):
    """Ski-PosePTZ dataset loader driven by an index-mapping JSON.

    Each entry in the index mapping corresponds to one (sequence, subject,
    cam_pair) and contains all directory paths needed to load frames, SAM3D
    2D/3D keypoints, and pseudo-GT.  Ground-truth 3D comes from labels.h5.

    Returned sample dict (keys present depend on load_* flags):
        {
            "frames":   {"cam1": Tensor[C,T,H,W], "cam2": Tensor[C,T,H,W]},
            "kpt2d":    {"cam1": Tensor[T,J,2],   "cam2": Tensor[T,J,2]},
            "kpt3d":    {"cam1": Tensor[T,J,3],   "cam2": Tensor[T,J,3]},
            "gt_kpt3d": Tensor[T,J,3],   # always present (from labels.h5)
            "frame_indices": Tensor[T],
            "meta": {"subj", "seq", "cam1", "cam2", "num_frames"},
        }

    Args:
        index_mapping: Path to a JSON file.  The JSON may be either a flat
            list of SkiPosePTZDataConfig dicts, or a dict with train/val/test
            keys — in the latter case pass `split` to select one.
        transform: Per-frame image transform applied to (C,H,W) tensors.
        load_frames: Whether to load RGB images.
        load_2d_kpt: Whether to load SAM3D 2D keypoints.
        load_3d_kpt: Whether to load SAM3D 3D keypoints.
        target_t: Number of frames to subsample each clip to (required).
        min_t: Minimum number of common frames required to include a clip.
        split: Select a split when the JSON root is a dict (e.g. "train").
    """

    def __init__(
        self,
        index_mapping: str | Path,
        transform,
        load_frames: bool = True,
        load_2d_kpt: bool = False,
        load_3d_kpt: bool = True,
        target_t: int = 32,
        min_t: int = 2,
        split: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._transform = transform
        self._load_frames = bool(load_frames)
        self._load_2d_kpt = bool(load_2d_kpt)
        self._load_3d_kpt = bool(load_3d_kpt)
        self._target_t = int(target_t)
        if self._target_t <= 0:
            raise ValueError("target_t must be > 0")
        self._min_t = int(min_t)
        if self._min_t <= 0:
            raise ValueError("min_t must be > 0")

        # ── Load index mapping ──────────────────────────────────────────────
        mapping_path = Path(index_mapping)
        if not mapping_path.exists():
            raise FileNotFoundError(f"index_mapping not found: {mapping_path}")
        with mapping_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            configs: List[Dict[str, Any]] = raw
        elif isinstance(raw, dict):
            if split is not None:
                if split not in raw:
                    raise KeyError(f"split '{split}' not found in index_mapping keys: {list(raw.keys())}")
                configs = raw[split]
            else:
                # Flatten all split lists (skip metadata keys)
                configs = []
                for v in raw.values():
                    if isinstance(v, list):
                        configs.extend(v)
        else:
            raise ValueError(f"Unsupported index_mapping format: {type(raw)}")

        if not configs:
            raise ValueError("index_mapping contains no entries.")

        # ── Cache labels.h5 data (one h5 file is shared by many configs) ───
        self._h5_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # ── Build temporal sample list ──────────────────────────────────────
        self._samples = self._build_samples(configs)
        if not self._samples:
            raise ValueError("No valid temporal samples found in index_mapping.")

    # ── labels.h5 helpers ────────────────────────────────────────────────────

    def _load_h5(self, labels_h5: str) -> Dict[str, np.ndarray]:
        if labels_h5 not in self._h5_cache:
            with h5py.File(labels_h5, "r") as f:
                seq   = np.asarray(f["seq"],   dtype=np.int32)
                cam   = np.asarray(f["cam"],   dtype=np.int32)
                frame = np.asarray(f["frame"], dtype=np.int32)
                subj  = np.asarray(f["subj"],  dtype=np.int32)
                pose_3d_raw = np.asarray(f["3D"], dtype=np.float32)
            if pose_3d_raw.ndim == 2:
                if pose_3d_raw.shape[1] % 3 != 0:
                    raise ValueError(f"Invalid 3D shape in labels.h5: {pose_3d_raw.shape}")
                pose_3d = pose_3d_raw.reshape(pose_3d_raw.shape[0], -1, 3)
            elif pose_3d_raw.ndim == 3 and pose_3d_raw.shape[-1] == 3:
                pose_3d = pose_3d_raw
            else:
                raise ValueError(f"Unsupported 3D keypoint shape: {pose_3d_raw.shape}")
            self._h5_cache[labels_h5] = {
                "seq": seq, "cam": cam, "frame": frame, "subj": subj, "pose_3d": pose_3d,
            }
        return self._h5_cache[labels_h5]

    # ── SAM3D helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _read_none_detected(sam_dir: Path) -> Set[int]:
        none_file = sam_dir / "none_detected_frames.txt"
        if not none_file.exists():
            return set()
        indices: Set[int] = set()
        for line in none_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                indices.add(int(line))
            except ValueError:
                pass
        return indices

    @staticmethod
    @lru_cache(maxsize=4096)
    def _read_sam3d_npz_cached(npz_path: str, key: str) -> Tuple[float, ...]:
        """Read one array from a SAM3D NPZ and return as a flat tuple (for caching)."""
        data = np.load(npz_path, allow_pickle=True)
        out_dict = data["output"].item()
        arr = np.asarray(out_dict[key], dtype=np.float32)
        return tuple(arr.flatten().tolist()), arr.shape  # type: ignore[return-value]

    @classmethod
    def _read_sam3d_kpt(cls, sam_dir: Path, frame_id: int, key: str) -> np.ndarray:
        npz_path = sam_dir / f"{frame_id:06d}_sam3d_body.npz"
        flat, shape = cls._read_sam3d_npz_cached(str(npz_path.resolve()), key)
        return np.array(flat, dtype=np.float32).reshape(shape)

    # ── Sample list construction ──────────────────────────────────────────────

    def _build_samples(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for cfg in configs:
            seq_id  = int(cfg["sequence_id"])
            subj_id = int(cfg["subject_id"])
            cam1_id = int(cfg["cam1_id"])
            cam2_id = int(cfg["cam2_id"])
            labels_h5 = cfg["labels_h5"]

            h5 = self._load_h5(labels_h5)

            # Row indices for this (subj, seq) combination
            mask = (h5["subj"] == subj_id) & (h5["seq"] == seq_id)
            rows = np.where(mask)[0]

            cam1_frame_to_row: Dict[int, int] = {}
            cam2_frame_to_row: Dict[int, int] = {}
            for row_idx in rows:
                c = int(h5["cam"][row_idx])
                f = int(h5["frame"][row_idx])
                if c == cam1_id:
                    cam1_frame_to_row[f] = int(row_idx)
                elif c == cam2_id:
                    cam2_frame_to_row[f] = int(row_idx)

            common_frames = sorted(set(cam1_frame_to_row) & set(cam2_frame_to_row))

            # Exclude frames with no SAM3D detection
            if self._load_2d_kpt or self._load_3d_kpt:
                none1 = self._read_none_detected(Path(cfg["cam1_sam3d_kpt3d_dir"]))
                none2 = self._read_none_detected(Path(cfg["cam2_sam3d_kpt3d_dir"]))
                common_frames = [f for f in common_frames if f not in none1 and f not in none2]

                # Keep only frames where the SAM3D NPZ file exists for both cameras
                if self._load_2d_kpt or self._load_3d_kpt:
                    sam1_dir = Path(cfg["cam1_sam3d_kpt3d_dir"])
                    sam2_dir = Path(cfg["cam2_sam3d_kpt3d_dir"])
                    common_frames = [
                        f for f in common_frames
                        if (sam1_dir / f"{f:06d}_sam3d_body.npz").exists()
                        and (sam2_dir / f"{f:06d}_sam3d_body.npz").exists()
                    ]

            if len(common_frames) < self._min_t:
                continue

            out.append({
                "seq":    seq_id,
                "subj":   subj_id,
                "cam1_id": cam1_id,
                "cam2_id": cam2_id,
                "labels_h5":        labels_h5,
                "cam1_frames_dir":  cfg["cam1_frames_dir"],
                "cam2_frames_dir":  cfg["cam2_frames_dir"],
                "cam1_sam3d_dir":   cfg["cam1_sam3d_kpt3d_dir"],
                "cam2_sam3d_dir":   cfg["cam2_sam3d_kpt3d_dir"],
                "pesudo_gt_kpt3d_dir": cfg["pesudo_gt_kpt3d_dir"],
                "frame_indices":    common_frames,
                "row_indices_cam1": [cam1_frame_to_row[f] for f in common_frames],
            })
        return out

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _temporal_select_indices(src_t: int, dst_t: int) -> torch.Tensor:
        if src_t <= 0 or dst_t <= 0:
            raise ValueError(f"src_t and dst_t must be > 0, got src_t={src_t}, dst_t={dst_t}")
        if src_t == dst_t:
            return torch.arange(src_t, dtype=torch.long)
        return torch.linspace(0, src_t - 1, steps=dst_t).round().long()

    # ── Frame/keypoint readers ────────────────────────────────────────────────

    @staticmethod
    def _read_frame(frames_dir: Path, frame_id: int) -> torch.Tensor:
        image_path = frames_dir / f"image_{frame_id:06d}.png"
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"image not found: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # (C,H,W)

    def _apply_transform_to_sequence(self, frames_tchw: torch.Tensor) -> torch.Tensor:
        if self._transform is None:
            return frames_tchw
        return torch.stack(
            [self._transform(frames_tchw[t]) for t in range(frames_tchw.shape[0])], dim=0
        )

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]

        frame_indices    = list(sample["frame_indices"])
        row_indices_cam1 = list(sample["row_indices_cam1"])

        # Temporal subsampling
        if len(frame_indices) != self._target_t:
            sel = self._temporal_select_indices(len(frame_indices), self._target_t).tolist()
            frame_indices    = [frame_indices[int(i)]    for i in sel]
            row_indices_cam1 = [row_indices_cam1[int(i)] for i in sel]

        cam1_dir = Path(sample["cam1_frames_dir"])
        cam2_dir = Path(sample["cam2_frames_dir"])
        sam1_dir = Path(sample["cam1_sam3d_dir"])
        sam2_dir = Path(sample["cam2_sam3d_dir"])

        result: Dict[str, Any] = {}

        # ── Frames ──────────────────────────────────────────────────────────
        if self._load_frames:
            frames_cam1 = torch.stack([self._read_frame(cam1_dir, f) for f in frame_indices], dim=0)
            frames_cam2 = torch.stack([self._read_frame(cam2_dir, f) for f in frame_indices], dim=0)
            frames_cam1 = self._apply_transform_to_sequence(frames_cam1)
            frames_cam2 = self._apply_transform_to_sequence(frames_cam2)
            result["frames"] = {
                "cam1": frames_cam1.permute(1, 0, 2, 3),  # (C,T,H,W)
                "cam2": frames_cam2.permute(1, 0, 2, 3),
            }

        # ── SAM3D 2D keypoints ───────────────────────────────────────────────
        if self._load_2d_kpt:
            kpt2d_cam1 = torch.stack([
                torch.from_numpy(filter_sam3d_body_kpts(
                    self._read_sam3d_kpt(sam1_dir, f, "pred_keypoints_2d")
                ))
                for f in frame_indices
            ], dim=0)  # (T,J,2)
            kpt2d_cam2 = torch.stack([
                torch.from_numpy(filter_sam3d_body_kpts(
                    self._read_sam3d_kpt(sam2_dir, f, "pred_keypoints_2d")
                ))
                for f in frame_indices
            ], dim=0)
            result["kpt2d"] = {"cam1": kpt2d_cam1, "cam2": kpt2d_cam2}

        # ── SAM3D 3D keypoints ───────────────────────────────────────────────
        if self._load_3d_kpt:
            kpt3d_cam1 = torch.stack([
                torch.from_numpy(filter_sam3d_body_kpts(
                    self._read_sam3d_kpt(sam1_dir, f, "pred_keypoints_3d")
                ))
                for f in frame_indices
            ], dim=0)  # (T,J,3)
            kpt3d_cam2 = torch.stack([
                torch.from_numpy(filter_sam3d_body_kpts(
                    self._read_sam3d_kpt(sam2_dir, f, "pred_keypoints_3d")
                ))
                for f in frame_indices
            ], dim=0)
            result["kpt3d"] = {"cam1": kpt3d_cam1, "cam2": kpt3d_cam2}

        # ── Ground-truth 3D from labels.h5 (cam1 rows) ──────────────────────
        h5 = self._load_h5(sample["labels_h5"])
        result["gt_kpt3d"] = torch.stack([
            torch.from_numpy(np.asarray(h5["pose_3d"][r], dtype=np.float32))
            for r in row_indices_cam1
        ], dim=0)  # (T,J,3)

        result["frame_indices"] = torch.tensor(frame_indices, dtype=torch.long)
        result["meta"] = {
            "subj":       sample["subj"],
            "seq":        sample["seq"],
            "cam1":       sample["cam1_id"],
            "cam2":       sample["cam2_id"],
            "num_frames": len(frame_indices),
        }
        return result


def ski_poseptz_dataset(
    index_mapping: str | Path,
    split: Optional[str] = None,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    load_frames: bool = True,
    load_2d_kpt: bool = False,
    load_3d_kpt: bool = True,
    target_t: int = 32,
    min_t: int = 2,
) -> LabeledSkiPosePTZDataset:
    return LabeledSkiPosePTZDataset(
        index_mapping=index_mapping,
        transform=transform,
        load_frames=load_frames,
        load_2d_kpt=load_2d_kpt,
        load_3d_kpt=load_3d_kpt,
        target_t=target_t,
        min_t=min_t,
        split=split,
    )


if __name__ == "__main__":
    dataset = ski_poseptz_dataset(
        index_mapping="/workspace/data/Ski-PosePTZ-CameraDataset-png/index_mapping/ski_pose_ptz_index_mapping.json",
        split="test",
        transform=None,
        load_frames=False,
        load_2d_kpt=True,
        load_3d_kpt=True,
        target_t=10,
        min_t=2,
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    if "kpt3d" in sample:
        print(f"kpt3d cam1 shape: {sample['kpt3d']['cam1'].shape}")
    print(f"gt_kpt3d shape:   {sample['gt_kpt3d'].shape}")
    print(f"frame_indices:    {sample['frame_indices']}")
    print(f"meta: {sample['meta']}")

