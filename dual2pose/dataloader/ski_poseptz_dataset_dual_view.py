#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import h5py  # type: ignore[import-untyped]
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class LabeledSkiPosePTZDataset(Dataset):
    """Ski-PosePTZ dataset loader (two cameras, temporal sequence).

    This dataset follows the loading style from `load_h5_example.py`:
    - labels are read from `<dataset_root>/<split>/labels.h5`
    - image path is resolved as
      `<dataset_root>/<split>/seq_{seq:03d}/cam_{cam:02d}/image_{frame:06d}.png`

    One dataset item is one `(subj, seq)` clip where both cameras exist at
    the same frame indices.

    Returned sample:
        {
            "frames": {
                "cam1": Tensor[C, T, H, W],
                "cam2": Tensor[C, T, H, W],
            },
            "kpt3d": Tensor[T, J, 3],
            "frame_indices": Tensor[T],
            "meta": {
                "subj", "seq", "cam1", "cam2", "num_frames",
            }
        }
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "test",
        cam1_id: int = 0,
        cam2_id: int = 1,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        labels_file: str = "labels.h5",
        target_t: Optional[int] = None,
        min_t: int = 2,
    ) -> None:
        super().__init__()
        self._dataset_root = Path(dataset_root)
        self._split = str(split)
        self._cam1_id = int(cam1_id)
        self._cam2_id = int(cam2_id)
        if self._cam1_id == self._cam2_id:
            raise ValueError("cam1_id and cam2_id must be different.")

        self._target_t = int(target_t) if target_t is not None else None
        if self._target_t is not None and self._target_t <= 0:
            raise ValueError("target_t must be > 0 when provided.")

        self._min_t = int(min_t)
        if self._min_t <= 0:
            raise ValueError("min_t must be > 0.")

        self._transform = transform

        self._split_root = self._dataset_root / self._split
        self._labels_path = self._split_root / labels_file
        if not self._labels_path.exists():
            raise FileNotFoundError(f"labels file not found: {self._labels_path}")

        with h5py.File(self._labels_path, "r") as f:
            self._seq = np.asarray(f["seq"], dtype=np.int32)
            self._cam = np.asarray(f["cam"], dtype=np.int32)
            self._frame = np.asarray(f["frame"], dtype=np.int32)
            self._subj = np.asarray(f["subj"], dtype=np.int32)

            pose_3d_raw = np.asarray(f["3D"], dtype=np.float32)
            if pose_3d_raw.ndim == 2:
                if pose_3d_raw.shape[1] % 3 != 0:
                    raise ValueError(
                        f"Invalid 3D shape in labels.h5: {pose_3d_raw.shape}"
                    )
                self._pose_3d = pose_3d_raw.reshape(pose_3d_raw.shape[0], -1, 3)
            elif pose_3d_raw.ndim == 3 and pose_3d_raw.shape[-1] == 3:
                self._pose_3d = pose_3d_raw
            else:
                raise ValueError(f"Unsupported 3D keypoint shape: {pose_3d_raw.shape}")

        if not (
            len(self._seq)
            == len(self._cam)
            == len(self._frame)
            == len(self._subj)
            == len(self._pose_3d)
        ):
            raise ValueError(
                "Inconsistent number of rows across label fields in h5 file."
            )

        self._samples = self._build_temporal_samples()
        if not self._samples:
            raise ValueError(
                f"No temporal samples found for camera pair ({self._cam1_id}, {self._cam2_id}) "
                f"in {self._labels_path}."
            )

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _temporal_select_indices(src_t: int, dst_t: int) -> torch.Tensor:
        if src_t <= 0 or dst_t <= 0:
            raise ValueError(
                f"src_t and dst_t must be > 0, got src_t={src_t}, dst_t={dst_t}"
            )
        if src_t == dst_t:
            return torch.arange(src_t, dtype=torch.long)
        return torch.linspace(0, src_t - 1, steps=dst_t).round().long()

    def _build_temporal_samples(self) -> List[Dict[str, Any]]:
        """Build per-(subj,seq) samples with synchronized cam1/cam2 frame indices."""
        grouped: Dict[tuple[int, int], Dict[int, Dict[int, int]]] = defaultdict(dict)
        for row_idx in range(len(self._seq)):
            subj = int(self._subj[row_idx])
            seq = int(self._seq[row_idx])
            frame = int(self._frame[row_idx])
            cam = int(self._cam[row_idx])

            by_frame = grouped[(subj, seq)]
            if frame not in by_frame:
                by_frame[frame] = {}
            by_frame[frame][cam] = row_idx

        out: List[Dict[str, Any]] = []
        for (subj, seq), by_frame in sorted(grouped.items(), key=lambda x: x[0]):
            common_frames = sorted(
                frame
                for frame, cam_to_row in by_frame.items()
                if self._cam1_id in cam_to_row and self._cam2_id in cam_to_row
            )
            if len(common_frames) < self._min_t:
                continue

            row_indices_cam1 = [by_frame[f][self._cam1_id] for f in common_frames]
            row_indices_cam2 = [by_frame[f][self._cam2_id] for f in common_frames]

            out.append(
                {
                    "subj": subj,
                    "seq": seq,
                    "frame_indices": common_frames,
                    "row_indices_cam1": row_indices_cam1,
                    "row_indices_cam2": row_indices_cam2,
                }
            )

        return out

    def _read_frame(self, seq: int, cam: int, frame: int) -> torch.Tensor:
        image_path = (
            self._split_root
            / f"seq_{seq:03d}"
            / f"cam_{cam:02d}"
            / f"image_{frame:06d}.png"
        )
        img = imageio.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"image not found or unreadable: {image_path}")

        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, 3), got {img.shape}"
            )

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return img_t

    def _apply_transform_to_sequence(self, frames_tchw: torch.Tensor) -> torch.Tensor:
        """Apply transform frame-wise on (T,C,H,W)."""
        if self._transform is None:
            return frames_tchw
        transformed = [
            self._transform(frames_tchw[t]) for t in range(frames_tchw.shape[0])
        ]
        return torch.stack(transformed, dim=0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._samples[index]
        subj = int(sample["subj"])
        seq = int(sample["seq"])

        frame_indices = list(sample["frame_indices"])
        row_indices_cam1 = list(sample["row_indices_cam1"])

        if self._target_t is not None and len(frame_indices) != self._target_t:
            select = self._temporal_select_indices(len(frame_indices), self._target_t)
            frame_indices = [frame_indices[int(i)] for i in select.tolist()]
            row_indices_cam1 = [row_indices_cam1[int(i)] for i in select.tolist()]

        frames_cam1: List[torch.Tensor] = []
        frames_cam2: List[torch.Tensor] = []
        kpt3d_list: List[torch.Tensor] = []

        for row_idx_cam1 in row_indices_cam1:
            frame_id = int(self._frame[row_idx_cam1])
            frames_cam1.append(
                self._read_frame(seq=seq, cam=self._cam1_id, frame=frame_id)
            )
            frames_cam2.append(
                self._read_frame(seq=seq, cam=self._cam2_id, frame=frame_id)
            )
            kpt3d_list.append(
                torch.from_numpy(
                    np.asarray(self._pose_3d[row_idx_cam1], dtype=np.float32)
                )
            )

        frames_cam1_t = torch.stack(frames_cam1, dim=0)  # (T,C,H,W)
        frames_cam2_t = torch.stack(frames_cam2, dim=0)  # (T,C,H,W)

        frames_cam1_t = self._apply_transform_to_sequence(frames_cam1_t)
        frames_cam2_t = self._apply_transform_to_sequence(frames_cam2_t)

        kpt3d_t = torch.stack(kpt3d_list, dim=0)  # (T,J,3)
        frame_indices_t = torch.tensor(frame_indices, dtype=torch.long)

        return {
            "frames": {
                "cam1": frames_cam1_t.permute(1, 0, 2, 3),  # (C,T,H,W)
                "cam2": frames_cam2_t.permute(1, 0, 2, 3),  # (C,T,H,W)
            },
            "kpt3d": kpt3d_t,
            "frame_indices": frame_indices_t,
            "meta": {
                "subj": subj,
                "seq": seq,
                "cam1": self._cam1_id,
                "cam2": self._cam2_id,
                "num_frames": int(frame_indices_t.numel()),
            },
        }


def ski_poseptz_dataset(
    dataset_root: str | Path,
    split: str = "test",
    cam1_id: int = 0,
    cam2_id: int = 1,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    labels_file: str = "labels.h5",
    target_t: Optional[int] = None,
    min_t: int = 2,
) -> LabeledSkiPosePTZDataset:
    return LabeledSkiPosePTZDataset(
        dataset_root=dataset_root,
        split=split,
        cam1_id=cam1_id,
        cam2_id=cam2_id,
        transform=transform,
        labels_file=labels_file,
        target_t=target_t,
        min_t=min_t,
    )


if __name__ == "__main__":
    # quick sanity check
    dataset = ski_poseptz_dataset(
        dataset_root="/workspace/data/Ski-PosePTZ-CameraDataset-png",
        split="test",
        cam1_id=0,
        cam2_id=1,
        transform=None,
        labels_file="labels.h5",
        target_t=10,
        min_t=2,
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Frames cam1 shape: {sample['frames']['cam1'].shape}")
    print(f"Frames cam2 shape: {sample['frames']['cam2'].shape}")
    print(f"Kpt3d shape: {sample['kpt3d'].shape}")
    print(f"Frame indices shape: {sample['frame_indices'].shape}")
    print(f"Meta: {sample['meta']}")
