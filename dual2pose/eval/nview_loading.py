"""Strict four-view temporal intersection and pose loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import torch

from dual2pose.map_config import filter_sam3d_body_kpts, filter_unity_kpts


_FRAME_PATTERN = re.compile(r"(\d+)")


class InsufficientCommonFrames(RuntimeError):
    def __init__(self, group_id: str, available: int, required: int) -> None:
        self.group_id = group_id
        self.available = int(available)
        self.required = int(required)
        super().__init__(
            f"Insufficient common frames for {group_id}: "
            f"available={available}, required={required}"
        )


@dataclass(frozen=True)
class MultiViewSample:
    group: Any
    frame_indices: torch.Tensor
    poses: dict[str, torch.Tensor]
    ground_truth: torch.Tensor


def _frame_map(directory: Path, pattern: str) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for path in sorted(Path(directory).glob(pattern)):
        matches = _FRAME_PATTERN.findall(path.stem)
        if not matches:
            raise ValueError(f"No frame index in {path}")
        long_matches = [value for value in matches if len(value) >= 6]
        frame = int(long_matches[0] if long_matches else matches[-1])
        if frame in output:
            raise ValueError(f"Duplicate frame {frame} in {directory}")
        output[frame] = path
    return output


def _none_detected(directory: Path) -> set[int]:
    path = Path(directory) / "none_detected_frames.txt"
    if not path.is_file():
        return set()
    output: set[int] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        try:
            frame = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid none-detected frame {value!r}: {path}") from exc
        if frame < 0:
            raise ValueError(f"Negative none-detected frame {frame}: {path}")
        output.add(frame)
    return output


def _filtered_pose(path: Path, *, source: str, person_id: str) -> torch.Tensor:
    array = np.asarray(np.load(path), dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] != 3:
        raise ValueError(f"Pose must have shape J x 3, got {array.shape}: {path}")
    if array.shape[0] == 15:
        filtered = array
    elif source == "sam3d":
        filtered = filter_sam3d_body_kpts(array)
    else:
        filtered = filter_unity_kpts(array, flag="3d", gender=person_id)
    filtered = np.asarray(filtered, dtype=np.float32)
    if filtered.shape != (15, 3):
        raise ValueError(f"Filtered pose must have shape 15 x 3: {path}")
    if not np.isfinite(filtered).all():
        raise ValueError(f"Pose contains non-finite values: {path}")
    return torch.from_numpy(np.ascontiguousarray(filtered))


def load_multiview_sample(
    group: Any,
    row_lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
    target_t: int = 30,
) -> MultiViewSample:
    """Load one shared time window for all four camera streams and one GT pose."""

    if target_t <= 0:
        raise ValueError("target_t must be positive")
    pose_maps: dict[str, dict[int, Path]] = {}
    gt_maps: list[dict[int, Path]] = []
    valid_sets: list[set[int]] = []
    for camera in group.cameras:
        key = (str(group.person_id), str(group.action_id), str(camera))
        if key not in row_lookup:
            raise KeyError(f"Missing N-view source for {key}")
        source = row_lookup[key]
        pose_dir = Path(str(source["sam3d_kpt3d_dir"]))
        gt_dir = Path(str(source["kpt3d_dir"]))
        pose_map = _frame_map(pose_dir, "kpt3d_*.npy")
        gt_map = _frame_map(gt_dir, "frame_*.npy")
        valid = set(pose_map).intersection(gt_map).difference(_none_detected(pose_dir))
        pose_maps[camera] = pose_map
        gt_maps.append(gt_map)
        valid_sets.append(valid)
    common = sorted(set.intersection(*valid_sets)) if valid_sets else []
    if len(common) < target_t:
        raise InsufficientCommonFrames(group.group_id, len(common), target_t)
    positions = torch.round(torch.linspace(0, len(common) - 1, target_t)).long()
    selected = [common[int(position)] for position in positions]
    frame_indices = torch.tensor(selected, dtype=torch.long)

    poses = {
        camera: torch.stack(
            [
                _filtered_pose(
                    pose_maps[camera][frame],
                    source="sam3d",
                    person_id=str(group.person_id),
                )
                for frame in selected
            ],
            dim=0,
        )
        for camera in group.cameras
    }
    ground_truth = torch.stack(
        [
            _filtered_pose(
                gt_maps[0][frame],
                source="unity",
                person_id=str(group.person_id),
            )
            for frame in selected
        ],
        dim=0,
    )
    return MultiViewSample(group, frame_indices, poses, ground_truth)
