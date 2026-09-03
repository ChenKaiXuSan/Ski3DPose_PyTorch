"""Pure frame-selection and deterministic RGB occlusion helpers for E5."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


# This intentionally matches eval_unity_masking.py, including its archived order.
DISTAL_JOINT_INDICES = (2, 3, 5, 6, 8, 9, 11, 12)
JOINT_COUNT = 15
_FRAME_INDEX_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ImageOcclusionSetting:
    pattern: str
    ratio: float
    seed: int = 42
    patch_fraction: float = 0.12
    minimum_side_px: int = 16
    temporal_span: int = 10

    def __post_init__(self) -> None:
        if self.pattern not in {"random", "distal", "temporal"}:
            raise ValueError("pattern must be random, distal, or temporal")
        if not math.isfinite(self.ratio) or not 0.0 <= self.ratio <= 1.0:
            raise ValueError("ratio must be finite and lie in [0, 1]")
        if not math.isfinite(self.patch_fraction) or self.patch_fraction <= 0.0:
            raise ValueError("patch_fraction must be finite and positive")
        if self.minimum_side_px <= 0:
            raise ValueError("minimum_side_px must be positive")
        if self.temporal_span <= 0:
            raise ValueError("temporal_span must be positive")


@dataclass(frozen=True)
class OcclusionFrameKey:
    person_id: str
    action_id: str
    camera_id: str
    frame_id: int


def _stable_unit_interval(*parts: object) -> float:
    message = "\x1f".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(message, digest_size=8).digest(), "big")
    return value / float(1 << 64)


def _stream_hash_parts(setting: ImageOcclusionSetting, key: OcclusionFrameKey):
    return (
        setting.seed,
        key.person_id,
        key.action_id,
        key.camera_id,
        setting.pattern,
        f"{setting.ratio:.12g}",
    )


def selected_joint_mask(
    setting: ImageOcclusionSetting,
    frame_key: OcclusionFrameKey,
    *,
    source_frame_ids: Iterable[int] | None = None,
) -> np.ndarray:
    """Return a deterministic 15-joint mask for one source image."""

    stream_parts = _stream_hash_parts(setting, frame_key)
    mask = np.zeros(JOINT_COUNT, dtype=bool)
    if setting.ratio == 0.0:
        return mask
    if setting.pattern == "random":
        for joint in range(JOINT_COUNT):
            mask[joint] = (
                _stable_unit_interval(*stream_parts, joint, frame_key.frame_id)
                < setting.ratio
            )
        return mask
    if setting.pattern == "distal":
        for joint in DISTAL_JOINT_INDICES:
            mask[joint] = (
                _stable_unit_interval(*stream_parts, joint, frame_key.frame_id)
                < setting.ratio
            )
        return mask

    if source_frame_ids is None:
        raise ValueError("temporal masking requires source_frame_ids")
    frame_ids = sorted({int(value) for value in source_frame_ids})
    if not frame_ids:
        raise ValueError("temporal masking requires non-empty source_frame_ids")
    selected_count = int(round(setting.ratio * JOINT_COUNT))
    joint_order = sorted(
        range(JOINT_COUNT),
        key=lambda joint: _stable_unit_interval(*stream_parts, "joint", joint),
    )
    minimum = frame_ids[0]
    maximum = frame_ids[-1]
    latest_start = max(minimum, maximum - setting.temporal_span + 1)
    start_count = latest_start - minimum + 1
    for joint in joint_order[:selected_count]:
        start_unit = _stable_unit_interval(*stream_parts, "start", joint)
        start = minimum + min(start_count - 1, int(start_unit * start_count))
        mask[joint] = start <= frame_key.frame_id < start + setting.temporal_span
    return mask


def apply_image_occlusion(
    image: np.ndarray,
    joints_2d: np.ndarray,
    setting: ImageOcclusionSetting,
    frame_key: OcclusionFrameKey,
    *,
    source_frame_ids: Iterable[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cover selected 2D joints with clipped, per-image-mean RGB squares."""

    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"image must have HxWx3 shape, got {source.shape}")
    if source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("image dimensions must be non-zero")
    joints = np.asarray(joints_2d, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[0] != JOINT_COUNT or joints.shape[1] not in {2, 3}:
        raise ValueError(f"joints_2d must have shape 15x2 or 15x3, got {joints.shape}")
    valid = np.isfinite(joints[:, :2]).all(axis=1)
    if joints.shape[1] == 3:
        valid &= np.isfinite(joints[:, 2]) & (joints[:, 2] > 0.0)
    if int(valid.sum()) < 2:
        raise ValueError("At least two valid projected joints are required")

    height, width = source.shape[:2]
    clipped_x = np.clip(joints[:, 0], 0.0, float(width - 1))
    clipped_y = np.clip(joints[:, 1], 0.0, float(height - 1))
    person_height = float(clipped_y[valid].max() - clipped_y[valid].min())
    if not math.isfinite(person_height) or person_height <= 0.0:
        raise ValueError("Projected person height must be finite and positive")
    side_px = max(
        setting.minimum_side_px,
        int(round(setting.patch_fraction * person_height)),
    )
    selected = selected_joint_mask(
        setting,
        frame_key,
        source_frame_ids=source_frame_ids,
    )
    selected &= valid

    fill = np.rint(source.astype(np.float64).mean(axis=(0, 1)))
    if np.issubdtype(source.dtype, np.integer):
        limits = np.iinfo(source.dtype)
        fill = np.clip(fill, limits.min, limits.max)
    fill = fill.astype(source.dtype, copy=False)
    masked = source.copy()
    boxes: list[list[int]] = []
    for joint in np.flatnonzero(selected):
        center_x = int(round(float(clipped_x[joint])))
        center_y = int(round(float(clipped_y[joint])))
        x0 = max(0, center_x - side_px // 2)
        y0 = max(0, center_y - side_px // 2)
        x1 = min(width, x0 + side_px)
        y1 = min(height, y0 + side_px)
        # Keep the requested side when clipping against the far edge when possible.
        x0 = max(0, x1 - side_px)
        y0 = max(0, y1 - side_px)
        masked[y0:y1, x0:x1] = fill
        boxes.append([x0, y0, x1, y1])
    return masked, {
        "person_id": frame_key.person_id,
        "action_id": frame_key.action_id,
        "camera_id": frame_key.camera_id,
        "frame_id": int(frame_key.frame_id),
        "pattern": setting.pattern,
        "ratio": float(setting.ratio),
        "person_height_px": person_height,
        "side_px": side_px,
        "selected_joint_indices": [int(value) for value in np.flatnonzero(selected)],
        "selected_joint_count": int(selected.sum()),
        "boxes_xyxy": boxes,
        "fill_rgb": fill.tolist(),
    }


def _extract_frame_index(path: Path) -> int:
    matches = _FRAME_INDEX_RE.findall(path.stem)
    if not matches:
        raise ValueError(f"No frame index found in filename: {path.name}")
    preferred = [value for value in matches if len(value) >= 6]
    return int(preferred[0] if preferred else matches[-1])


def _file_map(path: Path, pattern: str) -> dict[int, Path]:
    if not path.is_dir():
        return {}
    result: dict[int, Path] = {}
    for file_path in sorted(path.glob(pattern)):
        index = _extract_frame_index(file_path)
        if index in result:
            raise ValueError(f"Duplicate frame index {index} under {path}")
        result[index] = file_path
    return result


def _none_detected(path: Path) -> set[int]:
    none_path = path / "none_detected_frames.txt"
    if not none_path.is_file():
        return set()
    indices: set[int] = set()
    for raw in none_path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value:
            continue
        try:
            index = int(value)
        except ValueError as error:
            raise ValueError(f"Invalid none-detected frame {value!r}: {none_path}") from error
        if index < 0:
            raise ValueError(f"Negative none-detected frame {index}: {none_path}")
        indices.add(index)
    return indices


def _rewrite_path(
    raw_path: str,
    *,
    data_root: Path | None,
    rewrite_from: Path | None,
) -> Path:
    source = Path(raw_path)
    if rewrite_from is not None and data_root is not None:
        try:
            return data_root / source.relative_to(rewrite_from)
        except ValueError:
            pass
    if source.exists():
        return source
    if data_root is not None and "skiing_unity_dataset" in source.parts:
        marker = source.parts.index("skiing_unity_dataset")
        return data_root.joinpath(*source.parts[marker + 1 :])
    return source


def _load_test_rows(index_source: Path | Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(index_source, Mapping):
        payload = index_source
    else:
        payload = json.loads(Path(index_source).read_text(encoding="utf-8-sig"))
    rows = payload.get("test")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Fold index requires a non-empty test split")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("Every fold test row must be an object")
    return rows


def build_required_frames_manifest(
    index_source: Path | Mapping[str, Any],
    *,
    target_length: int = 30,
    data_root: Path | None = None,
    rewrite_from: Path | None = None,
) -> dict[str, Any]:
    """Derive the exact native-loader frame positions and per-stream union."""

    if target_length <= 0:
        raise ValueError("target_length must be positive")
    rows = _load_test_rows(index_source)
    map_cache: dict[tuple[Path, str], dict[int, Path]] = {}

    def cached_map(path: Path, pattern: str) -> dict[int, Path]:
        key = (path, pattern)
        if key not in map_cache:
            map_cache[key] = _file_map(path, pattern)
        return map_cache[key]

    stream_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    pair_sequences: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for row in rows:
        person_id = str(row["person_id"])
        action_id = str(row["action_id"])
        cam1_id = str(row["cam1_id"])
        cam2_id = str(row["cam2_id"])
        pair_id = "|".join(sorted((cam1_id, cam2_id)))
        pair_key = (action_id, person_id, pair_id)
        if pair_key in seen_pairs:
            raise ValueError(f"Duplicate fold test camera pair {pair_key!r}")
        seen_pairs.add(pair_key)

        paths = {
            key: _rewrite_path(
                str(row[key]), data_root=data_root, rewrite_from=rewrite_from
            )
            for key in (
                "cam1_frames_dir",
                "cam2_frames_dir",
                "cam1_kpt2d_dir",
                "cam2_kpt2d_dir",
                "kpt3d_dir",
                "sam3d_cam1_kpt3d_dir",
                "sam3d_cam2_kpt3d_dir",
            )
        }
        gt_indices = set(cached_map(paths["kpt3d_dir"], "frame_*.npy"))
        sam1_indices = set(
            cached_map(paths["sam3d_cam1_kpt3d_dir"], "kpt3d_*.npy")
        )
        sam2_indices = set(
            cached_map(paths["sam3d_cam2_kpt3d_dir"], "kpt3d_*.npy")
        )
        common = sorted(
            (gt_indices & sam1_indices & sam2_indices)
            - _none_detected(paths["sam3d_cam1_kpt3d_dir"])
            - _none_detected(paths["sam3d_cam2_kpt3d_dir"])
        )
        if not common:
            raise ValueError(f"No native valid frames for {pair_key!r}")
        positions = torch.round(
            torch.linspace(0, len(common) - 1, target_length, dtype=torch.float32)
        ).long()
        selected = [common[int(position)] for position in positions.tolist()]
        pair_sequences.append(
            {
                "person_id": person_id,
                "action_id": action_id,
                "cam1_id": cam1_id,
                "cam2_id": cam2_id,
                "camera_pair_id": pair_id,
                "native_common_frame_count": len(common),
                "frame_indices": selected,
            }
        )

        for camera_id, rgb_key, kpt_key in (
            (cam1_id, "cam1_frames_dir", "cam1_kpt2d_dir"),
            (cam2_id, "cam2_frames_dir", "cam2_kpt2d_dir"),
        ):
            key = (person_id, action_id, camera_id)
            rgb_dir = paths[rgb_key]
            kpt2d_dir = paths[kpt_key]
            current = stream_records.get(key)
            if current is None:
                current = {
                    "person_id": person_id,
                    "action_id": action_id,
                    "camera_id": camera_id,
                    "rgb_dir": str(rgb_dir),
                    "kpt2d_dir": str(kpt2d_dir),
                    "frame_indices": set(),
                }
                stream_records[key] = current
            elif current["rgb_dir"] != str(rgb_dir) or current["kpt2d_dir"] != str(
                kpt2d_dir
            ):
                raise ValueError(f"Conflicting source paths for stream {key!r}")
            current["frame_indices"].update(selected)

    streams: list[dict[str, Any]] = []
    for key in sorted(stream_records):
        record = stream_records[key]
        frames = sorted(int(value) for value in record["frame_indices"])
        rgb_map = cached_map(Path(record["rgb_dir"]), "frame_*.png")
        kpt2d_map = cached_map(Path(record["kpt2d_dir"]), "kpt2d_*.npy")
        missing_rgb = [frame for frame in frames if frame not in rgb_map]
        missing_kpt = [frame for frame in frames if frame not in kpt2d_map]
        if missing_rgb:
            raise FileNotFoundError(
                f"Stream {key!r} is missing required RGB frames {missing_rgb[:10]}"
            )
        if missing_kpt:
            raise FileNotFoundError(
                f"Stream {key!r} is missing required 2D joints {missing_kpt[:10]}"
            )
        streams.append({**record, "frame_indices": frames})

    unique_images = sum(len(record["frame_indices"]) for record in streams)
    return {
        "schema_version": 1,
        "selection_method": "native_loader_round_linspace_after_gt_sam3d_intersection",
        "target_length": target_length,
        "test_pair_count": len(pair_sequences),
        "pair_position_count": len(pair_sequences) * target_length,
        "stream_count": len(streams),
        "action_count": len({record["action_id"] for record in streams}),
        "unique_required_frame_count": unique_images,
        "unique_source_image_count": unique_images,
        "pair_sequences": pair_sequences,
        "streams": streams,
    }


def mask_protocol_payload(setting: ImageOcclusionSetting) -> dict[str, Any]:
    return {
        "setting": asdict(setting),
        "joint_count": JOINT_COUNT,
        "distal_joint_indices": list(DISTAL_JOINT_INDICES),
        "randomness": "BLAKE2b stable hash keyed by stream, condition, joint, and frame",
        "placement_signal": "Unity ground-truth 2D joints used only for mask placement",
        "fill": "rounded per-image RGB channel mean",
    }
