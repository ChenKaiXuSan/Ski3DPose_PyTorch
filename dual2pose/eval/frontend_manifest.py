"""Validated local-pose manifest support for front-end generalization studies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


ManifestKey = Tuple[str, str, str]


def _sample_key(meta: Mapping[str, Any], camera_id: str) -> ManifestKey:
    try:
        return (
            str(meta["person_id"]),
            str(meta["action_id"]),
            str(camera_id),
        )
    except KeyError as exc:
        raise ValueError(f"Sample metadata is missing {exc.args[0]!r}") from exc


def _resample_pose_length(pose: torch.Tensor, target_length: int) -> torch.Tensor:
    if target_length <= 0:
        raise ValueError("Target pose length must be positive")
    source_length = int(pose.shape[0])
    if source_length <= 0:
        raise ValueError("Front-end pose sequence is empty")
    if source_length == target_length:
        return pose.clone()
    positions = torch.linspace(
        0.0,
        float(source_length - 1),
        target_length,
        dtype=pose.dtype,
        device=pose.device,
    )
    lower = torch.floor(positions).long()
    upper = torch.ceil(positions).long()
    weight = (positions - lower.to(positions.dtype)).view(target_length, 1, 1)
    return (1.0 - weight) * pose.index_select(0, lower) + weight * pose.index_select(
        0, upper
    )


@dataclass(frozen=True)
class FrontEndManifest:
    frontend_name: str
    entries: Dict[ManifestKey, Path]
    source_path: Path
    joint_indices: Tuple[int, ...] | None = None
    metadata: Dict[str, Any] | None = None

    @classmethod
    def load(cls, path: Path) -> "FrontEndManifest":
        source_path = Path(path).resolve()
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Front-end manifest root must be an object")
        frontend_name = str(payload.get("frontend_name", "")).strip()
        if not frontend_name:
            raise ValueError("Front-end manifest requires frontend_name")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError("Front-end manifest requires a non-empty entries list")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError("metadata must be an object")

        raw_joint_indices = payload.get("joint_indices")
        joint_indices: Tuple[int, ...] | None = None
        if raw_joint_indices is not None:
            if not isinstance(raw_joint_indices, list):
                raise ValueError("joint_indices must be a list of non-negative integers")
            joint_indices = tuple(int(value) for value in raw_joint_indices)
            if (
                not joint_indices
                or any(value < 0 for value in joint_indices)
                or len(set(joint_indices)) != len(joint_indices)
            ):
                raise ValueError("joint_indices must be unique non-negative integers")

        entries: Dict[ManifestKey, Path] = {}
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                raise ValueError("Every front-end manifest entry must be an object")
            try:
                key = (
                    str(raw_entry["person_id"]),
                    str(raw_entry["action_id"]),
                    str(raw_entry["camera_id"]),
                )
                raw_pose_path = Path(str(raw_entry["pose_path"]))
            except KeyError as exc:
                raise ValueError(f"Manifest entry is missing {exc.args[0]!r}") from exc
            if key in entries:
                raise ValueError(f"Duplicate front-end manifest key: {key}")
            pose_path = (
                raw_pose_path
                if raw_pose_path.is_absolute()
                else source_path.parent / raw_pose_path
            ).resolve()
            if not pose_path.is_file():
                raise FileNotFoundError(f"Front-end pose file does not exist: {pose_path}")
            entries[key] = pose_path
        return cls(
            frontend_name=frontend_name,
            entries=entries,
            source_path=source_path,
            joint_indices=joint_indices,
            metadata=dict(raw_metadata),
        )

    def validate_coverage(self, samples: Iterable[Mapping[str, Any]]) -> None:
        missing: list[ManifestKey] = []
        for sample in samples:
            for camera_field in ("cam1_id", "cam2_id"):
                camera_id = str(sample.get(camera_field, ""))
                key = _sample_key(sample, camera_id)
                if key not in self.entries:
                    missing.append(key)
        if missing:
            preview = ", ".join(str(key) for key in missing[:5])
            raise KeyError(
                f"Front-end manifest is missing {len(missing)} required streams: {preview}"
            )

    def load_pose(
        self,
        meta: Mapping[str, Any],
        camera_id: str,
        target_length: int,
        expected_joint_count: int,
        target_frame_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = _sample_key(meta, camera_id)
        if key not in self.entries:
            raise KeyError(f"Front-end manifest has no entry for {key}")
        pose_path = self.entries[key]
        source_frame_indices: np.ndarray | None = None
        if pose_path.suffix.lower() == ".npy":
            array = np.load(pose_path, allow_pickle=False)
        elif pose_path.suffix.lower() == ".npz":
            with np.load(pose_path, allow_pickle=False) as archive:
                if "pose" not in archive:
                    raise ValueError(f"NPZ front-end pose requires key 'pose': {pose_path}")
                array = archive["pose"]
                if "frame_indices" in archive:
                    source_frame_indices = np.asarray(archive["frame_indices"])
        else:
            raise ValueError(f"Front-end pose must be .npy or .npz: {pose_path}")
        array = np.asarray(array)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Front-end pose must have shape T x J x 3, got {array.shape}: {pose_path}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Front-end pose contains non-finite values: {pose_path}")
        if source_frame_indices is not None:
            if (
                source_frame_indices.ndim != 1
                or len(source_frame_indices) != len(array)
                or len(set(int(value) for value in source_frame_indices))
                != len(source_frame_indices)
            ):
                raise ValueError(
                    f"frame_indices must be unique and match pose length: {pose_path}"
                )
        if self.joint_indices is not None:
            if max(self.joint_indices) >= array.shape[1]:
                raise ValueError(
                    f"joint_indices exceed pose joint count {array.shape[1]}: {pose_path}"
                )
            array = array[:, self.joint_indices, :]
        if int(array.shape[1]) != int(expected_joint_count):
            raise ValueError(
                "Front-end joint count does not match the model input: "
                f"{array.shape[1]} vs {expected_joint_count} ({pose_path})"
            )
        pose = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        if target_frame_indices is not None and source_frame_indices is not None:
            targets = [int(value) for value in target_frame_indices.detach().cpu().tolist()]
            positions = {int(frame): index for index, frame in enumerate(source_frame_indices)}
            missing = [frame for frame in targets if frame not in positions]
            if missing:
                raise KeyError(
                    f"Front-end pose is missing target frames {missing[:10]}: {pose_path}"
                )
            pose = pose.index_select(
                0, torch.tensor([positions[frame] for frame in targets], dtype=torch.long)
            )
            if len(pose) != target_length:
                raise ValueError(
                    f"Aligned pose length {len(pose)} does not match target {target_length}"
                )
            return pose
        return _resample_pose_length(pose, target_length=target_length)


def replace_frontend_inputs(
    sample: Dict[str, Any], manifest: FrontEndManifest
) -> Dict[str, Any]:
    meta = sample.get("meta")
    streams = sample.get("kpt3d_sam")
    if not isinstance(meta, dict) or not isinstance(streams, dict):
        raise ValueError("Sample requires meta and kpt3d_sam dictionaries")
    out = dict(sample)
    out_streams = dict(streams)
    target_frame_indices = sample.get("frame_indices")
    if target_frame_indices is not None and not isinstance(target_frame_indices, torch.Tensor):
        raise ValueError("frame_indices must be a torch.Tensor when provided")
    for camera_key, metadata_key in (("cam1", "cam1_id"), ("cam2", "cam2_id")):
        base_pose = streams.get(camera_key)
        if not isinstance(base_pose, torch.Tensor) or base_pose.ndim != 3:
            raise ValueError(f"Base {camera_key} pose must have shape T x J x 3")
        camera_id = str(meta.get(metadata_key, ""))
        out_streams[camera_key] = manifest.load_pose(
            meta,
            camera_id=camera_id,
            target_length=int(base_pose.shape[0]),
            expected_joint_count=int(base_pose.shape[1]),
            target_frame_indices=target_frame_indices,
        )
    out["kpt3d_sam"] = out_streams
    out["_frontend_name"] = manifest.frontend_name
    return out


class FrontEndPoseDataset(Dataset):
    """Dataset wrapper that replaces the two front-end 3D pose streams."""

    def __init__(self, base_dataset: Dataset, manifest: FrontEndManifest) -> None:
        self.base_dataset = base_dataset
        self.manifest = manifest
        raw_index = getattr(base_dataset, "_index_mapping", None)
        if isinstance(raw_index, list):
            normalized = [
                item if isinstance(item, dict) else vars(item) for item in raw_index
            ]
            self.manifest.validate_coverage(normalized)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.base_dataset[index]
        if not isinstance(sample, dict):
            raise TypeError("Unity front-end evaluation expects dictionary samples")
        return replace_frontend_inputs(sample, self.manifest)
