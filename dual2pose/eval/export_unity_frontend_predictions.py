#!/usr/bin/env python3
"""Export VideoPose3D, PoseFormer, or MotionBERT predictions for Unity test streams."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from dual2pose.eval.frontend_lifters import (
    MOTIONBERT_H36M_BBOX_CENTER,
    MOTIONBERT_H36M_BBOX_SCALE,
    MOTIONBERT_INFERENCE_FACTOR,
    h36m17_to_canonfuse15,
    load_official_lifter,
    load_unity_2d_stream,
)


logger = logging.getLogger(__name__)
DEFAULT_STALE_DATA_ROOT = Path(
    "/home/kaixu_chen/data/skiing/skiing_unity_dataset"
)


@dataclass(frozen=True)
class UnityFrontEndStream:
    person_id: str
    action_id: str
    camera_id: str
    keypoint_dir: Path
    joint_names_path: Path
    sequence_meta_path: Path
    split: str


def _rewrite_data_path(path: str, rewrite_from: Path | None, data_root: Path) -> Path:
    source = Path(path)
    if rewrite_from is not None:
        try:
            relative = source.relative_to(rewrite_from)
        except ValueError:
            pass
        else:
            return data_root / relative
    if source.exists():
        return source
    marker = "skiing_unity_dataset"
    if marker in source.parts:
        marker_index = source.parts.index(marker)
        return data_root.joinpath(*source.parts[marker_index + 1 :])
    return source


def _discover_unity_streams_one(
    fold_json: Path,
    split: str,
    rewrite_from: Path | None,
    data_root: Path,
) -> list[UnityFrontEndStream]:
    """Return every unique camera stream referenced by one fold split."""

    payload = json.loads(Path(fold_json).read_text(encoding="utf-8-sig"))
    rows = payload.get(split)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Fold {fold_json} has no non-empty {split!r} split")
    streams: dict[tuple[str, str, str], UnityFrontEndStream] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Fold rows must be objects")
        person_id = str(row["person_id"])
        action_id = str(row["action_id"])
        joint_names_path = _rewrite_data_path(
            str(row["joint_names_path"]), rewrite_from, data_root
        )
        sequence_meta_path = _rewrite_data_path(
            str(row["sequence_meta_path"]), rewrite_from, data_root
        )
        for camera_field, path_field in (
            ("cam1_id", "cam1_kpt2d_dir"),
            ("cam2_id", "cam2_kpt2d_dir"),
        ):
            camera_id = str(row[camera_field])
            stream = UnityFrontEndStream(
                person_id=person_id,
                action_id=action_id,
                camera_id=camera_id,
                keypoint_dir=_rewrite_data_path(
                    str(row[path_field]), rewrite_from, data_root
                ),
                joint_names_path=joint_names_path,
                sequence_meta_path=sequence_meta_path,
                split=str(split),
            )
            key = (person_id, action_id, camera_id)
            previous = streams.get(key)
            if previous is not None and previous != stream:
                raise ValueError(f"Conflicting paths for Unity stream {key}")
            streams[key] = stream
    return [streams[key] for key in sorted(streams)]


def discover_unity_streams(
    fold_json: Path,
    split: str,
    rewrite_from: Path | None,
    data_root: Path,
) -> list[UnityFrontEndStream]:
    if split != "all":
        return _discover_unity_streams_one(fold_json, split, rewrite_from, data_root)
    combined: dict[tuple[str, str, str], UnityFrontEndStream] = {}
    membership: dict[tuple[str, str, str], str] = {}
    for split_name in ("train", "val", "test"):
        for stream in _discover_unity_streams_one(
            fold_json, split_name, rewrite_from, data_root
        ):
            key = (stream.person_id, stream.action_id, stream.camera_id)
            if key in combined:
                raise ValueError(
                    f"Unity stream {key} is assigned to both "
                    f"{membership[key]} and {split_name}"
                )
            combined[key] = stream
            membership[key] = split_name
    return [combined[key] for key in sorted(combined)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not component:
        raise ValueError(f"Cannot construct output path from identifier {value!r}")
    return component


def _output_path(output_dir: Path, stream: UnityFrontEndStream) -> Path:
    return (
        output_dir
        / "poses"
        / _safe_component(stream.person_id)
        / _safe_component(stream.action_id)
        / f"{_safe_component(stream.camera_id)}.npz"
    )


def export_predictions(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root).resolve()
    fold_json = (
        Path(args.fold_json).resolve()
        if args.fold_json
        else data_root
        / "index_mapping/use_layer_camera_filter_disabled/"
        "camera_pairs_by_action_folds/fold_00.json"
    )
    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / f"{args.frontend.lower()}_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}; pass --overwrite to replace it"
        )
    streams = discover_unity_streams(
        fold_json=fold_json,
        split=args.split,
        rewrite_from=Path(args.rewrite_from) if args.rewrite_from else None,
        data_root=data_root,
    )
    if args.limit_streams is not None:
        streams = streams[: int(args.limit_streams)]
    if not streams:
        raise ValueError("No Unity camera streams selected")

    lifter = load_official_lifter(
        frontend=args.frontend,
        repo_path=Path(args.frontend_repo),
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
        batch_size=args.batch_size,
        motionbert_config=(
            Path(args.motionbert_config) if args.motionbert_config else None
        ),
        poseformer_frames=args.poseformer_frames,
        allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
        allow_numpy_checkpoint_state=args.allow_numpy_checkpoint_state,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for index, stream in enumerate(streams, start=1):
        keypoints, frame_indices, image_size = load_unity_2d_stream(
            keypoint_dir=stream.keypoint_dir,
            joint_names_path=stream.joint_names_path,
            sequence_meta_path=stream.sequence_meta_path,
        )
        h36m_pose = lifter.predict(keypoints, image_size=image_size)
        pose = h36m17_to_canonfuse15(h36m_pose)
        output_path = _output_path(output_dir, stream)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Prediction already exists: {output_path}; pass --overwrite"
            )
        np.savez_compressed(
            output_path,
            pose=pose,
            frame_indices=frame_indices,
            h36m17_pose=h36m_pose,
        )
        entries.append(
            {
                "person_id": stream.person_id,
                "action_id": stream.action_id,
                "camera_id": stream.camera_id,
                "pose_path": str(output_path.relative_to(output_dir)),
                "split": stream.split,
            }
        )
        logger.info(
            "[%d/%d] %s/%s/%s: %d frames",
            index,
            len(streams),
            stream.person_id,
            stream.action_id,
            stream.camera_id,
            len(frame_indices),
        )

    checkpoint = Path(args.checkpoint).resolve()
    repo_path = Path(args.frontend_repo).resolve()
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "split": args.split,
        "fold_json": str(fold_json),
        "input_2d_source": "unity_gt_h36m17",
        "input_coordinate_system": "h36m_normalized_screen",
        "output_joint_convention": "canonfuse15",
        "h36m_upper_torso_order": "spine_thorax_neck_head",
        "publication_metric_joint_subset": "common13_indices_2_to_14",
        "synthetic_eye_policy": "head_plus_body_forward_and_shoulder_axis",
        "official_repo": str(repo_path),
        "official_repo_commit": _git_revision(repo_path),
        "estimator_checkpoint": str(checkpoint),
        "estimator_checkpoint_sha256": _sha256(checkpoint),
        "stream_count": len(entries),
        "split_counts": {
            split_name: sum(entry["split"] == split_name for entry in entries)
            for split_name in ("train", "val", "test")
            if any(entry["split"] == split_name for entry in entries)
        },
        "limited_smoke_test": args.limit_streams is not None,
        "unsafe_checkpoint_loading_enabled": bool(args.allow_unsafe_checkpoint),
        "numpy_checkpoint_safe_globals_enabled": bool(args.allow_numpy_checkpoint_state),
    }
    if args.frontend.lower().replace("-", "") == "motionbert":
        metadata["motionbert_config"] = str(
            Path(args.motionbert_config).resolve()
            if args.motionbert_config
            else repo_path / "configs/pose3d/MB_ft_h36m.yaml"
        )
        metadata["motionbert_2d_preprocessing"] = "mmpose_h36m_average_bbox"
        metadata["motionbert_h36m_bbox_center"] = list(
            MOTIONBERT_H36M_BBOX_CENTER
        )
        metadata["motionbert_h36m_bbox_scale"] = MOTIONBERT_H36M_BBOX_SCALE
        metadata["motionbert_inference_factor"] = MOTIONBERT_INFERENCE_FACTOR
    manifest = {
        "frontend_name": lifter.name,
        "joint_indices": list(range(15)),
        "metadata": metadata,
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved %s manifest with %d streams", lifter.name, len(entries))
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frontend",
        required=True,
        choices=("videopose3d", "poseformer", "motionbert"),
    )
    parser.add_argument("--frontend-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold-json", type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rewrite-from", type=Path, default=DEFAULT_STALE_DATA_ROOT)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--poseformer-frames", type=int, default=81)
    parser.add_argument("--motionbert-config", type=Path)
    parser.add_argument(
        "--allow-unsafe-checkpoint",
        action="store_true",
        help=(
            "Allow torch pickle loading for a trusted official checkpoint whose "
            "metadata is incompatible with weights_only=True"
        ),
    )
    parser.add_argument(
        "--allow-numpy-checkpoint-state",
        action="store_true",
        help="Allowlist the NumPy RNG metadata used by trusted official PoseFormer checkpoints",
    )
    parser.add_argument("--limit-streams", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit_streams is not None and args.limit_streams <= 0:
        parser.error("--limit-streams must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(levelname)s %(message)s",
    )
    manifest = export_predictions(args)
    print(manifest)


if __name__ == "__main__":
    main()
