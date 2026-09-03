#!/usr/bin/env python3
"""Run resumable SAM3D inference on deterministically occluded Unity images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from dual2pose.eval.image_occlusion import (
    ImageOcclusionSetting,
    OcclusionFrameKey,
    apply_image_occlusion,
    build_required_frames_manifest,
    mask_protocol_payload,
)
from dual2pose.map_config import filter_sam3d_body_kpts, filter_unity_kpts


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(
    "/home/kaixu_chen/skiing/data/skiing_unity_dataset"
)
DEFAULT_REWRITE_FROM = Path(
    "/home/kaixu_chen/data/skiing/skiing_unity_dataset"
)
DEFAULT_SAM3D_ROOT = REPO_ROOT / "ckpt/sam-3d-body-dinov3"
DEFAULT_CHECKPOINT = DEFAULT_SAM3D_ROOT / "model.ckpt"
DEFAULT_MHR = DEFAULT_SAM3D_ROOT / "assets/mhr_model.pt"
DEFAULT_CONFIG = REPO_ROOT / "configs/sam3d_body.yaml"


def _safe_component(value: object) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not component:
        raise ValueError(f"Cannot construct an output path from {value!r}")
    return component


def stream_output_path(output_root: Path, stream: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "poses"
        / _safe_component(stream["person_id"])
        / _safe_component(stream["action_id"])
        / f"{_safe_component(stream['camera_id'])}.npz"
    )


def validate_stream_npz(path: Path, *, expected_frames: Iterable[int]) -> bool:
    """Return true only for a complete finite stream artifact with exact IDs."""

    expected = np.asarray(sorted({int(value) for value in expected_frames}), dtype=np.int64)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not {"pose", "frame_indices", "detection_failed"}.issubset(archive.files):
                return False
            pose = np.asarray(archive["pose"])
            frames = np.asarray(archive["frame_indices"])
            failed = np.asarray(archive["detection_failed"])
            selected_counts = (
                np.asarray(archive["selected_joint_count"])
                if "selected_joint_count" in archive
                else None
            )
    except (OSError, ValueError, EOFError):
        return False
    if pose.shape != (len(expected), 15, 3) or not np.isfinite(pose).all():
        return False
    if frames.shape != expected.shape or not np.array_equal(frames, expected):
        return False
    if failed.shape != expected.shape or failed.dtype != np.bool_:
        return False
    if selected_counts is not None and selected_counts.shape != expected.shape:
        return False
    return True


def _default_image_loader(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read RGB frame: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _default_joint_loader(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)


def _filtered_prediction(prediction: Any) -> np.ndarray:
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    filtered = filter_sam3d_body_kpts(np.asarray(prediction, dtype=np.float32))
    if filtered.shape != (15, 3) or not np.isfinite(filtered).all():
        raise ValueError(f"Filtered SAM3D prediction must be finite 15x3, got {filtered.shape}")
    return np.asarray(filtered, dtype=np.float32)


def infer_stream(
    stream: Mapping[str, Any],
    *,
    predictor: Callable[[np.ndarray], Any | None],
    setting: ImageOcclusionSetting,
    output_root: Path,
    image_loader: Callable[[Path], np.ndarray] = _default_image_loader,
    joint_loader: Callable[[Path], np.ndarray] = _default_joint_loader,
) -> Path:
    """Infer one stream, recording no-person detections as explicit zero poses."""

    frame_indices = sorted({int(value) for value in stream["frame_indices"]})
    if not frame_indices:
        raise ValueError("Stream frame_indices must be non-empty")
    output_path = stream_output_path(output_root, stream)
    if output_path.is_file() and validate_stream_npz(
        output_path, expected_frames=frame_indices
    ):
        return output_path

    person_id = str(stream["person_id"])
    action_id = str(stream["action_id"])
    camera_id = str(stream["camera_id"])
    rgb_dir = Path(str(stream["rgb_dir"]))
    kpt2d_dir = Path(str(stream["kpt2d_dir"]))
    poses: list[np.ndarray] = []
    detection_failed: list[bool] = []
    selected_counts: list[int] = []
    for frame_id in frame_indices:
        image_path = rgb_dir / f"frame_{frame_id:06d}.png"
        joint_path = kpt2d_dir / f"kpt2d_{frame_id:06d}.npy"
        image = image_loader(image_path)
        raw_joints = joint_loader(joint_path)
        joints = filter_unity_kpts(raw_joints, flag="2d", gender=person_id)
        frame_key = OcclusionFrameKey(person_id, action_id, camera_id, frame_id)
        masked, record = apply_image_occlusion(
            image,
            joints,
            setting,
            frame_key,
            source_frame_ids=frame_indices,
        )
        prediction = predictor(masked)
        failed = prediction is None
        pose = (
            np.zeros((15, 3), dtype=np.float32)
            if failed
            else _filtered_prediction(prediction)
        )
        poses.append(pose)
        detection_failed.append(failed)
        selected_counts.append(int(record["selected_joint_count"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        pose=np.stack(poses).astype(np.float32, copy=False),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        detection_failed=np.asarray(detection_failed, dtype=bool),
        selected_joint_count=np.asarray(selected_counts, dtype=np.int16),
    )
    if not validate_stream_npz(temporary, expected_frames=frame_indices):
        raise RuntimeError(f"Refusing to publish invalid stream artifact: {temporary}")
    temporary.replace(output_path)
    return output_path


class SAM3DPredictor:
    """Thin single-initialization adapter around the repository SAM3D estimator."""

    def __init__(self, *, config: Path, checkpoint: Path, mhr: Path, gpu: int) -> None:
        from omegaconf import OmegaConf

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cfg = OmegaConf.load(config)
        cfg.model.root_path = str(checkpoint.parent)
        cfg.model.checkpoint_path = str(checkpoint)
        cfg.model.mhr_path = str(mhr)
        cfg.infer.gpu = 0
        from SAM3Dbody.infer import setup_sam_3d_body

        self.estimator = setup_sam_3d_body(cfg)

    def __call__(self, image: np.ndarray) -> Any | None:
        from SAM3Dbody.infer import select_best_person

        outputs = self.estimator.process_one_image(img=image, bboxes=None)
        best, _ = select_best_person(outputs, verbose=False)
        if best is None:
            return None
        if "pred_keypoints_3d" not in best:
            raise KeyError("SAM3D best-person output lacks pred_keypoints_3d")
        return best["pred_keypoints_3d"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize_condition_outputs(
    streams: Sequence[Mapping[str, Any]], output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    entries: list[dict[str, Any]] = []
    completed_frames = 0
    failed_frames = 0
    for stream in streams:
        path = stream_output_path(output_root, stream)
        expected = stream["frame_indices"]
        if not path.is_file() or not validate_stream_npz(path, expected_frames=expected):
            continue
        with np.load(path, allow_pickle=False) as archive:
            failed = np.asarray(archive["detection_failed"], dtype=bool)
        completed_frames += len(expected)
        failed_frames += int(failed.sum())
        entries.append(
            {
                "person_id": str(stream["person_id"]),
                "action_id": str(stream["action_id"]),
                "camera_id": str(stream["camera_id"]),
                "pose_path": str(path.relative_to(output_root)),
                "frame_count": len(expected),
                "detection_failure_count": int(failed.sum()),
            }
        )
    return entries, {
        "completed_stream_count": len(entries),
        "completed_frame_count": completed_frames,
        "detection_failure_count": failed_frames,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rewrite-from", type=Path, default=DEFAULT_REWRITE_FROM)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pattern", choices=("random", "distal", "temporal"), required=True)
    parser.add_argument("--ratio", type=float, choices=(0.5, 1.0), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-streams", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--mhr", type=Path, default=DEFAULT_MHR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must lie in [0, shard-count)")
    data_root = args.data_root.resolve()
    index_path = (
        args.index_path.resolve()
        if args.index_path
        else data_root
        / "index_mapping/use_layer_camera_filter_disabled/"
        "camera_pairs_by_action_folds/fold_00.json"
    )
    output_root = args.output_root.resolve()
    setting = ImageOcclusionSetting(args.pattern, args.ratio, seed=args.seed)
    required = build_required_frames_manifest(
        index_path,
        target_length=30,
        data_root=data_root,
        rewrite_from=args.rewrite_from.resolve(),
    )
    _atomic_json(output_root / "required_frames_manifest.json", required)
    _atomic_json(output_root / "mask_protocol.json", mask_protocol_payload(setting))
    print(
        json.dumps(
            {
                key: required[key]
                for key in (
                    "test_pair_count",
                    "stream_count",
                    "action_count",
                    "unique_required_frame_count",
                )
            }
        )
    )
    if args.dry_run:
        return

    all_streams = list(required["streams"])
    selected_streams = all_streams[args.shard_index :: args.shard_count]
    if args.max_streams is not None:
        if args.max_streams <= 0:
            raise ValueError("max-streams must be positive")
        selected_streams = selected_streams[: args.max_streams]
    predictor = SAM3DPredictor(
        config=args.config.resolve(),
        checkpoint=args.checkpoint.resolve(),
        mhr=args.mhr.resolve(),
        gpu=args.gpu,
    )
    for index, stream in enumerate(selected_streams, start=1):
        path = infer_stream(
            stream,
            predictor=predictor,
            setting=setting,
            output_root=output_root,
        )
        print(f"[{index}/{len(selected_streams)}] {path}", flush=True)

    entries, counts = summarize_condition_outputs(all_streams, output_root)
    complete = len(entries) == len(all_streams)
    metadata = {
        "schema_version": 1,
        "pattern": setting.pattern,
        "ratio": setting.ratio,
        "seed": setting.seed,
        "failure_policy": "no SAM3D person -> all-zero 15x3 pose plus boolean flag",
        "expected_stream_count": required["stream_count"],
        "expected_frame_count": required["unique_required_frame_count"],
        **counts,
        "complete": complete,
        "limited_smoke_test": args.max_streams is not None,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "mhr": str(args.mhr.resolve()),
        "mhr_sha256": _sha256(args.mhr.resolve()),
        "command": " ".join(sys.argv),
    }
    payload = {
        "frontend_name": f"sam3d_image_occlusion_{setting.pattern}_{setting.ratio:g}",
        "joint_indices": None,
        "entries": entries,
        "metadata": metadata,
    }
    manifest_name = "frontend_manifest.json" if complete else "frontend_manifest_partial.json"
    _atomic_json(output_root / manifest_name, payload)
    if not complete and args.max_streams is None and args.shard_count == 1:
        raise RuntimeError("Condition inference finished without complete stream coverage")


if __name__ == "__main__":
    main()
