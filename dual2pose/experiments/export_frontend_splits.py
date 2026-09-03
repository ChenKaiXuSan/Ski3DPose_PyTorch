#!/usr/bin/env python3
"""Export leakage-safe Unity train/val front-end poses and merge split manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Sequence

from dual2pose.eval.export_unity_frontend_predictions import (
    DEFAULT_STALE_DATA_ROOT,
    export_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/kaixu_chen/skiing/data/skiing_unity_dataset")
OUTPUT_ROOT = REPO_ROOT / "logs/ivc_p1/frontend_adaptation/predictions"
FRONTEND_CONFIGS: dict[str, dict[str, Any]] = {
    "videopose3d": {
        "repo": REPO_ROOT / ".cache/frontend_estimators/VideoPose3D",
        "checkpoint": REPO_ROOT / "ckpt/frontends/pretrained_h36m_cpn.bin",
    },
    "poseformer": {
        "repo": REPO_ROOT / ".cache/frontend_estimators/PoseFormer",
        "checkpoint": REPO_ROOT / "ckpt/frontends/poseformer_detected81f.bin",
        "poseformer_frames": 81,
        "allow_numpy_checkpoint_state": True,
    },
    "motionbert": {
        "repo": REPO_ROOT / ".cache/frontend_estimators/MotionBERT",
        "checkpoint": REPO_ROOT / "ckpt/frontends/motionbert_ft_h36m_mmpose.pth",
        "motionbert_config": REPO_ROOT / ".cache/frontend_estimators/MotionBERT/configs/pose3d/MB_ft_h36m.yaml",
        "allow_unsafe_checkpoint": True,
    },
}
EXISTING_TEST_MANIFESTS = {
    name: REPO_ROOT / f"logs/eval_unity_frontend_generalization/predictions/{name}/{name}_manifest.json"
    for name in FRONTEND_CONFIGS
}


def merge_split_manifests(paths: Sequence[Path], output: Path) -> Path:
    if not paths:
        raise ValueError("At least one split manifest is required")
    frontend_name: str | None = None
    joint_indices: list[int] | None = None
    entries: list[dict[str, Any]] = []
    seen_keys: dict[tuple[str, str, str], str] = {}
    split_actions: dict[str, set[str]] = {}
    source_metadata: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        name = str(payload.get("frontend_name", "")).strip()
        metadata = payload.get("metadata")
        if not name or not isinstance(metadata, dict):
            raise ValueError(f"Invalid split manifest: {path}")
        split = str(metadata.get("split", "")).strip()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Manifest must identify train/val/test split: {path}")
        if frontend_name is None:
            frontend_name = name
            joint_indices = [int(value) for value in payload.get("joint_indices", [])]
        elif name != frontend_name or [int(value) for value in payload.get("joint_indices", [])] != joint_indices:
            raise ValueError("Split manifests disagree on front-end or joint convention")
        source_metadata[split] = dict(metadata)
        actions = split_actions.setdefault(split, set())
        for raw_entry in payload.get("entries", []):
            entry = dict(raw_entry)
            entry_split = str(entry.get("split", split))
            if entry_split != split:
                raise ValueError(f"Entry split {entry_split} disagrees with manifest {split}")
            key = (
                str(entry["person_id"]),
                str(entry["action_id"]),
                str(entry["camera_id"]),
            )
            if key in seen_keys:
                raise ValueError(
                    f"Camera stream {key} is assigned to both {seen_keys[key]} and {split}"
                )
            pose_path = Path(str(entry["pose_path"]))
            pose_path = pose_path if pose_path.is_absolute() else path.parent / pose_path
            if not pose_path.is_file():
                raise FileNotFoundError(f"Front-end pose file does not exist: {pose_path}")
            entry["pose_path"] = str(pose_path.resolve())
            entry["split"] = split
            entries.append(entry)
            seen_keys[key] = split
            actions.add(key[1])
    action_owner: dict[str, str] = {}
    for split, actions in split_actions.items():
        for action in actions:
            if action in action_owner and action_owner[action] != split:
                raise ValueError(
                    f"Action {action} is assigned to both {action_owner[action]} and {split}"
                )
            action_owner[action] = split
    split_counts = Counter(str(entry["split"]) for entry in entries)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frontend_name": frontend_name,
        "joint_indices": joint_indices,
        "metadata": {
            "schema_version": 3,
            "split": "all",
            "split_counts": dict(sorted(split_counts.items())),
            "split_actions": {
                split: sorted(actions) for split, actions in sorted(split_actions.items())
            },
            "source_manifests": [str(Path(path).resolve()) for path in paths],
            "source_metadata": source_metadata,
        },
        "entries": sorted(
            entries,
            key=lambda entry: (
                entry["split"], entry["person_id"], entry["action_id"], entry["camera_id"]
            ),
        ),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    return output


def _validate_existing_manifest(path: Path, frontend: str, split: str) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("frontend_name", "")).lower() != frontend:
        raise ValueError(f"Manifest front-end mismatch: {path}")
    if str(payload.get("metadata", {}).get("split", "")) != split:
        raise ValueError(f"Manifest split mismatch: {path}")
    return path


def export_frontend_split(
    frontend: str,
    split: str,
    device: str,
    limit_streams: int | None = None,
) -> Path:
    frontend = frontend.lower()
    if frontend not in FRONTEND_CONFIGS or split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported front-end/split: {frontend}/{split}")
    if split == "test" and EXISTING_TEST_MANIFESTS[frontend].is_file() and limit_streams is None:
        return _validate_existing_manifest(EXISTING_TEST_MANIFESTS[frontend], frontend, split)
    destination = OUTPUT_ROOT / frontend / split
    manifest_path = destination / f"{frontend}_manifest.json"
    if manifest_path.is_file():
        return _validate_existing_manifest(manifest_path, frontend, split)
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Refusing incomplete non-empty export directory: {destination}")
    config = FRONTEND_CONFIGS[frontend]
    args = argparse.Namespace(
        frontend=frontend,
        frontend_repo=Path(config["repo"]),
        checkpoint=Path(config["checkpoint"]),
        data_root=DATA_ROOT,
        fold_json=None,
        split=split,
        output_dir=destination,
        rewrite_from=DEFAULT_STALE_DATA_ROOT,
        device=device,
        batch_size=64,
        poseformer_frames=int(config.get("poseformer_frames", 81)),
        motionbert_config=config.get("motionbert_config"),
        allow_unsafe_checkpoint=bool(config.get("allow_unsafe_checkpoint", False)),
        allow_numpy_checkpoint_state=bool(config.get("allow_numpy_checkpoint_state", False)),
        limit_streams=limit_streams,
        overwrite=False,
    )
    return export_predictions(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontends", nargs="+", choices=tuple(FRONTEND_CONFIGS), default=list(FRONTEND_CONFIGS))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-streams", type=int)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    for frontend in args.frontends:
        manifests = {
            split: export_frontend_split(frontend, split, args.device, args.limit_streams)
            for split in args.splits
        }
        if args.limit_streams is None:
            for split in ("train", "val", "test"):
                manifests.setdefault(split, export_frontend_split(frontend, split, args.device))
            merged = merge_split_manifests(
                [manifests[split] for split in ("train", "val", "test")],
                OUTPUT_ROOT / frontend / f"{frontend}_all_manifest.json",
            )
            print(merged)
        else:
            for path in manifests.values():
                print(path)


if __name__ == "__main__":
    main()
