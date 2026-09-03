#!/usr/bin/env python3
"""Evaluate frozen CanonFuse3D with SAM3D predictions from occluded images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, seed_everything
from torch.utils.data import DataLoader, Dataset

from dual2pose.eval.extension_experiment_utils import (
    complete_test_dataloader,
    patch_index_mapping_path_rewrite,
)
from dual2pose.eval.frontend_manifest import FrontEndManifest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
)
DEFAULT_POSE_OCCLUSION = (
    REPO_ROOT / "logs/ivc_mmsports_extension/masking/occlusion_summary_combined.csv"
)


@dataclass(frozen=True)
class ImageOcclusionCell:
    pattern: str
    ratio: float
    view_mode: str

    @property
    def name(self) -> str:
        ratio = f"{self.ratio:.2f}".replace(".", "p")
        return f"{self.view_mode}_{self.pattern}_r{ratio}"


def build_image_occlusion_study(
    *,
    patterns: Iterable[str] = ("random", "distal", "temporal"),
    ratios: Iterable[float] = (0.5, 1.0),
    view_modes: Iterable[str] = ("left", "right", "both"),
) -> list[ImageOcclusionCell]:
    normalized_patterns = tuple(str(value) for value in patterns)
    normalized_ratios = tuple(float(value) for value in ratios)
    normalized_views = tuple(str(value) for value in view_modes)
    if set(normalized_patterns) != {"random", "distal", "temporal"}:
        raise ValueError("E5 requires random, distal, and temporal patterns")
    if set(normalized_ratios) != {0.5, 1.0}:
        raise ValueError("E5 requires ratios 0.5 and 1.0")
    if set(normalized_views) != {"left", "right", "both"}:
        raise ValueError("E5 requires left, right, and both view modes")
    cells = [
        ImageOcclusionCell(pattern, ratio, view_mode)
        for pattern in normalized_patterns
        for ratio in normalized_ratios
        for view_mode in normalized_views
    ]
    if len(cells) != 18:
        raise AssertionError("The E5 factorial grid must contain 18 cells")
    return cells


@lru_cache(maxsize=2048)
def _load_occlusion_archive(path_string: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate one condition stream once per data-loader process."""

    path = Path(path_string)
    with np.load(path, allow_pickle=False) as archive:
        if not {"pose", "frame_indices", "detection_failed"}.issubset(archive.files):
            raise ValueError(f"Image-occlusion NPZ lacks required arrays: {path}")
        pose = np.asarray(archive["pose"], dtype=np.float32)
        frames = np.asarray(archive["frame_indices"], dtype=np.int64)
        failed = np.asarray(archive["detection_failed"])
    if pose.ndim != 3 or pose.shape[-1] != 3 or not np.isfinite(pose).all():
        raise ValueError(f"Image-occlusion pose must be finite TxJx3: {path}")
    if (
        frames.ndim != 1
        or len(frames) != len(pose)
        or len(set(int(value) for value in frames)) != len(frames)
        or failed.shape != frames.shape
        or failed.dtype != np.bool_
    ):
        raise ValueError(f"Invalid frame/failure arrays: {path}")
    return pose, frames, failed


@dataclass(frozen=True)
class ImageOcclusionManifest:
    pose_manifest: FrontEndManifest

    @classmethod
    def load(cls, path: Path, *, require_complete: bool = True) -> "ImageOcclusionManifest":
        manifest = FrontEndManifest.load(path)
        metadata = manifest.metadata or {}
        if require_complete:
            if metadata.get("complete") is not True:
                raise ValueError(f"Image-occlusion manifest is not complete: {path}")
            expected = int(metadata.get("expected_stream_count", -1))
            if expected != 720 or len(manifest.entries) != expected:
                raise ValueError(
                    f"Image-occlusion manifest requires 720 streams, got {len(manifest.entries)}"
                )
            completed_frames = int(metadata.get("completed_frame_count", -1))
            expected_frames = int(metadata.get("expected_frame_count", -1))
            if completed_frames != expected_frames or expected_frames != 17_089:
                raise ValueError(
                    "Image-occlusion manifest requires exactly 17089 unique source images"
                )
        return cls(manifest)

    def validate_coverage(self, samples: Iterable[Mapping[str, Any]]) -> None:
        self.pose_manifest.validate_coverage(samples)

    def load_pose_and_failure(
        self,
        meta: Mapping[str, Any],
        *,
        camera_id: str,
        target_frame_indices: torch.Tensor,
        target_length: int,
        expected_joint_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(meta["person_id"]), str(meta["action_id"]), str(camera_id))
        path = self.pose_manifest.entries.get(key)
        if path is None:
            raise KeyError(f"Image-occlusion manifest has no entry for {key}")
        source_pose, source_frames, source_failed = _load_occlusion_archive(str(path))
        if self.pose_manifest.joint_indices is not None:
            source_pose = source_pose[:, self.pose_manifest.joint_indices, :]
        if source_pose.shape[1] != expected_joint_count:
            raise ValueError(
                f"Masked pose joint count {source_pose.shape[1]} does not match "
                f"model input {expected_joint_count}: {path}"
            )
        positions = {int(frame): index for index, frame in enumerate(source_frames)}
        targets = [int(value) for value in target_frame_indices.detach().cpu().tolist()]
        missing = [frame for frame in targets if frame not in positions]
        if missing:
            raise KeyError(f"Failure array is missing target frames {missing[:10]}: {path}")
        aligned = np.asarray([positions[frame] for frame in targets], dtype=np.int64)
        pose = torch.from_numpy(np.ascontiguousarray(source_pose[aligned]))
        failed = torch.from_numpy(np.ascontiguousarray(source_failed[aligned], dtype=bool))
        if len(pose) != target_length:
            raise ValueError(
                f"Aligned masked pose length {len(pose)} does not match {target_length}"
            )
        return pose, failed


def replace_image_occlusion_inputs(
    sample: Mapping[str, Any],
    manifest: ImageOcclusionManifest,
    view_mode: str,
) -> dict[str, Any]:
    """Replace selected streams after native frame filtering and selection."""

    if view_mode not in {"left", "right", "both"}:
        raise ValueError("view_mode must be left, right, or both")
    meta = sample.get("meta")
    streams = sample.get("kpt3d_sam")
    frame_indices = sample.get("frame_indices")
    if not isinstance(meta, Mapping) or not isinstance(streams, Mapping):
        raise ValueError("Sample requires meta and kpt3d_sam mappings")
    if not isinstance(frame_indices, torch.Tensor) or frame_indices.ndim != 1:
        raise ValueError("Sample requires one-dimensional frame_indices")
    output = dict(sample)
    output_streams = dict(streams)
    failures: dict[str, torch.Tensor] = {}
    selected_views = {
        "left": {"cam1"},
        "right": {"cam2"},
        "both": {"cam1", "cam2"},
    }[view_mode]
    for view_key, camera_key in (("cam1", "cam1_id"), ("cam2", "cam2_id")):
        base_pose = streams.get(view_key)
        if not isinstance(base_pose, torch.Tensor) or base_pose.ndim != 3:
            raise ValueError(f"Base {view_key} pose must have shape TxJx3")
        if view_key in selected_views:
            pose, failed = manifest.load_pose_and_failure(
                meta,
                camera_id=str(meta[camera_key]),
                target_frame_indices=frame_indices,
                target_length=int(base_pose.shape[0]),
                expected_joint_count=int(base_pose.shape[1]),
            )
            output_streams[view_key] = pose
            failures[view_key] = failed
        else:
            failures[view_key] = torch.zeros(len(frame_indices), dtype=torch.bool)
    output["kpt3d_sam"] = output_streams
    output["image_occlusion_failed"] = failures
    output["_image_occlusion_view_mode"] = view_mode
    return output


class ImageOcclusionDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        manifest: ImageOcclusionManifest,
        view_mode: str,
    ) -> None:
        self.base_dataset = base_dataset
        self.manifest = manifest
        self.view_mode = view_mode
        raw_index = getattr(base_dataset, "_index_mapping", None)
        if isinstance(raw_index, list):
            normalized = [
                item if isinstance(item, dict) else vars(item) for item in raw_index
            ]
            manifest.validate_coverage(normalized)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base_dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError("Image-occlusion evaluation requires dictionary samples")
        return replace_image_occlusion_inputs(sample, self.manifest, self.view_mode)


class ImageOcclusionUnityDataModule(LightningDataModule):
    def __init__(self, base_dm: LightningDataModule, manifest: ImageOcclusionManifest, view_mode: str):
        super().__init__()
        self.base_dm = base_dm
        self.manifest = manifest
        self.view_mode = view_mode

    def prepare_data(self) -> None:
        self.base_dm.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        self.base_dm.setup(stage)
        base_dataset = getattr(self.base_dm, "test_gait_dataset", None)
        if base_dataset is None:
            raise RuntimeError("UnityDataModule did not create a test dataset")
        self.base_dm.test_gait_dataset = ImageOcclusionDataset(
            base_dataset,
            self.manifest,
            self.view_mode,
        )

    def test_dataloader(self) -> DataLoader:
        return complete_test_dataloader(self.base_dm.test_dataloader())


def summarize_cell(
    test_outputs: list[dict[str, Any]],
    *,
    failure_flags: Mapping[str, torch.Tensor],
    failure_threshold: float,
) -> dict[str, Any]:
    """Summarize one E5 cell without dropping negative fusion gains."""

    from dual2pose.eval.eval_unity_masking import (
        _collect_alpha_tensor,
        _flatten_test_outputs,
        _summarize_outputs,
    )
    from dual2pose.eval.eval_unity_temporal_offset import (
        _summarize_gate_error_relationship,
    )

    flat = _flatten_test_outputs(test_outputs)
    metrics = _summarize_outputs(flat, failure_threshold=failure_threshold)
    if "fused" not in metrics or "canonical_avg" not in metrics:
        raise ValueError("No valid E5 model outputs were produced")
    fused = metrics["fused"]
    canonical = metrics["canonical_avg"]
    fused_mpjpe = float(fused["mpjpe"])
    canonical_mpjpe = float(canonical["mpjpe"])
    gain = canonical_mpjpe - fused_mpjpe
    alpha = _collect_alpha_tensor(test_outputs)
    failure_chunks = [value.reshape(-1).bool() for value in failure_flags.values()]
    if not failure_chunks:
        raise ValueError("At least one affected-view failure array is required")
    failure_rate = float(torch.cat(failure_chunks).float().mean().item())
    row = {
        "sample_count": int(flat["fused"].shape[0]),
        "frame_count": int(flat["fused"].shape[0] * flat["fused"].shape[1]),
        "joint_count": int(flat["fused"].shape[2]),
        "fused_mpjpe": fused_mpjpe,
        "canonical_avg_mpjpe": canonical_mpjpe,
        "fusion_gain_mpjpe": gain,
        "fusion_gain_percent": (
            100.0 * gain / canonical_mpjpe
            if canonical_mpjpe != 0.0
            else float("nan")
        ),
        "fused_acceleration_error": float(fused["acceleration_error"]),
        "canonical_avg_acceleration_error": float(canonical["acceleration_error"]),
        "fused_failure_rate": float(fused["failure_rate"]),
        "alpha_global_mean": float(alpha.mean().item()) if alpha is not None else float("nan"),
        "alpha_global_std": float(alpha.std().item()) if alpha is not None else float("nan"),
        "sam3d_detection_failure_rate": failure_rate,
        **_summarize_gate_error_relationship(test_outputs),
    }
    return row


def collect_pair_aligned_failure_flags(
    required_manifest: Mapping[str, Any],
    manifest: ImageOcclusionManifest,
    view_mode: str,
) -> dict[str, torch.Tensor]:
    selected = {"left": ("cam1",), "right": ("cam2",), "both": ("cam1", "cam2")}[view_mode]
    cache: dict[tuple[str, str, str], tuple[dict[int, bool], Path]] = {}
    chunks: dict[str, list[torch.Tensor]] = {key: [] for key in selected}
    for pair in required_manifest["pair_sequences"]:
        meta = {"person_id": pair["person_id"], "action_id": pair["action_id"]}
        targets = [int(value) for value in pair["frame_indices"]]
        for view_key in selected:
            camera_id = str(pair[f"{view_key}_id"])
            key = (str(meta["person_id"]), str(meta["action_id"]), camera_id)
            if key not in cache:
                path = manifest.pose_manifest.entries[key]
                with np.load(path, allow_pickle=False) as archive:
                    frames = np.asarray(archive["frame_indices"])
                    failed = np.asarray(archive["detection_failed"])
                cache[key] = (
                    {int(frame): bool(flag) for frame, flag in zip(frames, failed)},
                    path,
                )
            lookup, path = cache[key]
            missing = [frame for frame in targets if frame not in lookup]
            if missing:
                raise KeyError(f"Missing pair-aligned failures {missing[:10]}: {path}")
            chunks[view_key].append(
                torch.tensor([lookup[frame] for frame in targets], dtype=torch.bool)
            )
    return {key: torch.cat(value) for key, value in chunks.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_root(root: Path, pattern: str, ratio: float) -> Path:
    component = f"{ratio:g}".replace(".", "p")
    path = root / pattern / component
    if not path.is_dir():
        alternate = root / pattern / f"r{component}"
        if alternate.is_dir():
            return alternate
    return path


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _comparison_rows(image_rows: Sequence[Mapping[str, Any]], pose_csv: Path) -> list[dict[str, Any]]:
    with pose_csv.open("r", encoding="utf-8", newline="") as handle:
        pose_rows = list(csv.DictReader(handle))
    pose_by_key = {
        (row["view_mode"], row["pattern"], float(row["ratio"])): row
        for row in pose_rows
    }
    compared: list[dict[str, Any]] = []
    for image in image_rows:
        key = (str(image["view_mode"]), str(image["pattern"]), float(image["ratio"]))
        if key not in pose_by_key:
            raise KeyError(f"Pose-stream occlusion summary lacks E5 match {key}")
        pose = pose_by_key[key]
        pose_mpjpe = float(pose["fused_mpjpe"])
        image_mpjpe = float(image["fused_mpjpe"])
        compared.append(
            {
                "view_mode": key[0],
                "pattern": key[1],
                "ratio": key[2],
                "image_fused_mpjpe": image_mpjpe,
                "pose_fused_mpjpe": pose_mpjpe,
                "image_minus_pose_mpjpe": image_mpjpe - pose_mpjpe,
                "image_fusion_gain_percent": image["fusion_gain_percent"],
                "pose_delta_full_minus_avg": pose["delta_mpjpe_full_minus_avg"],
                "sam3d_detection_failure_rate": image["sam3d_detection_failure_rate"],
            }
        )
    return compared


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pose-occlusion-csv", type=Path, default=DEFAULT_POSE_OCCLUSION)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failure-threshold", type=float, default=0.15)
    parser.add_argument("--limit-cells", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    seed_everything(args.seed, workers=True)
    with initialize_config_dir(
        version_base=None,
        config_dir=str((REPO_ROOT / "configs").resolve()),
    ):
        base_config = compose(
            config_name="dual2pose",
            overrides=[
                f"train.gpu={args.gpu}",
                "train.fold=0",
                f"train.seed={args.seed}",
                f"data.batch_size={args.batch_size}",
                f"data.num_workers={args.num_workers}",
                "data.load_frames=false",
                "data.load_2d_kpt=false",
                "data.load_3d_kpt=true",
                "data.load_mask=false",
            ],
        )
    patch_index_mapping_path_rewrite(
        old_root=str(base_config.data.unity.index_path_rewrite_from),
        new_root=str(base_config.data.unity.root_path),
    )
    from dual2pose.eval import eval_unity_masking as helpers

    cells = build_image_occlusion_study()
    if args.limit_cells is not None:
        cells = cells[: args.limit_cells]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for cell in cells:
        condition_root = _condition_root(args.manifest_root.resolve(), cell.pattern, cell.ratio)
        manifest_path = condition_root / "frontend_manifest.json"
        required_path = condition_root / "required_frames_manifest.json"
        manifest = ImageOcclusionManifest.load(manifest_path)
        required = json.loads(required_path.read_text(encoding="utf-8"))
        failure_flags = collect_pair_aligned_failure_flags(required, manifest, cell.view_mode)

        config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        run_dir = output_root / "cells" / cell.name
        config.log_path = str(run_dir)
        model = helpers._build_model(config)
        base_dm = helpers.UnityDataModule(config)
        datamodule = ImageOcclusionUnityDataModule(base_dm, manifest, cell.view_mode)
        trainer = helpers._build_trainer(config, save_dir=run_dir)
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=str(args.checkpoint.resolve()),
            weights_only=False,
        )
        test_outputs = list(getattr(model, "test_outputs", []))
        summary = summarize_cell(
            test_outputs,
            failure_flags=failure_flags,
            failure_threshold=args.failure_threshold,
        )
        if summary["sample_count"] != 64_440 or summary["joint_count"] != 15:
            raise ValueError(f"Incomplete E5 cell {cell.name}: {summary}")
        finite_keys = (
            "fused_mpjpe",
            "canonical_avg_mpjpe",
            "fusion_gain_mpjpe",
            "fusion_gain_percent",
            "fused_acceleration_error",
            "canonical_avg_acceleration_error",
            "sam3d_detection_failure_rate",
        )
        if any(not math.isfinite(float(summary[key])) for key in finite_keys):
            raise ValueError(f"E5 cell {cell.name} contains non-finite primary metrics")
        row = {
            "setting": cell.name,
            "view_mode": cell.view_mode,
            "pattern": cell.pattern,
            "ratio": cell.ratio,
            **summary,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
            "frontend_manifest": str(manifest_path),
            "frontend_manifest_sha256": _sha256(manifest_path),
            "fold": 0,
            "seed": args.seed,
            "joint_subset": "all15",
            "units": "dataset_coordinate_units",
        }
        rows.append(row)
        details.append({"cell": row, "metrics_available": list(test_outputs[0]) if test_outputs else []})
        print(f"Completed {cell.name}: fused MPJPE={row['fused_mpjpe']:.6f}", flush=True)

    expected_rows = 18 if args.limit_cells is None else args.limit_cells
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} E5 rows, got {len(rows)}")
    _atomic_csv(output_root / "image_occlusion_summary_last.csv", rows)
    _atomic_json(
        output_root / "image_occlusion_summary_last.json",
        {
            "experiment": "unity_image_occlusion_through_sam3d",
            "cell_count": len(rows),
            "failure_policy": "no SAM3D person -> all-zero 15x3 pose plus boolean flag",
            "rows": rows,
            "details": details,
        },
    )
    comparison = _comparison_rows(rows, args.pose_occlusion_csv.resolve())
    _atomic_csv(output_root / "image_vs_pose_occlusion_last.csv", comparison)


if __name__ == "__main__":
    main()
