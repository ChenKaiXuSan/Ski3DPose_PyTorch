#!/usr/bin/env python3
"""Evaluate frozen pairwise CanonFuse3D composition for N=1,2,3,4 views."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from dual2pose.eval.nview_acceptance import collect_accepted_samples
from dual2pose.eval.nview_protocol import (
    InsufficientCommonFrames,
    MultiViewSample,
    build_nested_camera_groups,
    load_multiview_sample,
    nested_cameras,
)
from dual2pose.path_rewrite import rewrite_data_paths
from dual2pose.trainer.canonicalize import canonicalize_pose_torch
from dual2pose.training_protocol import resolve_fold_index_path, validate_fold_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "logs/train_unity/crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
DEFAULT_STALE_ROOT = "/home/kaixu_chen/data/skiing/skiing_unity_dataset"
OUTPUT_ROOT = REPO_ROOT / "logs/ivc_p1/nview"


def resolve_nview_inputs(
    configured_root: str | Path,
    explicit_root: str | None = None,
) -> tuple[Path, Path]:
    """Resolve and validate the relocated Unity root used by N-view evaluation."""

    data_root = Path(explicit_root or configured_root).expanduser().resolve()
    fold_path = resolve_fold_index_path(data_root, 0).resolve()
    validate_fold_metadata(fold_path, 0)
    return data_root, fold_path


def _canonicalize(model: Any, pose: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "predict_pair") and not hasattr(model, "models"):
        return pose
    canonical, _ = canonicalize_pose_torch(
        pose.unsqueeze(0),
        left_hip=int(model.left_hip_idx),
        right_hip=int(model.right_hip_idx),
        neck=int(model.neck_idx),
    )
    return canonical.squeeze(0)


def _predict_pair(
    model: Any,
    left_camera: str,
    right_camera: str,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
) -> torch.Tensor:
    if hasattr(model, "predict_pair"):
        return model.predict_pair(left_camera, right_camera, left_pose, right_pose)
    left = _canonicalize(model, left_pose)
    right = _canonicalize(model, right_pose)
    fused, _ = model.models(left.unsqueeze(0), right.unsqueeze(0))
    return fused.squeeze(0)


@torch.no_grad()
def evaluate_nview_group(
    model: Any,
    sample: MultiViewSample,
    n_views: int,
) -> dict[str, Any]:
    selected = nested_cameras(sample.group, n_views)
    canonical_inputs = {
        camera: _canonicalize(model, sample.poses[camera]) for camera in selected
    }
    gt = _canonicalize(model, sample.ground_truth)
    output: dict[str, Any] = {
        "group_id": sample.group.group_id,
        "n_views": int(n_views),
        "selected_cameras": selected,
        "gt": gt,
        "single_view": canonical_inputs[selected[0]],
        "pair_forward_count": 0,
        "inference_seconds": 0.0,
        "peak_gpu_bytes": 0,
    }
    if n_views == 1:
        return output
    output["nview_canonical_mean"] = torch.stack(
        [canonical_inputs[camera] for camera in selected], dim=0
    ).mean(dim=0)
    pairs = list(itertools.combinations(selected, 2))
    if gt.is_cuda:
        torch.cuda.synchronize(gt.device)
        torch.cuda.reset_peak_memory_stats(gt.device)
    started = time.perf_counter()
    pair_predictions = [
        _predict_pair(
            model,
            left_camera,
            right_camera,
            sample.poses[left_camera],
            sample.poses[right_camera],
        )
        for left_camera, right_camera in pairs
    ]
    pairwise_mean = torch.stack(pair_predictions, dim=0).mean(dim=0)
    if gt.is_cuda:
        torch.cuda.synchronize(gt.device)
    output["inference_seconds"] = time.perf_counter() - started
    output["peak_gpu_bytes"] = (
        int(torch.cuda.max_memory_allocated(gt.device)) if gt.is_cuda else 0
    )
    output["pair_forward_count"] = len(pair_predictions)
    output["pairwise_canonfuse_mean"] = pairwise_mean
    pair_errors = torch.stack(
        [torch.norm(prediction - gt, dim=-1).mean() for prediction in pair_predictions]
    )
    oracle_index = int(torch.argmin(pair_errors).item())
    output["pairwise_oracle_select"] = pair_predictions[oracle_index]
    output["pairwise_oracle_pair"] = pairs[oracle_index]
    return output


@torch.no_grad()
def warmup_pairwise_composition(
    model: Any,
    sample: MultiViewSample,
    iterations: int = 10,
) -> None:
    """Warm all six serial pair forwards used by the four-view condition."""

    if int(iterations) < 0:
        raise ValueError("Warm-up iterations must be non-negative")
    for _ in range(int(iterations)):
        evaluate_nview_group(model, sample, n_views=4)


def _acceleration_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape[0] < 3:
        return float("nan")
    pred_acc = prediction[2:] - 2.0 * prediction[1:-1] + prediction[:-2]
    gt_acc = target[2:] - 2.0 * target[1:-1] + target[:-2]
    return float(torch.norm(pred_acc - gt_acc, dim=-1).mean().item())


def summarize_nview_rows(
    rows: Sequence[Mapping[str, Any]],
    bootstrap_seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["n_views"]), str(row["method"])), []).append(
            float(row["mpjpe"])
        )
    rng = np.random.default_rng(int(bootstrap_seed))
    summary: list[dict[str, Any]] = []
    for (n_views, method), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            continue
        indices = rng.integers(0, len(array), size=(int(bootstrap_samples), len(array)))
        means = array[indices].mean(axis=1)
        summary.append(
            {
                "n_views": n_views,
                "method": method,
                "group_count": len(array),
                "mpjpe_mean": float(array.mean()),
                "mpjpe_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "mpjpe_ci95_low": float(np.quantile(means, 0.025)),
                "mpjpe_ci95_high": float(np.quantile(means, 0.975)),
                "upper_bound": method == "pairwise_oracle_select",
            }
        )
    return summary


def summarize_nview_efficiency(
    rows: Sequence[Mapping[str, Any]],
    *,
    warmup_iterations: int = 10,
    device_name: str = "unknown",
    torch_version: str = "unknown",
    cuda_version: str | None = None,
    sequence_frames: int = 30,
) -> list[dict[str, Any]]:
    """Summarize synchronized serial pairwise latency for deployable N-view rows."""

    selected = [
        row for row in rows if str(row.get("method")) == "pairwise_canonfuse_mean"
    ]
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(int(row["n_views"]), []).append(row)
    expected_views = {2, 3, 4}
    if set(grouped) != expected_views:
        raise ValueError(
            f"Efficiency summary views mismatch; expected={sorted(expected_views)}, "
            f"actual={sorted(grouped)}"
        )

    summary: list[dict[str, Any]] = []
    for n_views in sorted(grouped):
        n_rows = grouped[n_views]
        group_ids = [str(row["group_id"]) for row in n_rows]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError(f"Duplicate efficiency group rows for N={n_views}")
        expected_pairs = n_views * (n_views - 1) // 2
        pair_counts = {int(row["pair_forward_count"]) for row in n_rows}
        if pair_counts != {expected_pairs}:
            raise ValueError(
                f"N={n_views} pair count mismatch; expected={expected_pairs}, "
                f"actual={sorted(pair_counts)}"
            )
        latency_ms = np.asarray(
            [float(row["inference_seconds"]) * 1000.0 for row in n_rows],
            dtype=np.float64,
        )
        if np.any(latency_ms <= 0.0):
            raise ValueError(f"N={n_views} latency must be positive")
        mean_ms = float(latency_ms.mean())
        summary.append(
            {
                "n_views": n_views,
                "pair_forward_count": expected_pairs,
                "group_count": len(n_rows),
                "mpjpe_mean": float(np.mean([float(row["mpjpe"]) for row in n_rows])),
                "latency_mean_ms": mean_ms,
                "latency_std_ms": (
                    float(latency_ms.std(ddof=1)) if len(latency_ms) > 1 else 0.0
                ),
                "latency_median_ms": float(np.median(latency_ms)),
                "latency_p95_ms": float(np.quantile(latency_ms, 0.95)),
                "relative_latency_vs_two_view": 0.0,
                "throughput_groups_per_second": 1000.0 / mean_ms,
                "peak_gpu_memory_mib": max(
                    float(row.get("peak_gpu_bytes", 0)) for row in n_rows
                ) / (1024.0 * 1024.0),
                "mpjpe_reduction_vs_two_view_pct": 0.0,
                "warmup_iterations": int(warmup_iterations),
                "execution_mode": "serial_all_unique_pairs",
                "device_name": str(device_name),
                "torch_version": str(torch_version),
                "cuda_version": str(cuda_version or "none"),
                "sequence_frames": int(sequence_frames),
            }
        )

    two_view_latency = float(summary[0]["latency_mean_ms"])
    two_view_mpjpe = float(summary[0]["mpjpe_mean"])
    for row in summary:
        row["relative_latency_vs_two_view"] = (
            float(row["latency_mean_ms"]) / two_view_latency
        )
        row["mpjpe_reduction_vs_two_view_pct"] = (
            100.0 * (two_view_mpjpe - float(row["mpjpe_mean"])) / two_view_mpjpe
        )
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _camera_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        person = str(row["person_id"])
        action = str(row["action_id"])
        for camera_field, pose_field in (
            ("cam1_id", "sam3d_cam1_kpt3d_dir"),
            ("cam2_id", "sam3d_cam2_kpt3d_dir"),
        ):
            key = (person, action, str(row[camera_field]))
            value = {
                "sam3d_kpt3d_dir": str(row[pose_field]),
                "kpt3d_dir": str(row["kpt3d_dir"]),
            }
            previous = output.get(key)
            if previous is not None and previous != value:
                raise ValueError(f"Conflicting N-view stream paths for {key}")
            output[key] = value
    return output


def _load_model(config: DictConfig, checkpoint: Path, device: torch.device) -> Any:
    from dual2pose.trainer.train_crossview_fusion import CrossViewFusionTrainer

    model = CrossViewFusionTrainer(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


@hydra.main(version_base=None, config_path="../../configs", config_name="dual2pose.yaml")
def main(config: DictConfig) -> None:
    data_root, fold_path = resolve_nview_inputs(
        configured_root=str(config.data.unity.root_path),
        explicit_root=os.environ.get("UNITY_DATA_ROOT"),
    )
    payload = json.loads(fold_path.read_text(encoding="utf-8-sig"))
    rows = rewrite_data_paths(
        payload["test"],
        old_root=str(getattr(config.data.unity, "index_path_rewrite_from", DEFAULT_STALE_ROOT)),
        new_root=str(data_root),
    )
    groups = build_nested_camera_groups(rows)
    if len(groups) != 180:
        raise RuntimeError(f"N-view protocol requires 180 proposed groups, got {len(groups)}")
    limit = int(os.environ.get("NVIEW_LIMIT_GROUPS", "0"))
    checkpoint = Path(os.environ.get("EVAL_CKPT_PATH", str(DEFAULT_CHECKPOINT))).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(config, checkpoint, device)
    lookup = _camera_lookup(rows)
    output_root = Path(os.environ.get("EVAL_OUTPUT_ROOT", str(OUTPUT_ROOT))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    accepted_samples, rejected, evaluated_count = collect_accepted_samples(
        groups,
        lookup,
        target_t=int(config.data.time_window),
        limit_accepted=limit,
    )
    warmup_iterations = int(os.environ.get("NVIEW_WARMUP_ITERATIONS", "10"))
    if warmup_iterations < 0:
        raise ValueError("NVIEW_WARMUP_ITERATIONS must be non-negative")
    if accepted_samples:
        warmup_sample = accepted_samples[0][1]
        warmup_sample = MultiViewSample(
            group=warmup_sample.group,
            frame_indices=warmup_sample.frame_indices.to(device),
            poses={
                camera: pose.to(device)
                for camera, pose in warmup_sample.poses.items()
            },
            ground_truth=warmup_sample.ground_truth.to(device),
        )
        warmup_pairwise_composition(
            model,
            warmup_sample,
            iterations=warmup_iterations,
        )
    accepted: list[str] = []
    result_rows: list[dict[str, Any]] = []
    for group, sample in accepted_samples:
        sample = MultiViewSample(
            group=sample.group,
            frame_indices=sample.frame_indices.to(device),
            poses={camera: pose.to(device) for camera, pose in sample.poses.items()},
            ground_truth=sample.ground_truth.to(device),
        )
        accepted.append(group.group_id)
        for n_views in (1, 2, 3, 4):
            result = evaluate_nview_group(model, sample, n_views)
            methods = ["single_view"] if n_views == 1 else [
                "nview_canonical_mean",
                "pairwise_canonfuse_mean",
                "pairwise_oracle_select",
            ]
            for method in methods:
                prediction = result[method]
                mpjpe = float(torch.norm(prediction - result["gt"], dim=-1).mean().item())
                result_rows.append(
                    {
                        "group_id": group.group_id,
                        "person_id": group.person_id,
                        "action_id": group.action_id,
                        "layer": group.layer,
                        "n_views": n_views,
                        "method": method,
                        "mpjpe": mpjpe,
                        "acceleration_error": _acceleration_error(prediction, result["gt"]),
                        "pair_forward_count": result["pair_forward_count"],
                        "inference_seconds": result["inference_seconds"],
                        "peak_gpu_bytes": result["peak_gpu_bytes"],
                        "upper_bound": method == "pairwise_oracle_select",
                        "frame_start": int(sample.frame_indices.min().item()),
                        "frame_end": int(sample.frame_indices.max().item()),
                    }
                )
    manifest_payload = {
        "proposed_group_count": len(groups),
        "evaluated_group_count": evaluated_count,
        "accepted_group_count": len(accepted),
        "accepted_group_ids": accepted,
        "rejected": rejected,
        "limited_smoke_test": limit > 0,
    }
    (output_root / "nview_group_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2), encoding="utf-8"
    )
    if not accepted:
        raise RuntimeError(
            f"No N-view groups met the {int(config.data.time_window)}-frame requirement; "
            f"see {output_root / 'nview_group_manifest.json'}"
        )
    expected_rows = len(accepted) * 10
    if len(result_rows) != expected_rows:
        raise RuntimeError(f"Incomplete N-view rows: expected {expected_rows}, got {len(result_rows)}")
    summary = summarize_nview_rows(result_rows)
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    efficiency = summarize_nview_efficiency(
        result_rows,
        warmup_iterations=warmup_iterations,
        device_name=device_name,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        sequence_frames=int(config.data.time_window),
    )
    _write_csv(output_root / "nview_per_group.csv", result_rows)
    _write_csv(output_root / "nview_summary.csv", summary)
    _write_csv(output_root / "nview_efficiency.csv", efficiency)
    (output_root / "nview_group_manifest.json").write_text(
        json.dumps(
            {
                "proposed_group_count": len(groups),
                "evaluated_group_count": evaluated_count,
                "accepted_group_count": len(accepted),
                "accepted_group_ids": accepted,
                "rejected": rejected,
                "limited_smoke_test": limit > 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_root / "nview_provenance.json").write_text(
        json.dumps(
            {
                "experiment": "ivc_p1_nview",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "fold_json": str(fold_path),
                "fold": 0,
                "seed": 42,
                "joint_subset": "all15",
                "units": "dataset_coordinate_units",
                "pairwise_composition": "all_unique_pairs_arithmetic_mean",
                "timing_execution_mode": "serial_all_unique_pairs",
                "timing_scope": "canonicalization_pair_forwards_and_pair_mean_only",
                "timing_excludes": ["monocular_frontend", "data_loading", "oracle_selection"],
                "warmup_iterations": warmup_iterations,
                "measurement_groups": len(accepted),
                "sequence_frames": int(config.data.time_window),
                "device_name": device_name,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
