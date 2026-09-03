"""Pure helpers shared by the IVC extension experiment entry points."""

from __future__ import annotations

import hashlib
import importlib
import sys
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, cast

import torch
from torch.utils.data import DataLoader, Dataset


_UNITY_CAMERA_RE = re.compile(r"(?:^|_)L(?P<layer>\d+)_A(?P<azimuth>\d+(?:\.\d+)?)$")
DEFAULT_DATA_ROOT_IN_INDEX = "/home/kaixu_chen/data/skiing/skiing_unity_dataset"
LEGACY_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_legacy_import_path() -> None:
    project_root = str(LEGACY_PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _rewrite_data_paths(value: Any, old_root: str, new_root: str) -> Any:
    if old_root == new_root:
        return value
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    if isinstance(value, list):
        return [_rewrite_data_paths(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_data_paths(item, old_root, new_root)
            for key, item in value.items()
        }
    return value


def patch_index_mapping_path_rewrite(old_root: str, new_root: str) -> None:
    """Patch the repository legacy loader so stale absolute roots remain usable."""

    ensure_legacy_import_path()
    data_module = importlib.import_module("dataloader.data_loader")
    original_loader = data_module.load_index_mapping
    if getattr(original_loader, "_ivc_extension_path_rewrite", False):
        return

    def _load_index_mapping_with_rewrite(index_mapping_path: str) -> Dict[str, list]:
        mapping = original_loader(index_mapping_path)
        return cast(
            Dict[str, list],
            _rewrite_data_paths(mapping, old_root=old_root, new_root=new_root),
        )

    _load_index_mapping_with_rewrite._ivc_extension_path_rewrite = True  # type: ignore[attr-defined]
    data_module.load_index_mapping = _load_index_mapping_with_rewrite



def build_experiment_provenance(
    checkpoint: str | Path,
    sample_count: int,
    fold: int,
    seed: int,
    joint_subset: str,
    units: str,
) -> Dict[str, Any]:
    """Return deterministic provenance shared by extension experiments."""

    checkpoint_path = Path(checkpoint).resolve()
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest.hexdigest(),
        "sample_count": int(sample_count),
        "fold": int(fold),
        "seed": int(seed),
        "joint_subset": str(joint_subset),
        "units": str(units),
    }


def complete_test_dataloader(
    base_loader: DataLoader,
    collate_fn: Any | None = None,
    dataset: Dataset | None = None,
) -> DataLoader:
    """Clone a test loader while retaining its final partial batch."""

    kwargs: Dict[str, Any] = {
        "dataset": dataset if dataset is not None else base_loader.dataset,
        "batch_size": base_loader.batch_size,
        "shuffle": False,
        "num_workers": base_loader.num_workers,
        "pin_memory": base_loader.pin_memory,
        "drop_last": False,
        "collate_fn": collate_fn or base_loader.collate_fn,
        "worker_init_fn": base_loader.worker_init_fn,
    }
    if base_loader.num_workers > 0:
        kwargs["persistent_workers"] = base_loader.persistent_workers
        kwargs["prefetch_factor"] = base_loader.prefetch_factor
    return DataLoader(**kwargs)


def resample_pose_rate(
    pose: torch.Tensor,
    rate_error: float,
    anchor: str = "center",
) -> torch.Tensor:
    """Resample B x T x J x C poses under a fractional clock-rate error."""

    if pose.ndim != 4:
        raise ValueError(f"Expected pose shape BxTxJxC, got {tuple(pose.shape)}")
    scale = 1.0 + float(rate_error)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("rate_error must be finite and greater than -1")
    if anchor not in {"center", "start"}:
        raise ValueError("anchor must be center or start")
    time_len = int(pose.shape[1])
    if time_len == 0 or rate_error == 0.0:
        return pose.clone()

    anchor_position = (time_len - 1) / 2.0 if anchor == "center" else 0.0
    target = torch.arange(time_len, device=pose.device, dtype=pose.dtype)
    positions = anchor_position + (target - anchor_position) / scale
    positions = positions.clamp(0.0, float(time_len - 1))
    lower = torch.floor(positions).long()
    upper = torch.ceil(positions).long()
    weight = (positions - lower.to(dtype=positions.dtype)).view(1, time_len, 1, 1)
    lower_pose = pose.index_select(dim=1, index=lower)
    upper_pose = pose.index_select(dim=1, index=upper)
    return (1.0 - weight) * lower_pose + weight * upper_pose


def parse_unity_camera_id(camera_id: str) -> Tuple[int, float]:
    """Return ``(layer, azimuth_degrees)`` from a Unity camera identifier."""

    match = _UNITY_CAMERA_RE.search(str(camera_id))
    if match is None:
        raise ValueError(
            f"Unsupported Unity camera id {camera_id!r}; expected e.g. capture_L0_A090"
        )
    return int(match.group("layer")), float(match.group("azimuth")) % 360.0


def circular_angle_distance(angle_a: float, angle_b: float) -> float:
    """Return the unsigned shortest separation between two angles in degrees."""

    delta = abs((float(angle_a) - float(angle_b)) % 360.0)
    return min(delta, 360.0 - delta)


def _validated_bin_edges(bin_edges: Sequence[float]) -> List[float]:
    edges = [float(value) for value in bin_edges]
    if (
        len(edges) < 2
        or any(not math.isfinite(value) for value in edges)
        or any(right <= left for left, right in zip(edges, edges[1:]))
    ):
        raise ValueError("Angle bin edges must be finite and strictly increasing")
    if edges[0] > 0.0 or edges[-1] < 180.0:
        raise ValueError("Angle bins must cover the complete [0, 180] range")
    return edges


def _format_edge(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def assign_angle_bin(angle: float, bin_edges: Sequence[float]) -> str:
    """Assign an angle to a half-open bin, with the final right edge inclusive."""

    edges = _validated_bin_edges(bin_edges)
    value = float(angle)
    if not math.isfinite(value) or value < edges[0] or value > edges[-1]:
        raise ValueError(f"Angle {angle!r} is outside [{edges[0]}, {edges[-1]}]")
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        is_last = index == len(edges) - 2
        if left <= value < right or (is_last and value == right):
            return f"{_format_edge(left)}-{_format_edge(right)}"
    raise ValueError(f"Could not assign angle {value} to bins {edges}")


def _metric_velocity_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if pred.shape[1] < 2:
        return float("nan")
    return float(
        torch.norm(
            (pred[:, 1:] - pred[:, :-1]) - (gt[:, 1:] - gt[:, :-1]),
            dim=-1,
        )
        .mean()
        .item()
    )


def _metric_acceleration_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    if pred.shape[1] < 3:
        return float("nan")
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    gt_acc = gt[:, 2:] - 2.0 * gt[:, 1:-1] + gt[:, :-2]
    return float(torch.norm(pred_acc - gt_acc, dim=-1).mean().item())


def _meta_values(meta: Dict[str, Any], key: str, batch_size: int) -> List[str]:
    raw = meta.get(key)
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, Iterable):
        values = [str(value) for value in raw]
    else:
        raise ValueError(f"Batch metadata is missing {key!r}")
    if len(values) != batch_size:
        raise ValueError(
            f"Batch metadata {key!r} has {len(values)} values for batch size {batch_size}"
        )
    return values


def summarize_outputs_by_angle(
    test_outputs: List[Dict[str, Any]],
    failure_threshold: float,
    bin_edges: Sequence[float],
) -> List[Dict[str, Any]]:
    """Aggregate canonical-space evaluation metrics by camera azimuth separation."""

    edges = _validated_bin_edges(bin_edges)
    grouped: Dict[str, Dict[str, List[torch.Tensor]]] = defaultdict(
        lambda: {"fused": [], "left": [], "right": [], "gt": []}
    )
    sample_counts: Dict[str, int] = defaultdict(int)
    separations: Dict[str, List[float]] = defaultdict(list)

    for output in test_outputs:
        required = (
            "fused",
            "left_canonical",
            "right_canonical",
            "ground_truth_canonical",
        )
        if not all(isinstance(output.get(key), torch.Tensor) for key in required):
            continue
        fused = cast(torch.Tensor, output["fused"]).detach().cpu()
        left = cast(torch.Tensor, output["left_canonical"]).detach().cpu()
        right = cast(torch.Tensor, output["right_canonical"]).detach().cpu()
        gt = cast(torch.Tensor, output["ground_truth_canonical"]).detach().cpu()
        if not (fused.shape == left.shape == right.shape == gt.shape):
            raise ValueError("Angle evaluation tensors must share the same shape")
        meta = output.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("Angle evaluation requires per-sample batch metadata")
        cam1_ids = _meta_values(meta, "cam1_id", fused.shape[0])
        cam2_ids = _meta_values(meta, "cam2_id", fused.shape[0])

        for index, (cam1_id, cam2_id) in enumerate(zip(cam1_ids, cam2_ids)):
            _, azimuth_1 = parse_unity_camera_id(cam1_id)
            _, azimuth_2 = parse_unity_camera_id(cam2_id)
            separation = circular_angle_distance(azimuth_1, azimuth_2)
            label = assign_angle_bin(separation, edges)
            grouped[label]["fused"].append(fused[index : index + 1])
            grouped[label]["left"].append(left[index : index + 1])
            grouped[label]["right"].append(right[index : index + 1])
            grouped[label]["gt"].append(gt[index : index + 1])
            sample_counts[label] += 1
            separations[label].append(separation)

    rows: List[Dict[str, Any]] = []
    for left_edge, right_edge in zip(edges, edges[1:]):
        label = f"{_format_edge(left_edge)}-{_format_edge(right_edge)}"
        if not grouped[label]["fused"]:
            continue
        fused = torch.cat(grouped[label]["fused"], dim=0)
        left = torch.cat(grouped[label]["left"], dim=0)
        right = torch.cat(grouped[label]["right"], dim=0)
        gt = torch.cat(grouped[label]["gt"], dim=0)
        canonical_avg = 0.5 * (left + right)
        fused_frame_mpjpe = torch.norm(fused - gt, dim=-1).mean(dim=-1)
        fused_mpjpe = float(fused_frame_mpjpe.mean().item())
        canonical_avg_mpjpe = float(
            torch.norm(canonical_avg - gt, dim=-1).mean().item()
        )
        gain = canonical_avg_mpjpe - fused_mpjpe
        rows.append(
            {
                "angle_bin": label,
                "angle_min_deg": left_edge,
                "angle_max_deg": right_edge,
                "mean_separation_deg": sum(separations[label])
                / len(separations[label]),
                "sample_count": sample_counts[label],
                "frame_count": int(fused.shape[0] * fused.shape[1]),
                "fused_mpjpe": fused_mpjpe,
                "canonical_avg_mpjpe": canonical_avg_mpjpe,
                "fusion_gain_mpjpe": gain,
                "fusion_gain_percent": (
                    100.0 * gain / canonical_avg_mpjpe
                    if canonical_avg_mpjpe != 0.0
                    else float("nan")
                ),
                "velocity_error": _metric_velocity_error(fused, gt),
                "acceleration_error": _metric_acceleration_error(fused, gt),
                "failure_rate": float(
                    (fused_frame_mpjpe > float(failure_threshold)).float().mean().item()
                ),
            }
        )
    return rows


# Re-export the manifest boundary from the shared experiment namespace.
from dual2pose.eval.frontend_manifest import (  # noqa: E402
    FrontEndManifest,
    FrontEndPoseDataset,
    replace_frontend_inputs,
)
