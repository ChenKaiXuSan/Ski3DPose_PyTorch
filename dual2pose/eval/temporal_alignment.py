"""Pose-only temporal alignment with validation-gated lag correlation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from dual2pose.eval.eval_unity_temporal_offset import _shift_pose_sequence


@dataclass(frozen=True)
class AlignmentEstimate:
    correction_frames: torch.Tensor
    best_score: torch.Tensor
    zero_score: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class CalibrationRow:
    split: str
    confidence: float
    estimated_correction: float
    target_correction: float


def velocity_descriptor(pose: torch.Tensor) -> torch.Tensor:
    if pose.ndim != 4 or pose.shape[-1] != 3:
        raise ValueError(f"Expected pose shape BxTxJx3, got {tuple(pose.shape)}")
    if pose.shape[1] < 2:
        raise ValueError("Temporal alignment requires at least two frames")
    velocity = pose[:, 1:] - pose[:, :-1]
    return velocity - velocity.mean(dim=2, keepdim=True)


def correction_accuracy_masks(
    estimated: torch.Tensor,
    target: torch.Tensor,
    *,
    exact_tolerance: float = 0.25,
    within_half_tolerance: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if estimated.shape != target.shape:
        raise ValueError(
            f"Correction shapes differ: {tuple(estimated.shape)} vs {tuple(target.shape)}"
        )
    if exact_tolerance <= 0.0 or within_half_tolerance <= exact_tolerance:
        raise ValueError("Correction tolerances must satisfy 0 < exact < within-half")
    error = (estimated - target).abs()
    return (
        error < float(exact_tolerance),
        error < float(within_half_tolerance),
    )


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / (vector.norm(dim=-1, keepdim=True) + eps)


def _canonical_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.ndim != 4 or pose.shape[-1] != 3:
        raise ValueError(f"Expected pose shape BxTxJx3, got {tuple(pose.shape)}")
    if pose.shape[1] == 0:
        raise ValueError("Temporal alignment requires at least one frame")

    reference = pose[:, 0]
    left_hip = reference[:, 6]
    right_hip = reference[:, 7]
    neck = reference[:, 14]
    pelvis = 0.5 * (left_hip + right_hip)

    x_axis = _normalize(right_hip - left_hip)
    y_axis = _normalize(neck - pelvis)
    z_axis = _normalize(torch.linalg.cross(x_axis, y_axis))
    eye_mid = 0.5 * (reference[:, 0] + reference[:, 1])
    face_direction = _normalize(eye_mid - neck)
    sign = torch.sign((z_axis * face_direction).sum(dim=-1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    z_axis = z_axis * sign

    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    centered = pose - pelvis[:, None, None, :]
    return torch.einsum("btjc,bck->btjk", centered, rotation)


def _score_canonical(
    left_canonical: torch.Tensor,
    right_canonical: torch.Tensor,
    correction_frames: float,
) -> torch.Tensor:
    corrected = _shift_pose_sequence(right_canonical, float(correction_frames))
    left_velocity = velocity_descriptor(left_canonical)
    right_velocity = velocity_descriptor(corrected)
    transitions = torch.arange(
        left_canonical.shape[1] - 1,
        device=left_canonical.device,
        dtype=left_canonical.dtype,
    )
    source_start = transitions - float(correction_frames)
    source_end = transitions + 1.0 - float(correction_frames)
    valid = (source_start >= 0.0) & (source_end <= left_canonical.shape[1] - 1)
    if not bool(valid.any()):
        return left_canonical.new_full((left_canonical.shape[0],), float("nan"))
    left_flat = left_velocity[:, valid].reshape(left_canonical.shape[0], -1)
    right_flat = right_velocity[:, valid].reshape(right_canonical.shape[0], -1)
    numerator = (left_flat * right_flat).sum(dim=1)
    denominator = left_flat.norm(dim=1) * right_flat.norm(dim=1)
    score = numerator / denominator.clamp_min(1e-12)
    return torch.where(
        denominator > 1e-12,
        score,
        torch.full_like(score, float("nan")),
    )


def lag_score(
    left: torch.Tensor,
    right: torch.Tensor,
    correction_frames: float,
) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"Pose shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}")
    return _score_canonical(
        _canonical_pose(left),
        _canonical_pose(right),
        float(correction_frames),
    )


def estimate_temporal_correction(
    left: torch.Tensor,
    right: torch.Tensor,
    candidates: Sequence[float],
    confidence_threshold: float,
) -> AlignmentEstimate:
    if left.shape != right.shape:
        raise ValueError(f"Pose shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}")
    unique_candidates = sorted({float(value) for value in candidates})
    if 0.0 not in unique_candidates:
        raise ValueError("Temporal correction candidates must contain zero")
    left_canonical = _canonical_pose(left)
    right_canonical = _canonical_pose(right)
    score_columns = [
        _score_canonical(left_canonical, right_canonical, candidate)
        for candidate in unique_candidates
    ]
    scores = torch.stack(score_columns, dim=1)
    zero_index = unique_candidates.index(0.0)
    zero_score = scores[:, zero_index]
    corrections: list[float] = []
    best_scores: list[float] = []
    confidences: list[float] = []
    for batch_index in range(scores.shape[0]):
        ranked: list[tuple[float, float]] = []
        for candidate_index, candidate in enumerate(unique_candidates):
            score = float(scores[batch_index, candidate_index].item())
            if math.isfinite(score):
                ranked.append((candidate, score))
        zero = float(zero_score[batch_index].item())
        if not ranked or not math.isfinite(zero):
            corrections.append(0.0)
            best_scores.append(0.0)
            confidences.append(0.0)
            continue
        ranked.sort(
            key=lambda item: (
                -round(item[1], 12),
                abs(item[0]),
                0 if item[0] < 0 else 1,
            )
        )
        best_candidate, best_score = ranked[0]
        confidence = max(0.0, best_score - zero)
        corrections.append(
            best_candidate if confidence > float(confidence_threshold) else 0.0
        )
        best_scores.append(best_score)
        confidences.append(confidence)
    return AlignmentEstimate(
        correction_frames=left.new_tensor(corrections),
        best_score=left.new_tensor(best_scores),
        zero_score=torch.nan_to_num(zero_score, nan=0.0),
        confidence=left.new_tensor(confidences),
    )


def choose_confidence_threshold(
    rows: Sequence[CalibrationRow],
    candidates: Sequence[float],
) -> dict[str, object]:
    if not rows:
        raise ValueError("Confidence calibration requires validation rows")
    if any(row.split != "val" for row in rows):
        raise ValueError("Confidence calibration accepts validation rows only")
    thresholds = sorted({float(value) for value in candidates})
    if not thresholds:
        raise ValueError("Threshold candidates cannot be empty")
    sweep: list[dict[str, float]] = []
    for threshold in thresholds:
        applied = [
            row.estimated_correction if row.confidence > threshold else 0.0
            for row in rows
        ]
        errors = [
            abs(correction - row.target_correction)
            for correction, row in zip(applied, rows)
        ]
        zero_rows = [
            (correction, row)
            for correction, row in zip(applied, rows)
            if abs(row.target_correction) < 1e-12
        ]
        false_rate = (
            sum(abs(correction) > 1e-12 for correction, _ in zero_rows) / len(zero_rows)
            if zero_rows
            else 0.0
        )
        sweep.append(
            {
                "threshold": threshold,
                "offset_mae": sum(errors) / len(errors),
                "zero_offset_false_correction_rate": false_rate,
            }
        )
    selected = min(
        sweep,
        key=lambda row: (
            row["offset_mae"],
            row["zero_offset_false_correction_rate"],
            -row["threshold"],
        ),
    )
    return {
        "selected_threshold": selected["threshold"],
        "validation_row_count": len(rows),
        "selection_metric": "offset_mae",
        "tie_break": "lower_zero_false_rate_then_higher_threshold",
        "sweep": sweep,
    }
