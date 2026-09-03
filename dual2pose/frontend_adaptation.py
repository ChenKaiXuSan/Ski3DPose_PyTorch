"""Weights-only initialization and parameter scopes for front-end adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal

import torch

from dual2pose.map_config import FILTERED_15_TO_COMMON_13_INDICES


@dataclass(frozen=True)
class LoadReport:
    checkpoint: str
    checkpoint_sha256: str
    loaded_parameter_tensors: int
    source_epoch: int | None
    source_global_step: int | None
    optimizer_state_restored: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_weights_only(module: torch.nn.Module, checkpoint: Path) -> LoadReport:
    """Load only the exact state dict; never restore trainer or optimizer state."""

    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must be a dictionary: {checkpoint}")
    raw_state = payload.get("state_dict", payload)
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError(f"Checkpoint has no non-empty state_dict: {checkpoint}")
    expected = set(module.state_dict())
    actual = set(raw_state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint state keys differ; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    module.load_state_dict(raw_state, strict=True)
    return LoadReport(
        checkpoint=str(checkpoint),
        checkpoint_sha256=_sha256(checkpoint),
        loaded_parameter_tensors=len(raw_state),
        source_epoch=int(payload["epoch"]) if payload.get("epoch") is not None else None,
        source_global_step=(
            int(payload["global_step"]) if payload.get("global_step") is not None else None
        ),
    )


def configure_trainable_scope(
    module: torch.nn.Module,
    scope: Literal["heads_only", "full"],
) -> dict[str, int]:
    if scope not in {"heads_only", "full"}:
        raise ValueError("Adaptation scope must be heads_only or full")
    head_names: list[str] = []
    for name, parameter in module.named_parameters():
        trainable = scope == "full" or "gate_head" in name or "residual_head" in name
        parameter.requires_grad_(trainable)
        if trainable and scope == "heads_only":
            head_names.append(name)
    if scope == "heads_only" and not head_names:
        raise ValueError("No gate_head or residual_head parameters were found")
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
        "trainable_parameter_tensors": sum(
            parameter.requires_grad for parameter in module.parameters()
        ),
    }


def common13_mpjpe(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-2] < 15:
        raise ValueError(
            f"Common-13 MPJPE requires matching 15-joint tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    indices = torch.tensor(
        FILTERED_15_TO_COMMON_13_INDICES,
        dtype=torch.long,
        device=prediction.device,
    )
    prediction_common = prediction.index_select(-2, indices)
    target_common = target.index_select(-2, indices)
    return torch.norm(prediction_common - target_common, dim=-1).mean()
