"""Per-sample and per-action metric extraction for repeated training runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch


def summarize_test_outputs_by_action(
    outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_values: list[float] = []
    by_action: dict[str, list[float]] = defaultdict(list)
    for output in outputs:
        fused = output.get("fused")
        target = output.get("ground_truth_canonical")
        meta = output.get("meta")
        if not isinstance(fused, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("Test output requires fused and ground_truth_canonical tensors")
        if fused.shape != target.shape:
            raise ValueError(f"Test output tensor shapes differ: {fused.shape} vs {target.shape}")
        if not isinstance(meta, Mapping):
            raise ValueError("Test output requires action metadata")
        actions_raw = meta.get("action_id")
        if isinstance(actions_raw, str):
            actions = [actions_raw]
        elif isinstance(actions_raw, Sequence):
            actions = [str(value) for value in actions_raw]
        else:
            raise ValueError("Test output action_id must be a string sequence")
        if len(actions) != fused.shape[0]:
            raise ValueError("Action metadata count does not match test batch")
        sample_values = torch.norm(fused - target, dim=-1).mean(dim=(1, 2)).detach().cpu()
        for action, value in zip(actions, sample_values.tolist()):
            numeric = float(value)
            all_values.append(numeric)
            by_action[action].append(numeric)
    if not all_values:
        raise ValueError("No test outputs were provided")
    return {
        "sample_count": len(all_values),
        "mpjpe": sum(all_values) / len(all_values),
        "per_action": {
            action: {
                "sample_count": len(values),
                "mpjpe": sum(values) / len(values),
            }
            for action, values in sorted(by_action.items())
        },
    }
