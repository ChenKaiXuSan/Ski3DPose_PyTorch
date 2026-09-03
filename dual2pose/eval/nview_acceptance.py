"""Acceptance filtering shared by full and smoke N-view runs."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from dual2pose.eval.nview_protocol import InsufficientCommonFrames, load_multiview_sample


def collect_accepted_samples(
    groups: Iterable[Any],
    row_lookup: Mapping[Any, Any],
    target_t: int,
    limit_accepted: int = 0,
    sample_loader: Callable[..., Any] = load_multiview_sample,
) -> tuple[list[tuple[Any, Any]], list[dict[str, str]], int]:
    accepted: list[tuple[Any, Any]] = []
    rejected: list[dict[str, str]] = []
    evaluated = 0
    for group in groups:
        if limit_accepted > 0 and len(accepted) >= limit_accepted:
            break
        evaluated += 1
        try:
            sample = sample_loader(group, row_lookup, target_t=target_t)
        except (InsufficientCommonFrames, FileNotFoundError, KeyError, ValueError) as exc:
            rejected.append(
                {
                    "group_id": str(getattr(group, "group_id", group)),
                    "reason": str(exc),
                }
            )
            continue
        accepted.append((group, sample))
    return accepted, rejected, evaluated
