"""Build a protocol-checked comparison table for front-end experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


_PROTOCOL_FIELDS = (
    "checkpoint",
    "checkpoint_sha256",
    "fold",
    "sample_count",
    "joint_subset",
    "units",
)
_METRIC_FIELDS = (
    "common13_raw_avg_mpjpe",
    "common13_canonical_avg_mpjpe",
    "common13_fused_mpjpe",
)


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value for {field}: {value!r}")
    return result


def build_comparison_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise ValueError("No front-end result rows were provided")
    reference = normalized[0]
    for field in _PROTOCOL_FIELDS:
        expected = str(reference.get(field, ""))
        mismatched = [
            str(row.get("frontend_name", "unknown"))
            for row in normalized
            if str(row.get(field, "")) != expected
        ]
        if mismatched:
            raise ValueError(
                f"Cannot compare mixed {field} values; mismatched rows: {', '.join(mismatched)}"
            )
    for row in normalized:
        if not str(row.get("frontend_name", "")).strip():
            raise ValueError("Every result row requires frontend_name")
        for field in _METRIC_FIELDS:
            row[field] = _number(row.get(field), field)
        baseline = row["common13_canonical_avg_mpjpe"]
        fused = row["common13_fused_mpjpe"]
        gain = baseline - fused
        row["fusion_gain_common13_mpjpe"] = gain
        row["fusion_gain_common13_percent"] = (
            gain / baseline * 100.0 if baseline != 0 else math.nan
        )
    normalized.sort(
        key=lambda row: (row["common13_fused_mpjpe"], str(row["frontend_name"]))
    )
    for rank, row in enumerate(normalized, start=1):
        row["rank_common13_fused_mpjpe"] = rank
    return normalized


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_comparison_rows(_read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
