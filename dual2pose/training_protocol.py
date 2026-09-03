"""Shared provenance rules for the two-fold Unity training protocol."""

from __future__ import annotations

import os
from pathlib import Path
import re


_FOLD_DIRECTORY = Path(
    "index_mapping/use_layer_camera_filter_disabled/"
    "camera_pairs_by_action_folds"
)


def resolve_fold_index_path(data_root: Path, fold: int) -> Path:
    """Resolve one of the two legacy action folds used by the paper."""

    fold = int(fold)
    if fold not in (0, 1):
        raise ValueError(f"Unsupported fold {fold}; available folds are 0 and 1")
    return Path(data_root) / _FOLD_DIRECTORY / f"fold_{fold:02d}.json"


def validate_fold_metadata(index_path: Path, fold: int) -> None:
    """Validate trailing metadata without loading the multi-gigabyte index."""

    index_path = Path(index_path)
    if not index_path.is_file():
        raise FileNotFoundError(f"Fold index does not exist: {index_path}")
    with index_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 262_144), os.SEEK_SET)
        tail = handle.read().decode("utf-8-sig")
    metadata_start = tail.rfind('"_metadata"')
    if metadata_start < 0:
        raise ValueError(f"Fold index has no trailing _metadata object: {index_path}")
    match = re.search(r'"fold_idx"\s*:\s*(\d+)', tail[metadata_start:])
    if match is None:
        raise ValueError(f"Fold index metadata has no fold_idx: {index_path}")
    metadata_fold = int(match.group(1))
    if metadata_fold != int(fold):
        raise ValueError(
            f"Requested fold {fold}, but index metadata fold {metadata_fold}: {index_path}"
        )
