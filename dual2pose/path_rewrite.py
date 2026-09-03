"""Recursive path rewriting for legacy Unity index mappings."""

from __future__ import annotations

from typing import Any


def rewrite_data_paths(value: Any, old_root: str, new_root: str) -> Any:
    """Return a copy with every nested path string moved to ``new_root``."""

    if old_root == new_root:
        return value
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    if isinstance(value, list):
        return [rewrite_data_paths(item, old_root, new_root) for item in value]
    if isinstance(value, tuple):
        return tuple(rewrite_data_paths(item, old_root, new_root) for item in value)
    if isinstance(value, dict):
        return {
            key: rewrite_data_paths(item, old_root, new_root)
            for key, item in value.items()
        }
    return value
