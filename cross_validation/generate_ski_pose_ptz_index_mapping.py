#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Generate index mapping JSON files for Ski-PosePTZ.

The generated mapping follows the existing project style: a JSON object with
train/val/test lists, where each sample is compatible with
SkiPosePTZDataConfig.

Expected data layout:
    /workspace/data/Ski-PosePTZ-CameraDataset-png/data/{train,test}/seq_XXX/cam_YY/
    /workspace/data/Ski-PosePTZ-CameraDataset-png/pseudo_gt_exports/{train,test}/seqXXX_subjY/

The script reads each split's labels.h5, builds one sample per
(sequence, subject, camera_pair) and attaches the available 6-view
pseudo-GT export folder for that sequence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py  # type: ignore[import-untyped]
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dual2pose.map_config import SkiPosePTZDataConfig

PSEUDO_GT_DIR_PATTERN = re.compile(r"^seq(\d+)_subj(\d+)$")


def _read_split_labels(split_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_path = split_root / "labels.h5"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")

    with h5py.File(labels_path, "r") as f:
        seq = np.asarray(f["seq"], dtype=np.int32)
        subj = np.asarray(f["subj"], dtype=np.int32)
        cam = np.asarray(f["cam"], dtype=np.int32)

    return seq, subj, cam


def _resolve_pseudo_gt_dir(pseudo_gt_split_root: Path, seq_id: int, subj_id: int) -> Optional[Path]:
    for candidate in sorted(pseudo_gt_split_root.iterdir()):
        if not candidate.is_dir():
            continue
        match = PSEUDO_GT_DIR_PATTERN.match(candidate.name)
        if match is None:
            continue
        if int(match.group(1)) == seq_id and int(match.group(2)) == subj_id:
            return candidate
    return None


def _build_item(
    *,
    split_root: Path,
    pseudo_gt_split_root: Path,
    sam3d_split_root: Optional[Path],
    seq_id: int,
    subj_id: int,
    cam1_id: int,
    cam2_id: int,
) -> Optional[SkiPosePTZDataConfig]:
    seq_dir = split_root / f"seq_{seq_id:03d}"
    if not seq_dir.exists():
        return None

    cam1_dir = seq_dir / f"cam_{cam1_id:02d}"
    cam2_dir = seq_dir / f"cam_{cam2_id:02d}"
    if not cam1_dir.exists() or not cam2_dir.exists():
        return None

    labels_h5_path = split_root / "labels.h5"

    pseudo_gt_dir = _resolve_pseudo_gt_dir(pseudo_gt_split_root, seq_id, subj_id)
    if pseudo_gt_dir is None:
        return None

    # Resolve SAM3D inference dirs (2D and 3D both reside in the same cam folder)
    if sam3d_split_root is not None:
        sam3d_seq_dir = sam3d_split_root / f"seq_{seq_id:03d}"
        cam1_sam3d_dir = str((sam3d_seq_dir / f"cam_{cam1_id:02d}").resolve())
        cam2_sam3d_dir = str((sam3d_seq_dir / f"cam_{cam2_id:02d}").resolve())
    else:
        cam1_sam3d_dir = str(pseudo_gt_dir.resolve())
        cam2_sam3d_dir = str(pseudo_gt_dir.resolve())

    return SkiPosePTZDataConfig(
        subject_id=str(subj_id),
        sequence_id=str(seq_id),
        cam1_id=cam1_id,
        cam2_id=cam2_id,
        labels_h5=str(labels_h5_path.resolve()),
        cam1_frames_dir=str(cam1_dir.resolve()),
        cam2_frames_dir=str(cam2_dir.resolve()),
        cam1_sam3d_kpt2d_dir=cam1_sam3d_dir,
        cam2_sam3d_kpt2d_dir=cam2_sam3d_dir,
        cam1_sam3d_kpt3d_dir=cam1_sam3d_dir,
        cam2_sam3d_kpt3d_dir=cam2_sam3d_dir,
        pesudo_gt_kpt3d_dir=str(pseudo_gt_dir.resolve()),
    )


def _discover_cam_ids(seq_dir: Path) -> List[int]:
    """Return sorted list of camera IDs available in a sequence directory."""
    cam_ids = []
    for d in seq_dir.iterdir():
        if d.is_dir() and re.match(r'^cam_(\d+)$', d.name):
            cam_ids.append(int(d.name.split('_')[1]))
    return sorted(cam_ids)


def build_split_index(
    split_root: Path,
    pseudo_gt_split_root: Path,
    sam3d_split_root: Optional[Path] = None,
    cam_pairs: Optional[List[Tuple[int, int]]] = None,
) -> List[Dict[str, str]]:
    seq, subj, _cam = _read_split_labels(split_root)
    items: List[Dict[str, str]] = []

    unique_pairs = sorted({(int(seq[i]), int(subj[i])) for i in range(len(seq))})
    for seq_id, subj_id in unique_pairs:
        # Determine which camera pairs to generate for this sequence
        if cam_pairs is not None:
            pairs_to_use = cam_pairs
        else:
            # Auto-discover cameras from the data directory
            seq_dir = split_root / f"seq_{seq_id:03d}"
            if not seq_dir.exists():
                continue
            cam_ids = _discover_cam_ids(seq_dir)
            pairs_to_use = list(combinations(cam_ids, 2))

        for cam1_id, cam2_id in pairs_to_use:
            item = _build_item(
                split_root=split_root,
                pseudo_gt_split_root=pseudo_gt_split_root,
                sam3d_split_root=sam3d_split_root,
                seq_id=seq_id,
                subj_id=subj_id,
                cam1_id=cam1_id,
                cam2_id=cam2_id,
            )
            if item is not None:
                items.append(asdict(item))

    return items


def run(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    pseudo_gt_root = args.pseudo_gt_root.resolve()
    sam3d_root: Optional[Path] = args.sam3d_root.resolve() if args.sam3d_root is not None else None

    index_mapping_dir = args.output_dir.resolve()
    index_mapping_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, List[Dict[str, str]]] = {}
    for split in ["train", "test"]:
        split_root = data_root / split
        pseudo_gt_split_root = pseudo_gt_root / split
        if not split_root.exists():
            continue
        if not pseudo_gt_split_root.exists():
            continue

        sam3d_split_root = (sam3d_root / split) if sam3d_root is not None else None

        # Parse optional explicit camera pairs (e.g. "0,1 0,2 1,3")
        cam_pairs: Optional[List[Tuple[int, int]]] = None
        if args.cam_pairs:
            cam_pairs = []
            for pair_str in args.cam_pairs:
                a, b = pair_str.split(",")
                cam_pairs.append((int(a), int(b)))

        split_items = build_split_index(
            split_root=split_root,
            pseudo_gt_split_root=pseudo_gt_split_root,
            sam3d_split_root=sam3d_split_root,
            cam_pairs=cam_pairs,
        )
        result[split] = split_items

    summary = {
        "data_root": str(data_root),
        "pseudo_gt_root": str(pseudo_gt_root),
        "sam3d_root": str(sam3d_root) if sam3d_root is not None else None,
        "cam_pairs": args.cam_pairs if args.cam_pairs else "all_combinations",
        "num_train": len(result.get("train", [])),
        "num_val": len(result.get("val", [])),
        "num_test": len(result.get("test", [])),
        "index_mapping_file": str((index_mapping_dir / "ski_pose_ptz_index_mapping.json").resolve()),
    }

    payload = {
        "train": result.get("train", []),
        "val": [],
        "test": result.get("test", []),
        "_metadata": summary,
    }

    output_file = index_mapping_dir / "ski_pose_ptz_index_mapping.json"
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (index_mapping_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== Ski-PosePTZ Index Mapping Summary ===")
    print(f"train samples: {summary['num_train']}")
    print(f"test samples:  {summary['num_test']}")
    print(f"output file:   {output_file}")
    print(f"summary file:   {index_mapping_dir / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Ski-PosePTZ index mapping JSON.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/workspace/data/Ski-PosePTZ-CameraDataset-png/data"),
        help="Root containing train/test split folders.",
    )
    parser.add_argument(
        "--pseudo-gt-root",
        type=Path,
        default=Path("/workspace/data/Ski-PosePTZ-CameraDataset-png/pseudo_gt_exports"),
        help="Root containing pseudo_gt_exports/{train,test}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/Ski-PosePTZ-CameraDataset-png/index_mapping"),
        help="Directory where the JSON mapping will be written.",
    )
    parser.add_argument(
        "--sam3d-root",
        type=Path,
        default=Path("/workspace/data/Ski-PosePTZ-CameraDataset-png/sam3d_body_results/inference"),
        help="Root of SAM3D inference results containing {split}/seq_{XXX}/cam_{YY}/.",
    )
    parser.add_argument(
        "--cam-pairs",
        nargs="+",
        default=None,
        metavar="I,J",
        help="Explicit camera pairs to use, e.g. '0,1 0,2 1,3'. "
             "If omitted, all C(N,2) combinations are generated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())