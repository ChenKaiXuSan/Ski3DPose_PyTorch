#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Export 6-view pseudo ground truth from Ski-Pose PTZ labels.

The script groups annotations by (subj, seq, frame), fits one fixed similarity
transform per camera on a calibration subset of full-view frames, aligns all
camera-local 3D poses to a reference camera, and fuses the aligned poses by
joint-wise median.

Output:
    <output_root>/seqXXX_subjY/
        seqXXX_subjY_cams0-1-2-3-4-5_pseudo_gt.npz
        seqXXX_subjY_cams0-1-2-3-4-5_pseudo_gt.summary.json
    <output_root>/summary/pseudo_gt_export_summary.json

Each exported NPZ contains:
  - frames: int32 array of frame ids
  - poses: float32 array with shape (T, J, 3)
  - meta_json: a JSON string with sequence metadata

Typical usage:
  python analysis/export_6view_pseudo_gt.py \
      --labels-h5 /workspace/data/Ski-PosePTZ-CameraDataset-png/test/labels.h5 \
      --output-dir /workspace/data/Ski-PosePTZ-CameraDataset-png/pseudo_gt_exports
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import h5py  # type: ignore[import-untyped]
import numpy as np


@dataclass
class ExportStats:
    sequences: int = 0
    subjects: int = 0
    frames_total: int = 0
    frames_exported: int = 0
    camera_views: int = 0
    sequences_skipped: int = 0


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return similarity transform s, R, t with dst ~= s * (R @ src) + t."""
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: src={src.shape}, dst={dst.shape}")
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) point clouds, got {src.shape}")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(cov)

    sign_correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign_correction[-1, -1] = -1.0

    rotation = u @ sign_correction @ vt
    if estimate_scale:
        var_src = float((src_c ** 2).sum() / src.shape[0])
        scale = float(np.trace(np.diag(singular_values) @ sign_correction) / (var_src + 1e-12))
    else:
        scale = 1.0

    translation = mu_dst - scale * (rotation @ mu_src)
    return scale, rotation, translation


def apply_similarity(pose: np.ndarray, sim: Tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = sim
    return (scale * (rotation @ pose.T)).T + translation[None, :]


def load_h5_arrays(labels_h5: Path) -> Dict[str, np.ndarray]:
    with h5py.File(labels_h5, "r") as f:
        arrays = {
            "seq": np.asarray(f["seq"], dtype=np.int32),
            "cam": np.asarray(f["cam"], dtype=np.int32),
            "frame": np.asarray(f["frame"], dtype=np.int32),
            "subj": np.asarray(f["subj"], dtype=np.int32),
            "pose3d": np.asarray(f["3D"], dtype=np.float32).reshape(-1, 17, 3),
        }

    return arrays


def build_frame_groups(seq: np.ndarray, cam: np.ndarray, frame: np.ndarray, subj: np.ndarray) -> Dict[Tuple[int, int, int], Dict[int, int]]:
    groups: Dict[Tuple[int, int, int], Dict[int, int]] = defaultdict(dict)
    for idx in range(seq.shape[0]):
        key = (int(subj[idx]), int(seq[idx]), int(frame[idx]))
        groups[key][int(cam[idx])] = int(idx)
    return groups


def choose_reference_camera(calib_poses: Mapping[int, np.ndarray], camera_ids: Sequence[int]) -> int:
    pairwise = np.zeros((len(camera_ids), len(camera_ids)), dtype=np.float64)
    for i, cam_i in enumerate(camera_ids):
        for j, cam_j in enumerate(camera_ids):
            pairwise[i, j] = np.linalg.norm(calib_poses[cam_i] - calib_poses[cam_j], axis=-1).mean()
    ref_idx = int(np.argmin(pairwise.mean(axis=1)))
    return int(camera_ids[ref_idx])


def export_sequence(
    *,
    subj: int,
    seq_id: int,
    camera_ids: Sequence[int],
    frame_to_camidx: Mapping[int, Mapping[int, int]],
    pose3d: np.ndarray,
    output_dir: Path,
    num_calib_frames: int,
    overwrite: bool,
    min_eval_frames: int,
) -> Tuple[bool, Dict[str, float]]:
    valid_frames = [fr for fr, cam_map in frame_to_camidx.items() if all(c in cam_map for c in camera_ids)]
    valid_frames = sorted(valid_frames)
    if len(valid_frames) < max(num_calib_frames + 1, min_eval_frames):
        return False, {"valid_frames": float(len(valid_frames))}

    calib_frames = valid_frames[:num_calib_frames]
    eval_frames = valid_frames[num_calib_frames:]

    calib_poses_list: Dict[int, List[np.ndarray]] = {c: [] for c in camera_ids}
    for fr in calib_frames:
        cam_map = frame_to_camidx[fr]
        for cam_id in camera_ids:
            calib_poses_list[cam_id].append(pose3d[cam_map[cam_id]])
    calib_poses: Dict[int, np.ndarray] = {
        c: np.stack(v, axis=0) for c, v in calib_poses_list.items()
    }

    ref_cam = choose_reference_camera(calib_poses, camera_ids)
    transforms: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}
    ref_dst = calib_poses[ref_cam].reshape(-1, 3)
    for cam_id in camera_ids:
        src = calib_poses[cam_id].reshape(-1, 3)
        transforms[cam_id] = umeyama_similarity(src, ref_dst, estimate_scale=True)

    fused_by_frame: Dict[int, np.ndarray] = {}
    residual_stats: Dict[int, List[float]] = {c: [] for c in camera_ids}

    for fr in eval_frames:
        cam_map = frame_to_camidx[fr]
        aligned_views_list: List[np.ndarray] = []
        for cam_id in camera_ids:
            pose = pose3d[cam_map[cam_id]].astype(np.float64)
            aligned_views_list.append(apply_similarity(pose, transforms[cam_id]))

        aligned_views_arr = np.stack(aligned_views_list, axis=0)
        fused = np.median(aligned_views_arr, axis=0).astype(np.float32)
        fused_by_frame[int(fr)] = fused

        residuals = np.linalg.norm(aligned_views_arr - fused[None, :, :], axis=-1)
        for i, cam_id in enumerate(camera_ids):
            residual_stats[cam_id].append(float(residuals[i].mean()))

    residual_mean_per_cam = {
        str(c): float(np.mean(np.asarray(v, dtype=np.float32))) for c, v in residual_stats.items()
    }
    residual_median_per_cam = {
        str(c): float(np.median(np.asarray(v, dtype=np.float32))) for c, v in residual_stats.items()
    }
    all_residuals = np.concatenate([np.asarray(v, dtype=np.float32) for v in residual_stats.values()])
    residual_overall_mean = float(all_residuals.mean())
    residual_overall_median = float(np.median(all_residuals))

    export_frames = np.asarray(sorted(fused_by_frame.keys()), dtype=np.int32)
    export_poses = np.stack([fused_by_frame[int(fr)] for fr in export_frames], axis=0).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = output_dir / f"seq{seq_id:03d}_subj{subj}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    export_name = f"seq{seq_id:03d}_subj{subj}_cams{'-'.join(str(c) for c in camera_ids)}_pseudo_gt.npz"
    export_path = seq_dir / export_name
    if export_path.exists() and not overwrite:
        return True, {
            "frames": float(export_frames.shape[0]),
            "reference_cam": float(ref_cam),
            "skipped_existing": 1.0,
        }

    meta = {
        "seq": int(seq_id),
        "subj": int(subj),
        "cams": [int(c) for c in camera_ids],
        "reference_cam": int(ref_cam),
        "num_calib_frames": int(num_calib_frames),
        "num_valid_frames": int(len(valid_frames)),
        "num_eval_frames": int(len(eval_frames)),
        "pose_shape": [int(v) for v in export_poses.shape],
        "eval_residual_mean_per_cam": residual_mean_per_cam,
        "eval_residual_median_per_cam": residual_median_per_cam,
        "eval_residual_overall_mean": residual_overall_mean,
        "eval_residual_overall_median": residual_overall_median,
    }

    np.savez_compressed(
        export_path,
        frames=export_frames,
        poses=export_poses,
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )

    summary_path = export_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return True, {
        "frames": float(export_frames.shape[0]),
        "reference_cam": float(ref_cam),
        "exported": 1.0,
        "residual_overall_mean": residual_overall_mean,
        "residual_overall_median": residual_overall_median,
    }


def run(args: argparse.Namespace) -> None:
    arrays = load_h5_arrays(args.labels_h5)
    groups = build_frame_groups(arrays["seq"], arrays["cam"], arrays["frame"], arrays["subj"])

    seq_subj_to_frames: Dict[Tuple[int, int], Dict[int, Dict[int, int]]] = defaultdict(dict)
    seq_subj_to_cams: Dict[Tuple[int, int], set[int]] = defaultdict(set)

    for (subj, seq_id, frame_id), cam_map in groups.items():
        seq_subj_to_frames[(subj, seq_id)][frame_id] = cam_map
        seq_subj_to_cams[(subj, seq_id)].update(cam_map.keys())

    stats = ExportStats()
    stats.sequences = len(seq_subj_to_frames)
    stats.subjects = len({subj for subj, _ in seq_subj_to_frames.keys()})

    export_count = 0
    skipped_count = 0
    total_frames = 0
    overall_residual_means: List[float] = []
    overall_residual_medians: List[float] = []
    sequence_summaries: List[Dict[str, object]] = []

    for (subj, seq_id), frame_to_camidx in sorted(seq_subj_to_frames.items(), key=lambda item: (item[0][0], item[0][1])):
        camera_ids = sorted(int(c) for c in seq_subj_to_cams[(subj, seq_id)])
        if args.cameras is not None:
            camera_ids = [c for c in camera_ids if c in set(args.cameras)]
        if len(camera_ids) < args.min_cameras:
            skipped_count += 1
            continue

        ok, info = export_sequence(
            subj=subj,
            seq_id=seq_id,
            camera_ids=camera_ids,
            frame_to_camidx=frame_to_camidx,
            pose3d=arrays["pose3d"],
            output_dir=args.output_dir,
            num_calib_frames=args.num_calib_frames,
            overwrite=args.overwrite,
            min_eval_frames=args.min_eval_frames,
        )

        if not ok:
            skipped_count += 1
            continue

        export_count += 1
        total_frames += int(info.get("frames", 0.0))
        stats.camera_views += len(camera_ids)
        if "residual_overall_mean" in info:
            overall_residual_means.append(float(info["residual_overall_mean"]))
        if "residual_overall_median" in info:
            overall_residual_medians.append(float(info["residual_overall_median"]))

        sequence_summaries.append(
            {
                "subj": int(subj),
                "seq": int(seq_id),
                "cams": [int(c) for c in camera_ids],
                "frames": int(info.get("frames", 0.0)),
                "reference_cam": int(info.get("reference_cam", -1)),
                "residual_overall_mean": float(info.get("residual_overall_mean", float("nan"))),
                "residual_overall_median": float(info.get("residual_overall_median", float("nan"))),
                "output_dir": str(args.output_dir / f"seq{seq_id:03d}_subj{subj}"),
            }
        )

    stats.frames_exported = total_frames
    stats.sequences_skipped = skipped_count

    summary_path = args.output_dir / "summary" / "pseudo_gt_export_summary.json"
    summary_payload = {
        "labels_h5": str(args.labels_h5),
        "num_calib_frames": int(args.num_calib_frames),
        "min_eval_frames": int(args.min_eval_frames),
        "min_cameras": int(args.min_cameras),
        "camera_filter": [int(c) for c in args.cameras] if args.cameras is not None else None,
        "exports_written": int(export_count),
        "frames_exported": int(stats.frames_exported),
        "subjects_seen": int(stats.subjects),
        "sequences_seen": int(stats.sequences),
        "sequences_skipped": int(stats.sequences_skipped),
        "overall_residual_mean_across_exports": float(np.mean(np.asarray(overall_residual_means, dtype=np.float32))) if overall_residual_means else None,
        "overall_residual_median_across_exports": float(np.median(np.asarray(overall_residual_medians, dtype=np.float32))) if overall_residual_medians else None,
        "sequences": sequence_summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== Pseudo-GT Export Summary ===")
    print(f"sequences seen:   {stats.sequences}")
    print(f"subjects seen:    {stats.subjects}")
    print(f"exports written:  {export_count}")
    print(f"frames exported:  {stats.frames_exported}")
    print(f"camera views:     {stats.camera_views}")
    print(f"sequences skipped:{stats.sequences_skipped}")
    if overall_residual_means:
        print(f"overall residual mean across exports:   {summary_payload['overall_residual_mean_across_exports']:.4f}")
        print(f"overall residual median across exports: {summary_payload['overall_residual_median_across_exports']:.4f}")
    print(f"summary json:     {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 6-view pseudo ground truth from labels.h5.")
    parser.add_argument(
        "--labels-h5",
        type=Path,
        default=Path("/workspace/data/Ski-PosePTZ-CameraDataset-png/test/labels.h5"),
        help="Path to labels.h5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/Ski-PosePTZ-CameraDataset-png/pseudo_gt_exports/test"),
        help="Output directory for exported NPZ files.",
    )
    parser.add_argument(
        "--num-calib-frames",
        type=int,
        default=80,
        help="Number of full-view frames used to fit per-camera similarity transforms.",
    )
    parser.add_argument(
        "--min-eval-frames",
        type=int,
        default=1,
        help="Skip sequences with fewer than this many evaluation frames.",
    )
    parser.add_argument(
        "--min-cameras",
        type=int,
        default=6,
        help="Require at least this many cameras in a sequence.",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit camera ids to use (default: all cameras present in each sequence).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())