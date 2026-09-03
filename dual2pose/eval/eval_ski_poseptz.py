#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Evaluate Ski-PosePTZ checkpoints without masking sweeps.

This script runs a single clean evaluation and reports the fused result plus
baseline comparisons against the raw and canonical averages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from torch.utils.data import DataLoader, default_collate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DUAL2POSE_ROOT = REPO_ROOT / "dual2pose"
if str(DUAL2POSE_ROOT) not in sys.path:
    sys.path.insert(0, str(DUAL2POSE_ROOT))


@lru_cache(maxsize=1)
def _repo_symbols():
    dataset_module = importlib.import_module("dual2pose.dataloader.ski_poseptz_dataset_dual_view")
    map_module = importlib.import_module("dual2pose.map_config")
    fusion_module = importlib.import_module("dual2pose.trainer.train_crossview_fusion")
    dual_module = importlib.import_module("dual2pose.trainer.train_dual2pose")
    return (
        dataset_module.LabeledSkiPosePTZDataset,
        map_module.filter_filtered_kpts_to_common,
        map_module.filter_h36m_kpts,
        fusion_module.CrossViewFusionTrainer,
        dual_module.Dual2PoseTrainer,
    )


DEFAULT_CKPT = Path(
    "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_ski_poseptz/"
    "crossview_fusion/2026-05-25/14-13-23/checkpoints/last.ckpt"
)
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dual2pose.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained dual2pose/crossview_fusion checkpoint on Ski-PosePTZ test split."
    )
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT, help="Checkpoint path to evaluate.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG, help="Config file used to build the data module and model.")
    parser.add_argument("--backbone", type=str, default="crossview_fusion", choices=["crossview_fusion", "dual2pose"], help="Model backbone to instantiate.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=4, help="Test batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--time-window", type=int, default=30, help="Temporal window used by the dataset.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index to use when GPU is available.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for summaries. Overrides config.log_path.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace):
    config = OmegaConf.load(str(args.config_path))
    config.model.backbone = args.backbone
    config.data.left_hip_idx = 4
    config.data.right_hip_idx = 5
    config.data.neck_idx = 12
    config.data.load_frames = False
    config.data.load_2d_kpt = False
    config.data.load_3d_kpt = True
    config.data.time_window = int(args.time_window)
    config.data.batch_size = int(args.batch_size)
    config.data.num_workers = int(args.num_workers)
    ckpt_dir = args.ckpt_path.resolve().parent.parent
    config.log_path = str(
        Path(args.output_path).resolve() if args.output_path else ckpt_dir
    )
    return config


def _build_model(config):
    _, _, _, crossview_fusion_cls, dual2pose_cls = _repo_symbols()
    if config.model.backbone == "crossview_fusion":
        return crossview_fusion_cls(config)
    if config.model.backbone == "dual2pose":
        return dual2pose_cls(config)
    raise ValueError(f"Unsupported backbone: {config.model.backbone}")


def _collate_ski_poseptz_batch(batch):
    _, filter_filtered_kpts_to_common, filter_h36m_kpts, _, _ = _repo_symbols()
    collated = default_collate(batch)

    def _filter_pose_tensor_batch(pose_batch: torch.Tensor, filter_fn):
        flat_batch = pose_batch.reshape(-1, pose_batch.shape[-2], pose_batch.shape[-1])
        filtered = torch.stack([torch.from_numpy(filter_fn(sample.cpu().numpy())) for sample in flat_batch], dim=0)
        return filtered.view(*pose_batch.shape[:-2], -1, pose_batch.shape[-1])

    if "kpt3d" in collated:
        kpt3d_sam = collated.pop("kpt3d")
        collated["kpt3d_sam"] = {
            cam_name: _filter_pose_tensor_batch(cam_pose, filter_filtered_kpts_to_common)
            for cam_name, cam_pose in kpt3d_sam.items()
        }
    if "gt_kpt3d" in collated:
        gt = collated.pop("gt_kpt3d")
        if gt.ndim == 4 and gt.shape[-2] == 17:
            gt = _filter_pose_tensor_batch(gt, filter_h36m_kpts)
        collated["kpt3d_gt"] = gt
    return collated


def _safe_mpjpe(pred: torch.Tensor | None, label: torch.Tensor | None) -> float:
    if pred is None or label is None or pred.numel() == 0 or label.numel() == 0:
        return float("nan")
    return float(torch.norm(pred - label, dim=-1).mean().item())


def _safe_accel_err(pred: torch.Tensor | None, label: torch.Tensor | None) -> float:
    if pred is None or label is None or pred.shape[1] < 3 or label.shape[1] < 3:
        return float("nan")
    pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    label_acc = label[:, 2:] - 2.0 * label[:, 1:-1] + label[:, :-2]
    return float(torch.norm(pred_acc - label_acc, dim=-1).mean().item())


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_provenance(
    *,
    checkpoint: str | Path,
    index_mapping: str | Path,
    sample_count: int,
    split: str,
    seed: int,
    joint_subset: str,
    units: str,
    batch_size: int,
    time_window: int,
    path_rewrite_from: str | None = None,
    path_rewrite_to: str | None = None,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    index_mapping_path = Path(index_mapping).resolve()
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "index_mapping": str(index_mapping_path),
        "index_mapping_sha256": _file_sha256(index_mapping_path),
        "sample_count": int(sample_count),
        "split": str(split),
        "seed": int(seed),
        "joint_subset": str(joint_subset),
        "units": str(units),
        "batch_size": int(batch_size),
        "time_window": int(time_window),
        "drop_last": False,
        "path_rewrite_from": None if path_rewrite_from is None else str(path_rewrite_from),
        "path_rewrite_to": None if path_rewrite_to is None else str(path_rewrite_to),
    }


def _summarize_test_outputs(test_outputs: list[Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float | int]] = {
        name: {
            "mpjpe_sum": 0.0,
            "mpjpe_count": 0,
            "accel_sum": 0.0,
            "accel_count": 0,
        }
        for name in (
            "fused",
            "sam3d_left",
            "sam3d_right",
            "raw_avg",
            "left_canonical",
            "right_canonical",
            "canonical_avg",
        )
    }

    for batch_output in test_outputs:
        fused = batch_output.get("fused")
        p_left = batch_output.get("p_left")
        p_right = batch_output.get("p_right")
        left_canonical = batch_output.get("left_canonical")
        right_canonical = batch_output.get("right_canonical")
        ground_truth = batch_output.get("ground_truth")
        ground_truth_canonical = batch_output.get("ground_truth_canonical")

        raw_avg = None if p_left is None or p_right is None else 0.5 * (p_left + p_right)
        canonical_avg = None if left_canonical is None or right_canonical is None else 0.5 * (left_canonical + right_canonical)

        pairs = {
            "fused": (fused, ground_truth_canonical),
            "sam3d_left": (p_left, ground_truth),
            "sam3d_right": (p_right, ground_truth),
            "raw_avg": (raw_avg, ground_truth),
            "left_canonical": (left_canonical, ground_truth_canonical),
            "right_canonical": (right_canonical, ground_truth_canonical),
            "canonical_avg": (canonical_avg, ground_truth_canonical),
        }

        for name, (pred, label) in pairs.items():
            if pred is None or label is None or pred.numel() == 0 or label.numel() == 0:
                continue
            point_error = torch.norm(pred - label, dim=-1)
            stats[name]["mpjpe_sum"] += float(point_error.double().sum().item())
            stats[name]["mpjpe_count"] += int(point_error.numel())
            if pred.shape[1] >= 3 and label.shape[1] >= 3:
                pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
                label_acc = label[:, 2:] - 2.0 * label[:, 1:-1] + label[:, :-2]
                accel_error = torch.norm(pred_acc - label_acc, dim=-1)
                stats[name]["accel_sum"] += float(accel_error.double().sum().item())
                stats[name]["accel_count"] += int(accel_error.numel())

    return {
        name: {
            "mpjpe": (
                float(values["mpjpe_sum"]) / int(values["mpjpe_count"])
                if int(values["mpjpe_count"]) > 0
                else float("nan")
            ),
            "accel_err": (
                float(values["accel_sum"]) / int(values["accel_count"])
                if int(values["accel_count"]) > 0
                else float("nan")
            ),
        }
        for name, values in stats.items()
    }


def _format_float(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    LabeledSkiPosePTZDataset, _, _, _, _ = _repo_symbols()
    args = _parse_args()
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")

    seed_everything(42, workers=True)
    config = _build_config(args)

    ski_config = config.data.ski_pose_ptz
    path_rewrite_from = getattr(ski_config, "index_path_rewrite_from", None)
    path_rewrite_to = str(ski_config.root_path) if path_rewrite_from else None

    test_dataset = LabeledSkiPosePTZDataset(
        index_mapping=Path(config.data.ski_pose_ptz.index_mapping_path),
        transform=None,
        load_frames=False,
        load_2d_kpt=False,
        load_3d_kpt=True,
        target_t=int(args.time_window),
        split=args.split,
        path_rewrite_from=str(path_rewrite_from) if path_rewrite_from else None,
        path_rewrite_to=path_rewrite_to,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=not args.cpu,
        collate_fn=_collate_ski_poseptz_batch,
    )
    model = _build_model(config)

    use_gpu = torch.cuda.is_available() and not args.cpu
    trainer = Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=[int(args.gpu)] if use_gpu else 1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[RichProgressBar(refresh_rate=10, leave=True)],
    )

    metrics = trainer.test(
        model,
        dataloaders=test_loader,
        ckpt_path=str(args.ckpt_path),
        verbose=True,
        weights_only=False,
    )

    baseline_metrics = _summarize_test_outputs(list(getattr(model, "test_outputs", [])))
    fused_mpjpe = baseline_metrics.get("fused", {}).get("mpjpe", float("nan"))
    fused_accel_err = baseline_metrics.get("fused", {}).get("accel_err", float("nan"))
    canonical_avg_mpjpe = baseline_metrics.get("canonical_avg", {}).get("mpjpe", float("nan"))
    canonical_avg_accel = baseline_metrics.get("canonical_avg", {}).get("accel_err", float("nan"))
    raw_avg_mpjpe = baseline_metrics.get("raw_avg", {}).get("mpjpe", float("nan"))
    raw_avg_accel = baseline_metrics.get("raw_avg", {}).get("accel_err", float("nan"))

    summary_dir = Path(config.log_path) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    provenance = _build_provenance(
        checkpoint=args.ckpt_path,
        index_mapping=Path(config.data.ski_pose_ptz.index_mapping_path),
        sample_count=len(test_dataset),
        split=args.split,
        seed=42,
        joint_subset="common13",
        units="normalized_dataset_coordinates",
        batch_size=int(args.batch_size),
        time_window=int(args.time_window),
        path_rewrite_from=str(path_rewrite_from) if path_rewrite_from else None,
        path_rewrite_to=path_rewrite_to,
    )

    run_tag = args.ckpt_path.stem if getattr(args, "ckpt_path", None) else "run"
    comparison_csv = summary_dir / f"baseline_comparison_{run_tag}.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                *provenance.keys(),
                "fused_mpjpe",
                "canonical_avg_mpjpe",
                "raw_avg_mpjpe",
                "delta_mpjpe_full_minus_canonical_avg",
                "delta_mpjpe_full_minus_raw_avg",
                "fused_accel_err",
                "canonical_avg_accel_err",
                "raw_avg_accel_err",
                "delta_accel_full_minus_canonical_avg",
                "delta_accel_full_minus_raw_avg",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                **provenance,
                "fused_mpjpe": fused_mpjpe,
                "canonical_avg_mpjpe": canonical_avg_mpjpe,
                "raw_avg_mpjpe": raw_avg_mpjpe,
                "delta_mpjpe_full_minus_canonical_avg": fused_mpjpe - canonical_avg_mpjpe,
                "delta_mpjpe_full_minus_raw_avg": fused_mpjpe - raw_avg_mpjpe,
                "fused_accel_err": fused_accel_err,
                "canonical_avg_accel_err": canonical_avg_accel,
                "raw_avg_accel_err": raw_avg_accel,
                "delta_accel_full_minus_canonical_avg": fused_accel_err - canonical_avg_accel,
                "delta_accel_full_minus_raw_avg": fused_accel_err - raw_avg_accel,
            }
        )

    method_csv = summary_dir / f"method_comparison_{run_tag}.csv"
    with method_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["method", "mpjpe", "accel_err", *provenance.keys()],
        )
        writer.writeheader()
        for method, values in baseline_metrics.items():
            writer.writerow(
                {
                    "method": method,
                    "mpjpe": values["mpjpe"],
                    "accel_err": values["accel_err"],
                    **provenance,
                }
            )

    result_json = summary_dir / f"journal_evaluation_{run_tag}.json"
    result_json.write_text(
        json.dumps(
            {"provenance": provenance, "metrics": baseline_metrics},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = summary_dir / f"report_{run_tag}.txt"
    with report_path.open("w", encoding="utf-8") as fp:
        fp.write("Cross-View Fusion Test Report\n")
        fp.write("=" * 40 + "\n")
        fp.write("[Provenance]\n")
        for key, value in provenance.items():
            fp.write(f"{key}: {value}\n")
        fp.write("\n[Primary]\n")
        fp.write(f"fused_mpjpe: {_format_float(fused_mpjpe)}\n")
        fp.write(f"fused_accel_err: {_format_float(fused_accel_err)}\n")
        fp.write("\n[Baselines]\n")
        fp.write(f"canonical_avg_mpjpe: {_format_float(canonical_avg_mpjpe)}\n")
        fp.write(f"raw_avg_mpjpe: {_format_float(raw_avg_mpjpe)}\n")
        fp.write(f"delta_mpjpe_full_minus_canonical_avg: {_format_float(fused_mpjpe - canonical_avg_mpjpe)}\n")
        fp.write(f"delta_mpjpe_full_minus_raw_avg: {_format_float(fused_mpjpe - raw_avg_mpjpe)}\n")
        fp.write(f"canonical_avg_accel_err: {_format_float(canonical_avg_accel)}\n")
        fp.write(f"raw_avg_accel_err: {_format_float(raw_avg_accel)}\n")
        fp.write(f"delta_accel_full_minus_canonical_avg: {_format_float(fused_accel_err - canonical_avg_accel)}\n")
        fp.write(f"delta_accel_full_minus_raw_avg: {_format_float(fused_accel_err - raw_avg_accel)}\n")

    print("=== Test metrics ===")
    for item in metrics:
        for key, value in item.items():
            print(f"{key}: {value}")
    print(f"baseline comparison saved to: {comparison_csv}")
    print(f"method comparison saved to: {method_csv}")
    print(f"journal evaluation saved to: {result_json}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
