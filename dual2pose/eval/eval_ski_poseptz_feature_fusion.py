#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Run ablation variants for CrossView fusion on Ski-PosePTZ.

This script runs a set of ablation variants by toggling flags on the
`CrossViewCanonicalFusion` instance inside the Lightning `CrossViewFusionTrainer`.
It writes a CSV summary to `<ckpt_log>/summary/ablation_summary.csv`.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict

import importlib
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


def _parse_args():
    parser = argparse.ArgumentParser(description="Run ablation variants on Ski-PosePTZ")
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variants", nargs="*", default=[
        "full",
        "no_aligned",
        "no_residual",
        "no_velocity",
        "no_rotvec",
        "no_residual_no_rotvec",
    ])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--time-window", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional base output path for ablation summaries. Overrides config.log_path.",
    )
    return parser.parse_args()


def _build_config(args):
    config = OmegaConf.load(str(args.config_path))
    config.model.backbone = "crossview_fusion"
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
    config.log_path = str(ckpt_dir)
    return config


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


def _mean_ignore_nan(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def _summarize_test_outputs(test_outputs: list[Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, list[float]]] = {
        "fused": {"mpjpe": [], "accel_err": []},
        "sam3d_left": {"mpjpe": [], "accel_err": []},
        "sam3d_right": {"mpjpe": [], "accel_err": []},
        "raw_avg": {"mpjpe": [], "accel_err": []},
        "left_canonical": {"mpjpe": [], "accel_err": []},
        "right_canonical": {"mpjpe": [], "accel_err": []},
        "canonical_avg": {"mpjpe": [], "accel_err": []},
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
            stats[name]["mpjpe"].append(_safe_mpjpe(pred, label))
            stats[name]["accel_err"].append(_safe_accel_err(pred, label))

    return {
        name: {
            "mpjpe": _mean_ignore_nan(values["mpjpe"]),
            "accel_err": _mean_ignore_nan(values["accel_err"]),
        }
        for name, values in stats.items()
    }


def _format_float(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.6f}"


def main() -> None:
    LabeledSkiPosePTZDataset, _, _, crossview_cls, _ = _repo_symbols()
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
    base_log_path = Path(args.output_path) if getattr(args, "output_path", None) else Path(config.log_path)

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

    use_gpu = torch.cuda.is_available() and not args.cpu

    rows = []
    for variant in args.variants:
        setting_name = variant
        config.log_path = str(base_log_path / setting_name)
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(args.num_workers),
            pin_memory=not args.cpu,
            collate_fn=_collate_ski_poseptz_batch,
        )

        model = crossview_cls(config)

        # map variant name to flags on the inner fusion model
        flags = {
            "disable_aligned": False,
            "disable_residual": False,
            "disable_velocity": False,
            "disable_rotvec": False,
        }
        if variant == "no_aligned":
            flags["disable_aligned"] = True
        if variant == "no_residual":
            flags["disable_residual"] = True
        if variant == "no_velocity":
            flags["disable_velocity"] = True
        if variant == "no_rotvec":
            flags["disable_rotvec"] = True
        if variant == "no_residual_no_rotvec":
            flags["disable_residual"] = True
            flags["disable_rotvec"] = True

        # set flags if inner model exists
        if hasattr(model, "models") and model.models is not None:
            for k, v in flags.items():
                if hasattr(model.models, k):
                    setattr(model.models, k, v)

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
            verbose=False,
        )

        flat_metrics = metrics[0] if metrics else {}
        fused_mpjpe = flat_metrics.get("test/mpjpe", flat_metrics.get("mpjpe", float("nan")))
        accel_err = flat_metrics.get("test/accel_err", flat_metrics.get("accel_err", float("nan")))

        baseline_metrics = _summarize_test_outputs(list(getattr(model, "test_outputs", [])))
        canonical_avg_mpjpe = baseline_metrics.get("canonical_avg", {}).get("mpjpe", float("nan"))
        canonical_avg_accel = baseline_metrics.get("canonical_avg", {}).get("accel_err", float("nan"))
        raw_avg_mpjpe = baseline_metrics.get("raw_avg", {}).get("mpjpe", float("nan"))
        raw_avg_accel = baseline_metrics.get("raw_avg", {}).get("accel_err", float("nan"))

        rows.append({
            "variant": setting_name,
            "mpjpe": float(fused_mpjpe),
            "accel_err": float(accel_err),
            "canonical_avg_mpjpe": float(canonical_avg_mpjpe),
            "raw_avg_mpjpe": float(raw_avg_mpjpe),
            "delta_mpjpe_full_minus_canonical_avg": float(fused_mpjpe - canonical_avg_mpjpe),
            "delta_mpjpe_full_minus_raw_avg": float(fused_mpjpe - raw_avg_mpjpe),
            "canonical_avg_accel_err": float(canonical_avg_accel),
            "raw_avg_accel_err": float(raw_avg_accel),
            "delta_accel_full_minus_canonical_avg": float(accel_err - canonical_avg_accel),
            "delta_accel_full_minus_raw_avg": float(accel_err - raw_avg_accel),
        })

    # write summary CSV
    summary_dir = Path(config.log_path).resolve().parent / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.ckpt_path.stem if getattr(args, "ckpt_path", None) else "run"
    out_path = summary_dir / f"ablation_summary_{run_tag}.csv"
    if rows:
        with open(out_path, "w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    print(f"Wrote ablation summary to: {out_path}")


if __name__ == "__main__":
    main()
