#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Run ablation variants for Unity dual-view evaluation.

This script toggles the same `disable_*` flags on the fusion model and runs
`Trainer.test()` using the UnityDataModule, then writes a summary CSV to
`<log_path>/occlusion_eval/ablation_unity_summary.csv`.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import RichProgressBar

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@lru_cache(maxsize=1)
def _repo_symbols():
    data_module = importlib.import_module("dual2pose.dataloader.data_loader")
    eval_unity = importlib.import_module("dual2pose.eval_data.eval_unity_masking")
    fusion_module = importlib.import_module("dual2pose.trainer.train_crossview_fusion")
    dual_module = importlib.import_module("dual2pose.trainer.train_dual2pose")
    return data_module.UnityDataModule, eval_unity, fusion_module.CrossViewFusionTrainer, dual_module.Dual2PoseTrainer


DEFAULT_CKPT = (
    "/workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/logs/train_unity/"
    "crossview_fusion/2026-05-14/04-55-35/checkpoints/last.ckpt"
)
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dual2pose.yaml"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", type=Path, default=Path(DEFAULT_CKPT))
    p.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--variants", nargs="*", default=[
        "full",
        "no_aligned",
        "no_residual",
        "no_velocity",
        "no_rotvec",
        "no_residual_no_rotvec",
    ])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for Unity ablation results. Overrides config.log_path.",
    )
    return p.parse_args()


def _build_trainer(config, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    use_gpu = torch.cuda.is_available() and not getattr(config.train, "cpu", False)
    trainer = Trainer(
        devices=[int(config.train.gpu)] if use_gpu else 1,
        accelerator="gpu" if use_gpu else "cpu",
        max_epochs=int(config.train.max_epochs),
        logger=False,
        callbacks=[RichProgressBar(refresh_rate=10, leave=True)],
    )
    return trainer


def main() -> None:
    UnityDataModule, eval_unity, CrossViewFusionTrainer, Dual2PoseTrainer = _repo_symbols()
    args = _parse_args()
    if not args.ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")

    seed_everything(42, workers=True)
    config = OmegaConf.load(str(args.config_path))
    config.train.gpu = int(getattr(config.train, "gpu", 0))

    root_log = Path(str(config.log_path))
    results_root = Path(args.output_path) if getattr(args, "output_path", None) else root_log / "occlusion_eval"
    results_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for variant in args.variants:
        cfg = OmegaConf.to_container(config, resolve=True)
        run_cfg = OmegaConf.create(cfg)
        run_cfg.log_path = str(results_root / variant)

        model = CrossViewFusionTrainer(run_cfg)

        # set ablation flags on inner model if present
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

        if hasattr(model, "models") and model.models is not None:
            for k, v in flags.items():
                if hasattr(model.models, k):
                    setattr(model.models, k, v)

        base_dm = UnityDataModule(run_cfg)
        trainer = _build_trainer(run_cfg, save_dir=Path(run_cfg.log_path))

        trainer.test(model, datamodule=base_dm, ckpt_path=str(args.ckpt_path))

        test_outputs = list(getattr(model, "test_outputs", []))
        flat = eval_unity._flatten_test_outputs(test_outputs)
        metrics = eval_unity._summarize_outputs(flat=flat, failure_threshold=0.15)

        fused_mpjpe = metrics.get("fused", {}).get("mpjpe", math.nan)
        canonical_avg_mpjpe = metrics.get("canonical_avg", {}).get("mpjpe", math.nan)

        rows.append(
            {
                "variant": variant,
                "fused_mpjpe": fused_mpjpe,
                "canonical_avg_mpjpe": canonical_avg_mpjpe,
                "delta_mpjpe_full_minus_avg": fused_mpjpe - canonical_avg_mpjpe,
            }
        )

    run_tag = args.ckpt_path.stem if getattr(args, "ckpt_path", None) else "run"
    out_csv = results_root / f"ablation_unity_summary_{run_tag}.csv"
    if rows:
        with open(out_csv, "w", encoding="utf-8", newline="") as fp:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote Unity ablation summary to: {out_csv}")


if __name__ == "__main__":
    main()
