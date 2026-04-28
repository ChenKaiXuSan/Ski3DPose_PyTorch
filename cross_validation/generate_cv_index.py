#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
生成交叉验证索引文件
用于保存摄像头两两组合的交叉验证划分结果
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import hydra
from omegaconf import DictConfig, OmegaConf

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_validation.cross_validation_camera_pairs import CameraPairCrossValidation


def _serialize_sample(sample: Any) -> Dict[str, Any]:
    """Serialize a sample object to a JSON-friendly dict."""
    if hasattr(sample, "to_dict") and callable(sample.to_dict):
        return cast(Dict[str, Any], sample.to_dict())
    if isinstance(sample, dict):
        return sample
    raise TypeError(f"Unsupported sample type for serialization: {type(sample)}")


def _save_folds_separately(
    folds: Dict[int, Dict[str, Any]],
    strategy: str,
    index_mapping_dir: Path,
) -> List[Path]:
    """Save each fold into an individual JSON file."""
    fold_dir = index_mapping_dir / f"camera_pairs_{strategy}_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[Path] = []
    for fold_idx in sorted(folds.keys()):
        fold_data = folds[fold_idx]

        serialized_fold: Dict[str, Any] = {
            "train": [_serialize_sample(s) for s in fold_data["train"]],
            "val": [_serialize_sample(s) for s in fold_data["val"]],
            "test": [_serialize_sample(s) for s in fold_data.get("test", [])],
        }

        for key, value in fold_data.items():
            if key not in ["train", "val", "test"]:
                serialized_fold[key] = value

        serialized_fold["_metadata"] = {
            "strategy": strategy,
            "fold_idx": int(fold_idx),
            "num_train": len(fold_data["train"]),
            "num_val": len(fold_data["val"]),
            "num_test": len(fold_data.get("test", [])),
            "total": len(fold_data["train"])
            + len(fold_data["val"])
            + len(fold_data.get("test", [])),
        }

        fold_file = fold_dir / f"fold_{int(fold_idx):02d}.json"
        with open(fold_file, "w", encoding="utf-8") as f:
            json.dump(serialized_fold, f, ensure_ascii=False, indent=2)

        saved_files.append(fold_file.resolve())

    return saved_files


def _remove_legacy_aggregate_file(index_mapping_dir: Path, strategy: str) -> None:
    """Remove old aggregate JSON file if it exists.

    The new workflow keeps only split fold files.
    """
    aggregate_file = index_mapping_dir / f"camera_pairs_{strategy}.json"
    if aggregate_file.exists():
        aggregate_file.unlink()
        print(f"  已删除旧的整体索引文件: {aggregate_file}")


def generate_index_files(
    data_root: str,
    num_persons: int = 2,
    num_actions: int = 12,
    num_cameras: int = 108,
    use_layer_camera_filter: bool = False,
    selected_layers: Optional[List[int]] = None,
    selected_cameras_per_layer: Optional[Dict[str, List[int]]] = None,
    strategies: Optional[List[str]] = None,
    n_splits: int = 5,
    force_recreate: bool = False,
    sam3d_export_root: Optional[str] = None,
):
    """
    生成所有策略的索引文件

    Args:
        data_root: 数据根目录
        num_persons: 人物数量
        num_actions: 动作数量
        num_cameras: 摄像头数量
        use_layer_camera_filter: 是否启用按层/层内相机筛选
        selected_layers: 选中的层列表
        selected_cameras_per_layer: 每层选中的相机编号
        strategies: 要生成的策略列表，默认生成所有策略
        n_splits: K折数量
        force_recreate: 是否强制重新生成
    """
    if strategies is None:
        strategies = ["by_person", "by_action", "by_camera_pair"]

    index_mapping_dir = Path(data_root) / "index_mapping" / f"use_layer_camera_filter_{'enabled' if use_layer_camera_filter else 'disabled'}"
    index_mapping_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("生成交叉验证索引文件")
    print("=" * 80)
    print(f"数据根目录: {data_root}")
    print(f"人物数: {num_persons}")
    print(f"动作数: {num_actions}")
    print(f"摄像头数: {num_cameras}")
    print(
        "层/相机筛选: "
        f"{use_layer_camera_filter} (layers={selected_layers or []}, "
        f"cameras_per_layer={selected_cameras_per_layer or {}})"
    )
    print(f"策略: {', '.join(strategies)}")
    print(f"保存目录: {index_mapping_dir}")
    print("=" * 80 + "\n")

    results: Dict[str, Dict[str, Any]] = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"生成策略: {strategy}")
        print(f"{'='*60}")

        # 创建交叉验证对象
        cv = CameraPairCrossValidation(
            data_root=data_root,
            num_persons=num_persons,
            num_actions=num_actions,
            num_cameras=num_cameras,
            use_layer_camera_filter=use_layer_camera_filter,
            selected_layers=selected_layers,
            selected_cameras_per_layer=selected_cameras_per_layer,
            split_strategy=strategy,
            n_splits=n_splits,
            sam3d_export_root=sam3d_export_root,
            index_save_path=str(index_mapping_dir / f"camera_pairs_{strategy}.json"),
        )

        # 仅生成 fold 划分文件，不再保存整体聚合 json。
        fold_dir = index_mapping_dir / f"camera_pairs_{strategy}_folds"
        if fold_dir.exists() and not force_recreate:
            fold_files = sorted(p.resolve() for p in fold_dir.glob("fold_*.json"))
            if fold_files:
                print("✓ 发现已存在的 fold 划分文件，直接复用")
                results[strategy] = {
                    "fold_dir": str(fold_dir.resolve()),
                    "fold_files": fold_files,
                    "n_folds": len(fold_files),
                    "total_samples": "unknown",
                }
                print(f"  Fold目录: {results[strategy]['fold_dir']}")
                print(f"  Fold文件数: {len(fold_files)}")
                continue

        folds = cv.prepare_folds()
        fold_files = _save_folds_separately(
            folds=folds,
            strategy=strategy,
            index_mapping_dir=index_mapping_dir,
        )
        _remove_legacy_aggregate_file(index_mapping_dir, strategy)

        # 记录结果
        strategy_fold_dir = (
            index_mapping_dir / f"camera_pairs_{strategy}_folds"
        ).resolve()

        results[strategy] = {
            "fold_dir": str(strategy_fold_dir),
            "fold_files": fold_files,
            "n_folds": len(folds),
            "total_samples": sum(
                len(fold["train"]) + len(fold["val"]) + len(fold.get("test", []))
                for fold in folds.values()
            )
            // len(folds),
        }

        print(f"\n✓ 策略 '{strategy}' fold 索引文件已生成")
        print(f"  Fold目录: {results[strategy]['fold_dir']}")
        print(f"  Fold文件数: {len(fold_files)}")
        for fold_file in fold_files:
            print(f"    - {fold_file}")
        print(f"  折数: {results[strategy]['n_folds']}")
        print(f"  每折样本数: {results[strategy]['total_samples']}")

    # 打印汇总信息
    print("\n" + "=" * 80)
    print("✅ 所有索引文件生成完成")
    print("=" * 80)

    for strategy, info in results.items():
        fold_files_summary = cast(List[Path], info.get("fold_files", []))
        print(f"\n[{strategy}]")
        print(f"  Fold目录: {info['fold_dir']}")
        print(f"  Fold文件数: {len(fold_files_summary)}")
        print(f"  折数: {info['n_folds']}")
        total_samples = info["total_samples"]
        if isinstance(total_samples, int):
            print(f"  样本数: {total_samples:,}")

    print("\n" + "=" * 80)
    print("📁 所有索引文件位置:")
    print(f"   {index_mapping_dir}")
    print("=" * 80 + "\n")

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def hydra_main(cfg: Optional[DictConfig] = None) -> None:
    """
    通过 Hydra 配置生成 index mapping。

    默认读取 configs/train.yaml 中 data.cross_validation 字段，
    也支持命令行 override，例如：
    python cross_validation/generate_cv_index.py data.cross_validation.force_recreate=true
    """
    if cfg is None:
        raise ValueError("Hydra cfg is required")

    cv_cfg = cfg.data.cross_validation

    # 打印当前配置，便于排查。
    print("\nHydra cross validation config:")
    print(OmegaConf.to_yaml(cv_cfg))

    generate_index_files(
        data_root=str(cfg.data.root_path),
        num_persons=int(cv_cfg.num_persons),
        num_actions=int(cv_cfg.num_actions),
        num_cameras=int(cv_cfg.num_cameras),
        use_layer_camera_filter=bool(cv_cfg.get("use_layer_camera_filter", False)),
        selected_layers=[
            int(x) for x in cast(List[Any], list(cv_cfg.get("selected_layers", [])))
        ],
        selected_cameras_per_layer={
            str(layer): [int(cam_id) for cam_id in cast(List[Any], cam_ids)]
            for layer, cam_ids in cast(
                Dict[str, Any],
                OmegaConf.to_container(
                    cv_cfg.get("selected_cameras_per_layer", {}),
                    resolve=True,
                ),
            ).items()
        },
        strategies=list(cv_cfg.strategies),
        n_splits=int(cv_cfg.n_splits),
        force_recreate=bool(cv_cfg.force_recreate),
        sam3d_export_root=str(cfg.data.get("sam3d_export_path", "")),
    )


if __name__ == "__main__":
    cast(Any, hydra_main)()
