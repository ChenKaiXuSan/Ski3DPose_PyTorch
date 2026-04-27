#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project/cross_validation_camera_pairs.py
Project: /workspace/Skiing_Canonical_DualView_3D_Pose_PyTorch/project
Created Date: Sunday March 9th 2026
Author: Kaixu Chen
-----
Comment:
交叉验证脚本 - 用于摄像头两两组合的场景
针对2个人物、12个动作、每个动作108个摄像头的数据集。
支持三种划分策略：
1. by_person: 按人物划分（Leave-One-Person-Out）
2. by_action: 按动作划分（K-Fold on actions）
3. by_camera_pair: 按摄像头对划分（K-Fold on camera pairs）

Have a good code time :)
-----
Copyright (c) 2026 The University of Tsukuba
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, fields
from itertools import combinations

import numpy as np
from sklearn.model_selection import KFold


KPT_VARIANTS = ("character", "pole", "ski")


@dataclass
class CameraPairSample:
    """表示一个训练样本：人物 + 动作 + 摄像头对"""

    person_id: str  # "01", "02"
    action_id: str  # "action_01", "action_02", ..., "action_12"
    cam1_id: str  # "cam_001", "cam_002", ..., "cam_108"
    cam2_id: str  # "cam_001", "cam_002", ..., "cam_108"
    cam1_path: Optional[str] = None  # 兼容旧字段：camera1 frames目录
    cam2_path: Optional[str] = None  # 兼容旧字段：camera2 frames目录
    label_path: Optional[str] = None  # 兼容旧字段：sequence.json路径
    cam1_frames_dir: Optional[str] = None
    cam2_frames_dir: Optional[str] = None
    cam1_kpt2d_dirs: Optional[Dict[str, str]] = None
    cam2_kpt2d_dirs: Optional[Dict[str, str]] = None

    kpt3d_dirs: Optional[Dict[str, str]] = None

    sam3d_cam1_kpt2d_dir: Optional[str] = None
    sam3d_cam2_kpt2d_dir: Optional[str] = None

    sam3d_cam1_kpt3d_dir: Optional[str] = None
    sam3d_cam2_kpt3d_dir: Optional[str] = None

    sequence_meta_path: Optional[str] = None
    joint_names_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class CameraPairCrossValidation:
    """
    针对摄像头对的交叉验证策略

    参数:
        data_root: 数据根目录
        num_persons: 人物数量（默认2）
        num_actions: 动作数量（默认12）
        num_cameras: 每个动作的摄像头数量（默认108）
        split_strategy: 划分策略，可选 'by_person', 'by_action', 'by_camera_pair'
        n_splits: K折交叉验证的折数（仅用于 by_action 和 by_camera_pair 策略）
        index_save_path: 索引文件保存路径
    """

    def __init__(
        self,
        data_root: str,
        num_persons: int = 2,
        num_actions: int = 12,
        num_cameras: int = 108,
        split_strategy: str = "by_person",  # by_person, by_action, by_camera_pair
        n_splits: int = 5,
        sam3d_export_root: Optional[str] = None,
        index_save_path: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        self.data_dir = self._resolve_data_dir()
        self.sam3d_export_root = (
            Path(sam3d_export_root)
            if sam3d_export_root
            else self.data_root / "modalities_from_sam3d"
        )
        self.num_persons = num_persons
        self.num_actions = num_actions
        self.num_cameras = num_cameras
        self.split_strategy = split_strategy
        self.n_splits = n_splits

        if index_save_path is None:
            self.index_save_path: Path = (
                self.data_root / "index_mapping" / f"camera_pairs_{split_strategy}.json"
            )
        else:
            self.index_save_path = Path(index_save_path)
        self.index_save_path = self.index_save_path.resolve()

        self.index_save_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_data_dir(self) -> Path:
        candidate_names = ["data_pole_ski", "data"]
        for name in candidate_names:
            candidate = self.data_root / name
            if candidate.exists():
                return candidate
        return self.data_root / "data"

    @staticmethod
    def _capture_to_kpt2d_id(capture_name: str) -> str:
        # frames目录是 capture_Lx_Ayyy；kpt2d目录是 Lx_Ayyy
        return capture_name.replace("capture_", "", 1)

    @staticmethod
    def _build_variant_dir_map(base_dir: Path) -> Dict[str, str]:
        variant_dirs: Dict[str, str] = {}
        for variant in KPT_VARIANTS:
            variant_dir = base_dir / variant
            if variant_dir.exists():
                variant_dirs[variant] = str(variant_dir.absolute())
        return variant_dirs

    @staticmethod
    def _select_default_variant_dir(
        variant_dirs: Dict[str, str], fallback_dir: Path
    ) -> str:
        if "character" in variant_dirs:
            return variant_dirs["character"]
        if variant_dirs:
            return variant_dirs[sorted(variant_dirs.keys())[0]]
        return str(fallback_dir.absolute())

    @staticmethod
    def _to_abs_path(path: Path) -> str:
        return str(path.absolute())

    def _discover_people_actions(self) -> Dict[str, List[str]]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data目录不存在: {self.data_dir}")

        people_actions: Dict[str, List[str]] = {}
        for person_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            # 跳过辅助目录
            if person_dir.name.lower() in {"logs", "cameras"}:
                continue
            action_names: List[str] = []
            for action_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
                if (action_dir / "frames").exists():
                    action_names.append(action_dir.name)
            if action_names:
                people_actions[person_dir.name] = action_names
        return people_actions

    def build_all_samples(self) -> List[CameraPairSample]:
        """
        扫描真实目录，构建样本：person × action × camera_capture_pairs

        Returns:
            所有样本的列表
        """
        samples: List[CameraPairSample] = []
        people_actions = self._discover_people_actions()

        action_count_total = 0
        per_action_pair_count: List[int] = []

        for person_id, actions in people_actions.items():
            for action_id in actions:
                action_count_total += 1
                action_dir = self.data_dir / person_id / action_id
                frames_root = action_dir / "frames"
                kpt2d_root = action_dir / "kpt2d"
                kpt3d_dir = action_dir / "kpt3d"
                meta_dir = action_dir / "meta"
                sam3d_export_action = self.sam3d_export_root / person_id / action_id

                capture_dirs = sorted(
                    p
                    for p in frames_root.iterdir()
                    if p.is_dir() and p.name.startswith("capture_")
                )
                if len(capture_dirs) < 2:
                    continue

                per_action_pair_count.append(
                    len(capture_dirs) * (len(capture_dirs) - 1) // 2
                )

                for cam1_dir, cam2_dir in combinations(capture_dirs, 2):
                    cam1_id = cam1_dir.name
                    cam2_id = cam2_dir.name

                    kpt2d_cam1 = kpt2d_root / self._capture_to_kpt2d_id(cam1_id)
                    kpt2d_cam2 = kpt2d_root / self._capture_to_kpt2d_id(cam2_id)
                    kpt2d_cam1_dirs = self._build_variant_dir_map(kpt2d_cam1)
                    kpt2d_cam2_dirs = self._build_variant_dir_map(kpt2d_cam2)
                    kpt3d_dirs = self._build_variant_dir_map(kpt3d_dir)

                    sam3d_cam1_kpt2d = sam3d_export_action / "kpt2d" / cam1_id
                    sam3d_cam2_kpt2d = sam3d_export_action / "kpt2d" / cam2_id
                    sam3d_cam1_kpt3d = sam3d_export_action / "kpt3d" / cam1_id
                    sam3d_cam2_kpt3d = sam3d_export_action / "kpt3d" / cam2_id

                    sequence_meta = meta_dir / "sequence.json"
                    joint_meta = meta_dir / "joint_names.json"

                    sample = CameraPairSample(
                        person_id=person_id,
                        action_id=action_id,
                        cam1_id=cam1_id,
                        cam2_id=cam2_id,
                        cam1_path=self._to_abs_path(cam1_dir),
                        cam2_path=self._to_abs_path(cam2_dir),
                        label_path=self._to_abs_path(sequence_meta),
                        cam1_frames_dir=self._to_abs_path(cam1_dir),
                        cam2_frames_dir=self._to_abs_path(cam2_dir),
                        cam1_kpt2d_dirs=kpt2d_cam1_dirs or None,
                        cam2_kpt2d_dirs=kpt2d_cam2_dirs or None,
                        kpt3d_dirs=kpt3d_dirs or None,
                        sam3d_cam1_kpt2d_dir=self._to_abs_path(sam3d_cam1_kpt2d),
                        sam3d_cam2_kpt2d_dir=self._to_abs_path(sam3d_cam2_kpt2d),
                        sam3d_cam1_kpt3d_dir=self._to_abs_path(sam3d_cam1_kpt3d),
                        sam3d_cam2_kpt3d_dir=self._to_abs_path(sam3d_cam2_kpt3d),
                        sequence_meta_path=self._to_abs_path(sequence_meta),
                        joint_names_path=self._to_abs_path(joint_meta),
                    )
                    samples.append(sample)

        people_count = len(people_actions)
        action_count = sum(len(v) for v in people_actions.values())
        avg_pairs = int(np.mean(per_action_pair_count)) if per_action_pair_count else 0

        print(f"✓ 总共生成 {len(samples)} 个样本")
        print(f"  - {people_count} 个人物")
        print(f"  - {action_count} 个动作")
        print(f"  - 每个动作平均 {avg_pairs} 个摄像头对")

        return samples

    def split_by_person(
        self, samples: List[CameraPairSample]
    ) -> Dict[int, Dict[str, Any]]:
        """
        策略1: 按人物划分 (Leave-One-Person-Out with train/val/test split)
        每一折使用一个人的数据进行 7/2/1 train/val/test 划分
        其他人的数据全部作为额外的训练数据
        """
        fold_dict: Dict[int, Dict[str, Any]] = {}

        person_ids = sorted(set(s.person_id for s in samples))

        if len(person_ids) <= 1:
            return {
                0: {
                    "train": samples,
                    "val": [],
                    "test": [],
                    "val_person": person_ids[0] if person_ids else "unknown",
                }
            }

        rng = np.random.default_rng(42)

        for fold_idx, test_person in enumerate(person_ids):
            # 将测试人物的数据分成 train:val:test = 7:2:1
            test_person_samples = [s for s in samples if s.person_id == test_person]

            # 随机打乱测试人物的样本
            shuffled_test_samples = test_person_samples.copy()
            rng_fold = np.random.default_rng(
                int(rng.integers(0, 10_000_000)) + fold_idx
            )
            rng_fold.shuffle(shuffled_test_samples)

            n_total = len(shuffled_test_samples)
            n_train_person = int(round(n_total * 0.7))
            n_val_person = int(round(n_total * 0.2))
            n_test_person = n_total - n_train_person - n_val_person

            # 保证非空
            if n_total >= 3:
                if n_train_person <= 0:
                    n_train_person = 1
                if n_val_person <= 0:
                    n_val_person = 1
                n_test_person = n_total - n_train_person - n_val_person
                if n_test_person <= 0:
                    n_test_person = 1
                    if n_train_person > n_val_person:
                        n_train_person -= 1
                    else:
                        n_val_person -= 1

            train_person_samples = shuffled_test_samples[:n_train_person]
            val_person_samples = shuffled_test_samples[
                n_train_person : n_train_person + n_val_person
            ]
            test_person_samples_split = shuffled_test_samples[
                n_train_person + n_val_person :
            ]

            # 其他人的数据全部作为训练数据
            other_samples = [s for s in samples if s.person_id != test_person]

            final_train = train_person_samples + other_samples
            final_val = val_person_samples
            final_test = test_person_samples_split

            fold_dict[fold_idx] = {
                "train": final_train,
                "val": final_val,
                "test": final_test,
                "val_person": test_person,
                "ratio": "7/2/1",
            }

            print(
                f"Fold {fold_idx}: train={len(final_train)}, val={len(final_val)}, test={len(final_test)} (person={test_person})"
            )

        return fold_dict

    def split_by_action(
        self, samples: List[CameraPairSample]
    ) -> Dict[int, Dict[str, Any]]:
        """
        策略2: 按动作划分 (K-Fold on actions with train/val/test split)
        将动作分成K折，每折某些动作用于训练、验证和测试
        """
        fold_dict: Dict[int, Dict[str, Any]] = {}

        action_ids = sorted(set(s.action_id for s in samples))
        n_splits = min(self.n_splits, len(action_ids))
        if n_splits <= 1:
            return {
                0: {
                    "train": samples,
                    "val": [],
                    "test": [],
                    "val_actions": [],
                    "test_actions": [],
                }
            }

        # 方案：将所有actions分成多组进行K折CV
        # 对于每一fold，使用不同的actions作为train/val/test
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        action_ids_array = np.array(action_ids)

        fold_splits = list(kf.split(action_ids_array))
        rng = np.random.default_rng(42)

        for fold_idx in range(n_splits):
            # 方式：当前fold的train_action_indices为train
            # val_test_action_indices再分出2/3为val，1/3为test
            train_action_indices, val_test_action_indices = fold_splits[fold_idx]

            train_actions_set = set(action_ids_array[train_action_indices])
            val_test_actions = action_ids_array[val_test_action_indices]

            # 随机打乱验证+测试actions
            shuffled_val_test = val_test_actions.copy()
            rng_fold = np.random.default_rng(
                int(rng.integers(0, 10_000_000)) + fold_idx
            )
            rng_fold.shuffle(shuffled_val_test)

            n_val_test = len(shuffled_val_test)
            n_val = max(1, int(round(n_val_test * 0.67)))  # 2/3 for val

            val_actions = set(shuffled_val_test[:n_val])
            test_actions = set(shuffled_val_test[n_val:])

            train_samples = [s for s in samples if s.action_id in train_actions_set]
            val_samples = [s for s in samples if s.action_id in val_actions]
            test_samples = [s for s in samples if s.action_id in test_actions]

            fold_dict[fold_idx] = {
                "train": train_samples,
                "val": val_samples,
                "test": test_samples,
                "val_actions": sorted(list(val_actions)),
                "test_actions": sorted(list(test_actions)),
                "ratio": "7/2/1",
            }

            print(
                f"Fold {fold_idx}: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)} (actions: train={len(train_actions_set)}, val={len(val_actions)}, test={len(test_actions)})"
            )

        return fold_dict

    def split_by_camera_pair(
        self, samples: List[CameraPairSample]
    ) -> Dict[int, Dict[str, Any]]:
        """
        策略3: 按摄像头对划分（每个fold内按7/2/1切分train/val/test）
        每个fold都会基于相机对产生独立的 train/val/test。

        注意：这种方式会产生大量的样本，可能需要采样或分层
        """
        fold_dict: Dict[int, Dict[str, Any]] = {}

        # 按照 (person, action, cam_pair) 分组
        # 为了简化，我们可以只按摄像头对来划分
        camera_pairs = list(set((s.cam1_id, s.cam2_id) for s in samples))

        # 如果摄像头对太多，可以考虑采样
        if len(camera_pairs) > self.n_splits * 100:
            print(f"⚠ 摄像头对数量过多 ({len(camera_pairs)})，考虑使用随机采样")
            np.random.seed(42)
            sampled_indices = np.random.choice(
                len(samples), size=min(len(samples), 10000), replace=False
            )
            samples = [samples[i] for i in sampled_indices]
            camera_pairs = list(set((s.cam1_id, s.cam2_id) for s in samples))

        n_splits = max(1, min(self.n_splits, len(camera_pairs)))

        rng = np.random.default_rng(42)
        camera_pairs = list(camera_pairs)

        for fold_idx in range(n_splits):
            shuffled_pairs = camera_pairs.copy()
            rng_fold = np.random.default_rng(
                int(rng.integers(0, 10_000_000)) + fold_idx
            )
            rng_fold.shuffle(shuffled_pairs)

            n_total_pairs = len(shuffled_pairs)
            n_train_pairs = int(round(n_total_pairs * 0.7))
            n_val_pairs = int(round(n_total_pairs * 0.2))
            n_test_pairs = n_total_pairs - n_train_pairs - n_val_pairs

            # Keep all splits non-empty when possible.
            if n_total_pairs >= 3:
                if n_train_pairs <= 0:
                    n_train_pairs = 1
                if n_val_pairs <= 0:
                    n_val_pairs = 1
                n_test_pairs = n_total_pairs - n_train_pairs - n_val_pairs
                if n_test_pairs <= 0:
                    n_test_pairs = 1
                    if n_train_pairs > n_val_pairs:
                        n_train_pairs -= 1
                    else:
                        n_val_pairs -= 1

            train_pairs = set(shuffled_pairs[:n_train_pairs])
            val_pairs = set(shuffled_pairs[n_train_pairs : n_train_pairs + n_val_pairs])
            test_pairs = set(shuffled_pairs[n_train_pairs + n_val_pairs :])

            train_samples = [
                s for s in samples if (s.cam1_id, s.cam2_id) in train_pairs
            ]
            val_samples = [s for s in samples if (s.cam1_id, s.cam2_id) in val_pairs]
            test_samples = [s for s in samples if (s.cam1_id, s.cam2_id) in test_pairs]

            fold_dict[fold_idx] = {
                "train": train_samples,
                "val": val_samples,
                "test": test_samples,
                "ratio": "7/2/1",
            }

            print(
                f"Fold {fold_idx}: train={len(train_samples)}, "
                f"val={len(val_samples)}, test={len(test_samples)}"
            )

        return fold_dict

    def prepare_folds(self) -> Dict[int, Dict[str, Any]]:
        """
        根据选择的策略准备交叉验证的折
        """
        print(f"\n{'='*60}")
        print("准备交叉验证数据集")
        print(f"{'='*60}")
        print(f"策略: {self.split_strategy}")
        print(f"数据根目录: {self.data_root}")
        print(f"{'='*60}\n")

        samples = self.build_all_samples()

        if self.split_strategy == "by_person":
            fold_dict = self.split_by_person(samples)
        elif self.split_strategy == "by_action":
            fold_dict = self.split_by_action(samples)
        elif self.split_strategy == "by_camera_pair":
            fold_dict = self.split_by_camera_pair(samples)
        else:
            raise ValueError(f"未知的划分策略: {self.split_strategy}")

        return fold_dict

    def save_folds(self, fold_dict: Dict[int, Dict[str, Any]]):
        """
        保存交叉验证的划分结果到JSON文件
        """
        # 序列化
        serialized: Dict[str, Any] = {}
        for fold_idx, fold_data in fold_dict.items():
            serialized[str(fold_idx)] = {
                "train": [s.to_dict() for s in fold_data["train"]],
                "val": [s.to_dict() for s in fold_data["val"]],
                "test": [s.to_dict() for s in fold_data.get("test", [])],
            }
            # 保存额外信息（如验证集的人物或动作）
            for key in fold_data:
                if key not in ["train", "val", "test"]:
                    serialized[str(fold_idx)][key] = fold_data[key]

        # 添加元数据
        serialized["_metadata"] = {
            "num_persons": self.num_persons,
            "num_actions": self.num_actions,
            "num_cameras": self.num_cameras,
            "split_strategy": self.split_strategy,
            "n_splits": len(fold_dict),
            "total_samples": sum(
                len(fold_data["train"])
                + len(fold_data["val"])
                + len(fold_data.get("test", []))
                for fold_data in fold_dict.values()
            )
            // len(fold_dict),
        }

        with open(self.index_save_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 交叉验证索引已保存到: {self.index_save_path}")

    def load_folds(self) -> Dict[int, Dict[str, Any]]:
        """
        从JSON文件加载交叉验证的划分结果
        """
        if not self.index_save_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {self.index_save_path}")

        with open(self.index_save_path, "r", encoding="utf-8") as f:
            serialized = json.load(f)

        # 提取元数据
        metadata = serialized.pop("_metadata", {})
        print("\n加载交叉验证数据:")
        print(f"  策略: {metadata.get('split_strategy', 'unknown')}")
        print(f"  折数: {metadata.get('n_splits', 'unknown')}")
        print(f"  总样本数: {metadata.get('total_samples', 'unknown')}")

        # 反序列化
        fold_dict: Dict[int, Dict[str, Any]] = {}
        for fold_idx_str, fold_data in serialized.items():
            fold_idx = int(fold_idx_str)
            fold_dict[fold_idx] = {
                "train": [CameraPairSample.from_dict(d) for d in fold_data["train"]],
                "val": [CameraPairSample.from_dict(d) for d in fold_data["val"]],
                "test": [
                    CameraPairSample.from_dict(d) for d in fold_data.get("test", [])
                ],
            }
            # 恢复额外信息
            for key in fold_data:
                if key not in ["train", "val", "test"]:
                    fold_dict[fold_idx][key] = fold_data[key]

        return fold_dict

    def __call__(self, force_recreate: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        主入口：创建或加载交叉验证划分

        Args:
            force_recreate: 是否强制重新创建索引文件
        """
        if self.index_save_path.exists() and not force_recreate:
            print("✓ 发现已存在的索引文件，直接加载")
            return self.load_folds()
        else:
            print("✓ 创建新的交叉验证划分")
            fold_dict = self.prepare_folds()
            self.save_folds(fold_dict)
            return fold_dict


def main():
    """
    示例使用
    """
    for strategy in ["by_person", "by_action", "by_camera_pair"]:
        cv = CameraPairCrossValidation(
            data_root="/workspace/data/skiing_unity_dataset",
            split_strategy=strategy,
            n_splits=5,
            index_save_path=f"/workspace/data/skiing_unity_dataset/index_mapping/camera_pairs_{strategy}.json",
        )
        folds = cv(force_recreate=True)
        print(f"\n示例 - {strategy}:")
        print(f"折数: {len(folds)}")


if __name__ == "__main__":
    main()
