#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
简化的测试脚本 - 快速验证交叉验证功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from cross_validation.cross_validation_camera_pairs import CameraPairCrossValidation


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "="*80)
    print("测试 1: 基本功能测试")
    print("="*80)
    
    # 创建一个小规模测试（减少摄像头数量以加快测试）
    cv = CameraPairCrossValidation(
        data_root="/workspace/data/skiing_unity_dataset",
        num_persons=2,
        num_actions=3,  # 只用3个动作进行测试
        num_cameras=10,  # 只用10个摄像头进行测试
        split_strategy="by_person",
        index_save_path="/tmp/test_camera_pairs_by_person.json"
    )
    
    # 生成交叉验证划分
    folds = cv(force_recreate=True)
    
    # 验证结果
    print(f"\n✓ 生成了 {len(folds)} 折")
    
    for fold_idx, fold_data in folds.items():
        train_size = len(fold_data['train'])
        val_size = len(fold_data['val'])
        print(f"\nFold {fold_idx}:")
        print(f"  训练集: {train_size} 样本")
        print(f"  验证集: {val_size} 样本")
        
        # 检查训练集和验证集的人物是否不同
        train_persons = set(s.person_id for s in fold_data['train'])
        val_persons = set(s.person_id for s in fold_data['val'])
        print(f"  训练人物: {train_persons}")
        print(f"  验证人物: {val_persons}")
        
        # 确保没有重叠
        assert len(train_persons & val_persons) == 0, "训练集和验证集人物有重叠！"
        
        # 查看几个样本
        if len(fold_data['train']) > 0:
            sample = fold_data['train'][0]
            print(f"\n  训练样本示例:")
            print(f"    Person: {sample.person_id}")
            print(f"    Action: {sample.action_id}")
            print(f"    Camera 1: {sample.cam1_id}")
            print(f"    Camera 2: {sample.cam2_id}")
    
    print("\n✓ 测试通过！")


def test_different_strategies():
    """测试不同的划分策略"""
    print("\n" + "="*80)
    print("测试 2: 不同划分策略对比")
    print("="*80)
    
    strategies = {
        "by_person": "按人物划分",
        "by_action": "按动作划分",
    }
    
    for strategy, desc in strategies.items():
        print(f"\n{'='*60}")
        print(f"策略: {desc} ({strategy})")
        print(f"{'='*60}")
        
        cv = CameraPairCrossValidation(
            data_root="/workspace/data/skiing_unity_dataset",
            num_persons=2,
            num_actions=12,
            num_cameras=108,
            split_strategy=strategy,
            n_splits=5,
            index_save_path=f"/tmp/test_{strategy}.json"
        )
        
        folds = cv(force_recreate=True)
        
        print(f"\n生成 {len(folds)} 折:")
        total_train = 0
        total_val = 0
        
        for fold_idx in range(min(3, len(folds))):  # 只显示前3折
            fold_data = folds[fold_idx]
            train_size = len(fold_data['train'])
            val_size = len(fold_data['val'])
            total_train += train_size
            total_val += val_size
            
            print(f"  Fold {fold_idx}: train={train_size:>6}, val={val_size:>6}", end="")
            
            if 'val_person' in fold_data:
                print(f" (person={fold_data['val_person']})", end="")
            elif 'val_actions' in fold_data:
                print(f" (actions={len(fold_data['val_actions'])})", end="")
            
            print()
        
        if len(folds) > 3:
            print(f"  ... (还有 {len(folds) - 3} 折)")
        
        avg_train = total_train / min(3, len(folds))
        avg_val = total_val / min(3, len(folds))
        print(f"\n平均每折: train={avg_train:.0f}, val={avg_val:.0f}")


def test_sample_access():
    """测试样本访问和数据结构"""
    print("\n" + "="*80)
    print("测试 3: 样本访问")
    print("="*80)
    
    cv = CameraPairCrossValidation(
        data_root="/workspace/data/skiing_unity_dataset",
        num_persons=2,
        num_actions=2,
        num_cameras=5,
        split_strategy="by_person",
        index_save_path="/tmp/test_sample_access.json"
    )
    
    folds = cv(force_recreate=True)
    
    # 获取第一折的训练样本
    train_samples = folds[0]['train']
    
    print(f"\n训练集前10个样本:")
    print(f"{'#':<4} {'Person':<8} {'Action':<12} {'Cam1':<10} {'Cam2':<10}")
    print("-" * 60)
    
    for i, sample in enumerate(train_samples[:10]):
        print(f"{i+1:<4} {sample.person_id:<8} {sample.action_id:<12} "
              f"{sample.cam1_id:<10} {sample.cam2_id:<10}")
    
    # 测试样本序列化和反序列化
    sample = train_samples[0]
    sample_dict = sample.to_dict()
    
    print(f"\n样本字典格式:")
    for key, value in sample_dict.items():
        print(f"  {key}: {value}")
    
    # 从字典恢复
    from dual2pose.map_config import UnityDataConfig
    recovered_sample = UnityDataConfig.from_dict(sample_dict)
    assert recovered_sample.person_id == sample.person_id
    assert recovered_sample.action_id == sample.action_id
    
    print("\n✓ 样本序列化/反序列化测试通过！")


def test_load_existing():
    """测试加载已存在的索引文件"""
    print("\n" + "="*80)
    print("测试 4: 加载已存在的索引")
    print("="*80)
    
    # 第一次创建
    cv1 = CameraPairCrossValidation(
        data_root="/workspace/data/skiing_unity_dataset",
        num_persons=2,
        num_actions=2,
        num_cameras=5,
        split_strategy="by_person",
        index_save_path="/tmp/test_load_existing.json"
    )
    
    print("\n第一次创建...")
    folds1 = cv1(force_recreate=True)
    print(f"✓ 创建了 {len(folds1)} 折")
    
    # 第二次加载
    cv2 = CameraPairCrossValidation(
        data_root="/workspace/data/skiing_unity_dataset",
        num_persons=2,
        num_actions=2,
        num_cameras=5,
        split_strategy="by_person",
        index_save_path="/tmp/test_load_existing.json"
    )
    
    print("\n第二次加载...")
    folds2 = cv2(force_recreate=False)
    print(f"✓ 加载了 {len(folds2)} 折")
    
    # 验证两次结果一致
    assert len(folds1) == len(folds2), "折数不一致"
    
    for fold_idx in range(len(folds1)):
        train1 = folds1[fold_idx]['train']
        train2 = folds2[fold_idx]['train']
        assert len(train1) == len(train2), f"Fold {fold_idx} 训练集大小不一致"
    
    print("\n✓ 加载测试通过！两次结果一致。")


def main():
    """运行所有测试"""
    print("\n" + "🎿"*40)
    print("摄像头两两组合交叉验证 - 功能测试")
    print("🎿"*40)
    
    try:
        test_basic_functionality()
        test_different_strategies()
        test_sample_access()
        test_load_existing()
        
        print("\n" + "="*80)
        print("✅ 所有测试通过！")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
