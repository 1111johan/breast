"""
测试数据集加载器
"""
import sys
from pathlib import Path
from dataset import BUSCoTDataset

# 设置路径（根据实际情况修改）
DATA_ROOT = "BUS-CoT/BUS-CoT"
LESION_DATASET = "BUS-CoT/BUS-CoT/DatasetFiles/lesion_dataset.json"
BUS_EXPERT = "BUS-CoT/BUS-CoT/DatasetFiles/BUS-Expert_dataset.json"

def test_dataset():
    """测试数据集加载"""
    print("测试训练集...")
    train_dataset = BUSCoTDataset(
        lesion_dataset_path=LESION_DATASET,
        bus_expert_path=BUS_EXPERT,
        data_root=DATA_ROOT,
        split='train',
        train_ratio=0.8,
        image_size=256,
        seed=42
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"图像形状: {sample['image'].shape}")
        print(f"标签: {sample['label']}")
        print(f"路径: {sample['path']}")
    
    print("\n测试验证集...")
    val_dataset = BUSCoTDataset(
        lesion_dataset_path=LESION_DATASET,
        bus_expert_path=BUS_EXPERT,
        data_root=DATA_ROOT,
        split='val',
        train_ratio=0.8,
        image_size=256,
        seed=42
    )
    
    print(f"验证集大小: {len(val_dataset)}")
    if len(val_dataset) > 0:
        sample = val_dataset[0]
        print(f"图像形状: {sample['image'].shape}")
        print(f"标签: {sample['label']}")
        print(f"路径: {sample['path']}")

if __name__ == '__main__':
    test_dataset()

