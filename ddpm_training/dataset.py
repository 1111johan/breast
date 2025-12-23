"""
BUS-CoT数据集加载器，支持按患者ID划分数据集
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def extract_patient_id_from_image_path(image_path: str) -> str:
    """
    从图像路径中提取患者ID
    图像路径格式: BUS-Lesion/trainval/000000@0.png
    返回: 000000 (图像ID作为患者ID的代理，因为同一图像ID可能对应同一患者)
    """
    match = re.search(r'(\d{6})@', image_path)
    if match:
        return match.group(1)
    return "unknown"


def load_patient_mapping(bus_expert_path: str) -> Dict[str, str]:
    """
    从BUS-Expert_dataset.json加载图像ID到患者ID的映射
    """
    patient_mapping = {}
    with open(bus_expert_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for image_id, info in data.items():
            if 'patient_id' in info:
                # 提取图像ID（例如从"000000"）
                patient_mapping[image_id] = info['patient_id']
    return patient_mapping


class BUSCoTDataset(Dataset):
    """
    BUS-CoT数据集，支持按患者ID划分
    """
    
    def __init__(
        self,
        lesion_dataset_path: str,
        bus_expert_path: str,
        data_root: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        image_size: int = 256,
        use_patient_split: bool = True,
        seed: int = 42
    ):
        """
        Args:
            lesion_dataset_path: lesion_dataset.json路径
            bus_expert_path: BUS-Expert_dataset.json路径
            data_root: 数据集根目录
            split: 'train' 或 'val'
            train_ratio: 训练集比例（按患者划分）
            image_size: 图像大小
            use_patient_split: 是否按患者ID划分
            seed: 随机种子
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        
        # 加载数据集
        with open(lesion_dataset_path, 'r', encoding='utf-8') as f:
            lesion_data = json.load(f)
        
        # 加载患者映射
        patient_mapping = {}
        if os.path.exists(bus_expert_path):
            patient_mapping = load_patient_mapping(bus_expert_path)
        
        # 提取所有样本，并检查文件是否存在
        samples = []
        data_root_path = Path(data_root)
        missing_files = 0
        
        for key, item in lesion_data.items():
            image_path = item.get('image_path', '')
            pathology = item.get('pathology_histology', {}).get('pathology', '')
            
            if not image_path or not pathology:
                continue
            
            # 检查文件是否存在
            full_image_path = data_root_path / image_path
            if not full_image_path.exists():
                missing_files += 1
                continue
            
            # 转换为二分类标签: 0=良性, 1=恶性
            label = 1 if pathology.lower() == 'malignant' else 0
            
            # 提取患者ID
            image_id = extract_patient_id_from_image_path(image_path)
            patient_id = patient_mapping.get(image_id, image_id)  # 如果找不到映射，使用图像ID
            
            samples.append({
                'image_path': image_path,
                'label': label,
                'patient_id': patient_id,
                'image_id': image_id
            })
        
        if missing_files > 0:
            print(f"警告: 跳过了 {missing_files} 个不存在的图像文件")
        
        # 按患者ID划分数据集
        if use_patient_split:
            train_samples, val_samples = self._split_by_patient(samples, train_ratio, seed)
            self.samples = train_samples if split == 'train' else val_samples
        else:
            # 简单随机划分
            import random
            random.seed(seed)
            random.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            self.samples = samples[:split_idx] if split == 'train' else samples[split_idx:]
        
        # 数据增强
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        print(f"加载{split}集: {len(self.samples)}个样本")
        print(f"良性样本: {sum(1 for s in self.samples if s['label'] == 0)}")
        print(f"恶性样本: {sum(1 for s in self.samples if s['label'] == 1)}")
    
    def _split_by_patient(self, samples: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
        """
        按患者ID划分数据集，确保训练集和验证集患者不重叠
        """
        import random
        random.seed(seed)
        
        # 按患者ID分组
        patient_to_samples = {}
        for sample in samples:
            patient_id = sample['patient_id']
            if patient_id not in patient_to_samples:
                patient_to_samples[patient_id] = []
            patient_to_samples[patient_id].append(sample)
        
        # 获取所有患者ID并随机打乱
        patient_ids = list(patient_to_samples.keys())
        random.shuffle(patient_ids)
        
        # 按比例划分患者
        split_idx = int(len(patient_ids) * train_ratio)
        train_patient_ids = set(patient_ids[:split_idx])
        val_patient_ids = set(patient_ids[split_idx:])
        
        # 根据患者ID分配样本
        train_samples = []
        val_samples = []
        for sample in samples:
            if sample['patient_id'] in train_patient_ids:
                train_samples.append(sample)
            elif sample['patient_id'] in val_patient_ids:
                val_samples.append(sample)
        
        print(f"患者级别划分: 训练集{len(train_patient_ids)}个患者, 验证集{len(val_patient_ids)}个患者")
        
        return train_samples, val_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            sample = self.samples[idx]
            image_path = self.data_root / sample['image_path']
            
            # 检查文件是否存在
            if not image_path.exists():
                retry_count += 1
                if retry_count < max_retries:
                    # 尝试下一个样本
                    idx = (idx + 1) % len(self.samples)
                    continue
                else:
                    raise FileNotFoundError(f"无法找到图像文件: {image_path}")
            
            # 加载图像
            try:
                image = Image.open(image_path).convert('RGB')
                # BUS-Lesion图像已经是cropped ROI，但可能需要增强对比度
                # 先应用transform（resize + normalize）
                image = self.transform(image)
                # 确保数值在合理范围内（虽然Normalize应该已经处理了）
                image = torch.clamp(image, -1, 1)
                
                # 标签
                label = torch.tensor(sample['label'], dtype=torch.long)
                
                return {
                    'image': image,
                    'label': label,
                    'path': str(image_path)
                }
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    # 尝试下一个样本
                    idx = (idx + 1) % len(self.samples)
                    continue
                else:
                    raise RuntimeError(f"无法加载图像 {image_path}: {e}")
        
        raise RuntimeError(f"尝试加载图像失败，已重试 {max_retries} 次")

