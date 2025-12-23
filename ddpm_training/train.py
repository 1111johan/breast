"""
条件DDPM训练脚本
"""
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from dataset import BUSCoTDataset
from model import ConditionalDDPM, SimpleUNet


def save_tensor_image(x: torch.Tensor, path: Path, assume_range: str = "-1_1", to_gray: bool = True):
    """
    统一的保存函数，防止重复归一化导致过曝：
    - assume_range = "-1_1": 张量范围为[-1, 1]（模型输出）
    - assume_range = "0_1" : 张量范围为[0, 1]
    - to_gray: 将3通道均值成1通道，避免彩色花屏
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if to_gray and x.shape[1] == 3:
        x = x.mean(dim=1, keepdim=True)
    if assume_range == "-1_1":
        x = x.clamp(-1, 1)
        save_image(x, path, normalize=True, value_range=(-1, 1))
    else:
        x = x.clamp(0, 1)
        save_image(x, path, normalize=False)


def resolve_path(path_str, script_dir):
    """
    解析路径，如果是相对路径则基于脚本目录或项目根目录
    """
    path = Path(path_str)
    
    # 如果是绝对路径，直接返回
    if path.is_absolute():
        return str(path)
    
    # 尝试相对于脚本目录
    script_path = script_dir / path
    if script_path.exists():
        return str(script_path)
    
    # 尝试相对于项目根目录（脚本的父目录的父目录）
    project_root = script_dir.parent.parent
    project_path = project_root / path
    if project_path.exists():
        return str(project_path)
    
    # 尝试相对于当前工作目录
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path)
    
    # 如果都不存在，返回原始路径（让后续代码报错）
    return str(path)


def train_epoch(model, dataloader, optimizer, device, epoch, max_grad_norm=1.0):
    """
    训练一个epoch
    添加梯度裁剪防止训练不稳定
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        loss = model(images, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            loss = model(images, labels)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def save_samples(model, save_dir, epoch, device, image_size=256):
    """生成并保存样本"""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # 生成良性样本
        benign_cond = torch.zeros(4, dtype=torch.long).to(device)
        benign_samples = model.sample(benign_cond, (4, 3, image_size, image_size))
        
        # 生成恶性样本
        malignant_cond = torch.ones(4, dtype=torch.long).to(device)
        malignant_samples = model.sample(malignant_cond, (4, 3, image_size, image_size))
        
        # 保存图像
        from torchvision.utils import save_image
        
        benign_dir = save_dir / 'benign'
        malignant_dir = save_dir / 'malignant'
        benign_dir.mkdir(exist_ok=True)
        malignant_dir.mkdir(exist_ok=True)
        
        for i in range(4):
            save_tensor_image(benign_samples[i], benign_dir / f'epoch_{epoch}_sample_{i}.png', to_gray=True)
            save_tensor_image(malignant_samples[i], malignant_dir / f'epoch_{epoch}_sample_{i}.png', to_gray=True)


def main():
    # 获取脚本所在目录，用于解析相对路径
    script_dir = Path(__file__).parent
    # 项目根目录是脚本的父目录（因为脚本在 ddpm_training/ 目录下）
    project_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description='训练条件DDPM')
    parser.add_argument('--data_root', type=str, 
                        default=str(project_root / 'BUS-CoT' / 'BUS-CoT'),
                        help='数据集根目录（默认: 基于脚本位置自动查找）')
    parser.add_argument('--lesion_dataset', type=str,
                        default=str(project_root / 'BUS-CoT' / 'BUS-CoT' / 'DatasetFiles' / 'lesion_dataset.json'),
                        help='lesion_dataset.json路径')
    parser.add_argument('--bus_expert', type=str,
                        default=str(project_root / 'BUS-CoT' / 'BUS-CoT' / 'DatasetFiles' / 'BUS-Expert_dataset.json'),
                        help='BUS-Expert_dataset.json路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--image_size', type=int, default=256,
                        help='图像大小')
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='扩散步数')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='保存检查点的间隔（epoch）')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='生成样本的间隔（epoch）')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 如果用户提供了相对路径，尝试解析
    if not Path(args.data_root).is_absolute():
        args.data_root = resolve_path(args.data_root, script_dir)
    if not Path(args.lesion_dataset).is_absolute():
        args.lesion_dataset = resolve_path(args.lesion_dataset, script_dir)
    if not Path(args.bus_expert).is_absolute():
        args.bus_expert = resolve_path(args.bus_expert, script_dir)
    
    # 验证路径是否存在
    data_root_path = Path(args.data_root)
    lesion_dataset_path = Path(args.lesion_dataset)
    
    if not data_root_path.exists():
        print(f"❌ 错误: 找不到数据根目录: {data_root_path}")
        print(f"   请检查路径是否正确，或使用 --data_root 指定正确的路径")
        print(f"   示例: --data_root D:\\cursor_file\\1\\BUS-CoT\\BUS-CoT")
        return
    
    if not lesion_dataset_path.exists():
        print(f"❌ 错误: 找不到lesion数据集文件: {lesion_dataset_path}")
        print(f"   请检查路径是否正确，或使用 --lesion_dataset 指定正确的路径")
        print(f"   示例: --lesion_dataset D:\\cursor_file\\1\\BUS-CoT\\BUS-CoT\\DatasetFiles\\lesion_dataset.json")
        return
    
    print(f"✓ 数据根目录: {data_root_path}")
    print(f"✓ Lesion数据集: {lesion_dataset_path}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    sample_dir = output_dir / 'samples'
    sample_dir.mkdir(exist_ok=True)
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # 数据集
    print("加载数据集...")
    train_dataset = BUSCoTDataset(
        lesion_dataset_path=args.lesion_dataset,
        bus_expert_path=args.bus_expert,
        data_root=args.data_root,
        split='train',
        train_ratio=args.train_ratio,
        image_size=args.image_size,
        seed=args.seed
    )
    
    val_dataset = BUSCoTDataset(
        lesion_dataset_path=args.lesion_dataset,
        bus_expert_path=args.bus_expert,
        data_root=args.data_root,
        split='val',
        train_ratio=args.train_ratio,
        image_size=args.image_size,
        seed=args.seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 模型
    print("初始化模型...")
    network = SimpleUNet(
        image_channels=3,
        time_emb_dim=32,
        cond_dim=2
    )
    model = ConditionalDDPM(
        network=network,
        n_steps=args.n_steps,
        device=device
    )
    
    # 优化器（添加weight decay）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # EMA (Exponential Moving Average) 用于稳定推理
    from copy import deepcopy
    ema_model = deepcopy(model)
    ema_decay = 0.9999
    
    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # 训练循环
    print("开始训练...")
    print("=" * 80)
    print(f"初始学习率: {args.lr}")
    print(f"使用梯度裁剪: max_norm=1.0")
    print(f"使用EMA: decay={ema_decay}")
    print("=" * 80)
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, max_grad_norm=1.0)
        
        # 更新EMA
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
        
        # 验证
        val_loss = validate(model, val_loader, device)
        
        # 学习率调度
        scheduler.step()
        
        # 记录
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存检查点（包含EMA）
        if epoch % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),  # 保存EMA模型
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  保存检查点: {checkpoint_path}")
        
        # 保存最佳模型（使用EMA模型）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),  # 保存EMA模型
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f"  保存最佳模型: {best_path} (验证损失: {val_loss:.4f})")
        
        # 生成样本（使用EMA模型以获得更稳定的结果）
        if epoch % args.sample_interval == 0:
            print("  生成样本（使用EMA模型）...")
            save_samples(ema_model, sample_dir, epoch, device, args.image_size)
    
    writer.close()
    print("训练完成！")


if __name__ == '__main__':
    main()

