"""
小样本过拟合测试 - 验证模型能否学习到数据分布
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from dataset import BUSCoTDataset
from model import ConditionalDDPM, SimpleUNet
from torchvision.utils import save_image


def save_tensor_image(x, path, to_gray=True, assume_range="-1_1"):
    """
    统一保存函数
    assume_range: "-1_1" 表示输入范围是[-1,1]（模型输出或dataloader输出）
                   "0_1" 表示输入范围是[0,1]
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    
    # 确保real图像来自dataloader，范围是[-1,1]
    if assume_range == "-1_1":
        x = x.clamp(-1, 1)
        if to_gray and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        save_image(x, path, normalize=True, value_range=(-1, 1))
    else:
        x = x.clamp(0, 1)
        if to_gray and x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        save_image(x, path, normalize=False)


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
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def generate_and_save_samples(model, dataset, save_dir, epoch, device, num_samples=8, fixed_seed=42):
    """
    生成样本并与真实图像对比
    使用固定seed确保可复现性
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取真实样本用于对比（直接从dataloader获取，确保是干净的x0）
    real_samples = []
    real_labels = []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        # 确保real图像是干净的x0，来自dataloader，范围是[-1,1]
        real_img = sample['image'].clone()  # 克隆避免修改原始数据
        real_samples.append(real_img)
        real_labels.append(sample['label'].item())
    
    # 生成对应条件的样本（使用固定seed确保可复现）
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fixed_seed)
    
    with torch.no_grad():
        for i, (real_img, label) in enumerate(zip(real_samples, real_labels)):
            # 真实图像（明确指定范围是[-1,1]，来自dataloader）
            real_path = save_dir / f'epoch_{epoch}_real_{i}_label{label}.png'
            save_tensor_image(real_img, real_path, assume_range="-1_1")
            
            # 生成图像（模型输出范围也是[-1,1]）
            cond = torch.tensor([label], dtype=torch.long).to(device)
            generated = model.sample(cond, (1, 3, 256, 256), n_steps=1000, eta=0.0)
            gen_path = save_dir / f'epoch_{epoch}_gen_{i}_label{label}.png'
            save_tensor_image(generated[0], gen_path, assume_range="-1_1")


def main():
    parser = argparse.ArgumentParser(description='小样本过拟合测试')
    parser.add_argument('--data_root', type=str, 
                        default='BUS-CoT/BUS-CoT',
                        help='数据集根目录')
    parser.add_argument('--lesion_dataset', type=str,
                        default='BUS-CoT/BUS-CoT/DatasetFiles/lesion_dataset.json',
                        help='lesion_dataset.json路径')
    parser.add_argument('--bus_expert', type=str,
                        default='BUS-CoT/BUS-CoT/DatasetFiles/BUS-Expert_dataset.json',
                        help='BUS-Expert_dataset.json路径')
    parser.add_argument('--output_dir', type=str, default='./test_overfit_outputs',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='使用的小样本数量（每类）')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率（降低到5e-5以提高稳定性）')
    parser.add_argument('--image_size', type=int, default=256,
                        help='图像大小')
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='扩散步数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
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
    
    # 加载完整数据集
    print("加载数据集...")
    full_dataset = BUSCoTDataset(
        lesion_dataset_path=args.lesion_dataset,
        bus_expert_path=args.bus_expert,
        data_root=args.data_root,
        split='train',
        train_ratio=0.8,
        image_size=args.image_size,
        seed=args.seed
    )
    
    print(f"完整数据集大小: {len(full_dataset)}")
    
    # 创建小样本数据集（每类取num_samples个）
    benign_indices = []
    malignant_indices = []
    
    for i in range(len(full_dataset)):
        sample = full_dataset.samples[i]
        if sample['label'] == 0:  # 良性
            benign_indices.append(i)
        else:  # 恶性
            malignant_indices.append(i)
    
    # 每类取前num_samples个
    selected_indices = benign_indices[:args.num_samples] + malignant_indices[:args.num_samples]
    
    small_dataset = Subset(full_dataset, selected_indices)
    
    print(f"小样本数据集大小: {len(small_dataset)}")
    print(f"  良性: {args.num_samples} 个")
    print(f"  恶性: {args.num_samples} 个")
    
    train_loader = DataLoader(
        small_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
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
    
    # 优化器（降低初始学习率，添加weight decay）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    # 学习率调度器（cosine decay with warmup）
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # EMA (Exponential Moving Average) 用于稳定推理
    from copy import deepcopy
    ema_model = deepcopy(model)
    ema_decay = 0.9999
    best_loss = float('inf')
    best_epoch = 0
    
    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # 训练循环
    print("开始小样本过拟合测试...")
    print("=" * 80)
    print(f"初始学习率: {args.lr}")
    print(f"使用梯度裁剪: max_norm=1.0")
    print(f"使用EMA: decay={ema_decay}")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, max_grad_norm=1.0)
        
        # 更新EMA
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  训练损失: {train_loss:.4f}, 学习率: {current_lr:.6f}")
        
        # 跟踪最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
        
        # 每5个epoch生成一次样本用于对比（使用EMA模型）
        if epoch % 5 == 0 or epoch == 1:
            print("  生成对比样本（使用EMA模型）...")
            generate_and_save_samples(
                ema_model, small_dataset, sample_dir, epoch, device, num_samples=8, fixed_seed=42
            )
        
        # 每10个epoch保存检查点（包含EMA和最佳模型信息）
        if epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),  # 保存EMA模型
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
            }, checkpoint_path)
            print(f"  保存检查点: {checkpoint_path}")
            print(f"  最佳epoch: {best_epoch}, 最佳损失: {best_loss:.4f}")
    
    writer.close()
    print("=" * 80)
    print("小样本过拟合测试完成！")
    print(f"结果保存在: {output_dir}")
    print("\n检查要点:")
    print("1. 如果生成的图像能逐渐接近真实图像 → 代码没问题，需要更多训练")
    print("2. 如果生成的图像始终是噪声 → 代码有问题，需要修复")


if __name__ == '__main__':
    main()

