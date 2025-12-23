"""
条件DDPM推理脚本 - 从训练好的模型生成图像
"""
import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from model import ConditionalDDPM, SimpleUNet


def save_tensor_image(
    x: torch.Tensor,
    path: Path,
    assume_range: str = "-1_1",
    nrow: int | None = None,
    to_gray: bool = True,
):
    """
    统一的保存函数，避免重复归一化导致过曝：
    - assume_range = "-1_1": 张量范围为[-1,1]（模型输出）
    - assume_range = "0_1" : 张量范围为[0,1]
    - to_gray: 将3通道均值转成1通道灰度，避免彩色花屏
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if to_gray and x.shape[1] == 3:
        x = x.mean(dim=1, keepdim=True)
    if assume_range == "-1_1":
        x = x.clamp(-1, 1)
        save_image(x, path, normalize=True, value_range=(-1, 1), nrow=nrow)
    else:
        x = x.clamp(0, 1)
        save_image(x, path, normalize=False, nrow=nrow)


def load_model(checkpoint_path, device, n_steps=1000):
    """
    加载训练好的模型
    """
    print(f"加载模型: {checkpoint_path}")
    
    # 初始化模型
    network = SimpleUNet(
        image_channels=3,
        time_emb_dim=32,
        cond_dim=2
    )
    model = ConditionalDDPM(
        network=network,
        n_steps=n_steps,
        device=device
    )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 优先使用EMA模型（如果存在），否则使用普通模型
    if 'ema_model_state_dict' in checkpoint:
        print("使用EMA模型（更稳定）...")
        model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        print("使用标准模型...")
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    print(f"✓ 模型加载成功")
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']}")
    if 'best_epoch' in checkpoint:
        print(f"  最佳epoch: {checkpoint['best_epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  验证损失: {checkpoint['val_loss']:.4f}")
    
    return model


def generate_samples(
    model,
    condition,
    num_samples=4,
    image_size=256,
    n_steps=None,
    eta=0.0,
    seed=None
):
    """
    生成样本
    
    Args:
        model: 训练好的DDPM模型
        condition: 条件标签 (0=良性, 1=恶性)
        num_samples: 生成样本数量
        image_size: 图像大小
        n_steps: 采样步数（None则使用模型默认值）
        seed: 随机种子
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    device = next(model.parameters()).device
    
    # 创建条件标签
    if isinstance(condition, int):
        cond_tensor = torch.full((num_samples,), condition, dtype=torch.long).to(device)
    else:
        cond_tensor = torch.tensor([condition] * num_samples, dtype=torch.long).to(device)
    
    print(f"生成 {num_samples} 个样本 (条件: {'恶性' if condition == 1 else '良性'})...")
    
    # 生成样本
    with torch.no_grad():
        samples = model.sample(
            condition=cond_tensor,
            shape=(num_samples, 3, image_size, image_size),
            n_steps=n_steps,
            eta=eta
        )
    
    return samples


def save_samples_grid(samples, save_path, nrow=2):
    """
    保存样本网格
    """
    save_tensor_image(samples, save_path, assume_range="-1_1", nrow=nrow, to_gray=True)


def main():
    parser = argparse.ArgumentParser(description='条件DDPM推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='./inference_outputs',
                        help='输出目录')
    parser.add_argument('--condition', type=int, default=0,
                        choices=[0, 1],
                        help='生成条件: 0=良性, 1=恶性')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='生成样本数量')
    parser.add_argument('--image_size', type=int, default=256,
                        help='图像大小')
    parser.add_argument('--n_steps', type=int, default=None,
                        help='采样步数（子采样原始1000步，如250/400；默认None即全程）')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM随机系数：0=确定性，1=接近DDPM，加噪更随机')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--save_individual', action='store_true',
                        help='是否单独保存每个样本')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 错误: 找不到检查点文件: {checkpoint_path}")
        return
    
    model = load_model(checkpoint_path, device, n_steps=args.n_steps or 1000)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    condition_name = 'malignant' if args.condition == 1 else 'benign'
    condition_dir = output_dir / condition_name
    condition_dir.mkdir(exist_ok=True)
    
    # 生成样本
    samples = generate_samples(
        model=model,
        condition=args.condition,
        num_samples=args.num_samples,
        image_size=args.image_size,
        n_steps=args.n_steps,
        eta=args.eta,
        seed=args.seed
    )
    
    # 保存网格
    grid_path = condition_dir / f'samples_grid_{condition_name}.png'
    save_samples_grid(samples, grid_path, nrow=min(4, args.num_samples))
    print(f"✓ 保存网格: {grid_path}")
    
    # 单独保存每个样本
    if args.save_individual:
        for i, sample in enumerate(samples):
            sample_path = condition_dir / f'sample_{i:03d}.png'
            save_tensor_image(sample, sample_path, assume_range="-1_1", to_gray=True)
        print(f"✓ 保存 {len(samples)} 个单独样本到: {condition_dir}")
    
    print("推理完成！")


if __name__ == '__main__':
    main()

