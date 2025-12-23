"""
验证条件是否生效：同seed生成label0和label1，对比差异
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from torchvision.utils import save_image

from model import ConditionalDDPM, SimpleUNet


def save_tensor_image(x, path, to_gray=True):
    """统一保存函数"""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.clamp(-1, 1)
    if to_gray and x.shape[1] == 3:
        x = x.mean(dim=1, keepdim=True)
    save_image(x, path, normalize=True, value_range=(-1, 1))


def main():
    parser = argparse.ArgumentParser(description='验证条件是否生效')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='./condition_verify',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--image_size', type=int, default=256,
                        help='图像大小')
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='采样步数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 错误: 找不到检查点文件: {checkpoint_path}")
        return
    
    print(f"加载模型: {checkpoint_path}")
    network = SimpleUNet(image_channels=3, time_emb_dim=32, cond_dim=2)
    model = ConditionalDDPM(network=network, n_steps=args.n_steps, device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用相同seed生成label0和label1
    print("\n使用相同seed生成label0和label1...")
    
    for label in [0, 1]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        condition = torch.tensor([label], dtype=torch.long).to(device)
        label_name = 'benign' if label == 0 else 'malignant'
        
        with torch.no_grad():
            sample = model.sample(
                condition=condition,
                shape=(1, 3, args.image_size, args.image_size),
                n_steps=args.n_steps,
                eta=0.0
            )
        
        save_path = output_dir / f'label{label}_{label_name}_seed{args.seed}.png'
        save_tensor_image(sample[0], save_path)
        print(f"✓ 保存 {label_name} (label={label}): {save_path}")
        
        # 打印统计信息
        print(f"  统计: min={sample.min().item():.4f}, max={sample.max().item():.4f}, "
              f"mean={sample.mean().item():.4f}, std={sample.std().item():.4f}")
    
    # 计算差异
    print("\n加载两张图像计算差异...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    cond0 = torch.tensor([0], dtype=torch.long).to(device)
    cond1 = torch.tensor([1], dtype=torch.long).to(device)
    
    with torch.no_grad():
        sample0 = model.sample(cond0, (1, 3, args.image_size, args.image_size), n_steps=args.n_steps, eta=0.0)
        sample1 = model.sample(cond1, (1, 3, args.image_size, args.image_size), n_steps=args.n_steps, eta=0.0)
    
    diff = torch.abs(sample0 - sample1)
    diff_mean = diff.mean().item()
    diff_max = diff.max().item()
    
    print(f"\n差异统计:")
    print(f"  平均绝对差异: {diff_mean:.4f}")
    print(f"  最大绝对差异: {diff_max:.4f}")
    
    if diff_mean < 0.01:
        print("⚠️  警告: 差异极小，条件可能没有生效！")
    elif diff_mean < 0.1:
        print("⚠️  警告: 差异较小，条件可能部分生效")
    else:
        print("✓ 差异明显，条件应该生效了")
    
    # 保存差异图
    diff_path = output_dir / f'diff_label0_vs_label1_seed{args.seed}.png'
    save_tensor_image(diff[0], diff_path)
    print(f"✓ 保存差异图: {diff_path}")


if __name__ == '__main__':
    main()


