"""
检查训练结果脚本
"""
import argparse
from pathlib import Path
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def check_checkpoints(checkpoint_dir):
    """检查所有检查点"""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"❌ 检查点目录不存在: {checkpoint_dir}")
        return
    
    checkpoints = sorted(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        print(f"❌ 未找到检查点文件")
        return
    
    print(f"找到 {len(checkpoints)} 个检查点:")
    print("-" * 80)
    
    for ckpt_path in checkpoints:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 'N/A')
            val_loss = checkpoint.get('val_loss', 'N/A')
            train_loss = checkpoint.get('train_loss', 'N/A')
            
            print(f"{ckpt_path.name}")
            print(f"  Epoch: {epoch}")
            if train_loss != 'N/A':
                print(f"  训练损失: {train_loss:.4f}")
            if val_loss != 'N/A':
                print(f"  验证损失: {val_loss:.4f}")
            print()
        except Exception as e:
            print(f"❌ 无法加载 {ckpt_path.name}: {e}")
            print()


def check_samples(sample_dir):
    """检查生成的样本"""
    sample_dir = Path(sample_dir)
    
    if not sample_dir.exists():
        print(f"❌ 样本目录不存在: {sample_dir}")
        return
    
    benign_dir = sample_dir / 'benign'
    malignant_dir = sample_dir / 'malignant'
    
    benign_samples = list(benign_dir.glob('*.png')) if benign_dir.exists() else []
    malignant_samples = list(malignant_dir.glob('*.png')) if malignant_dir.exists() else []
    
    print(f"生成的样本:")
    print("-" * 80)
    print(f"良性样本: {len(benign_samples)} 个")
    print(f"恶性样本: {len(malignant_samples)} 个")
    
    if benign_samples:
        print(f"\n良性样本示例:")
        for sample in sorted(benign_samples)[:5]:
            print(f"  {sample.name}")
    
    if malignant_samples:
        print(f"\n恶性样本示例:")
        for sample in sorted(malignant_samples)[:5]:
            print(f"  {sample.name}")


def visualize_samples(sample_dir, output_path=None, max_samples=16):
    """可视化生成的样本"""
    sample_dir = Path(sample_dir)
    
    benign_dir = sample_dir / 'benign'
    malignant_dir = sample_dir / 'malignant'
    
    # 获取最新的样本
    benign_samples = sorted(benign_dir.glob('*.png'))[-max_samples:] if benign_dir.exists() else []
    malignant_samples = sorted(malignant_dir.glob('*.png'))[-max_samples:] if malignant_dir.exists() else []
    
    if not benign_samples and not malignant_samples:
        print("❌ 未找到样本图像")
        return
    
    fig, axes = plt.subplots(2, min(8, max(len(benign_samples), len(malignant_samples))), 
                            figsize=(20, 5))
    
    if len(axes.shape) == 1:
        axes = axes.reshape(2, -1)
    
    # 显示良性样本
    for i, sample_path in enumerate(benign_samples[:8]):
        img = Image.open(sample_path)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('良性样本', fontsize=12)
    
    # 显示恶性样本
    for i, sample_path in enumerate(malignant_samples[:8]):
        img = Image.open(sample_path)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('恶性样本', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存可视化结果: {output_path}")
    else:
        plt.show()


def check_tensorboard_logs(log_dir):
    """提示如何查看TensorBoard日志"""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return
    
    log_files = list(log_dir.glob('events.out.tfevents.*'))
    
    if log_files:
        print(f"找到 {len(log_files)} 个TensorBoard日志文件")
        print(f"\n查看训练曲线，运行:")
        print(f"  tensorboard --logdir {log_dir}")
        print(f"然后在浏览器打开 http://localhost:6006")
    else:
        print("❌ 未找到TensorBoard日志文件")


def main():
    parser = argparse.ArgumentParser(description='检查训练结果')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='训练输出目录')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化生成的样本')
    parser.add_argument('--save_viz', type=str, default=None,
                        help='保存可视化结果到文件')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    sample_dir = output_dir / 'samples'
    log_dir = output_dir / 'logs'
    
    print("=" * 80)
    print("训练结果检查")
    print("=" * 80)
    print()
    
    # 检查检查点
    print("1. 检查点信息:")
    check_checkpoints(checkpoint_dir)
    print()
    
    # 检查样本
    print("2. 生成的样本:")
    check_samples(sample_dir)
    print()
    
    # 检查TensorBoard日志
    print("3. TensorBoard日志:")
    check_tensorboard_logs(log_dir)
    print()
    
    # 可视化样本
    if args.visualize:
        print("4. 可视化样本:")
        visualize_samples(sample_dir, args.save_viz)
        print()


if __name__ == '__main__':
    main()

