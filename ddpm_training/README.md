# 条件DDPM训练 - 阶段一

在BUS-CoT数据集上训练条件DDPM模型，条件输入为病理类别的粗粒度二分类（良性/恶性）。

## 数据集信息

- **图像数量**: 10,897张
- **患者数量**: 4,838例
- **病理类型**: 99种
- **条件输入**: 良性/恶性二分类（0=良性, 1=恶性）
- **图像分辨率**: 256×256
- **扩散步数**: 1000步
- **数据集划分**: 按患者级别8:2划分，确保训练集和验证集患者不重叠

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
python train.py \
    --data_root /path/to/BUS-CoT/BUS-CoT \
    --lesion_dataset BUS-CoT/BUS-CoT/DatasetFiles/lesion_dataset.json \
    --bus_expert BUS-CoT/BUS-CoT/DatasetFiles/BUS-Expert_dataset.json \
    --output_dir ./outputs \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --image_size 256 \
    --n_steps 1000 \
    --train_ratio 0.8 \
    --seed 42 \
    --device cuda
```

### 参数说明

- `--data_root`: 数据集根目录（包含BUS-Lesion和BUS-Expert文件夹）
- `--lesion_dataset`: lesion_dataset.json路径
- `--bus_expert`: BUS-Expert_dataset.json路径（用于患者ID映射）
- `--output_dir`: 输出目录（检查点、样本、日志）
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--image_size`: 图像大小（256）
- `--n_steps`: 扩散过程步数（1000）
- `--train_ratio`: 训练集比例（0.8）
- `--seed`: 随机种子
- `--device`: 设备（cuda/cpu）
- `--save_interval`: 保存检查点的间隔（默认10个epoch）
- `--sample_interval`: 生成样本的间隔（默认5个epoch）

## 模型架构

- **Backbone**: UNet
- **条件嵌入**: Embedding层（2维，对应良性/恶性）
- **时间嵌入**: Sinusoidal位置编码
- **扩散过程**: 1000步线性噪声调度

## 输出

训练过程会生成：

1. **检查点**: `outputs/checkpoints/checkpoint_epoch_*.pth`
2. **最佳模型**: `outputs/checkpoints/best_model.pth`
3. **生成样本**: `outputs/samples/benign/` 和 `outputs/samples/malignant/`
4. **训练日志**: `outputs/logs/` (TensorBoard)

## 查看训练进度

```bash
tensorboard --logdir outputs/logs
```

