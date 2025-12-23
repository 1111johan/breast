"""
条件DDPM模型 - UNet backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """UNet基础块"""
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.cond_mlp = nn.Linear(cond_dim, out_ch)
        
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x, t, cond):
        # 第一个卷积
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # 时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]  # 扩展维度
        h = h + time_emb
        
        # 条件嵌入
        cond_emb = self.relu(self.cond_mlp(cond))
        cond_emb = cond_emb[(..., ) + (None, ) * 2]  # 扩展维度
        h = h + cond_emb
        
        # 第二个卷积
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        # 下采样或上采样
        return self.transform(h)


class SimpleUNet(nn.Module):
    """简化的UNet，用于条件DDPM"""
    def __init__(self, image_channels=3, time_emb_dim=32, cond_dim=2):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # 条件嵌入（良性/恶性二分类）
        self.cond_embedding = nn.Embedding(2, cond_dim)  # 0=良性, 1=恶性
        
        # 下采样
        self.down1 = Block(image_channels, 64, time_emb_dim, cond_dim)
        self.down2 = Block(64, 128, time_emb_dim, cond_dim)
        self.down3 = Block(128, 256, time_emb_dim, cond_dim)
        
        # 瓶颈层
        self.bot1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bot2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bot3 = nn.Conv2d(512, 256, 3, padding=1)
        
        # 上采样
        self.up1 = Block(256, 128, time_emb_dim, cond_dim, up=True)
        self.up2 = Block(128, 64, time_emb_dim, cond_dim, up=True)
        self.up3 = Block(64, 64, time_emb_dim, cond_dim, up=True)
        self.out = nn.Conv2d(64, image_channels, 1)
    
    def forward(self, x, timestep, condition):
        """
        Args:
            x: 噪声图像 [B, C, H, W]
            timestep: 时间步 [B]
            condition: 条件标签 [B] (0=良性, 1=恶性)
        """
        # 时间嵌入
        t = self.time_mlp(timestep)
        
        # 条件嵌入
        cond_emb = self.cond_embedding(condition)  # [B, cond_dim]
        
        # U-Net
        down1 = self.down1(x, t, cond_emb)
        down2 = self.down2(down1, t, cond_emb)
        down3 = self.down3(down2, t, cond_emb)
        
        bot1 = F.relu(self.bot1(down3))
        bot2 = F.relu(self.bot2(bot1))
        bot3 = F.relu(self.bot3(bot2))
        
        up1 = self.up1(torch.cat((bot3, down3), dim=1), t, cond_emb)
        up2 = self.up2(torch.cat((up1, down2), dim=1), t, cond_emb)
        up3 = self.up3(torch.cat((up2, down1), dim=1), t, cond_emb)
        
        output = self.out(up3)
        return output


class ConditionalDDPM(nn.Module):
    """条件DDPM模型"""
    def __init__(self, network, n_steps=1000, beta_1=1e-4, beta_T=0.02, device="cuda"):
        super().__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device)
        
        # 线性噪声调度
        self.beta = torch.linspace(beta_1, beta_T, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def forward(self, x_0, condition):
        """
        前向扩散过程
        Args:
            x_0: 原始图像 [B, C, H, W]
            condition: 条件标签 [B]
        Returns:
            loss: 训练损失
        """
        n = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (n,)).to(self.device)
        
        # 采样噪声
        eps = torch.randn_like(x_0)
        
        # 添加噪声
        alpha_bar_t = self.alpha_bar[t].reshape(n, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps
        
        # 预测噪声
        eps_theta = self.network(x_t, t, condition)
        
        # 计算损失
        loss = F.mse_loss(eps_theta, eps)
        return loss
    
    def sample(self, condition, shape, n_steps=None, timesteps=None, eta: float = 0.0):
        """
        反向扩散过程 - 生成样本
        Args:
            condition: 条件标签 [B]
            shape: 生成图像形状 (B, C, H, W)
            n_steps: 期望的采样步数（可与训练时不同，通过子采样实现）
            timesteps: 自定义时间步序列（基于训练的总步数索引，如[999, 995, ... ,0]）
            eta: DDIM风格的随机系数（0=确定性，1=接近DDPM，默认0）
        Returns:
            x_0: 生成的图像
        """
        total_steps = self.n_steps
        device = self.device
        batch_size = shape[0]

        # 构造时间步序列：默认用训练时的全长；如果指定 n_steps 则做子采样
        if timesteps is None:
            if n_steps is not None and n_steps < total_steps:
                timesteps = torch.linspace(total_steps - 1, 0, n_steps, dtype=torch.long)
            else:
                timesteps = torch.arange(total_steps - 1, -1, -1, dtype=torch.long)
        timesteps = timesteps.to(device)

        x = torch.randn(shape, device=device)

        for i in range(len(timesteps)):
            t = timesteps[i]
            eps_theta = self.network(x, torch.full((batch_size,), t, device=device, dtype=torch.long), condition)

            alpha_bar_t = self.alpha_bar[t].view(1, 1, 1, 1)
            alpha_t = self.alpha[t].view(1, 1, 1, 1)
            beta_t = self.beta[t].view(1, 1, 1, 1)

            if i == len(timesteps) - 1:
                # 最后一步，直接预测 x0（数值稳定版本）
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
                x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_theta) / sqrt_alpha_bar_t
                x = x0_pred.clamp(-1, 1)  # 确保在训练范围内
                break

            t_prev = timesteps[i + 1]
            alpha_bar_prev = self.alpha_bar[t_prev].view(1, 1, 1, 1)

            # 预测 x0（数值稳定版本）
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_theta) / sqrt_alpha_bar_t
            x0_pred = x0_pred.clamp(-1, 1)  # 确保在训练范围内

            # DDIM/跳步：eta 控制噪声；eta=0 -> 确定性，eta>0 -> 加噪
            if eta > 0:
                # 带噪声的DDPM风格更新
                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8) * 
                    (1 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))
                )
                sigma = torch.clamp(sigma, min=0.0, max=1.0)
                dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0)) * eps_theta
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
            else:
                # 确定性 DDIM 更新
                x = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_theta

        return x.clamp(-1, 1)  # 最终clamp确保在训练范围内

