import os
import math
import json
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm


# ----------------------------
# Schedule / Model (reuse from Stage1)
# ----------------------------
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + self.cond_proj(c_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 32, emb_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.input_proj = nn.Conv2d(in_channels, base, 3, padding=1)
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.null_emb = nn.Parameter(torch.zeros(emb_dim))

        self.down1_b1 = ResBlock(base, base, emb_dim)
        self.down1_b2 = ResBlock(base, base, emb_dim)
        self.down1_pool = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)

        self.down2_b1 = ResBlock(base * 2, base * 2, emb_dim)
        self.down2_b2 = ResBlock(base * 2, base * 2, emb_dim)
        self.down2_pool = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)

        self.down3_b1 = ResBlock(base * 4, base * 4, emb_dim)
        self.down3_b2 = ResBlock(base * 4, base * 4, emb_dim)
        self.down3_pool = nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1)

        self.mid1 = ResBlock(base * 8, base * 8, emb_dim)
        self.mid2 = ResBlock(base * 8, base * 8, emb_dim)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1)
        self.up3_b1 = ResBlock(base * 8, base * 4, emb_dim)
        self.up3_b2 = ResBlock(base * 4, base * 4, emb_dim)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up2_b1 = ResBlock(base * 4, base * 2, emb_dim)
        self.up2_b2 = ResBlock(base * 2, base * 2, emb_dim)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up1_b1 = ResBlock(base * 2, base, emb_dim)
        self.up1_b2 = ResBlock(base, base, emb_dim)

        self.output_proj = nn.Conv2d(base, in_channels, 3, padding=1)

    def get_c_emb(self, y):
        if y is None:
            return self.null_emb.unsqueeze(0)
        return self.class_emb(y)

    def forward(self, x, t, y):
        t_emb = self.time_emb(t)
        c_emb = self.get_c_emb(y)
        if c_emb.shape[0] == 1 and x.shape[0] > 1:
            c_emb = c_emb.expand(x.shape[0], -1)

        x0 = self.input_proj(x)
        d1 = self.down1_b2(self.down1_b1(x0, t_emb, c_emb), t_emb, c_emb)
        x1 = self.down1_pool(d1)
        d2 = self.down2_b2(self.down2_b1(x1, t_emb, c_emb), t_emb, c_emb)
        x2 = self.down2_pool(d2)
        d3 = self.down3_b2(self.down3_b1(x2, t_emb, c_emb), t_emb, c_emb)
        x3 = self.down3_pool(d3)

        m = self.mid2(self.mid1(x3, t_emb, c_emb), t_emb, c_emb)

        u3 = self.up3(m)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.up3_b2(self.up3_b1(u3, t_emb, c_emb), t_emb, c_emb)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2_b2(self.up2_b1(u2, t_emb, c_emb), t_emb, c_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up1_b2(self.up1_b1(u1, t_emb, c_emb), t_emb, c_emb)

        return self.output_proj(u1)


@torch.no_grad()
def ddim_sample(model, betas, shape, device, y, guidance=7.5, steps=50):
    T = betas.shape[0]
    t_seq = torch.linspace(T - 1, 0, steps, device=device).long()
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    sqrt_acp = torch.sqrt(acp)
    sqrt_om = torch.sqrt(1.0 - acp)

    x = torch.randn(shape, device=device)
    for i in range(len(t_seq)):
        t = t_seq[i].repeat(shape[0])
        eps_cond = model(x, t, y)
        if guidance != 1.0:
            eps_uncond = model(x, t, None)
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        x0 = (x - sqrt_om[t].view(-1, 1, 1, 1) * eps) / sqrt_acp[t].view(-1, 1, 1, 1)
        x0 = x0.clamp(-1, 1)

        if i == len(t_seq) - 1:
            x = x0
        else:
            t_next = t_seq[i + 1].repeat(shape[0])
            x = sqrt_acp[t_next].view(-1, 1, 1, 1) * x0 + sqrt_om[t_next].view(-1, 1, 1, 1) * eps
    return x


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    in_channels = int(cfg.get("in_channels", 1))
    timesteps = int(cfg.get("timesteps", 1000))
    base = int(cfg.get("base", 32))
    emb_dim = int(cfg.get("emb_dim", 128))
    img_size = int(cfg.get("img_size", 256))

    model = ConditionalUNet(in_channels=in_channels, base=base, emb_dim=emb_dim).to(device)
    sd = ckpt.get("ema", None) or ckpt["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    betas = cosine_beta_schedule(timesteps).to(device)
    return model, betas, in_channels, img_size


def generate_for_class(
    model,
    betas,
    in_channels: int,
    img_size: int,
    device: torch.device,
    cond_id: int,
    total: int,
    batch_size: int,
    steps: int,
    guidance: float,
    out_dir: Path,
    start_seed: int,
    manifest: List[dict],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = 0
    pbar = tqdm(total=total, desc=f"gen-{out_dir.name}")
    seed = start_seed
    while produced < total:
        bs = min(batch_size, total - produced)
        torch.manual_seed(seed)
        y = torch.full((bs,), cond_id, device=device, dtype=torch.long)
        x = ddim_sample(
            model,
            betas,
            (bs, in_channels, img_size, img_size),
            device,
            y,
            guidance=guidance,
            steps=steps,
        )
        for i in range(bs):
            idx = produced + i
            fname = f"{out_dir.name}_seed{seed}_idx{idx:06d}.png"
            fpath = out_dir / fname
            save_image(x[i : i + 1], fpath, normalize=True, value_range=(-1, 1))
            manifest.append({"path": str(fpath), "cond": "benign" if cond_id == 0 else "malignant", "seed": seed})
        produced += bs
        seed += 1
        pbar.update(bs)
    pbar.close()


def parse_sizes(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Stage1 checkpoint (.pt)")
    parser.add_argument("--out_dir", type=str, default="stage2_outputs", help="Root output dir")
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sizes", type=str, default="1000,2000,4000,8000,16000,32000,64000", help="Comma sep counts per class")
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sizes = parse_sizes(args.sizes)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model, betas, in_channels, img_size = load_model(args.ckpt, device)

    manifest = []
    for n in sizes:
        for cond_id, cond_name in [(0, "benign"), (1, "malignant")]:
            subdir = out_root / f"{cond_name}_{n}"
            generate_for_class(
                model=model,
                betas=betas,
                in_channels=in_channels,
                img_size=img_size,
                device=device,
                cond_id=cond_id,
                total=n,
                batch_size=args.batch_size,
                steps=args.steps,
                guidance=args.guidance,
                out_dir=subdir,
                start_seed=args.start_seed,
                manifest=manifest,
            )
    # write manifest
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"guidance": args.guidance, "sizes": sizes, "items": manifest}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
