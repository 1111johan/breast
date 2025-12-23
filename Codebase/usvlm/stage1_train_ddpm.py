import os
import json
import math
import random
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ----------------------------
# Diffusion schedule (cosine)
# ----------------------------
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    # Nichol & Dhariwal cosine schedule
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


# ----------------------------
# Embeddings
# ----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] int64
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # [B, dim]


# ----------------------------
# ResBlock (time + cond)
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch))

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + self.cond_proj(c_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ----------------------------
# UNet (small, fast)
# ----------------------------
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 32, emb_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.base = base
        self.emb_dim = emb_dim

        self.input_proj = nn.Conv2d(in_channels, base, 3, padding=1)

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.null_emb = nn.Parameter(torch.zeros(emb_dim))  # unconditional embedding

        # Down
        self.down1_b1 = ResBlock(base, base, emb_dim)
        self.down1_b2 = ResBlock(base, base, emb_dim)
        self.down1_pool = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)

        self.down2_b1 = ResBlock(base * 2, base * 2, emb_dim)
        self.down2_b2 = ResBlock(base * 2, base * 2, emb_dim)
        self.down2_pool = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)

        self.down3_b1 = ResBlock(base * 4, base * 4, emb_dim)
        self.down3_b2 = ResBlock(base * 4, base * 4, emb_dim)
        self.down3_pool = nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1)

        # Bottleneck
        self.mid1 = ResBlock(base * 8, base * 8, emb_dim)
        self.mid2 = ResBlock(base * 8, base * 8, emb_dim)

        # Up
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1)
        self.up3_b1 = ResBlock(base * 8, base * 4, emb_dim)  # concat: (base*4 + base*4) = base*8
        self.up3_b2 = ResBlock(base * 4, base * 4, emb_dim)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up2_b1 = ResBlock(base * 4, base * 2, emb_dim)  # concat: base*2 + base*2
        self.up2_b2 = ResBlock(base * 2, base * 2, emb_dim)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up1_b1 = ResBlock(base * 2, base, emb_dim)  # concat: base + base
        self.up1_b2 = ResBlock(base, base, emb_dim)

        self.output_proj = nn.Conv2d(base, in_channels, 3, padding=1)

    def get_c_emb(self, y: Optional[torch.Tensor]) -> torch.Tensor:
        # y: [B] long, or None for unconditional
        if y is None:
            return self.null_emb.unsqueeze(0)
        return self.class_emb(y)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
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


# ----------------------------
# EMA
# ----------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow


# ----------------------------
# Dataset: reads lesion_dataset.json or BUS-Expert_dataset.json
# ----------------------------
def _iter_records(data: Any):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                v["_key"] = k
                yield v
    elif isinstance(data, list):
        for v in data:
            if isinstance(v, dict):
                yield v


class BUSCoTDataset(Dataset):
    def __init__(
        self,
        ann_paths: List[str],
        root_dir: str,
        split: str,
        train_frac: float = 0.8,
        img_size: int = 256,
        in_channels: int = 1,
        include_test: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.in_channels = in_channels

        samples = []
        for ann_path in ann_paths:
            with open(ann_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, dict) and "data" in items:
                items = items["data"]

            for it in _iter_records(items):
                ph = (it.get("pathology_histology") or {}).get("pathology", None)
                if ph is None:
                    continue
                ph = str(ph).strip().lower()
                if ph not in ["benign", "malignant"]:
                    continue
                label = 0 if ph == "benign" else 1

                # split filtering (lesion_dataset.json uses split)
                split_tag = it.get("split")
                if split_tag is not None and str(split_tag).lower() == "test" and not include_test:
                    continue

                # image path candidates
                img_rel = it.get("image_path") or it.get("cropped_image") or it.get("image") or it.get("img")
                if img_rel is None and isinstance(it.get("image_file"), dict):
                    img_rel = it["image_file"].get("cropped_image") or it["image_file"].get("raw_image")
                if img_rel is None:
                    continue

                img_path = img_rel
                if not os.path.isabs(img_path):
                    img_path = os.path.join(root_dir, img_rel)
                if not os.path.exists(img_path):
                    continue

                # patient key
                pid = it.get("patient_id") or it.get("case_id") or it.get("id") or it.get("_key")
                if pid is None:
                    pid = os.path.basename(img_path)
                pid = str(pid)

                samples.append((img_path, label, pid))

        # patient-level deterministic split
        train, val = [], []
        for img_path, label, pid in samples:
            h = stable_hash(pid) % 10000
            is_train = (h / 10000.0) < train_frac
            if is_train:
                train.append((img_path, label))
            else:
                val.append((img_path, label))

        self.samples = train if split == "train" else val

        # transforms: resize -> to tensor -> normalize to [-1,1]
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),  # [0,1], shape [1,H,W] for L
                transforms.Normalize(mean=[0.5], std=[0.5]),  # -> [-1,1]
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        # ultrasound usually grayscale; unify to 'L'
        img = img.convert("L")
        x = self.tf(img)  # [1,H,W]
        if self.in_channels == 3:
            x = x.repeat(3, 1, 1)
        return x, torch.tensor(label, dtype=torch.long)


# ----------------------------
# DDPM training
# ----------------------------
@dataclass
class TrainConfig:
    ann_paths: List[str]
    root_dir: str
    out_dir: str = "outputs_stage1"
    img_size: int = 256
    in_channels: int = 1
    timesteps: int = 1000
    base: int = 32
    emb_dim: int = 128
    batch_size: int = 8
    lr: float = 2e-4
    epochs: int = 50
    num_workers: int = 0  # safer on Windows; increase if you need speed and no pickling issues
    ema_decay: float = 0.999
    cond_drop_prob: float = 0.1  # crucial for CFG
    sample_every_steps: int = 500
    save_every_epochs: int = 5
    seed: int = 42
    train_frac: float = 0.8


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    # x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*noise
    b = x0.shape[0]
    s1 = sqrt_alphas_cumprod[t].view(b, 1, 1, 1)
    s2 = sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1, 1)
    return s1 * x0 + s2 * noise


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    betas: torch.Tensor,
    shape: Tuple[int, int, int, int],
    device: torch.device,
    y: Optional[torch.Tensor],
    guidance: float = 1.0,
    steps: int = 50,
) -> torch.Tensor:
    """
    Fast DDIM sampler (deterministic). Supports CFG by running cond/uncond.
    """
    T = betas.shape[0]
    # choose a subset of timesteps
    t_seq = torch.linspace(T - 1, 0, steps, device=device).long()

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_acp = torch.sqrt(alphas_cumprod)
    sqrt_om = torch.sqrt(1.0 - alphas_cumprod)

    x = torch.randn(shape, device=device)

    for i in range(len(t_seq)):
        t = t_seq[i].repeat(shape[0])

        # predict eps
        eps_cond = model(x, t, y)
        if guidance != 1.0:
            eps_uncond = model(x, t, None)
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        # x0 pred
        x0 = (x - sqrt_om[t].view(-1, 1, 1, 1) * eps) / sqrt_acp[t].view(-1, 1, 1, 1)
        x0 = x0.clamp(-1, 1)

        if i == len(t_seq) - 1:
            x = x0
        else:
            t_next = t_seq[i + 1].repeat(shape[0])
            # DDIM deterministic update
            x = sqrt_acp[t_next].view(-1, 1, 1, 1) * x0 + sqrt_om[t_next].view(-1, 1, 1, 1) * eps

    return x


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_path",
        type=str,
        action="append",
        help="Repeatable. e.g. --ann_path DatasetFiles/lesion_dataset.json --ann_path DatasetFiles/BUS-Expert_dataset.json",
    )
    parser.add_argument("--root_dir", type=str, help="dataset root directory")
    parser.add_argument("--out_dir", type=str, default="outputs_stage1")
    parser.add_argument("--in_channels", type=int, default=1, choices=[1, 3])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--sample_every_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=1.0, help="Set 1.0 to use all training data.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Force device; default cuda.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def find_root_candidates():
        cands = []
        cwd = os.getcwd()
        cands.append(cwd)
        cands.append(os.path.join(cwd, "BUS-CoT", "BUS-CoT"))
        cands.append(script_dir)
        cands.append(os.path.abspath(os.path.join(script_dir, "..", "..", "BUS-CoT", "BUS-CoT")))
        return cands

    def first_existing_root():
        for r in find_root_candidates():
            if os.path.exists(os.path.join(r, "DatasetFiles", "lesion_dataset.json")):
                return r
        return None

    # resolve root_dir
    root_dir = args.root_dir
    if root_dir is None:
        root_dir = first_existing_root() or "."

    # resolve ann_paths
    if args.ann_path:
        ann_paths = []
        for p in args.ann_path:
            if os.path.exists(p):
                ann_paths.append(p)
            else:
                cand = os.path.join(root_dir, p)
                if os.path.exists(cand):
                    ann_paths.append(cand)
        if not ann_paths:
            ann_paths = [os.path.join(root_dir, "DatasetFiles", "lesion_dataset.json"),
                         os.path.join(root_dir, "DatasetFiles", "BUS-Expert_dataset.json")]
    else:
        ann_paths = [
            os.path.join(root_dir, "DatasetFiles", "lesion_dataset.json"),
            os.path.join(root_dir, "DatasetFiles", "BUS-Expert_dataset.json"),
        ]

    # final existence check (best effort)
    ann_paths = [p for p in ann_paths if os.path.exists(p)]
    if not ann_paths:
        raise FileNotFoundError("No valid annotation files found. Please set --ann_path and --root_dir.")

    cfg = TrainConfig(
        ann_paths=ann_paths,
        root_dir=root_dir,
        out_dir=args.out_dir,
        in_channels=args.in_channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        cond_drop_prob=args.cond_drop_prob,
        sample_every_steps=args.sample_every_steps,
        seed=args.seed,
        train_frac=args.train_frac,
    )

    set_seed(cfg.seed)
    # device resolution
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested (--device cuda) but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ensure_dir(cfg.out_dir)
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    samp_dir = os.path.join(cfg.out_dir, "samples")
    ensure_dir(ckpt_dir)
    ensure_dir(samp_dir)

    # diffusion constants
    betas = cosine_beta_schedule(cfg.timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_acp = torch.sqrt(alphas_cumprod)
    sqrt_om = torch.sqrt(1.0 - alphas_cumprod)

    # dataset
    ds_train = BUSCoTDataset(
        cfg.ann_paths,
        cfg.root_dir,
        split="train",
        train_frac=cfg.train_frac,
        img_size=cfg.img_size,
        in_channels=cfg.in_channels,
    )
    ds_val = BUSCoTDataset(
        cfg.ann_paths,
        cfg.root_dir,
        split="val",
        train_frac=cfg.train_frac,
        img_size=cfg.img_size,
        in_channels=cfg.in_channels,
    )
    print(f"[DATA] train={len(ds_train)} val={len(ds_val)} (patient-level split)")

    dl = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    model = ConditionalUNet(
        in_channels=cfg.in_channels, base=cfg.base, emb_dim=cfg.emb_dim, num_classes=2
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ema = EMA(model, decay=cfg.ema_decay)

    global_step = 0
    model.train()

    for epoch in range(1, cfg.epochs + 1):
        pbar = tqdm(dl, desc=f"epoch {epoch}/{cfg.epochs}")
        for x0, y in pbar:
            x0 = x0.to(device)
            y = y.to(device)

            b = x0.shape[0]
            t = torch.randint(0, cfg.timesteps, (b,), device=device).long()
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise, sqrt_acp, sqrt_om)

            # condition dropout for CFG
            if cfg.cond_drop_prob > 0:
                drop_mask = torch.rand(b, device=device) < cfg.cond_drop_prob
                eps_pred_cond = model(xt, t, y)
                eps_pred_uncond = model(xt, t, None)
                eps_pred = torch.where(
                    drop_mask.view(-1, 1, 1, 1), eps_pred_uncond, eps_pred_cond
                )
            else:
                eps_pred = model(xt, t, y)

            loss = F.mse_loss(eps_pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))

            # periodic sampling preview (this is your main "is it working?" signal)
            if global_step % cfg.sample_every_steps == 0:
                model.eval()
                # use EMA weights for preview
                shadow = ema.state_dict()
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(shadow, strict=True)

                with torch.no_grad():
                    for cond_name, cond_id in [("benign", 0), ("malignant", 1)]:
                        y_s = torch.full((16,), cond_id, device=device, dtype=torch.long)
                        x_s = ddim_sample(
                            model,
                            betas,
                            (16, cfg.in_channels, cfg.img_size, cfg.img_size),
                            device,
                            y=y_s,
                            guidance=7.5,
                            steps=50,
                        )
                        # save as grid, normalize from [-1,1] to [0,1]
                        save_path = os.path.join(
                            samp_dir, f"step{global_step}_{cond_name}_g7p5.png"
                        )
                        save_image(x_s, save_path, nrow=4, normalize=True, value_range=(-1, 1))

                # restore weights and train mode
                model.load_state_dict(backup, strict=True)
                model.train()

        if epoch % cfg.save_every_epochs == 0 or epoch == cfg.epochs:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "config": asdict(cfg),
            }
            torch.save(ckpt, os.path.join(ckpt_dir, f"ddpm_bus_epoch{epoch}.pt"))
            print(f"[CKPT] saved epoch {epoch}")


if __name__ == "__main__":
    main()
