import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from stage3_train_classifier import MultiHeadClassifier, ATTR_KEYS


class SynthDataset(Dataset):
    def __init__(self, manifest_items, img_size=256):
        self.items = manifest_items
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Lambda(lambda im: im.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["path"])
        x = self.tf(img)
        cond = it.get("cond", "benign")
        label = 0 if cond.lower() == "benign" else 1
        return x, label, it


def load_classifier(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    attr_maps = ckpt["attr_maps"]
    num_attrs = {k: max(1, len(v)) for k, v in attr_maps.items()}
    model = MultiHeadClassifier(num_attrs=num_attrs).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    inv_maps = {k: {v: k2 for k2, v in m.items()} for k, m in attr_maps.items()}
    return model, inv_maps


def filter_manifest(manifest_path: str, model, inv_maps, device, batch_size: int, guidance: float = 7.5):
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    items = m.get("items", m)  # allow raw list
    ds = SynthDataset(items)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    kept = []
    total = len(ds)
    with torch.no_grad():
        for x, label, raw in tqdm(dl, desc="filter"):
            x = x.to(device)
            out = model(x)
            pred = out["path"].argmax(dim=1)
            match = pred.cpu() == label
            for i in range(x.shape[0]):
                if not match[i]:
                    continue
                attrs = {}
                for k in ATTR_KEYS:
                    logits = out.get(k)
                    if logits is None:
                        continue
                    pid = int(logits[i].argmax().item())
                    attrs[k] = inv_maps[k].get(pid, None)
                item = dict(raw[i])
                item["pseudo_attrs"] = attrs
                item["pred_path"] = "benign" if int(pred[i]) == 0 else "malignant"
                kept.append(item)
    return kept, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Trained classifier checkpoint (stage3)")
    parser.add_argument("--manifest", type=str, required=True, help="Stage2 manifest.json")
    parser.add_argument("--out_manifest", type=str, default="filtered_manifest.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, inv_maps = load_classifier(args.ckpt, device)
    kept, total = filter_manifest(args.manifest, model, inv_maps, device, args.batch_size)
    out_path = Path(args.out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"items": kept}, f, ensure_ascii=False, indent=2)
    keep_rate = len(kept) / max(1, total)
    print(f"[RESULT] kept {len(kept)}/{total} ({keep_rate*100:.2f}%) -> {out_path}")


if __name__ == "__main__":
    main()
