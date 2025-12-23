import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from stage3_train_classifier import (
    MultiHeadClassifier,
    ATTR_KEYS,
    resolve_root_and_anns,
    load_records,
    build_attr_vocabs,
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MixedDataset(Dataset):
    def __init__(self, real_items, syn_items, attr_maps, img_size=256):
        self.items = []
        for p, y, attrs in real_items:
            self.items.append({"path": p, "label": y, "attrs": attrs, "source": "real"})
        for it in syn_items:
            attrs = it.get("pseudo_attrs", {})
            label = 0 if it.get("cond", "benign").lower() == "benign" else 1
            self.items.append({"path": it["path"], "label": label, "attrs": attrs, "source": "syn"})

        self.attr_maps = attr_maps
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
        attr_ids = {}
        for k in ATTR_KEYS:
            amap = self.attr_maps.get(k, {})
            val = it["attrs"].get(k)
            if val is None or val == "" or val not in amap:
                attr_ids[k] = -100
            else:
                attr_ids[k] = amap[val]
        return x, torch.tensor(it["label"], dtype=torch.long), attr_ids


def load_filtered_manifest(path: str):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return m.get("items", m)


def sample_synthetic(filtered_items, target_total: int):
    benign = [it for it in filtered_items if it.get("cond", "benign").lower() == "benign"]
    malignant = [it for it in filtered_items if it.get("cond", "benign").lower() == "malignant"]
    half = target_total // 2
    random.shuffle(benign)
    random.shuffle(malignant)
    res = benign[:half] + malignant[:target_total - half]
    random.shuffle(res)
    return res


def compute_f1(preds: List[int], gts: List[int]):
    eps = 1e-9
    f1s = []
    for c in [0, 1]:
        tp = sum((p == c) and (g == c) for p, g in zip(preds, gts))
        fp = sum((p == c) and (g != c) for p, g in zip(preds, gts))
        fn = sum((p != c) and (g == c) for p, g in zip(preds, gts))
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1s.append(2 * prec * rec / (prec + rec + eps))
    return sum(f1s) / len(f1s)


def eval_pathology(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y, attr_ids in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            p = out["path"].argmax(dim=1)
            preds.extend(p.cpu().tolist())
            gts.extend(y.cpu().tolist())
    return compute_f1(preds, gts)


def train_scale(scale_size: int, args, root_dir, ann_paths, filtered_syn):
    set_seed(args.seed)
    # real split
    train_real, val_real = load_records(ann_paths, root_dir, train_frac=args.train_frac)
    attr_maps = build_attr_vocabs(train_real)
    num_attrs = {k: max(1, len(v)) for k, v in attr_maps.items()}

    syn_sample = sample_synthetic(filtered_syn, target_total=scale_size)

    ds_train = MixedDataset(train_real + [], syn_sample, attr_maps)
    ds_val = MixedDataset(val_real, [], attr_maps)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    model = MultiHeadClassifier(num_attrs=num_attrs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir) / f"scale_{scale_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"scale {scale_size} epoch {epoch}/{args.epochs}")
        for x, y, attr_ids in pbar:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = F.cross_entropy(out["path"], y)
            for k, logits in out.items():
                if k in ["feat", "path"]:
                    continue
                target = torch.tensor([a[k] for a in attr_ids], device=device)
                loss = loss + F.cross_entropy(logits, target, ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        f1 = eval_pathology(model, dl_val, device)
        if f1 > best:
            best = f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "attr_maps": attr_maps,
                    "config": vars(args),
                    "scale_size": scale_size,
                    "best_f1": best,
                },
                out_dir / "classifier_best.pt",
            )
        print(f"[VAL] scale={scale_size} epoch={epoch} f1={f1:.4f} best={best:.4f}")

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_manifest", type=str, required=True, help="Filtered synthetic manifest (stage3)")
    parser.add_argument("--ann_path", type=str, action="append")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="stage4_multitask")
    parser.add_argument("--scale_sizes", type=str, default="800,1600,3200,6400,12800,25600,51200")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir, ann_paths = resolve_root_and_anns(args.root_dir, args.ann_path, script_dir)

    filtered_syn = load_filtered_manifest(args.filtered_manifest)
    scales = [int(s) for s in args.scale_sizes.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for s in scales:
        f1 = train_scale(s, args, root_dir, ann_paths, filtered_syn)
        results[s] = f1

    with open(Path(args.out_dir) / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[DONE] results:", results)


if __name__ == "__main__":
    main()
