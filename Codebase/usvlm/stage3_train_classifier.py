import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Constants
# ----------------------------
ATTR_KEYS = [
    "BIRADS",
    "LesionEdge",
    "LesionBoundary",
    "LesionCalcificationFeatures",
    "EchoCharacteristics",
]


# ----------------------------
# Data utilities
# ----------------------------
def stable_hash(s: str) -> int:
    import hashlib

    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def find_root_candidates(script_dir: str):
    cwd = os.getcwd()
    return [
        cwd,
        os.path.join(cwd, "BUS-CoT", "BUS-CoT"),
        script_dir,
        os.path.abspath(os.path.join(script_dir, "..", "..", "BUS-CoT", "BUS-CoT")),
    ]


def resolve_root_and_anns(args_root: str, args_ann: List[str], script_dir: str):
    def first_root():
        for r in find_root_candidates(script_dir):
            if os.path.exists(os.path.join(r, "DatasetFiles", "lesion_dataset.json")):
                return r
        return None

    root_dir = args_root or first_root() or "."

    if args_ann:
        ann_paths = []
        for p in args_ann:
            if os.path.exists(p):
                ann_paths.append(p)
            else:
                cand = os.path.join(root_dir, p)
                if os.path.exists(cand):
                    ann_paths.append(cand)
        if not ann_paths:
            ann_paths = [
                os.path.join(root_dir, "DatasetFiles", "lesion_dataset.json"),
                os.path.join(root_dir, "DatasetFiles", "BUS-Expert_dataset.json"),
            ]
    else:
        ann_paths = [
            os.path.join(root_dir, "DatasetFiles", "lesion_dataset.json"),
            os.path.join(root_dir, "DatasetFiles", "BUS-Expert_dataset.json"),
        ]
    ann_paths = [p for p in ann_paths if os.path.exists(p)]
    if not ann_paths:
        raise FileNotFoundError("No valid annotation files found; please pass --ann_path/--root_dir.")
    return root_dir, ann_paths


def iter_records(items: Any):
    if isinstance(items, dict):
        for k, v in items.items():
            if isinstance(v, dict):
                v = dict(v)
                v["_key"] = k
                yield v
    elif isinstance(items, list):
        for v in items:
            if isinstance(v, dict):
                yield v


def extract_us_report(it: Dict[str, Any]) -> Dict[str, Any]:
    rep = it.get("us_report")
    if isinstance(rep, dict):
        # BUS-Expert has nested dict {"0": {...}}
        if len(rep) > 0:
            first_val = next(iter(rep.values()))
            if isinstance(first_val, dict) and any(k in first_val for k in ATTR_KEYS):
                return first_val
        return rep
    return {}


def load_records(ann_paths: List[str], root_dir: str, train_frac: float = 0.8):
    samples = []
    for ann_path in ann_paths:
        with open(ann_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, dict) and "data" in items:
            items = items["data"]
        for it in iter_records(items):
            ph = (it.get("pathology_histology") or {}).get("pathology", None)
            if ph is None:
                continue
            ph = str(ph).strip().lower()
            if ph not in ["benign", "malignant"]:
                continue
            label = 0 if ph == "benign" else 1
            split_tag = it.get("split")
            if split_tag is not None and str(split_tag).lower() == "test":
                continue
            img_rel = (
                it.get("image_path")
                or it.get("cropped_image")
                or it.get("image")
                or it.get("img")
            )
            if img_rel is None and isinstance(it.get("image_file"), dict):
                img_rel = it["image_file"].get("cropped_image") or it["image_file"].get("raw_image")
            if img_rel is None:
                continue
            img_path = img_rel if os.path.isabs(img_rel) else os.path.join(root_dir, img_rel)
            if not os.path.exists(img_path):
                continue

            pid = it.get("patient_id") or it.get("case_id") or it.get("id") or it.get("_key")
            if pid is None:
                pid = os.path.basename(img_path)
            pid = str(pid)

            us = extract_us_report(it)
            attrs = {}
            for k in ATTR_KEYS:
                val = us.get(k)
                if isinstance(val, str):
                    attrs[k] = val.strip()
                else:
                    attrs[k] = None
            samples.append((img_path, pid, label, attrs))

    train, val = [], []
    for img_path, pid, label, attrs in samples:
        h = stable_hash(pid) % 10000
        is_train = (h / 10000.0) < train_frac
        if is_train:
            train.append((img_path, label, attrs))
        else:
            val.append((img_path, label, attrs))
    return train, val


def build_attr_vocabs(samples: List[Tuple[str, int, Dict[str, Any]]]):
    vocabs = {k: set() for k in ATTR_KEYS}
    for _, _, attrs in samples:
        for k, v in attrs.items():
            if v is not None and v != "":
                vocabs[k].add(v)
    idx_maps = {}
    for k, vals in vocabs.items():
        vals = sorted(list(vals))
        idx_maps[k] = {v: i for i, v in enumerate(vals)}
    return idx_maps


class BUSMultiAttrDataset(Dataset):
    def __init__(self, items, attr_maps, img_size=256):
        self.items = items
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
        path, label, attrs = self.items[idx]
        img = Image.open(path)
        x = self.tf(img)
        attr_ids = {}
        for k in ATTR_KEYS:
            amap = self.attr_maps.get(k, {})
            val = attrs.get(k)
            if val is None or val == "" or val not in amap:
                attr_ids[k] = -100  # ignore
            else:
                attr_ids[k] = amap[val]
        return x, torch.tensor(label, dtype=torch.long), attr_ids


# ----------------------------
# Model
# ----------------------------
class MultiHeadClassifier(nn.Module):
    def __init__(self, num_attrs: Dict[str, int], proj_dim: int = 1280):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        backbone = efficientnet_b4(weights=weights)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        feat_dim = 1792  # efficientnet_b4 penultimate dim
        self.proj = nn.Linear(feat_dim, proj_dim)
        self.head_path = nn.Linear(proj_dim, 2)
        self.head_attrs = nn.ModuleDict()
        for k, n in num_attrs.items():
            self.head_attrs[k] = nn.Linear(proj_dim, n)

    def forward(self, x):
        feats = self.backbone(x)
        z = self.proj(feats)
        z = F.relu(z)
        out = {"feat": z, "path": self.head_path(z)}
        for k, head in self.head_attrs.items():
            out[k] = head(z)
        return out


def compute_f1(preds: List[int], gts: List[int], num_classes: int):
    # simple macro F1 without sklearn
    eps = 1e-9
    f1s = []
    for c in range(num_classes):
        tp = sum((p == c) and (g == c) for p, g in zip(preds, gts))
        fp = sum((p == c) and (g != c) for p, g in zip(preds, gts))
        fn = sum((p != c) and (g == c) for p, g in zip(preds, gts))
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def evaluate(model, loader, device, attr_maps):
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
    f1 = compute_f1(preds, gts, 2)
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_path", type=str, action="append")
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--out_dir", type=str, default="stage3_classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir, ann_paths = resolve_root_and_anns(args.root_dir, args.ann_path, script_dir)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_items, val_items = load_records(ann_paths, root_dir, train_frac=args.train_frac)
    attr_maps = build_attr_vocabs(train_items)

    ds_train = BUSMultiAttrDataset(train_items, attr_maps)
    ds_val = BUSMultiAttrDataset(val_items, attr_maps)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_attrs = {k: max(1, len(v)) for k, v in attr_maps.items()}
    model = MultiHeadClassifier(num_attrs=num_attrs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"epoch {epoch}/{args.epochs}")
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

        val_f1 = evaluate(model, dl_val, device, attr_maps)
        if val_f1 > best_f1:
            best_f1 = val_f1
            ckpt = {
                "model": model.state_dict(),
                "attr_maps": attr_maps,
                "config": vars(args),
                "best_f1": best_f1,
            }
            torch.save(ckpt, out_dir / "classifier_best.pt")
        print(f"[VAL] epoch={epoch} f1={val_f1:.4f} best={best_f1:.4f}")

    with open(out_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(attr_maps, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved to {out_dir}")


if __name__ == "__main__":
    main()
