import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from stage3_train_classifier import MultiHeadClassifier, ATTR_KEYS


class ImgDataset(Dataset):
    def __init__(self, items, img_size=256):
        self.items = items
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
        return x, it


def load_classifier(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    attr_maps = ckpt["attr_maps"]
    num_attrs = {k: max(1, len(v)) for k, v in attr_maps.items()}
    model = MultiHeadClassifier(num_attrs=num_attrs).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    inv_maps = {k: {v: k2 for k2, v in m.items()} for k, m in attr_maps.items()}
    return model, inv_maps, attr_maps


@torch.no_grad()
def build_index(model, device, manifest_path: str, out_path: str, batch_size: int = 64):
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    items = m.get("items", m)
    ds = ImgDataset(items)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feats, metas = [], []
    for x, meta in tqdm(dl, desc="index"):
        x = x.to(device)
        out = model(x)
        z = out["feat"]
        z = F.normalize(z, dim=1)
        feats.append(z.cpu())
        metas.extend(meta)
    feats = torch.cat(feats, dim=0)
    torch.save({"feats": feats, "meta": metas}, out_path)
    print(f"[INDEX] saved {len(metas)} entries to {out_path}")


@torch.no_grad()
def query(model, inv_maps, index_path: str, query_path: str, device: torch.device, topk: int = 5):
    # load index
    data = torch.load(index_path, map_location="cpu")
    feats_db = data["feats"]  # [N, D]
    metas = data["meta"]

    # preprocess query
    tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(query_path)
    x = tf(img).unsqueeze(0).to(device)
    out = model(x)
    z = F.normalize(out["feat"], dim=1)  # [1,D]
    path_pred = int(out["path"].argmax(dim=1).item())

    # attribute predictions
    attrs_pred = {}
    for k in ATTR_KEYS:
        logits = out.get(k)
        if logits is None:
            continue
        pid = int(logits.argmax(dim=1).item())
        attrs_pred[k] = inv_maps[k].get(pid, None)

    # filter by pathology
    mask = []
    for m in metas:
        cond = m.get("cond") or m.get("pred_path") or m.get("label") or "benign"
        lbl = 0 if str(cond).lower().startswith("b") else 1
        mask.append(lbl == path_pred)
    mask = torch.tensor(mask, dtype=torch.bool)
    feats_f = feats_db[mask]
    metas_f = [m for m, keep in zip(metas, mask) if keep]
    if feats_f.shape[0] == 0:
        print("No entries match pathology; returning top from all.")
        feats_f = feats_db
        metas_f = metas

    sims = torch.matmul(feats_f, z.T).squeeze(1)  # [M]
    topk = min(topk, sims.shape[0])
    vals, idxs = torch.topk(sims, k=topk, dim=0)

    results = []
    for score, idx in zip(vals.tolist(), idxs.tolist()):
        m = metas_f[idx]
        # compute attribute overlap
        overlap = 0
        total = 0
        if "pseudo_attrs" in m:
            for k in ATTR_KEYS:
                if k in attrs_pred and m["pseudo_attrs"].get(k) == attrs_pred[k]:
                    overlap += 1
                if k in attrs_pred:
                    total += 1
        attr_score = overlap / total if total > 0 else 0.0
        results.append(
            {
                "path": m.get("path"),
                "score": float(score),
                "attr_overlap": attr_score,
                "meta": m,
            }
        )
    return {
        "query_pathology": "benign" if path_pred == 0 else "malignant",
        "query_attrs": attrs_pred,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Trained classifier checkpoint")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--build_index", action="store_true")
    parser.add_argument("--manifest", type=str, help="Manifest for index building")
    parser.add_argument("--index_out", type=str, default="retrieval_index.pt")
    parser.add_argument("--query", type=str, help="Query image path")
    parser.add_argument("--index", type=str, help="Index file for querying")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, inv_maps, attr_maps = load_classifier(args.ckpt, device)

    if args.build_index:
        if not args.manifest:
            raise ValueError("--manifest required when --build_index is set")
        build_index(model, device, args.manifest, args.index_out)
        return

    if args.query:
        if not args.index:
            raise ValueError("--index required for querying")
        res = query(model, inv_maps, args.index, args.query, device, topk=args.topk)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    raise ValueError("Specify --build_index or --query")


if __name__ == "__main__":
    main()
