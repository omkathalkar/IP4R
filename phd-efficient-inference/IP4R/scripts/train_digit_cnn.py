"""Train the DigitCNN on synthetic defects generated from the good bank.

Usage:
    python scripts/train_digit_cnn.py [--epochs 30] [--out models/digit_cnn.pt]

No labels required — intact patches come from the good bank, defective patches
are synthesised by erasing 1-3 random segments using the known segment layout.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ip4r.config import Config
from ip4r.preprocess import preprocess
from ip4r.registration import register
from ip4r.pipeline import Inspector
from ip4r.inspect import inspect_rois
from ip4r.digit_cnn import (
    DigitCNN, DigitPatchDataset, DEVICE, PATCH_SIZE,
    crop_patch, make_defective,
)


def collect_patches(cfg: Config, inspector: Inspector) -> list[np.ndarray]:
    """Extract aligned digit patches from every good-bank image."""
    rois     = [r for r in inspector.rois if r.kind == "digit"]
    good_dir = cfg.path("tier_b.good_images_dir")
    golden   = inspector.golden_proc
    gh, gw   = golden.shape[:2]

    patches: list[np.ndarray] = []
    total = 0
    for cam_dir in sorted(good_dir.iterdir()):
        for img_path in sorted(cam_dir.glob("*.jpg")):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            sp      = preprocess(frame, cfg)
            aligned, _ = register(sp, golden, cfg)
            for r in rois:
                x, y, w, h = r.to_pixels(gw, gh)
                patch = aligned[y:y + h, x:x + w]
                if patch.size == 0:
                    continue
                patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE),
                                   interpolation=cv2.INTER_LINEAR)
                patches.append(patch)
            total += 1
    print(f"Collected {len(patches)} patches from {total} good-bank images")
    return patches


def build_dataset(good_patches: list[np.ndarray],
                  defect_ratio: float = 1.0,
                  augment: bool = True) -> DigitPatchDataset:
    """Build balanced dataset: good patches + synthetic defectives."""
    n_defect = int(len(good_patches) * defect_ratio)
    defect_patches = [
        make_defective(random.choice(good_patches))
        for _ in range(n_defect)
    ]
    all_patches = good_patches + defect_patches
    all_labels  = [0] * len(good_patches) + [1] * n_defect
    # shuffle
    combined = list(zip(all_patches, all_labels))
    random.shuffle(combined)
    patches, labels = zip(*combined)
    return DigitPatchDataset(list(patches), list(labels), augment=augment)


def train(args):
    cfg       = Config.load(REPO_ROOT / "config" / "default.yaml")
    inspector = Inspector(cfg)

    print(f"Device: {DEVICE}")
    print("Collecting good-bank patches …")
    good_patches = collect_patches(cfg, inspector)
    if not good_patches:
        print("ERROR: no patches found — check tier_b.good_images_dir in config")
        sys.exit(1)

    dataset   = build_dataset(good_patches, defect_ratio=1.5, augment=True)
    n_val     = max(1, int(0.15 * len(dataset)))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    print(f"Dataset: {n_train} train  {n_val} val  (good={len(good_patches)}  "
          f"synthetic_defect={len(dataset)-len(good_patches)})")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = DigitCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = train_correct = train_total = 0
        for patches, labels in train_loader:
            patches, labels = patches.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(patches)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(labels)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += len(labels)
        scheduler.step()

        # ── val ──
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches, labels = patches.to(DEVICE), labels.to(DEVICE)
                logits = model(patches)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += len(labels)

        t_acc = 100 * train_correct / train_total
        v_acc = 100 * val_correct   / val_total
        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss/train_total:.4f}  "
              f"train_acc={t_acc:.1f}%  val_acc={v_acc:.1f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), str(out_path))

    print(f"\nBest val acc: {best_val_acc:.1f}%")
    print(f"Model saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",  type=int,   default=40)
    ap.add_argument("--batch",   type=int,   default=64)
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument("--out",     default="models/digit_cnn.pt")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
