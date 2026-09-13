"""v6a training script.

Usage:
    python -m server_v6a.train \\
        --jun27-good   /path/to/June-27-2026/good \\
        --jun27-bad    /path/to/June-27-2026/not_good \\
        --aug28-good   /path/to/August-28-2026/good \\
        --aug28-bad    /path/to/August-28-2026/not_good \\
        --cache-dir    data/v6a_crops \\
        --out-dir      models/v6a \\
        --epochs       30 \\
        --batch-size   16 \\
        --seed         42

Two-phase training:
  Phase 1 (epochs 0..WARMUP-1): backbone frozen, only classifier head trains at LR=1e-3.
  Phase 2 (epochs WARMUP..N-1): full model fine-tunes at LR=3e-5.

Best model is saved by val AUC-ROC.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset import SplashCropDataset
from .model import (
    TRAIN_TRANSFORM, INFER_TRANSFORM,
    build_model, freeze_backbone, unfreeze_all,
    get_device, save_checkpoint,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

WARMUP_EPOCHS = 5    # head-only
LR_HEAD       = 1e-3
LR_FULL       = 3e-5
WEIGHT_DECAY  = 1e-4
VAL_FRAC      = 0.20
SEED          = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    n = len(labels)
    n_good   = labels.count(0)
    n_bad    = labels.count(1)
    w_good   = n / (2 * n_good)  if n_good  else 0.0
    w_bad    = n / (2 * n_bad)   if n_bad   else 0.0
    weights  = [w_good if l == 0 else w_bad for l in labels]
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)


def _split_indices(
    dataset: SplashCropDataset,
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Stratified split by (session, label) so both sessions appear in val."""
    labels = [l for _, l in dataset.samples]
    paths  = [str(p) for p, _ in dataset.samples]

    # Build a stratum key from label + path (session embedded in path via cache subdir)
    def _stratum(i: int) -> int:
        # 0=jun27_good, 1=jun27_bad, 2=aug28_good, 3=aug28_bad
        is_jun27 = "jun27" in paths[i]
        return labels[i] + (0 if is_jun27 else 2)

    strata   = [_stratum(i) for i in range(len(dataset))]
    sss      = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(sss.split(strata, strata))
    return list(train_idx), list(val_idx)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels_t = labels.float().unsqueeze(1).to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels_t)
        total_loss += loss.item() * len(imgs)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc


def train(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    device  = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Device: %s", device)

    # ── Build sources list ────────────────────────────────────────────────────
    sources: list[dict] = []
    if args.jun27_good:
        sources.append({"root": args.jun27_good,  "label": 0, "session": "jun27"})
    if args.jun27_bad:
        sources.append({"root": args.jun27_bad,   "label": 1, "session": "jun27"})
    if args.aug28_good:
        sources.append({"root": args.aug28_good,  "label": 0, "session": "aug28"})
    if args.aug28_bad:
        sources.append({"root": args.aug28_bad,   "label": 1, "session": "aug28"})
    if not sources:
        raise ValueError("At least one training source must be specified.")

    # ── Harvest crops (cached) ────────────────────────────────────────────────
    log.info("Harvesting splash crops (cached in %s) …", args.cache_dir)
    full_ds = SplashCropDataset(
        sources=sources,
        cache_dir=args.cache_dir,
        transform=None,            # transforms applied per-split below
        force_rebuild=args.rebuild,
        n_candidates=args.n_candidates,
    )
    if len(full_ds) == 0:
        raise RuntimeError("No valid crops found. Check source paths and video files.")

    full_ds.save_manifest(out_dir / "manifest_full.json")

    # ── Train / val split ─────────────────────────────────────────────────────
    train_idx, val_idx = _split_indices(full_ds, VAL_FRAC, args.seed)
    log.info("Split: %d train, %d val", len(train_idx), len(val_idx))

    # Lightweight wrapper that applies a per-split transform without subclassing
    # Subset (avoids PyTorch ≥2.5 __getitems__ requirement).
    class _TransformView(Dataset):
        def __init__(self, ds, idx, tf):
            self._ds, self._idx, self._tf = ds, idx, tf
        def __len__(self):
            return len(self._idx)
        def __getitem__(self, i):
            img, label = self._ds[self._idx[i]]
            if self._tf is not None:
                img = self._tf(img)
            return img, label

    train_ds = _TransformView(full_ds, train_idx, TRAIN_TRANSFORM)
    val_ds   = _TransformView(full_ds, val_idx,   INFER_TRANSFORM)

    # Weighted sampler to balance GOOD / NOT_GOOD in each mini-batch
    train_labels = [full_ds.samples[i][1] for i in train_idx]
    sampler = _make_weighted_sampler(train_labels)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(pretrained=True).to(device)
    freeze_backbone(model)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=WARMUP_EPOCHS,
    )

    best_auc   = 0.0
    best_epoch = 0
    history    = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        t0 = time.perf_counter()

        # Switch to full fine-tuning at WARMUP_EPOCHS
        if epoch == WARMUP_EPOCHS:
            log.info("Epoch %d: unfreezing backbone, LR → %.1e", epoch, LR_FULL)
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - WARMUP_EPOCHS,
            )

        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = _val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        phase = "head" if epoch < WARMUP_EPOCHS else "full"
        log.info(
            "Epoch %3d/%d [%s] train_loss=%.4f val_loss=%.4f val_auc=%.4f  %.1fs",
            epoch + 1, args.epochs, phase,
            train_loss, val_loss, val_auc, elapsed,
        )

        history.append({
            "epoch": epoch + 1, "phase": phase,
            "train_loss": round(train_loss, 5),
            "val_loss":   round(val_loss, 5),
            "val_auc":    round(val_auc, 5),
        })

        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch + 1
            save_checkpoint(
                model, out_dir / "best.pth",
                meta={"epoch": best_epoch, "val_auc": round(best_auc, 5)},
            )
            log.info("  ↑ New best AUC=%.4f → saved best.pth", best_auc)

    # Always save the last checkpoint too
    save_checkpoint(
        model, out_dir / "last.pth",
        meta={"epoch": args.epochs, "val_auc": round(val_auc, 5)},
    )

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info("Training complete. Best val_auc=%.4f at epoch %d", best_auc, best_epoch)
    log.info("Model saved to %s/best.pth", out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description="IP4R v6a — train splash CNN")
    p.add_argument("--jun27-good",    default=None)
    p.add_argument("--jun27-bad",     default=None)
    p.add_argument("--aug28-good",    default=None)
    p.add_argument("--aug28-bad",     default=None)
    p.add_argument("--cache-dir",     default="data/v6a_crops")
    p.add_argument("--out-dir",       default="models/v6a")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=SEED)
    p.add_argument("--n-candidates",  type=int,   default=6,
                   help="frames to sample per video for best-splash selection")
    p.add_argument("--rebuild",       action="store_true",
                   help="force re-extract all crops (ignore cache)")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
