"""SplashCropDataset — harvest and cache the best splash crop from each training video.

Usage:
    ds = SplashCropDataset(
        sources=[
            {"root": "/data/June-27-2026/good",     "label": 0, "session": "jun27"},
            {"root": "/data/June-27-2026/not_good",  "label": 1, "session": "jun27"},
            {"root": "/data/August-28-2026/good",    "label": 0, "session": "aug28"},
            {"root": "/data/August-28-2026/not_good","label": 1, "session": "aug28"},
        ],
        cache_dir="data/v6a_crops",
        transform=TRAIN_TRANSFORM,
        force_rebuild=False,
    )

Crops are extracted once and stored as <cache_dir>/<session>_<label>/<stem>.jpg.
Subsequent loads skip extraction and just read the cached JPEGs.

Label convention: 0 = GOOD, 1 = NOT_GOOD  (see model.py)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .frame_extract import extract_best_splash_frame

log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


class SplashCropDataset(Dataset):
    def __init__(
        self,
        sources: list[dict],
        cache_dir: str | Path,
        transform: transforms.Compose | None = None,
        force_rebuild: bool = False,
        n_candidates: int = 6,
    ) -> None:
        """
        sources: list of dicts with keys:
            root    (str | Path) — directory containing video files
            label   (int)        — 0=GOOD, 1=NOT_GOOD
            session (str)        — "jun27" | "aug28" (informational only)
        """
        self.transform = transform
        self.cache_dir = Path(cache_dir)
        self.samples: list[tuple[Path, int]] = []  # (crop_path, label)

        for src in sources:
            root    = Path(src["root"])
            label   = int(src["label"])
            session = src.get("session", "unknown")

            sub_cache = self.cache_dir / f"{session}_{'good' if label == 0 else 'bad'}"
            sub_cache.mkdir(parents=True, exist_ok=True)

            videos = sorted(p for p in root.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
            if not videos:
                log.warning("No videos found in %s", root)

            for vid in videos:
                crop_path = sub_cache / (vid.stem + ".jpg")
                if crop_path.exists() and not force_rebuild:
                    self.samples.append((crop_path, label))
                    continue

                crop, info = extract_best_splash_frame(vid, n_candidates=n_candidates)
                if crop is None:
                    log.warning("No splash crop for %s: %s", vid.name, info.get("error"))
                    continue

                cv2.imwrite(str(crop_path), crop)
                self.samples.append((crop_path, label))
                log.debug("Cached %s → score=%.3f session=%s", vid.name,
                          info.get("score", 0), info.get("session"))

        log.info("Dataset ready: %d samples (%d GOOD, %d NOT_GOOD)",
                 len(self.samples),
                 sum(1 for _, l in self.samples if l == 0),
                 sum(1 for _, l in self.samples if l == 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        crop_path, label = self.samples[idx]
        img = Image.open(crop_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def save_manifest(self, path: str | Path) -> None:
        manifest = [{"crop": str(p), "label": l} for p, l in self.samples]
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("Manifest saved to %s", path)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        transform: transforms.Compose | None = None,
    ) -> "SplashCropDataset":
        """Load a pre-built dataset from a manifest JSON (skips extraction)."""
        with open(manifest_path) as f:
            manifest = json.load(f)
        inst = object.__new__(cls)
        inst.transform = transform
        inst.cache_dir = Path(manifest_path).parent
        inst.samples   = [(Path(m["crop"]), int(m["label"])) for m in manifest]
        return inst
