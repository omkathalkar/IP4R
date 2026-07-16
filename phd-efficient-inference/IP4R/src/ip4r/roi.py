"""Region-of-interest model. Coords are normalised [0,1] so they survive resolution changes."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import yaml


@dataclass
class ROI:
    name: str           # snake_case, descriptive, e.g. "icon_snowflake"
    kind: str           # "icon" | "label" | "digit" | "segment_group" | "custom"
    x: float            # normalised top-left x  [0,1]
    y: float            # normalised top-left y  [0,1]
    w: float            # normalised width        [0,1]
    h: float            # normalised height       [0,1]
    ssim_min: float | None = None        # per-ROI SSIM floor; falls back to config global if None
    diff_max: float | None = None        # per-ROI diff-density ceiling; falls back to config global if None
    cov_ratio_min: float | None = None   # per-ROI coverage ratio floor; falls back to config global if None
    n_digits: int | None = None          # number of sub-digits in a wide digit strip; triggers per-sub-digit coverage check

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x = int(round(self.x * width))
        y = int(round(self.y * height))
        w = max(1, int(round(self.w * width)))
        h = max(1, int(round(self.h * height)))
        # clamp inside frame
        x = min(max(0, x), width - 1)
        y = min(max(0, y), height - 1)
        w = min(w, width - x)
        h = min(h, height - y)
        return x, y, w, h


def load_rois(path: str | Path) -> list[ROI]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return [ROI(**item) for item in data.get("rois", [])]


def save_rois(rois: list[ROI], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"rois": [asdict(r) for r in rois]}, f, sort_keys=False)
