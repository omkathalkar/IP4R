"""21-element / 111-submask ROI atlas for v6b Method 1.

Loads ROI definitions from a YAML file, maps them onto the 480×640 registered
LCD crop, and computes normalized per-element coverage.

Coverage convention
───────────────────
  After CLAHE equalisation on the registered crop, active LCD segments are dark
  (low intensity).  "Lit" = pixel intensity < LIT_THRESHOLD (default 128).

  raw_nc(e)        = fraction of lit pixels within ROI e
  global_coverage  = fraction of lit pixels across the whole LCD crop
  normalized_nc(e) = raw_nc(e) / global_coverage

  Domain-shift rationale: absolute intensity varies across sessions (Jun-27
  light jig vs Aug-28 dark jig), but the ratio of element coverage to global
  coverage is stable because both the numerator and denominator shift together.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

LCD_W, LCD_H = 480, 640
LIT_THRESHOLD = 128
MIN_GLOBAL_COVERAGE = 0.01  # below this → frame is blank; return empty dict


@dataclass
class ROI:
    name: str
    x: float   # normalised left   ∈ [0, 1] (column direction, axis 1 in numpy)
    y: float   # normalised top    ∈ [0, 1] (row direction, axis 0 in numpy)
    w: float   # normalised width  ∈ [0, 1]
    h: float   # normalised height ∈ [0, 1]

    def pixel_box(
        self, img_w: int = LCD_W, img_h: int = LCD_H
    ) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) pixel coords for use as image[y1:y2, x1:x2]."""
        x1 = max(0, int(self.x * img_w))
        y1 = max(0, int(self.y * img_h))
        x2 = min(img_w, int((self.x + self.w) * img_w))
        y2 = min(img_h, int((self.y + self.h) * img_h))
        return x1, y1, x2, y2


def load_atlas(yaml_path: str | Path) -> list[ROI]:
    """Load ROI definitions from YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [
        ROI(
            name=entry["name"],
            x=float(entry["x"]),
            y=float(entry["y"]),
            w=float(entry["w"]),
            h=float(entry["h"]),
        )
        for entry in data.get("rois", [])
    ]


def compute_normalized_nc(
    warped_bgr: np.ndarray,
    atlas: list[ROI],
    lit_threshold: int = LIT_THRESHOLD,
) -> dict[str, float]:
    """Return per-element normalized coverage from a registered LCD crop.

    Returns {} if global_coverage < MIN_GLOBAL_COVERAGE (blank / unlit frame).
    normalized_nc(e) = raw_nc(e) / global_coverage ∈ [0, ∞)

    A well-lit element on a healthy unit yields normalized_nc ≈ 1 or above
    (element is at least as dense as the global average).  A missing or very
    dim element yields values much closer to 0.

    Uses actual image dimensions for pixel_box so the function works for any
    stored orientation of the golden/warped image.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)

    lit_mask = (eq < lit_threshold).astype(np.float32)
    global_coverage = float(lit_mask.mean())

    if global_coverage < MIN_GLOBAL_COVERAGE:
        return {}

    # Derive dimensions from the actual image so ROI pixel boxes are always correct
    img_h, img_w = warped_bgr.shape[:2]  # numpy shape = (rows=height, cols=width)

    results: dict[str, float] = {}
    for roi in atlas:
        x1, y1, x2, y2 = roi.pixel_box(img_w=img_w, img_h=img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        region = lit_mask[y1:y2, x1:x2]
        raw_nc = float(region.mean())
        results[roi.name] = raw_nc / global_coverage

    return results
