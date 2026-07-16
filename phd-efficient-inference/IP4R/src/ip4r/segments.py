"""Optional digit-aware layer: verify each of the 7 strokes of a seven-segment '8' is lit.

In the all-on splash state every segment must be ON. We model the 7 strokes as fixed
sub-regions of a digit's bbox and check lit-coverage per stroke. Enable in config
(tier_a.segments.enabled) after authoring digit ROIs of kind == "digit".

Standard segment layout:
       aaa
      f   b
      f   b
       ggg
      e   c
      e   c
       ddd
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .roi import ROI

# Each segment as a normalised (x, y, w, h) sub-box within the digit bbox.
# Calibrated from column/row occupancy analysis on the golden reference image:
#   Left verticals (e,f) at column x=[0.27, 0.47]  (pixel ~12-20 on a 42px digit)
#   Right verticals (b,c) at column x=[0.72, 0.92]  (pixel ~30-38)
#   Horizontals span      x=[0.27, 0.79]
#   Top segment           y=[0.09, 0.25]
#   Middle segment        y=[0.46, 0.60]
#   Bottom segment        y=[0.83, 0.97]
_SEG_BOXES = {
    "a": (0.27, 0.09, 0.52, 0.16),   # top horizontal
    "b": (0.72, 0.26, 0.20, 0.26),   # top-right vertical
    "c": (0.72, 0.60, 0.20, 0.27),   # bottom-right vertical
    "d": (0.27, 0.83, 0.52, 0.14),   # bottom horizontal
    "e": (0.27, 0.60, 0.20, 0.27),   # bottom-left vertical
    "f": (0.27, 0.26, 0.20, 0.26),   # top-left vertical
    "g": (0.27, 0.46, 0.52, 0.14),   # middle horizontal
}


def check_digit_segments(patch: np.ndarray, cfg: Config) -> dict[str, bool]:
    """Return {segment: is_on} for a single digit patch (grayscale, registered)."""
    on_min = float(cfg.get("tier_a.segments.on_coverage_min", 0.25))
    active_is_dark = bool(cfg.get("tier_a.coverage.active_is_dark", True))
    block = int(cfg.get("tier_a.coverage.adaptive_block", 31))
    cc = int(cfg.get("tier_a.coverage.adaptive_C", 5))
    H, W = patch.shape[:2]
    out: dict[str, bool] = {}
    mode = cv2.THRESH_BINARY_INV if active_is_dark else cv2.THRESH_BINARY
    for seg, (sx, sy, sw, sh) in _SEG_BOXES.items():
        x, y = int(sx * W), int(sy * H)
        w, h = max(1, int(sw * W)), max(1, int(sh * H))
        sub = patch[y:y + h, x:x + w]
        if sub.size == 0:
            out[seg] = False
            continue
        b = block if block % 2 == 1 else block + 1
        b = max(3, min(b, (min(sub.shape[:2]) // 2) * 2 + 1))
        binar = cv2.adaptiveThreshold(
            sub, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, b, cc
        )
        out[seg] = float((binar > 0).mean()) >= on_min
    return out


def missing_segments(patch: np.ndarray, cfg: Config) -> list[str]:
    """Segments expected ON (all-on state) but detected OFF."""
    return [seg for seg, on in check_digit_segments(patch, cfg).items() if not on]
