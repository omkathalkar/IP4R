"""Synthesise defective units from the golden image so the whole pipeline can be
validated before any real defective samples exist (erase/dim a random ROI, add blur,
or knock the pose to test registration)."""
from __future__ import annotations

import random

import cv2
import numpy as np

from .roi import ROI


def make_defect(golden_bgr: np.ndarray, rois: list[ROI], mode: str = "erase",
                seed: int | None = None) -> tuple[np.ndarray, str]:
    """Return (defective_image, description). Modes: erase | dim | blur | shift."""
    rng = random.Random(seed)
    img = golden_bgr.copy()
    h, w = img.shape[:2]

    if mode in ("erase", "dim") and rois:
        roi = rng.choice(rois)
        x, y, bw, bh = roi.to_pixels(w, h)
        if mode == "erase":
            # paint the element out with the panel's light background -> segment goes missing
            bg = int(np.percentile(img, 80))
            img[y:y + bh, x:x + bw] = bg
            return img, f"erased:{roi.name}"
        else:
            patch = img[y:y + bh, x:x + bw].astype(np.float32) * 0.45
            img[y:y + bh, x:x + bw] = patch.astype(np.uint8)
            return img, f"dimmed:{roi.name}"

    if mode == "blur":
        k = rng.choice([5, 7, 9])
        return cv2.GaussianBlur(img, (k, k), 0), f"blur:k{k}"

    if mode == "shift":
        dx, dy = rng.randint(-12, 12), rng.randint(-12, 12)
        ang = rng.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE), \
            f"shift:dx{dx}_dy{dy}_ang{ang:.1f}"

    return img, "none"
