"""Image preprocessing: tame lighting/glare drift before comparison."""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def preprocess(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Return a single-channel float-friendly uint8 image ready for registration/scoring."""
    out = img
    if cfg.get("preprocess.to_gray", True) and out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    if cfg.get("preprocess.denoise", True):
        # bilateral filter preserves segment/icon edges while killing sensor noise
        out = cv2.bilateralFilter(out, d=5, sigmaColor=50, sigmaSpace=50)

    if cfg.get("preprocess.illumination_normalize", True):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(out)

    return out
