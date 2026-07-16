"""Align a sample image to the golden frame. Alignment is the make-or-break step.

Strategy: robust feature homography (ORB + RANSAC) to absorb translation/rotation/scale,
then ECC refinement for sub-pixel accuracy (SSIM is not shift-invariant, so this matters).
Falls back to ECC-only when feature matches are too sparse.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def _orb_homography(sample: np.ndarray, golden: np.ndarray, cfg: Config) -> np.ndarray | None:
    n = int(cfg.get("registration.orb_features", 2000))
    orb = cv2.ORB_create(nfeatures=n)
    kp1, des1 = orb.detectAndCompute(sample, None)
    kp2, des2 = orb.detectAndCompute(golden, None)
    if des1 is None or des2 is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    min_matches = int(cfg.get("registration.min_matches", 12))
    if len(matches) < min_matches:
        return None

    matches = sorted(matches, key=lambda m: m.distance)
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(
        src, dst, cv2.RANSAC, float(cfg.get("registration.ransac_reproj_thresh", 5.0))
    )
    if H is None or mask is None or int(mask.sum()) < min_matches:
        return None
    return H


def _ecc_refine(sample: np.ndarray, golden: np.ndarray, cfg: Config,
                warp_init: np.ndarray | None = None) -> np.ndarray | None:
    """Refine alignment with ECC (affine). Returns 3x3 homography-compatible matrix."""
    iters = int(cfg.get("registration.ecc_iterations", 100))
    eps = float(cfg.get("registration.ecc_eps", 1e-5))
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        _, warp = cv2.findTransformECC(
            golden.astype(np.float32), sample.astype(np.float32),
            warp, cv2.MOTION_AFFINE, criteria, None, 5,
        )
    except cv2.error:
        return None
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = warp
    return H


def register(sample: np.ndarray, golden: np.ndarray, cfg: Config) -> tuple[np.ndarray, dict]:
    """Warp `sample` into `golden`'s frame. Returns (aligned_sample, info)."""
    h, w = golden.shape[:2]
    method = cfg.get("registration.method", "orb_ecc")
    info: dict = {"method": method, "homography": None, "fallback": False}

    if not cfg.get("registration.enabled", True) or method == "none":
        info["method"] = "none"
        return sample.copy(), info

    H = None
    if method in ("orb_ecc", "orb_only"):
        H = _orb_homography(sample, golden, cfg)
        if H is None:
            info["fallback"] = True

    if method in ("orb_ecc", "ecc_only") and (H is None or method == "orb_ecc"):
        # warp by H first if we have it, then ECC-refine the residual
        if H is not None:
            base = cv2.warpPerspective(sample, H, (w, h))
        elif sample.shape[:2] != (h, w):
            # ORB failed and sizes differ — resize to golden dims so ECC can run
            base = cv2.resize(sample, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            base = sample
        H_ecc = _ecc_refine(base, golden, cfg)
        if H_ecc is not None:
            H = H_ecc if H is None else H_ecc @ H

    if H is None:
        info["fallback"] = True
        return sample.copy(), info

    aligned = cv2.warpPerspective(sample, H, (w, h))
    info["homography"] = H.tolist()
    return aligned, info
