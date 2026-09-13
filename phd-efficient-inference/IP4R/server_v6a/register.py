"""Register-once splash frame to golden reference with multi-candidate retry.

Both paths produce a 2×3 affine warp aligned to the golden reference:
  Light background (frame_median ≥ DARK_THRESHOLD):  ORB keypoints → RANSAC → ECC affine
  Dark  background (frame_median <  DARK_THRESHOLD):  CLAHE → phase-correlation init → ECC affine

The ECC correlation coefficient (rho ∈ [-1, 1]) is the quality signal.
Candidates are tried in order; the one with the highest rho above SANITY_FLOOR is kept.
If all candidates fail the floor, the caller receives (None, rho, path) and must ABSTAIN.

Both paths warp the crop with cv2.WARP_INVERSE_MAP so the output is in golden space.
"""
from __future__ import annotations

import cv2
import numpy as np

from .lcd_crop import LCD_W, LCD_H

SANITY_FLOOR    = 0.30   # minimum ECC rho to accept a registration
DARK_THRESHOLD  = 63     # frame_median of raw grayscale crop below this → dark path

_ECC_ITER       = 200
_ECC_EPS        = 1e-5
_ORB_N          = 1000
_RANSAC_THRESH  = 5.0
_MIN_MATCHES    = 10


def _ecc_affine(
    src_gray: np.ndarray,
    dst_gray: np.ndarray,
    init_warp: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Run ECC affine refinement. Returns (warp, rho) or (None, -1.0) on failure.

    findTransformECC(template=dst, input=src) finds W such that
    warpAffine(src, W, size, WARP_INVERSE_MAP) ≈ dst.
    W maps template (golden) coords → input (crop) coords.
    """
    warp = init_warp.copy() if init_warp is not None else np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, _ECC_ITER, _ECC_EPS)
    try:
        rho, warp_out = cv2.findTransformECC(
            dst_gray.astype(np.float32),
            src_gray.astype(np.float32),
            warp, cv2.MOTION_AFFINE, criteria,
        )
        return warp_out, float(rho)
    except cv2.error:
        return None, -1.0


def _register_light(
    crop_gray: np.ndarray,
    golden_gray: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Light-background: ORB+RANSAC init → ECC affine."""
    orb = cv2.ORB_create(_ORB_N)
    kp1, des1 = orb.detectAndCompute(crop_gray, None)
    kp2, des2 = orb.detectAndCompute(golden_gray, None)

    if des1 is None or des2 is None or len(kp1) < _MIN_MATCHES or len(kp2) < _MIN_MATCHES:
        return _ecc_affine(crop_gray, golden_gray)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    good = matches[:min(50, len(matches))]

    if len(good) < _MIN_MATCHES:
        return _ecc_affine(crop_gray, golden_gray)

    # We need H mapping golden (template) → crop (input) for ECC init
    src_pts = np.float32([kp2[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)  # golden
    dst_pts = np.float32([kp1[m.queryIdx].pt  for m in good]).reshape(-1, 1, 2)  # crop

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, _RANSAC_THRESH)
    if H is None or mask is None or mask.sum() < _MIN_MATCHES:
        return _ecc_affine(crop_gray, golden_gray)

    init_warp = H[:2, :].astype(np.float32)
    return _ecc_affine(crop_gray, golden_gray, init_warp)


def _register_dark(
    crop_gray: np.ndarray,
    golden_gray: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Dark-background: CLAHE equalise → phase-correlation translation init → ECC affine."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_crop   = clahe.apply(crop_gray)
    eq_golden = clahe.apply(golden_gray)

    # phaseCorrelate(src1, src2) → shift such that src2 ≈ src1 shifted by shift
    # i.e. golden coord p → crop coord p + (dx, dy)
    shift, _ = cv2.phaseCorrelate(
        eq_golden.astype(np.float64),
        eq_crop.astype(np.float64),
    )
    dx, dy = float(shift[0]), float(shift[1])
    init_warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

    return _ecc_affine(eq_crop, eq_golden, init_warp)


def register_crop(
    crop_bgr: np.ndarray,
    golden_bgr: np.ndarray,
) -> tuple[np.ndarray | None, float, str]:
    """Register one LCD crop to the golden reference.

    The crop is resized to the golden's dimensions before ECC so this function
    works even when lcd_crop and the stored golden have different aspect ratios
    (e.g. portrait 480×640 crop vs landscape 640×480 golden).

    Returns:
        warped_bgr:  BGR image in golden's dimensions, registered to golden (or None)
        ecc_rho:     ECC correlation score ∈ [-1, 1]
        path_used:   "light" | "dark"
    """
    g_h, g_w = golden_bgr.shape[:2]

    # Resize crop to golden dimensions if they differ
    if crop_bgr.shape[:2] != (g_h, g_w):
        crop_bgr = cv2.resize(crop_bgr, (g_w, g_h), interpolation=cv2.INTER_LINEAR)

    crop_gray   = cv2.cvtColor(crop_bgr,   cv2.COLOR_BGR2GRAY)
    golden_gray = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY)

    if float(np.median(crop_gray)) >= DARK_THRESHOLD:
        warp, rho = _register_light(crop_gray, golden_gray)
        path = "light"
    else:
        warp, rho = _register_dark(crop_gray, golden_gray)
        path = "dark"

    if warp is None or rho < SANITY_FLOOR:
        return None, float(rho) if rho is not None else -1.0, path

    warped = cv2.warpAffine(
        crop_bgr, warp, (g_w, g_h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return warped, float(rho), path


def register_best_candidate(
    candidates: list[np.ndarray],
    golden_bgr: np.ndarray,
) -> tuple[np.ndarray | None, float, str, int]:
    """Try each 480×640 candidate crop; keep the one with the highest ECC rho.

    Returns:
        best_warped:  registered BGR crop aligned to golden (None if all fail SANITY_FLOOR)
        best_rho:     ECC score of the chosen candidate
        best_path:    "light" | "dark"
        best_idx:     index into candidates of the chosen frame (-1 if all failed)
    """
    best_warped: np.ndarray | None = None
    best_rho    = -1.0
    best_path   = "unknown"
    best_idx    = -1

    for i, crop in enumerate(candidates):
        warped, rho, path = register_crop(crop, golden_bgr)
        if warped is not None and rho > best_rho:
            best_warped = warped
            best_rho    = rho
            best_path   = path
            best_idx    = i

    return best_warped, best_rho, best_path, best_idx
