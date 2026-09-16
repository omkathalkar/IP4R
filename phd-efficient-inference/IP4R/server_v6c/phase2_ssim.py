"""Phase 2 — Per-ROI SSIM content check (Fix 2 v2).

Replaces Otsu-binarize + IoU from phase2_template.py.

The binarization step in phase2_template.py throws away graylevel structure,
making a dim-but-present segment and a truly absent segment look identical
(both produce a sparse binary mask).  SSIM on raw grayscale keeps that
distinction: a dim-but-present segment still correlates structurally with the
golden (high structure term) whereas an absent segment breaks local correlation.

Key design choices:
  - CLAHE normalisation per ROI before SSIM to equalise local contrast across
    sessions; applied to BOTH live and golden at comparison time.
  - Per-ROI SSIM thresholds stored in the .npz template (key ssim_thr_<name>),
    calibrated at the 1st-percentile of confirmed-GOOD scores during template
    building.  Never a single global cut.
  - Ghost check unchanged from phase2_template.py.
  - Do NOT normalise the SSIM score by a map statistic (median/MAD) — that
    inverts the ranking for dim frames.

Algorithm:
  1. Load golden template (.npz, must have gray_* and ssim_thr_* keys).
  2. Sample 5 timepoints around T* (same windowing as other Phase 2 modules).
  3. For each frame: detect_lcd() → 480×640 crop.
  4. For each ROI: CLAHE(crop_roi) → SSIM vs CLAHE(golden_roi).
     ON   if SSIM >= ssim_thr[roi]
     WEAK if ssim_thr[roi] * WEAK_FRAC <= SSIM < ssim_thr[roi]
     MISS otherwise
  5. Persistence vote (same as phase2_roi / phase2_template).
  6. Ghost check.

Verdict: PASS / FAIL / AMBIGUOUS — same semantics as other Phase 2 modules.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

LCD_W = 480
LCD_H = 640

GHOST_THR        = 0.40
N_TIMEPOINTS     = 5
CONFIRM_FRAC     = 0.75
CONFIRM_MIN_HITS = 2

CLAHE_CLIP  = 2.0
CLAHE_GRID  = (8, 8)
SSIM_WIN    = 7      # box-filter window size for SSIM (odd, ≥3)
WEAK_FRAC   = 0.85   # SSIM in [thr*WEAK_FRAC, thr) → WEAK (not hard fail)

# Default fallback when .npz has no ssim_thr_* keys
SSIM_THR_FALLBACK = 0.50

ROI_ATLAS: list[tuple] = [
    ("top_icons",         0,  90,  85, 400),
    ("icon_strip",       90, 140,  85, 400),
    ("left_clock",      140, 205,  70, 195),
    ("right_clock",     140, 205, 235, 380),
    ("center_88",       215, 305,  70, 190),
    ("signal_bars",     250, 305, 310, 390),
    ("secondary_icons", 305, 405,  45, 435),
    ("bottom_88888",    405, 455, 175, 395),
]

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


# ---------------------------------------------------------------------------
# SSIM (no external deps — uses cv2.filter2D as the box filter)
# ---------------------------------------------------------------------------

def _ssim(a: np.ndarray, b: np.ndarray, win: int = SSIM_WIN) -> float:
    """Mean windowed SSIM on uint8 images.

    Uses a box filter instead of Gaussian (cheap; good enough for ROI-level
    scores).  Returns the mean over the valid (non-border) region.
    """
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    k = np.ones((win, win), np.float32) / (win * win)

    def filt(x: np.ndarray) -> np.ndarray:
        return cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

    mu1, mu2 = filt(a), filt(b)
    sig1  = filt(a * a) - mu1 * mu1
    sig2  = filt(b * b) - mu2 * mu2
    sig12 = filt(a * b) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1 + sig2 + C2))

    p = win // 2
    return float(ssim_map[p:-p, p:-p].mean())


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase2SSIMResult:
    passed:              bool
    verdict:             str              # PASS | FAIL | AMBIGUOUS
    roi_states:          list[dict]       # per-ROI: name, verdict, ssim_scores, votes, ssim_thr
    ghost_frac:          float
    ghost_fail:          bool
    confirmed_on_count:  int
    confirmed_fail_rois: list[str]
    ambiguous_rois:      list[str]
    n_timepoints_valid:  int
    frame_indices:       list[int]
    error:               str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ssim_template(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return (golden_gray, ssim_thr) from an .npz built by build_golden_template.py."""
    data = np.load(str(path), allow_pickle=True)
    golden: dict[str, np.ndarray] = {}
    thrs:   dict[str, float]      = {}
    for key in data.files:
        if key.startswith("gray_"):
            golden[key[5:]] = data[key].astype(np.uint8)
        elif key.startswith("ssim_thr_"):
            thrs[key[9:]] = float(data[key].flat[0])
    if not golden:
        raise ValueError(
            f"{path} has no gray_* keys. "
            "Re-run build_golden_template.py — it now stores grayscale ROIs."
        )
    log.debug("SSIM template: %d ROIs, thresholds: %s",
              len(golden), {k: f"{v:.4f}" for k, v in thrs.items()})
    return golden, thrs


def _roi_ssim(crop_bgr: np.ndarray,
              y1: int, y2: int, x1: int, x2: int,
              golden_gray: np.ndarray) -> float:
    roi = crop_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    live_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    live_c = _clahe.apply(live_g)
    gold_c = _clahe.apply(golden_gray)
    if live_c.shape != gold_c.shape:
        gold_c = cv2.resize(gold_c, (live_c.shape[1], live_c.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    win = min(SSIM_WIN, live_c.shape[0] - 2, live_c.shape[1] - 2)
    win = max(3, win if win % 2 == 1 else win - 1)
    return _ssim(live_c, gold_c, win=win)


def _ghost_frac(crop_bgr: np.ndarray, dark_thresh: int = 130) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < dark_thresh).view(np.uint8)
    mask = np.zeros((LCD_H, LCD_W), dtype=np.uint8)
    for _, y1, y2, x1, x2 in ROI_ATLAS:
        mask[y1:y2, x1:x2] = 1
    outside = mask == 0
    n = int(outside.sum())
    return 0.0 if n == 0 else float(dark[outside].sum()) / n


def _persistence_verdict(counts: dict[str, int], n_valid: int) -> str:
    """
    Classify ROI state from per-frame votes.

    "on"   = SSIM ≥ threshold           → clearly present
    "weak" = SSIM in [thr*WEAK_FRAC, thr) → dim-but-present, treated as neutral
    "miss" = SSIM < thr*WEAK_FRAC       → structurally absent

    CONFIRMED_FAIL only when enough frames are "miss" (truly absent).
    CONFIRMED_ON when enough frames are "on" or "weak" (present at any brightness).
    This avoids penalising dim-LCD GOOD units whose SSIM sits in the weak band.
    """
    if n_valid == 0:
        return "AMBIGUOUS"
    thresh    = max(CONFIRM_MIN_HITS, math.ceil(n_valid * CONFIRM_FRAC))
    thresh    = min(thresh, n_valid)
    pos_hits  = counts.get("on", 0) + counts.get("weak", 0)
    miss_hits = counts.get("miss", 0)
    if pos_hits >= thresh:
        return "CONFIRMED_ON"
    if miss_hits >= thresh:
        return "CONFIRMED_FAIL"
    return "AMBIGUOUS"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase2_ssim(
    video_path:    str | Path,
    T_star:        int,
    template_path: str | Path,
    fps:           float = 12.0,
    ghost_thr:     float = GHOST_THR,
    n_timepoints:  int   = N_TIMEPOINTS,
    ssim_thr_override: dict[str, float] | None = None,
    ssim_thr_margin:   float = 0.97,   # multiply calibrated thr by this before applying
) -> Phase2SSIMResult:
    """
    Args:
        video_path        : path to video file
        T_star            : frame index from Phase 1
        template_path     : .npz with gray_* and ssim_thr_* keys
        ssim_thr_override : optional per-ROI threshold override (replaces .npz thresholds)
        ssim_thr_margin   : scale factor on calibrated thresholds (default 0.97 = 3% margin)
    """
    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    try:
        golden_gray, ssim_thr_cal = _load_ssim_template(template_path)
    except Exception as exc:
        return Phase2SSIMResult(
            passed=False, verdict="FAIL",
            roi_states=[], ghost_frac=0.0, ghost_fail=False,
            confirmed_on_count=0, confirmed_fail_rois=[],
            ambiguous_rois=[], n_timepoints_valid=0,
            frame_indices=[], error=f"template_load_error:{exc}",
        )

    # Build effective per-ROI threshold map
    ssim_thr: dict[str, float] = {}
    for name, *_ in ROI_ATLAS:
        if ssim_thr_override and name in ssim_thr_override:
            ssim_thr[name] = ssim_thr_override[name]
        elif name in ssim_thr_cal:
            ssim_thr[name] = ssim_thr_cal[name] * ssim_thr_margin
        else:
            ssim_thr[name] = SSIM_THR_FALLBACK

    # Always sample consecutive frames: the splash state is brief (~0.4s at 12fps),
    # so stride-based sampling jumps past it and lands on dark/off-state frames.
    frame_idxs = list(range(T_star, T_star + n_timepoints))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return Phase2SSIMResult(
            passed=False, verdict="FAIL",
            roi_states=[], ghost_frac=0.0, ghost_fail=False,
            confirmed_on_count=0, confirmed_fail_rois=[],
            ambiguous_rois=[], n_timepoints_valid=0,
            frame_indices=frame_idxs, error=f"cannot_open:{video_path}",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vote_counts: dict[str, dict[str, int]] = {
        name: {"on": 0, "weak": 0, "miss": 0} for name, *_ in ROI_ATLAS
    }
    ssim_log:      dict[str, list[float]] = {name: [] for name, *_ in ROI_ATLAS}
    ghost_samples: list[float] = []
    valid_idxs:    list[int]   = []

    for fidx in frame_idxs:
        if fidx >= total_frames:
            log.debug("Phase2SSIM: frame %d beyond video length — skip", fidx)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        crop, _ = detect_lcd(frame)
        if crop is None:
            log.debug("Phase2SSIM: detect_lcd failed at frame %d — skip", fidx)
            continue

        for name, y1, y2, x1, x2 in ROI_ATLAS:
            g = golden_gray.get(name)
            if g is None:
                continue
            score = _roi_ssim(crop, y1, y2, x1, x2, g)
            thr   = ssim_thr[name]
            weak  = thr * WEAK_FRAC
            if score >= thr:
                state = "on"
            elif score >= weak:
                state = "weak"
            else:
                state = "miss"
            vote_counts[name][state] += 1
            ssim_log[name].append(score)

        ghost_samples.append(_ghost_frac(crop))
        valid_idxs.append(fidx)

    cap.release()

    n_valid = len(valid_idxs)
    if n_valid == 0:
        return Phase2SSIMResult(
            passed=False, verdict="FAIL",
            roi_states=[], ghost_frac=0.0, ghost_fail=False,
            confirmed_on_count=0, confirmed_fail_rois=[],
            ambiguous_rois=[], n_timepoints_valid=0,
            frame_indices=frame_idxs, error="no_valid_lcd_crops",
        )

    roi_states:     list[dict] = []
    confirmed_on   = 0
    confirmed_fail: list[str]  = []
    ambiguous:      list[str]  = []

    for name, *_ in ROI_ATLAS:
        counts = vote_counts[name]
        verd   = _persistence_verdict(counts, n_valid)
        roi_states.append({
            "name":        name,
            "verdict":     verd,
            "votes":       dict(counts),
            "ssim_scores": [round(v, 4) for v in ssim_log[name]],
            "ssim_thr":    round(ssim_thr[name], 4),
            "n_valid":     n_valid,
        })
        if verd == "CONFIRMED_ON":
            confirmed_on += 1
        elif verd == "CONFIRMED_FAIL":
            confirmed_fail.append(name)
        else:
            ambiguous.append(name)

    median_ghost = float(np.median(ghost_samples))
    ghost_fail   = median_ghost > ghost_thr

    log.info(
        "Phase2SSIM: confirmed_on=%d/%d  fail=%s  ambiguous=%s  ghost=%.4f%s",
        confirmed_on, len(ROI_ATLAS),
        confirmed_fail or "[]",
        ambiguous or "[]",
        median_ghost,
        "  [GHOST FAIL]" if ghost_fail else "",
    )

    if confirmed_fail or ghost_fail:
        verdict, passed = "FAIL", False
    elif ambiguous:
        verdict, passed = "AMBIGUOUS", False
    else:
        verdict, passed = "PASS", True

    return Phase2SSIMResult(
        passed              = passed,
        verdict             = verdict,
        roi_states          = roi_states,
        ghost_frac          = round(median_ghost, 4),
        ghost_fail          = ghost_fail,
        confirmed_on_count  = confirmed_on,
        confirmed_fail_rois = confirmed_fail,
        ambiguous_rois      = ambiguous,
        n_timepoints_valid  = n_valid,
        frame_indices       = valid_idxs,
    )
