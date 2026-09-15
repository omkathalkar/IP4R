"""Phase 2 — Full 8-region ROI coverage check (all-icons verification).

Design: IP4R_Stage2_21Icon_ROI_Design.md

Registration: detect_lcd() already warps each video frame to a canonical
480×640 portrait crop, so no additional homography is needed beyond what
Phase 3 already uses.  The 8 group ROIs cover all 21 icons:
  top_icons       → 5-6 mode row icons (auto/cool/dry/fan/heat/IR)
  icon_strip      → E / Lock / Turbo / ion row
  left_clock      → timer-off digit block
  right_clock     → timer-on digit block
  center_88       → large temperature digits
  signal_bars     → signal / fan-speed bars
  secondary_icons → battery / light / H-swing / V-swing / sleep
  bottom_88888    → 5-digit footer display

Algorithm:
1. T* guard: if T*=0 the LCD may still be transitioning, so shift the
   sampling window forward by one stride (0.5 s) before taking any frames.
2. Sample 5 timepoints: window_start + [0, 0.5s, 1.0s, 1.5s, 2.0s]
   At 12 FPS that is frame offsets [0, 6, 12, 18, 24] from window_start.
3. For each sampled frame: detect_lcd → 480×640 portrait crop.
4. Per-ROI coverage = fraction of pixels < DARK_THRESH (80).
   ON  if cov >= on_min
   DIM if dim_min <= cov < on_min
   OFF if cov < dim_min
5. Persistence vote across all valid frames:
   CONFIRMED_ON   if ON  in >= ceil(0.75 * n_valid), min 3
   CONFIRMED_FAIL if ON  in <  (n_valid - ceil(0.75*n_valid) + 1)
   AMBIGUOUS      otherwise
6. Ghost-pixel check: dark pixels outside the union of all 8 ROI masks.
   If ghost fraction > ghost_thr → FAIL.

Verdict:
  PASS      — all ROIs CONFIRMED_ON, no ghost pixels
  FAIL      — any ROI CONFIRMED_FAIL OR ghost pixels detected
  AMBIGUOUS — some ROIs unresolved, no hard failures → route to Phase 3
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

LCD_W       = 480
LCD_H       = 640
# Calibrated 2026-09-15 on Sep-15 GOOD splash frame (224312 frame 0).
# LCD segments appear as mid-gray (~80-130) on bright background (~160-200).
# DARK_THRESH=130 correctly captures segment pixels; old value of 80 was too low.
DARK_THRESH = 130
N_TIMEPOINTS     = 5        # frames to sample per video
CONFIRM_FRAC     = 0.75     # fraction of valid frames needed to confirm
CONFIRM_MIN_HITS = 2        # lowered from 3: splash window may be <0.5s
GHOST_THR        = 0.40     # fraction of non-ROI dark pixels triggering ghost fail.
                            # ~0.25 is normal for full-remote crop (bezel+buttons).
                            # Only flag genuinely anomalous leakage above 0.40.

# ROI atlas — all coordinates on the 480×640 portrait LCD crop from detect_lcd().
# Recalibrated 2026-09-15 from actual Sep-15 GOOD splash frame pixel stats.
# Format: (name, y1, y2, x1, x2, on_cov_min, dim_cov_min)
#   on_cov_min  : fraction of pixels < DARK_THRESH=130 for state=ON
#                 Set to ~60% of measured splash value to allow dimmer frames.
#   dim_cov_min : ~30% of measured splash value — below this → OFF
# Measured splash lt130 values:
#   top_icons=0.284, icon_strip=0.768, left_clock=0.565, right_clock=0.496
#   center_88=0.879, signal_bars=0.729, secondary_icons=0.729, bottom_88888=0.762
ROI_ATLAS: list[tuple] = [
    # name               y1   y2   x1   x2  on_min dim_min
    ("top_icons",         0,  90,  85, 400,  0.16,  0.06),   # mode row icons
    ("icon_strip",       90, 140,  85, 400,  0.45,  0.15),   # E/Lock/Turbo/ion
    ("left_clock",      140, 205,  70, 195,  0.33,  0.10),   # timer-off digits
    ("right_clock",     140, 205, 235, 380,  0.28,  0.08),   # timer-on digits
    ("center_88",       215, 305,  70, 190,  0.52,  0.18),   # temperature digits
    ("signal_bars",     250, 305, 310, 390,  0.43,  0.14),   # signal/fan bars
    ("secondary_icons", 305, 405,  45, 435,  0.43,  0.12),   # batt/light/swing/sleep
    ("bottom_88888",    405, 455, 175, 395,  0.45,  0.15),   # footer 5-digit display
]


@dataclass
class Phase2ROIResult:
    passed:              bool
    verdict:             str            # PASS | FAIL | AMBIGUOUS
    roi_states:          list[dict]     # per-ROI persistence verdict + vote counts
    ghost_frac:          float          # median dark fraction outside ROI union
    ghost_fail:          bool
    confirmed_on_count:  int
    confirmed_fail_rois: list[str]
    ambiguous_rois:      list[str]
    n_timepoints_valid:  int
    frame_indices:       list[int]      # frames that produced valid LCD crops
    error:               str | None = None


def _roi_state(gray: np.ndarray, y1: int, y2: int, x1: int, x2: int,
               on_min: float, dim_min: float) -> str:
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return "off"
    cov = float((roi < DARK_THRESH).mean())
    if cov >= on_min:
        return "on"
    if cov >= dim_min:
        return "dim"
    return "off"


def _classify_frame(crop_bgr: np.ndarray) -> dict[str, str]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return {
        name: _roi_state(gray, y1, y2, x1, x2, on_min, dim_min)
        for name, y1, y2, x1, x2, on_min, dim_min in ROI_ATLAS
    }


def _ghost_frac(crop_bgr: np.ndarray) -> float:
    """Fraction of dark pixels outside the union of all ROI masks."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < DARK_THRESH).view(np.uint8)

    roi_mask = np.zeros((LCD_H, LCD_W), dtype=np.uint8)
    for _, y1, y2, x1, x2, _, _ in ROI_ATLAS:
        roi_mask[y1:y2, x1:x2] = 1

    outside       = roi_mask == 0
    outside_total = int(outside.sum())
    if outside_total == 0:
        return 0.0
    return float(dark[outside].sum()) / outside_total


def _persistence_verdict(counts: dict[str, int], n_valid: int) -> str:
    """4-of-5 persistence rule, scaled to n_valid available frames."""
    if n_valid == 0:
        return "AMBIGUOUS"
    thresh = max(CONFIRM_MIN_HITS, math.ceil(n_valid * CONFIRM_FRAC))
    thresh = min(thresh, n_valid)
    on_hits = counts.get("on", 0)
    if on_hits >= thresh:
        return "CONFIRMED_ON"
    # CONFIRMED_FAIL: too few ON frames, with enough contrary evidence
    if on_hits <= n_valid - thresh:
        return "CONFIRMED_FAIL"
    return "AMBIGUOUS"


def run_phase2_roi(
    video_path:   str | Path,
    T_star:       int,
    fps:          float = 12.0,
    ghost_thr:    float = GHOST_THR,
    n_timepoints: int   = N_TIMEPOINTS,
    t_star_guard: bool  = True,
) -> Phase2ROIResult:
    """
    Args:
        video_path   : path to video file
        T_star       : frame index from Phase 1 (YOLO-confirmed splash start)
        fps          : video frames per second
        ghost_thr    : max fraction of non-ROI dark pixels before ghost FAIL
        n_timepoints : number of sample frames (default 5)
        t_star_guard : if True, shift window by 0.5s when T*=0
    """
    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    stride = max(1, int(round(fps / 2)))   # 0.5 s between samples

    # Early-T* guard: if T* is within the first 1s of the video (T* < fps),
    # stride-based sampling extends past the (short) splash window into dark background.
    # Use consecutive-frame sampling to keep all samples within the splash.
    # Covers T*=0 (recording starts mid-splash), T*=2 (Fix 1 windowed anchor),
    # and T*=8 (YOLO needed 8 frames to accumulate all 5 classes).
    if t_star_guard and T_star < int(fps):
        frame_idxs = list(range(T_star, T_star + n_timepoints))
    else:
        frame_idxs = [T_star + i * stride for i in range(n_timepoints)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return Phase2ROIResult(
            passed=False, verdict="FAIL",
            roi_states=[], ghost_frac=0.0, ghost_fail=False,
            confirmed_on_count=0, confirmed_fail_rois=[],
            ambiguous_rois=[], n_timepoints_valid=0,
            frame_indices=frame_idxs,
            error=f"cannot_open:{video_path}",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vote_counts: dict[str, dict[str, int]] = {
        name: {"on": 0, "dim": 0, "off": 0} for name, *_ in ROI_ATLAS
    }
    ghost_samples: list[float] = []
    valid_idxs:    list[int]   = []

    for fidx in frame_idxs:
        if fidx >= total_frames:
            log.debug("Phase2ROI: frame %d beyond video length %d — skip", fidx, total_frames)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        crop, _ = detect_lcd(frame)
        if crop is None:
            log.debug("Phase2ROI: detect_lcd failed at frame %d — skip", fidx)
            continue

        for name, state in _classify_frame(crop).items():
            vote_counts[name][state] += 1

        ghost_samples.append(_ghost_frac(crop))
        valid_idxs.append(fidx)

    cap.release()

    n_valid = len(valid_idxs)
    if n_valid == 0:
        return Phase2ROIResult(
            passed=False, verdict="FAIL",
            roi_states=[], ghost_frac=0.0, ghost_fail=False,
            confirmed_on_count=0, confirmed_fail_rois=[],
            ambiguous_rois=[], n_timepoints_valid=0,
            frame_indices=frame_idxs,
            error="no_valid_lcd_crops",
        )

    roi_states:      list[dict] = []
    confirmed_on     = 0
    confirmed_fail:  list[str]  = []
    ambiguous:       list[str]  = []

    for name, *_ in ROI_ATLAS:
        counts  = vote_counts[name]
        verd    = _persistence_verdict(counts, n_valid)
        roi_states.append({
            "name":    name,
            "verdict": verd,
            "votes":   dict(counts),
            "n_valid": n_valid,
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
        "Phase2ROI: confirmed_on=%d/%d  fail=%s  ambiguous=%s  ghost=%.4f%s",
        confirmed_on, len(ROI_ATLAS),
        confirmed_fail or "[]",
        ambiguous or "[]",
        median_ghost,
        "  [GHOST FAIL]" if ghost_fail else "",
    )

    if confirmed_fail or ghost_fail:
        verdict = "FAIL"
        passed  = False
    elif ambiguous:
        verdict = "AMBIGUOUS"
        passed  = False   # routes to Phase 3
    else:
        verdict = "PASS"
        passed  = True

    return Phase2ROIResult(
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
