"""Phase 2 — Golden-template content check (Fix 2).

Replaces the coverage-counting approach in phase2_roi.py.

Key differences from phase2_roi.py:
  - Per-ROI state is measured by IoU against a domain-matched golden binary mask,
    NOT by pixel coverage fraction. This detects wrong icon content, not just presence.
  - Binarization is local Otsu (per-crop) so the check adapts to scene brightness
    rather than requiring a globally-tuned DARK_THRESH.
  - Ghost check is unchanged (dark pixels outside ROI union > GHOST_THR).

Algorithm:
  1. Load golden template (.npz) built by build_golden_template.py.
  2. Sample 5 timepoints: T*, T*+0.5s, T*+1.0s, T*+1.5s, T*+2.0s.
  3. For each frame: detect_lcd() → 480×640 crop.
  4. For each ROI: Otsu-binarize the crop patch → IoU with golden mask.
     MATCH  if IoU >= iou_thr
     WEAK   if iou_weak_thr <= IoU < iou_thr
     MISMATCH otherwise
  5. Persistence vote (same as phase2_roi.py):
     CONFIRMED_ON   if MATCH in >= ceil(0.75 * n_valid), min 2
     CONFIRMED_FAIL if MATCH in < (n_valid - ceil(0.75*n_valid) + 1)
     AMBIGUOUS      otherwise
  6. Ghost check (unchanged from phase2_roi.py).

Verdict:
  PASS      — all ROIs CONFIRMED_ON, no ghost
  FAIL      — any ROI CONFIRMED_FAIL OR ghost
  AMBIGUOUS — unresolved ROIs, no hard failures → routes to Phase 3
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

# Default IoU thresholds — calibrated 2026-09-15 (Sep-15 eval sweep)
IOC_THR_DEFAULT      = 0.20   # IoU >= this → MATCH  (GOOD=93%, NOT_GOOD=29%)
IOC_WEAK_THR_DEFAULT = 0.10   # IoU in [weak, thr) → WEAK (not a hard fail)
GHOST_THR            = 0.40   # fraction of non-ROI dark pixels triggering ghost FAIL

# Per-ROI overrides (applied on top of the global iou_thr).
# NOTE 2026-09-15: sep-15 eval shows GOOD outliers (224421, 235841) have
# dim-LCD-induced low IoU (0.25-0.31) that overlaps with NOT_GOOD units in
# the same range. Per-ROI boost does NOT improve F1; flat iou_thr=0.20 is
# preferred until SSIM-based scoring replaces binary IoU.
RECOMMENDED_PER_ROI_THR: dict[str, float] = {}  # no boost: use flat iou_thr

N_TIMEPOINTS     = 5
CONFIRM_FRAC     = 0.75
CONFIRM_MIN_HITS = 2

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


@dataclass
class Phase2TemplateResult:
    passed:              bool
    verdict:             str              # PASS | FAIL | AMBIGUOUS
    roi_states:          list[dict]       # per-ROI: name, verdict, iou_scores, votes
    ghost_frac:          float
    ghost_fail:          bool
    confirmed_on_count:  int
    confirmed_fail_rois: list[str]
    ambiguous_rois:      list[str]
    n_timepoints_valid:  int
    frame_indices:       list[int]
    error:               str | None = None


def _load_template(template_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(str(template_path), allow_pickle=True)
    golden: dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("roi_"):
            roi_name = key[4:]   # strip "roi_" prefix
            golden[roi_name] = data[key].astype(np.uint8)
    log.debug("Loaded golden template: %d ROIs from %s", len(golden), template_path)
    return golden


def _otsu_binarize(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def _iou(live_mask: np.ndarray, golden_mask: np.ndarray) -> float:
    if live_mask.shape != golden_mask.shape:
        golden_mask = cv2.resize(golden_mask, (live_mask.shape[1], live_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    inter = np.logical_and(live_mask, golden_mask).sum()
    union = np.logical_or(live_mask, golden_mask).sum()
    return float(inter) / max(float(union), 1.0)


def _roi_iou(crop_bgr: np.ndarray, y1: int, y2: int, x1: int, x2: int,
             golden_mask: np.ndarray) -> float:
    roi = crop_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    live_mask = _otsu_binarize(roi)
    return _iou(live_mask, golden_mask)


def _roi_state(iou: float, iou_thr: float, iou_weak_thr: float) -> str:
    if iou >= iou_thr:
        return "on"
    if iou >= iou_weak_thr:
        return "dim"
    return "off"


def _ghost_frac(crop_bgr: np.ndarray, dark_thresh: int = 130) -> float:
    gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    dark  = (gray < dark_thresh).view(np.uint8)
    roi_mask = np.zeros((LCD_H, LCD_W), dtype=np.uint8)
    for _, y1, y2, x1, x2 in ROI_ATLAS:
        roi_mask[y1:y2, x1:x2] = 1
    outside       = roi_mask == 0
    outside_total = int(outside.sum())
    if outside_total == 0:
        return 0.0
    return float(dark[outside].sum()) / outside_total


def _persistence_verdict(counts: dict[str, int], n_valid: int) -> str:
    if n_valid == 0:
        return "AMBIGUOUS"
    thresh  = max(CONFIRM_MIN_HITS, math.ceil(n_valid * CONFIRM_FRAC))
    thresh  = min(thresh, n_valid)
    on_hits = counts.get("on", 0)
    if on_hits >= thresh:
        return "CONFIRMED_ON"
    if on_hits <= n_valid - thresh:
        return "CONFIRMED_FAIL"
    return "AMBIGUOUS"


def run_phase2_template(
    video_path:        str | Path,
    T_star:            int,
    template_path:     str | Path,
    fps:               float = 12.0,
    iou_thr:           float = IOC_THR_DEFAULT,
    iou_weak_thr:      float = IOC_WEAK_THR_DEFAULT,
    ghost_thr:         float = GHOST_THR,
    n_timepoints:      int   = N_TIMEPOINTS,
    t_star_guard:      bool  = True,
    per_roi_iou_thr:   dict[str, float] | None = None,
) -> Phase2TemplateResult:
    """
    Args:
        video_path      : path to video file
        T_star          : frame index from Phase 1
        template_path   : path to golden_template.npz
        iou_thr         : global IoU ≥ this → ROI is MATCH (on)
        iou_weak_thr    : IoU in [weak, thr) → WEAK (dim), not a hard fail
        ghost_thr       : max fraction of non-ROI dark pixels
        per_roi_iou_thr : optional per-ROI overrides for iou_thr, e.g.
                          {"left_clock": 0.35, "center_88": 0.30}
    """
    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    try:
        golden = _load_template(template_path)
    except Exception as exc:
        return Phase2TemplateResult(
            passed=False, verdict="FAIL",
            roi_states=[], ghost_frac=0.0, ghost_fail=False,
            confirmed_on_count=0, confirmed_fail_rois=[],
            ambiguous_rois=[], n_timepoints_valid=0,
            frame_indices=[],
            error=f"template_load_error:{exc}",
        )

    stride = max(1, int(round(fps / 2)))

    # Same early-T* guard as phase2_roi: any T* within first 1s uses consecutive frames.
    if t_star_guard and T_star < int(fps):
        frame_idxs = list(range(T_star, T_star + n_timepoints))
    else:
        frame_idxs = [T_star + i * stride for i in range(n_timepoints)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return Phase2TemplateResult(
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
    iou_log:     dict[str, list[float]] = {name: [] for name, *_ in ROI_ATLAS}
    ghost_samples: list[float] = []
    valid_idxs:    list[int]   = []

    for fidx in frame_idxs:
        if fidx >= total_frames:
            log.debug("Phase2Template: frame %d beyond video length — skip", fidx)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        crop, _ = detect_lcd(frame)
        if crop is None:
            log.debug("Phase2Template: detect_lcd() failed at frame %d — skip", fidx)
            continue

        for name, y1, y2, x1, x2 in ROI_ATLAS:
            g_mask = golden.get(name)
            if g_mask is None:
                continue
            roi_thr      = (per_roi_iou_thr or {}).get(name, iou_thr)
            roi_weak_thr = min(iou_weak_thr, roi_thr * 0.5)
            iou   = _roi_iou(crop, y1, y2, x1, x2, g_mask)
            state = _roi_state(iou, roi_thr, roi_weak_thr)
            vote_counts[name][state] += 1
            iou_log[name].append(iou)

        ghost_samples.append(_ghost_frac(crop))
        valid_idxs.append(fidx)

    cap.release()

    n_valid = len(valid_idxs)
    if n_valid == 0:
        return Phase2TemplateResult(
            passed=False, verdict="FAIL",
            roi_states=[], ghost_frac=0.0, ghost_fail=False,
            confirmed_on_count=0, confirmed_fail_rois=[],
            ambiguous_rois=[], n_timepoints_valid=0,
            frame_indices=frame_idxs,
            error="no_valid_lcd_crops",
        )

    roi_states:     list[dict] = []
    confirmed_on   = 0
    confirmed_fail: list[str]  = []
    ambiguous:      list[str]  = []

    for name, *_ in ROI_ATLAS:
        counts = vote_counts[name]
        verd   = _persistence_verdict(counts, n_valid)
        roi_states.append({
            "name":      name,
            "verdict":   verd,
            "votes":     dict(counts),
            "iou_scores": [round(v, 4) for v in iou_log[name]],
            "n_valid":   n_valid,
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
        "Phase2Template: confirmed_on=%d/%d  fail=%s  ambiguous=%s  ghost=%.4f%s",
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
        passed  = False
    else:
        verdict = "PASS"
        passed  = True

    return Phase2TemplateResult(
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
