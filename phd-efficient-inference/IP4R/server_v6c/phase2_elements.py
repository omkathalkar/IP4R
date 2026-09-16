"""Phase 2A — Per-element presence check (primary gate).

Root cause fix for the 32/49 NOT_GOOD false-pass rate: the previous 8-region
SSIM/IoU/ROI checks average over large groups (e.g. all 6 mode icons in one
"top_icons" box), diluting a single missing icon below threshold. This module
tests each of the 28 elements independently.

Algorithm (one video):
  1. Extract 5 consecutive frames from T* (splash is brief; consecutive is correct).
  2. For each frame: detect_lcd() → 480×640 crop → grayscale.
  3. Per element: compute fill_factor = fraction of pixels whose intensity
     differs from the local background collar by > contrast_thr.
     ON   if fill >= fill_thr
     WEAK if fill >= fill_thr * WEAK_FRAC
     MISS otherwise
  4. Persistence vote (same 4/5 rule as other Phase 2 modules).
  5. FAIL  if any element has CONFIDENT_MISS, or >= WEAK_K elements are WEAK.
     PASS  otherwise.

Calibration (run once on Jul-14 GOOD + Sep-09 GOOD):
    python -m server_v6c.phase2_elements \\
        --dataset '/path/Jul-14/GOOD,/path/Sep-09/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --out models/element_thresholds.json

IMPORTANT: calibrate ONLY on Jul-14 + Sep-09 GOOD (18 videos).
           Sep-15 is the locked eval set — never touch during calibration.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .element_rois import ELEMENTS, ElementROI

log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

N_TIMEPOINTS     = 5
CONFIRM_FRAC     = 0.75
CONFIRM_MIN_HITS = 2
CONTRAST_THR     = 15      # intensity delta vs collar median to count as "lit"
COLLAR_PX        = 6       # ring width in pixels around each element ROI
WEAK_FRAC        = 0.6     # fill in [fill_thr*WEAK_FRAC, fill_thr) → WEAK
WEAK_K           = 3       # fail if >= this many elements are WEAK (no hard miss)
FILL_PCT         = 5.0     # percentile of GOOD fill factors used as threshold
FILL_MARGIN      = 0.95    # threshold = percentile * this (small safety margin)


# ---------------------------------------------------------------------------
# Per-element fill-factor computation
# ---------------------------------------------------------------------------

def _element_state(
    gray: np.ndarray,
    elem: ElementROI,
    fill_thr: float,
    contrast_thr: int = CONTRAST_THR,
    collar: int = COLLAR_PX,
) -> tuple[str, float]:
    """Return (state, fill_factor) for one element in one frame.

    fill_factor: fraction of ROI pixels whose intensity deviates from local
    background by more than contrast_thr.  Raw grayscale (no CLAHE) per plan.

    States: "ON" | "WEAK" | "MISS"
    """
    y1, y2, x1, x2 = elem.y1, elem.y2, elem.x1, elem.x2
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return "MISS", 0.0

    cy1 = max(0, y1 - collar)
    cy2 = min(gray.shape[0], y2 + collar)
    cx1 = max(0, x1 - collar)
    cx2 = min(gray.shape[1], x2 + collar)
    outer = gray[cy1:cy2, cx1:cx2]

    collar_mask = np.ones(outer.shape, dtype=bool)
    iy1 = y1 - cy1; iy2 = iy1 + (y2 - y1)
    ix1 = x1 - cx1; ix2 = ix1 + (x2 - x1)
    collar_mask[iy1:iy2, ix1:ix2] = False
    bg_pixels = outer[collar_mask]
    bg_med = float(np.median(bg_pixels)) if bg_pixels.size > 0 else 128.0

    fill = float((np.abs(roi.astype(np.int32) - bg_med) > contrast_thr).mean())

    if fill >= fill_thr:
        return "ON", fill
    if fill >= fill_thr * WEAK_FRAC:
        return "WEAK", fill
    return "MISS", fill


# ---------------------------------------------------------------------------
# Persistence vote
# ---------------------------------------------------------------------------

def _persistence_verdict(counts: dict[str, int], n_valid: int) -> str:
    if n_valid == 0:
        return "AMBIGUOUS"
    thresh = max(CONFIRM_MIN_HITS, math.ceil(n_valid * CONFIRM_FRAC))
    thresh = min(thresh, n_valid)
    pos_hits  = counts.get("ON", 0) + counts.get("WEAK", 0)
    miss_hits = counts.get("MISS", 0)
    if pos_hits >= thresh:
        return "CONFIRMED_ON"
    if miss_hits >= thresh:
        return "CONFIRMED_MISS"
    return "AMBIGUOUS"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Phase2ElementsResult:
    passed:             bool
    verdict:            str            # PASS | FAIL
    element_states:     list[dict]     # per-element name + verdict + votes + scores
    confident_miss:     list[str]      # elements with CONFIRMED_MISS
    weak_elements:      list[str]      # elements in AMBIGUOUS (dim but not confirmed on)
    n_weak:             int
    n_timepoints_valid: int
    frame_indices:      list[int]
    error:              str | None = None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_phase2_elements(
    video_path:      str | Path,
    T_star:          int,
    thresholds_path: str | Path,
    fps:             float = 12.0,
    n_timepoints:    int   = N_TIMEPOINTS,
    contrast_thr:    int   = CONTRAST_THR,
    weak_k:          int   = WEAK_K,
) -> Phase2ElementsResult:
    """
    Args:
        video_path      : path to video file
        T_star          : frame index from Phase 1
        thresholds_path : JSON produced by calibrate_elements()
        fps             : video FPS
        n_timepoints    : consecutive frames to sample from T*
        contrast_thr    : intensity delta vs collar to count as lit
        weak_k          : fail if >= this many AMBIGUOUS elements
    """
    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    try:
        with open(thresholds_path) as f:
            calib = json.load(f)
        thresholds: dict[str, float] = {
            k: v["fill_thr"] for k, v in calib["thresholds"].items()
        }
    except Exception as exc:
        return Phase2ElementsResult(
            passed=False, verdict="FAIL",
            element_states=[], confident_miss=[], weak_elements=[],
            n_weak=0, n_timepoints_valid=0, frame_indices=[],
            error=f"threshold_load_error:{exc}",
        )

    frame_idxs = list(range(T_star, T_star + n_timepoints))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return Phase2ElementsResult(
            passed=False, verdict="FAIL",
            element_states=[], confident_miss=[], weak_elements=[],
            n_weak=0, n_timepoints_valid=0, frame_indices=frame_idxs,
            error=f"cannot_open:{video_path}",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vote_counts: dict[str, dict[str, int]] = {
        e.name: {"ON": 0, "WEAK": 0, "MISS": 0} for e in ELEMENTS
    }
    fill_log: dict[str, list[float]] = {e.name: [] for e in ELEMENTS}
    valid_idxs: list[int] = []

    for fidx in frame_idxs:
        if fidx >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        crop, _ = detect_lcd(frame)
        if crop is None:
            log.debug("Phase2Elements: detect_lcd failed at frame %d — skip", fidx)
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for elem in ELEMENTS:
            thr = thresholds.get(elem.name, 0.10)
            state, fill = _element_state(gray, elem, thr, contrast_thr)
            vote_counts[elem.name][state] += 1
            fill_log[elem.name].append(round(fill, 4))

        valid_idxs.append(fidx)

    cap.release()

    n_valid = len(valid_idxs)
    if n_valid == 0:
        return Phase2ElementsResult(
            passed=False, verdict="FAIL",
            element_states=[], confident_miss=[], weak_elements=[],
            n_weak=0, n_timepoints_valid=0, frame_indices=frame_idxs,
            error="no_valid_lcd_crops",
        )

    element_states: list[dict] = []
    confident_miss: list[str]  = []
    weak_elements:  list[str]  = []

    for elem in ELEMENTS:
        counts = vote_counts[elem.name]
        verd   = _persistence_verdict(counts, n_valid)
        element_states.append({
            "name":        elem.name,
            "group":       elem.group,
            "verdict":     verd,
            "votes":       dict(counts),
            "fill_scores": fill_log[elem.name],
            "fill_thr":    round(thresholds.get(elem.name, 0.0), 4),
        })
        if verd == "CONFIRMED_MISS":
            confident_miss.append(elem.name)
        elif verd == "AMBIGUOUS":
            weak_elements.append(elem.name)

    n_weak = len(weak_elements)
    if confident_miss or n_weak >= weak_k:
        verdict, passed = "FAIL", False
    else:
        verdict, passed = "PASS", True

    log.info(
        "Phase2Elements: verdict=%s  miss=%s  weak=%d(%s)  valid_frames=%d",
        verdict, confident_miss or "[]", n_weak, weak_elements[:3] or "[]", n_valid,
    )

    return Phase2ElementsResult(
        passed             = passed,
        verdict            = verdict,
        element_states     = element_states,
        confident_miss     = confident_miss,
        weak_elements      = weak_elements,
        n_weak             = n_weak,
        n_timepoints_valid = n_valid,
        frame_indices      = valid_idxs,
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_elements(
    dataset_dirs:    list[Path],
    yolo_model_path: Path,
    n_samples:       int   = 20,
    contrast_thr:    int   = CONTRAST_THR,
    fill_pct:        float = FILL_PCT,
    fill_margin:     float = FILL_MARGIN,
    out_path:        Path  = Path("models/element_thresholds.json"),
    domain:          str   = "Jul14+Sep09",
) -> Path:
    """Build per-element fill thresholds from confirmed-GOOD calibration videos.

    IMPORTANT: dataset_dirs must contain ONLY Jul-14 GOOD and Sep-09 GOOD videos.
               Sep-15 is the locked eval set — never include here.
    """
    from .yolo_phase1 import run_yolo_phase1
    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    videos: list[Path] = []
    for d in dataset_dirs:
        videos.extend(sorted(p for p in Path(d).iterdir() if p.suffix.lower() in _VIDEO_EXTS))

    if not videos:
        raise RuntimeError(f"No video files found in {dataset_dirs}")

    log.info("Calibrating %d elements on up to %d/%d videos",
             len(ELEMENTS), n_samples, len(videos))

    fill_scores: dict[str, list[float]] = {e.name: [] for e in ELEMENTS}
    used = 0

    for vid in videos[:n_samples]:
        log.info("  Processing %s …", vid.name)
        p1 = run_yolo_phase1(vid, yolo_model_path)
        if not p1.complete or p1.T_star is None:
            log.warning("    Phase1 FAIL — skipping")
            continue

        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_FRAMES, p1.T_star)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            log.warning("    Cannot read T*=%d — skipping", p1.T_star)
            continue

        crop, _ = detect_lcd(frame)
        if crop is None:
            log.warning("    detect_lcd failed at T*=%d — skipping", p1.T_star)
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for elem in ELEMENTS:
            _, ff = _element_state(gray, elem, fill_thr=0.0, contrast_thr=contrast_thr)
            fill_scores[elem.name].append(ff)
        log.info("    OK  T*=%d (%.1fs)", p1.T_star, p1.T_star_sec)
        used += 1

    if used == 0:
        raise RuntimeError("No valid calibration samples collected")

    sep = "-" * 62
    log.info(sep)
    log.info("  %-25s  %6s  %6s  %6s  %6s  n", "Element", "thr", "p5", "mean", "max")
    log.info(sep)

    thresholds: dict = {}
    for elem in ELEMENTS:
        scores = fill_scores[elem.name]
        if not scores:
            log.warning("  %s: no scores — skipping", elem.name)
            continue
        pct_val = float(np.percentile(scores, fill_pct))
        thr     = pct_val * fill_margin
        thresholds[elem.name] = {
            "fill_thr": round(thr, 4),
            "p5":       round(pct_val, 4),
            "mean":     round(float(np.mean(scores)), 4),
            "max":      round(float(np.max(scores)), 4),
            "n":        len(scores),
            "scores":   [round(s, 4) for s in scores],
        }
        log.info("  %-25s  %.4f  %.4f  %.4f  %.4f  %d",
                 elem.name, thr, pct_val,
                 float(np.mean(scores)), float(np.max(scores)), len(scores))
    log.info(sep)

    out_data = {
        "meta": {
            "n_videos":    used,
            "domain":      domain,
            "contrast_thr": contrast_thr,
            "fill_pct":    fill_pct,
            "fill_margin": fill_margin,
        },
        "thresholds": thresholds,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    log.info("Element thresholds saved → %s  (%d elements, %d videos)", out_path, len(thresholds), used)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate per-element fill thresholds on GOOD videos. "
                    "IMPORTANT: use ONLY Jul-14 GOOD + Sep-09 GOOD (18 videos). "
                    "Sep-15 is the locked eval set."
    )
    ap.add_argument("--dataset",    required=True,
                    help="Comma-separated list of dirs containing GOOD MP4 videos "
                         "(e.g. '/path/Jul-14/GOOD,/path/Sep-09/GOOD')")
    ap.add_argument("--yolo-model", required=True, help="Path to YOLO best.pt")
    ap.add_argument("--n-samples",  type=int,   default=20)
    ap.add_argument("--contrast-thr", type=int, default=CONTRAST_THR)
    ap.add_argument("--fill-pct",   type=float, default=FILL_PCT)
    ap.add_argument("--fill-margin", type=float, default=FILL_MARGIN)
    ap.add_argument("--out",        default="models/element_thresholds.json")
    ap.add_argument("--domain",     default="Jul14+Sep09")
    args = ap.parse_args()

    dirs = [Path(d.strip()) for d in args.dataset.split(",") if d.strip()]
    calibrate_elements(
        dataset_dirs    = dirs,
        yolo_model_path = Path(args.yolo_model),
        n_samples       = args.n_samples,
        contrast_thr    = args.contrast_thr,
        fill_pct        = args.fill_pct,
        fill_margin     = args.fill_margin,
        out_path        = Path(args.out),
        domain          = args.domain,
    )


if __name__ == "__main__":
    main()
