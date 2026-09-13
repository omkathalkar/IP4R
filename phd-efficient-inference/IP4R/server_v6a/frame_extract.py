"""Timing-based splash frame extraction for v6a.

Session detection — by video duration:
  Jun-27 (25 s): duration > DURATION_SPLIT → splash window [18, 22] s
  Aug-28 (18 s): duration ≤ DURATION_SPLIT → splash window [12, 16] s

Scoring — pick the frame where both digit segments AND icons are most
lit simultaneously. The splash state is the only moment both are on at once.
After CLAHE-equalisation on the crop, active segments are dark (< 128).
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

# ── Session timing ────────────────────────────────────────────────────────────
DURATION_SPLIT = 22.0          # seconds; above = Jun-27, at-or-below = Aug-28
JUN27_WINDOW   = (18.0, 22.0)  # Jun-27 splash window
AUG28_WINDOW   = (12.0, 16.0)  # Aug-28 splash window

# ── Scoring zones on 480×640 crop ────────────────────────────────────────────
# Digit zones (same as server_v3 _SEG_ROIS coords)
_DIGIT_ZONES: list[tuple[int, int, int, int]] = [
    (140, 205,  70, 195),   # left_clock
    (140, 205, 235, 380),   # right_clock
    (215, 305,  70, 190),   # center_88
    (405, 455, 175, 395),   # bottom_88888
]
# Icon strip (present in splash but not in Phase-A / operating mode)
_ICON_ZONE = (90, 140, 85, 400)


def _video_duration(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    n   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return float(n / fps)


def _score_crop(crop_bgr: np.ndarray) -> float:
    """Higher score = more LCD contrast in splash zones = closer to splash state.

    Contrast (std dev) rather than dark-fraction is used because dark-fraction
    incorrectly scores a uniformly-dark (no-display) frame at 1.0.  A real
    splash frame has alternating dark segments and bright background → high std.
    A dark or washed-out frame is low-contrast → low std → low score.

    Uses CLAHE-equalised grayscale so the score is robust to global brightness
    differences between Jun-27 (light jig) and Aug-28 (dark jig).
    Requires both digit zones AND icon zone to be high (harmonic mean) to
    uniquely select the splash frame over operating-mode frames (digits only).
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray).astype(np.float32)

    # Normalised std per zone ∈ [0, 1] (max std for 8-bit bimodal ≈ 128)
    _STD_NORM = 128.0
    digit_stds = [
        float(eq[y1:y2, x1:x2].std() / _STD_NORM)
        for y1, y2, x1, x2 in _DIGIT_ZONES
    ]
    icon_std = float(
        eq[_ICON_ZONE[0]:_ICON_ZONE[1], _ICON_ZONE[2]:_ICON_ZONE[3]].std() / _STD_NORM
    )

    digit_mean = float(np.mean(digit_stds))
    # Harmonic mean: requires both digits AND icons to have high contrast.
    # Pre-splash (digits lit, icons off): low icon_std → low score.
    # Post-splash (operating mode): icon layout changes → lower combined score.
    if digit_mean + icon_std == 0:
        return 0.0
    return float(2.0 * digit_mean * icon_std / (digit_mean + icon_std))


def infer_session(duration: float) -> tuple[str, tuple[float, float]]:
    if duration > DURATION_SPLIT:
        return "jun27", JUN27_WINDOW
    return "aug28", AUG28_WINDOW


def extract_best_splash_frame(
    video_path: str | Path,
    n_candidates: int = 6,
    window_override: tuple[float, float] | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Return (480×640 BGR crop, info) for the best splash-state frame.

    Returns (None, info) with info["error"] set if extraction fails.
    Caller should treat None as ABSTAIN.
    """
    from .lcd_crop import detect_lcd

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, {"error": "cannot_open", "video": str(video_path)}

    fps      = cap.get(cv2.CAP_PROP_FPS) or 12.0
    duration = _video_duration(cap)

    if window_override is not None:
        t_start, t_end = window_override
        session = "manual"
    else:
        session, (t_start, t_end) = infer_session(duration)

    sample_times = np.linspace(t_start, t_end, n_candidates)

    best_crop:  np.ndarray | None = None
    best_score: float             = -1.0
    best_info:  dict              = {}
    n_detected = 0

    for t in sample_times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        crop, crop_info = detect_lcd(frame)
        if crop is None:
            continue
        n_detected += 1

        score = _score_crop(crop)
        if score > best_score:
            best_score = score
            best_crop  = crop.copy()
            best_info  = {
                "t_sec":      float(t),
                "frame_idx":  frame_idx,
                "score":      round(score, 4),
                "duration":   round(duration, 2),
                "session":    session,
                "lcd_method": crop_info.get("method", "unknown"),
                "n_detected": n_detected,
            }

    cap.release()

    if best_crop is None:
        return None, {
            "error":    "no_valid_crop",
            "duration": round(duration, 2),
            "session":  session,
        }

    best_info["n_detected"] = n_detected
    return best_crop, best_info


# ── v6b Method 1: global-coverage candidate extraction ───────────────────────

def _global_coverage(crop_bgr: np.ndarray) -> float:
    """Fraction of lit (dark-segment) pixels in the 480×640 LCD crop after CLAHE.

    Consistent with atlas.compute_normalized_nc: lit = pixel < 128 after CLAHE.
    A true splash frame has a local maximum here because all segments are on.
    A blank or uniformly-bright frame scores near 0.
    """
    gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq    = clahe.apply(gray)
    return float((eq < 128).mean())


def extract_splash_candidates(
    video_path: str | Path,
    n_candidates: int = 3,
    window_override: tuple[float, float] | None = None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Sample ALL frames in the splash timing window; return top n by global coverage.

    Step A: read every frame in the timing window (one per video frame at native FPS).
    Step B: score each detected crop by global lit-pixel coverage; rank descending.
            The true splash state is a local maximum in coverage — self-corrects timing jitter.

    Returns:
        candidates:      list of up to n_candidates 480×640 BGR crops
        candidate_infos: per-candidate metadata dicts
        session_info:    {"session", "duration", "window", "n_frames_sampled",
                          "n_detected", "error" (if no crops found)}
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], {"error": "cannot_open", "video": str(video_path)}

    fps      = cap.get(cv2.CAP_PROP_FPS) or 12.0
    duration = _video_duration(cap)

    if window_override is not None:
        t_start, t_end = window_override
        session = "manual"
    else:
        session, (t_start, t_end) = infer_session(duration)

    # Sample every frame in the window (one-per-frame at native FPS)
    frame_times = np.arange(t_start, t_end, 1.0 / fps)

    scored: list[tuple[float, np.ndarray, dict]] = []
    n_detected = 0

    from .lcd_crop import detect_lcd

    for t in frame_times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        crop, crop_info = detect_lcd(frame)
        if crop is None:
            continue
        n_detected += 1
        # Use std-based harmonic score (penalises transients where icons are off)
        # rather than raw global coverage which can be high during partial-on states.
        score    = _score_crop(crop)
        coverage = _global_coverage(crop)
        info = {
            "t_sec":      round(float(t), 3),
            "frame_idx":  frame_idx,
            "coverage":   round(coverage, 4),
            "score":      round(score, 4),
            "duration":   round(duration, 2),
            "session":    session,
            "lcd_method": crop_info.get("method", "unknown"),
        }
        scored.append((score, crop.copy(), info))

    cap.release()

    session_info = {
        "session":          session,
        "duration":         round(duration, 2),
        "window":           (t_start, t_end),
        "n_frames_sampled": len(frame_times),
        "n_detected":       n_detected,
    }

    if not scored:
        session_info["error"] = "no_valid_crop"
        return [], [], session_info

    # Rank by coverage descending, keep top n_candidates
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:n_candidates]

    candidates     = [c for _, c, _ in top]
    cand_infos     = [i for _, _, i in top]
    return candidates, cand_infos, session_info
