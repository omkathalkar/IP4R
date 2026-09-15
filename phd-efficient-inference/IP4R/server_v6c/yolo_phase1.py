"""Phase 1 — YOLO Dynamic Segment Checklist.

Runs YOLO on every frame within a hard time cap. Uses a sliding-window union
to require that all 5 classes are collectively detected within a short window
(not just once ever), preventing late stray detections from becoming T*.

Fix 1 (2026-09-15): windowed confirmation + hard T* cap.
  - WINDOW_FRAMES=6 (~0.5s): union of classes detected in any frame within window
  - MIN_CONSECUTIVE=3: window must have ≥3 frames before triggering
  - T_STAR_MAX_SEC=5.0: hard cap — splash spec is 2-4s; anything later is mis-anchor
  - FALLBACK_MIN_CLS=4: if 5/5 window not found within cap, accept best 4/5 window

YOLO model: data/macro_dataset/runs/macro_test/weights/best.pt
Classes   : top_left_block, top_right_block, middle_block, signal_icon, footer_digits
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Checklist classes (order must match YOLO model training) ──────────────────
CHECKLIST_CLASSES = [
    "top_left_block",
    "top_right_block",
    "middle_block",
    "signal_icon",
    "footer_digits",
]
_ALL_5 = set(CHECKLIST_CLASSES)

DEFAULT_CONF_THRESHOLD = 0.50   # below this → class not confirmed in frame
MAX_FRAMES_TO_SCAN    = None    # None = scan whole video

# Fix 1 — windowed T* anchoring
WINDOW_FRAMES    = 6     # sliding window size (~0.5s at 12 FPS)
MIN_CONSECUTIVE  = 3     # window must accumulate ≥ this many frames before firing
T_STAR_MAX_SEC   = 5.0   # hard cap: splash spec is 2-4s; T* beyond this → mis-anchor
FALLBACK_MIN_CLS = 4     # accept best 4/5-class window if 5/5 not found within cap


@dataclass
class Phase1Result:
    complete: bool                        # True if T* found (5/5 or 4/5 fallback)
    T_star: int | None                    # frame index when checklist completed
    T_star_sec: float | None             # T* in seconds
    checklist: dict[str, float]           # class → best confidence seen (full scan)
    confirmed: dict[str, bool]            # class → seen ≥ conf_threshold within cap
    bboxes: dict[str, list[int]]          # class → [x1,y1,x2,y2] near T*
    total_frames_scanned: int
    fps: float
    error: str | None = None


def run_yolo_phase1(
    video_path: str | Path,
    model_path: str | Path,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    max_frames: int | None = MAX_FRAMES_TO_SCAN,
) -> Phase1Result:
    """
    Run YOLO on a video with windowed T* anchoring (Fix 1).

    Args:
        video_path    : path to MP4
        model_path    : path to best.pt (YOLO weights)
        conf_threshold: minimum confidence to count a class as present in a frame
        max_frames    : stop scanning after this many frames (None = full video)

    Returns:
        Phase1Result — check .complete and .T_star
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    video_path = Path(video_path)
    model_path = Path(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return Phase1Result(
            complete=False, T_star=None, T_star_sec=None,
            checklist={c: 0.0 for c in CHECKLIST_CLASSES},
            confirmed={c: False for c in CHECKLIST_CLASSES},
            bboxes={}, total_frames_scanned=0, fps=12.0,
            error=f"cannot_open:{video_path}",
        )

    fps         = cap.get(cv2.CAP_PROP_FPS) or 12.0
    total_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_scan      = total_video if max_frames is None else min(max_frames, total_video)
    cap_frames  = min(int(T_STAR_MAX_SEC * fps), n_scan)

    yolo = YOLO(str(model_path))

    # Running best confidence and bbox across full scan
    best_conf: dict[str, float]     = {c: 0.0 for c in CHECKLIST_CLASSES}
    best_bbox: dict[str, list[int]] = {}

    # Per-frame data within the hard cap (for fallback search)
    per_frame: list[tuple[int, set, dict]] = []   # (fidx, detected_set, frame_bboxes)

    window: deque[set] = deque(maxlen=WINDOW_FRAMES)
    T_star:     int | None = None
    T_bboxes:   dict[str, list[int]] = {}
    frame_idx = 0

    # ── First pass: scan within hard cap ──────────────────────────────────────
    while frame_idx < cap_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)[0]
        detected:     set             = set()
        frame_bboxes: dict[str, list] = {}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id >= len(CHECKLIST_CLASSES):
                continue
            cls_name = CHECKLIST_CLASSES[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

            if conf > best_conf[cls_name]:
                best_conf[cls_name] = conf
                best_bbox[cls_name] = xyxy

            if conf >= conf_threshold:
                detected.add(cls_name)
                frame_bboxes[cls_name] = xyxy

        per_frame.append((frame_idx, detected, frame_bboxes))
        window.append(detected)

        union = set().union(*window)
        if union == _ALL_5 and len(window) >= MIN_CONSECUTIVE:
            T_star   = frame_idx
            # Bboxes: best seen so far + anything detected this frame
            T_bboxes = {**best_bbox, **frame_bboxes}
            log.info("Phase 1 complete at frame %d (%.1fs)", T_star, T_star / fps)
            break

        frame_idx += 1

    # ── Continue scanning past cap to update best_conf (no T* assignment) ─────
    if T_star is None:
        while frame_idx < n_scan:
            ret, frame = cap.read()
            if not ret:
                break
            results = yolo(frame, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id >= len(CHECKLIST_CLASSES):
                    continue
                cls_name = CHECKLIST_CLASSES[cls_id]
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                if conf > best_conf[cls_name]:
                    best_conf[cls_name] = conf
                    best_bbox[cls_name] = xyxy
            frame_idx += 1

    cap.release()

    # ── Fallback: find best union window within per_frame ─────────────────────
    fallback_cls = 0
    if T_star is None and per_frame:
        best_n = 0
        best_end = None
        for i in range(len(per_frame)):
            win_start = max(0, i - WINDOW_FRAMES + 1)
            win_union = set().union(*(d for _, d, _ in per_frame[win_start:i + 1]))
            if len(win_union) > best_n:
                best_n   = len(win_union)
                best_end = i

        if best_n >= FALLBACK_MIN_CLS and best_end is not None:
            T_star      = per_frame[best_end][0]
            fallback_cls = best_n
            T_bboxes    = dict(best_bbox)
            log.warning(
                "Phase1 fallback: %d/5 classes in best window, T*=%d (%.1fs)",
                best_n, T_star, T_star / fps,
            )
        else:
            log.warning("Phase1 no valid window (best=%d/5) within %.1fs cap", best_n, T_STAR_MAX_SEC)

    # ── Summarise what was confirmed within the cap scan ──────────────────────
    all_detected_in_cap: set[str] = set()
    for _, d, _ in per_frame:
        all_detected_in_cap.update(d)
    confirmed = {c: c in all_detected_in_cap for c in CHECKLIST_CLASSES}

    complete   = T_star is not None
    T_star_sec = T_star / fps if T_star is not None else None

    if not complete:
        missing = [c for c, v in confirmed.items() if not v]
        log.warning("Phase 1 INCOMPLETE — missing: %s | best_conf: %s", missing, best_conf)
    elif fallback_cls:
        missing = [c for c, v in confirmed.items() if not v]
        log.warning("Phase 1 FALLBACK (%d/5) — unconfirmed in cap: %s", fallback_cls, missing)

    return Phase1Result(
        complete             = complete,
        T_star               = T_star,
        T_star_sec           = T_star_sec,
        checklist            = best_conf,
        confirmed            = confirmed,
        bboxes               = T_bboxes,
        total_frames_scanned = frame_idx + 1,
        fps                  = fps,
    )
