"""Phase 1 — YOLO Dynamic Segment Checklist.

Runs YOLO on every frame. Maintains a per-class running-max confidence
checklist. Phase 1 is complete when all 5 classes are confirmed (conf >= threshold).
T* = frame index at which the checklist becomes 100% complete.

YOLO model: data/macro_dataset/runs/macro_test/weights/best.pt
Classes   : top_left_block, top_right_block, middle_block, signal_icon, footer_digits
"""
from __future__ import annotations

import logging
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

DEFAULT_CONF_THRESHOLD = 0.50   # below this → class not confirmed
MAX_FRAMES_TO_SCAN    = None    # None = scan whole video


@dataclass
class Phase1Result:
    complete: bool                        # True if all 5 classes confirmed
    T_star: int | None                    # frame index when checklist completed
    T_star_sec: float | None             # T* in seconds
    checklist: dict[str, float]           # class → best confidence seen
    confirmed: dict[str, bool]            # class → confirmed?
    bboxes: dict[str, list[int]]          # class → [x1,y1,x2,y2] at T* frame
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
    Run YOLO on a video, build the 5-class checklist, return Phase1Result.

    Args:
        video_path   : path to MP4
        model_path   : path to best.pt (YOLO weights)
        conf_threshold: minimum confidence to confirm a class
        max_frames   : stop early after this many frames (None = full video)

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

    yolo = YOLO(str(model_path))

    # Per-class state
    best_conf: dict[str, float]      = {c: 0.0   for c in CHECKLIST_CLASSES}
    confirmed: dict[str, bool]       = {c: False for c in CHECKLIST_CLASSES}
    best_bbox: dict[str, list[int]]  = {}

    T_star: int | None       = None
    T_star_sec: float | None = None
    frame_idx = 0

    while frame_idx < n_scan:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame, verbose=False)[0]

        frame_bboxes: dict[str, list[int]] = {}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id >= len(CHECKLIST_CLASSES):
                continue
            cls_name = CHECKLIST_CLASSES[cls_id]

            if conf > best_conf[cls_name]:
                best_conf[cls_name] = conf
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                best_bbox[cls_name] = xyxy

            if conf >= conf_threshold:
                confirmed[cls_name] = True
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                frame_bboxes[cls_name] = xyxy

        # Check if checklist just completed this frame
        if T_star is None and all(confirmed.values()):
            T_star     = frame_idx
            T_star_sec = frame_idx / fps
            # Capture bboxes from THIS frame for masking in Phase 2
            best_bbox.update(frame_bboxes)
            log.info("Phase 1 complete at frame %d (%.1fs)", T_star, T_star_sec)
            break   # no need to scan further

        frame_idx += 1

    cap.release()

    complete = all(confirmed.values())
    if not complete:
        missing = [c for c, v in confirmed.items() if not v]
        log.warning("Phase 1 INCOMPLETE — missing: %s | best_conf: %s", missing, best_conf)

    return Phase1Result(
        complete              = complete,
        T_star                = T_star,
        T_star_sec            = T_star_sec,
        checklist             = best_conf,
        confirmed             = confirmed,
        bboxes                = best_bbox,
        total_frames_scanned  = frame_idx + 1,
        fps                   = fps,
    )
