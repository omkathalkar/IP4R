"""Phase 2A — YOLO-based per-element presence check.

Replaces the fill-factor approach (phase2_elements.py) which was brittle due to
brightness domain gaps between calibration (Jul-14) and eval (Sep-15) datasets.

Algorithm:
  1. Extract N_TIMEPOINTS consecutive frames from T*.
  2. Run the element-detection YOLO model on each full frame.
  3. Per element class: DETECTED if any box with confidence >= conf_thr is found.
  4. Persistence vote over N_TIMEPOINTS frames:
       CONFIRMED_MISS if element absent in >= CONFIRM_HITS frames.
  5. FAIL if any element has CONFIRMED_MISS.

The model (yolov8s, 21 classes) is trained on Aug-28 GOOD labeled frames where
every element is annotated. At inference, a missing detection = element absent.

Usage:
    result = run_phase2_yolo_elements(video_path, T_star,
                 model_path='models/elem_yolo/best.pt')
    print(result.verdict, result.confident_miss)

CLI (calibration — not needed, model is trained once):
    No calibration CLI. Train via train_phase2_elem.py.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
N_TIMEPOINTS  = 5
CONFIRM_HITS  = 4      # element absent in >= this many frames → CONFIRMED_MISS
CONF_THR      = 0.25   # YOLO detection confidence threshold

# All 21 element classes (must match classes.txt order used during training)
ELEMENT_NAMES: list[str] = [
    "Auto_Mode", "Battery", "Clock", "Cool_Mode", "Dry_Mode",
    "Energy-Save_Mode", "Fan_Mode", "Fan_Speed", "Foot_Display",
    "H_Swing", "Heat_Mode", "IR_Transmission", "Light", "Lock",
    "Sleep_Mode", "Temperature", "Timer_OFF", "Timer_ON",
    "Turbo", "V_Swing", "ion",
]

# Module-level model cache: (model_path_str) → loaded model
_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_path: str | Path):
    key = str(model_path)
    if key not in _MODEL_CACHE:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        log.info("Loading element YOLO model: %s", key)
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class Phase2YoloElementsResult:
    passed:         bool
    verdict:        str          # PASS | FAIL
    confident_miss: list[str]    # element names with CONFIRMED_MISS
    element_hits:   dict[str, int]   # name → number of frames where detected
    n_timepoints:   int
    frame_indices:  list[int]
    conf_thr:       float
    error:          str | None = None


# ── Main entry point ─────────────────────────────────────────────────────────

def run_phase2_yolo_elements(
    video_path:   str | Path,
    T_star:       int,
    model_path:   str | Path,
    n_timepoints: int   = N_TIMEPOINTS,
    confirm_hits: int   = CONFIRM_HITS,
    conf_thr:     float = CONF_THR,
    imgsz:        int   = 640,
) -> Phase2YoloElementsResult:
    """Run YOLO element check on N_TIMEPOINTS frames starting from T_star.

    Args:
        video_path:   path to video
        T_star:       frame index from Phase 1
        model_path:   path to trained element YOLO best.pt
        n_timepoints: number of consecutive frames to check
        confirm_hits: element absent in this many frames → FAIL
        conf_thr:     YOLO detection confidence threshold
        imgsz:        inference image size
    """
    try:
        model = _get_model(model_path)
    except Exception as exc:
        return Phase2YoloElementsResult(
            passed=False, verdict="FAIL",
            confident_miss=[], element_hits={}, n_timepoints=0,
            frame_indices=[], conf_thr=conf_thr,
            error=f"model_load_error:{exc}",
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return Phase2YoloElementsResult(
            passed=False, verdict="FAIL",
            confident_miss=[], element_hits={}, n_timepoints=0,
            frame_indices=[], conf_thr=conf_thr,
            error=f"cannot_open:{video_path}",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs   = list(range(T_star, min(T_star + n_timepoints, total_frames)))

    # element_hits[name] = number of frames where element was detected
    element_hits: dict[str, int] = {n: 0 for n in ELEMENT_NAMES}
    valid_idxs:   list[int]      = []

    for fidx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        preds = model.predict(
            frame,
            conf=conf_thr,
            imgsz=imgsz,
            verbose=False,
        )

        detected_classes: set[int] = set()
        if preds and len(preds[0].boxes):
            for cls_id in preds[0].boxes.cls.cpu().numpy().astype(int):
                detected_classes.add(int(cls_id))

        for cls_id, name in enumerate(ELEMENT_NAMES):
            if cls_id in detected_classes:
                element_hits[name] += 1

        valid_idxs.append(fidx)

    cap.release()

    n_valid = len(valid_idxs)
    if n_valid == 0:
        return Phase2YoloElementsResult(
            passed=False, verdict="FAIL",
            confident_miss=list(ELEMENT_NAMES), element_hits=element_hits,
            n_timepoints=0, frame_indices=frame_idxs, conf_thr=conf_thr,
            error="no_valid_frames",
        )

    # CONFIRMED_MISS: absent in >= confirm_hits frames
    absent_counts = {name: n_valid - hits for name, hits in element_hits.items()}
    confident_miss = [
        name for name in ELEMENT_NAMES
        if absent_counts[name] >= confirm_hits
    ]

    verdict = "FAIL" if confident_miss else "PASS"
    passed  = not confident_miss

    log.info(
        "Phase2YoloElements: verdict=%s  miss=%s  valid_frames=%d/%d",
        verdict, confident_miss or "[]", n_valid, n_timepoints,
    )

    return Phase2YoloElementsResult(
        passed         = passed,
        verdict        = verdict,
        confident_miss = confident_miss,
        element_hits   = element_hits,
        n_timepoints   = n_valid,
        frame_indices  = valid_idxs,
        conf_thr       = conf_thr,
    )
