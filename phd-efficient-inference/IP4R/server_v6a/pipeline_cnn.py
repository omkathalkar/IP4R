"""v6a inference pipeline: video → PASS / FAIL / ABSTAIN.

  video
    │
    ▼ extract_best_splash_frame()  (timing-based, n=6 candidates)
  best 480×640 LCD crop
    │
    ▼ predict_crop()               (EfficientNet-B0)
  prob_fail ∈ [0, 1]
    │
    ▼ threshold
  PASS / FAIL / ABSTAIN
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .frame_extract import extract_best_splash_frame
from .model import load_model, predict_crop, get_device

# Default FAIL threshold; above this prob_fail → FAIL.
# ABSTAIN zone: [ABSTAIN_LO, ABSTAIN_HI] — model is uncertain.
DEFAULT_THRESHOLD   = 0.50
ABSTAIN_LO          = 0.40
ABSTAIN_HI          = 0.60


def predict_video(
    video_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
    n_candidates: int = 6,
    window_override: tuple[float, float] | None = None,
    use_abstain: bool = False,
) -> dict:
    """Run the v6a pipeline on one video.

    Returns a result dict with keys:
        verdict     str   "PASS" | "FAIL" | "ABSTAIN"
        passed      bool  True iff verdict == "PASS"
        prob_fail   float ∈ [0, 1]; None if ABSTAIN
        frame_info  dict  from extract_best_splash_frame
        error       str | None  set if extraction failed
        inference_ms float
    """
    t0 = time.perf_counter()
    video_path = Path(video_path)

    crop, frame_info = extract_best_splash_frame(
        video_path,
        n_candidates=n_candidates,
        window_override=window_override,
    )

    if crop is None:
        ms = (time.perf_counter() - t0) * 1000
        return {
            "video":        str(video_path),
            "verdict":      "ABSTAIN",
            "passed":       False,
            "prob_fail":    None,
            "frame_info":   frame_info,
            "error":        frame_info.get("error", "no_crop"),
            "inference_ms": round(ms, 1),
        }

    prob_fail = predict_crop(crop, model, device)

    if use_abstain and ABSTAIN_LO < prob_fail < ABSTAIN_HI:
        verdict = "ABSTAIN"
        passed  = False
    elif prob_fail >= threshold:
        verdict = "FAIL"
        passed  = False
    else:
        verdict = "PASS"
        passed  = True

    ms = (time.perf_counter() - t0) * 1000
    return {
        "video":        str(video_path),
        "verdict":      verdict,
        "passed":       passed,
        "prob_fail":    round(prob_fail, 4),
        "threshold":    threshold,
        "frame_info":   frame_info,
        "error":        None,
        "inference_ms": round(ms, 1),
    }


def save_crop_overlay(
    crop_bgr: np.ndarray,
    result: dict,
    out_path: str | Path,
) -> None:
    """Write an annotated overlay JPEG for a single result."""
    img    = crop_bgr.copy()
    H, W   = img.shape[:2]
    verdict = result["verdict"]
    prob    = result.get("prob_fail")
    GREEN, RED, AMBER = (40, 200, 40), (30, 30, 220), (0, 165, 255)
    color   = GREEN if verdict == "PASS" else (AMBER if verdict == "ABSTAIN" else RED)

    cv2.rectangle(img, (0, 0), (W, 44), (0, 0, 0), -1)
    label = f"v6a: {verdict}"
    if prob is not None:
        label += f"  p_fail={prob:.3f}"
    cv2.putText(img, label, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

    fi = result.get("frame_info", {})
    sub = f"t={fi.get('t_sec', '?')}s  score={fi.get('score', '?')}  {fi.get('session', '')}"
    cv2.putText(img, sub, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


class V6aPipeline:
    """Convenience wrapper: load once, call repeatedly."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        device: torch.device | None = None,
        n_candidates: int = 6,
        use_abstain: bool = False,
    ) -> None:
        self.device      = device or get_device()
        self.model       = load_model(checkpoint_path, self.device)
        self.threshold   = threshold
        self.n_candidates = n_candidates
        self.use_abstain = use_abstain

    def predict(
        self,
        video_path: str | Path,
        window_override: tuple[float, float] | None = None,
    ) -> dict:
        return predict_video(
            video_path, self.model, self.device,
            threshold=self.threshold,
            n_candidates=self.n_candidates,
            window_override=window_override,
            use_abstain=self.use_abstain,
        )
