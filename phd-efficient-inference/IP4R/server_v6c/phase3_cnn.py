"""Phase 3 — EfficientNet Full-Screen 21-Icon Verification.

Extracts 3 LCD crops at T*, T*+1s, T*+2s (YOLO-anchored splash frames).
Runs EfficientNet-B0 (v6a/best.pth) on each crop → prob_fail per frame.
Verdict: FAIL if median(prob_fail) > threshold (default 0.5).

This catches:
- Missing static icons (Auto_Mode, Fan_Speed, Battery, Lock, etc.)
- Dim / low-contrast icons
- Partial segment strokes
- Any full-screen defect that YOLO's 5-class detector did not cover

Model: models/v6a/best.pth  (EfficientNet-B0, prob_fail convention)
       prob_fail ∈ [0,1] — higher = more likely NOT GOOD
       FAIL if prob_fail > 0.5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD  = 0.50   # prob_fail > this → FAIL
FRAMES_PER_SECOND  = 12     # video FPS (adjust if different)
N_PHASE3_FRAMES    = 3      # how many frames to sample around T*


@dataclass
class Phase3Result:
    passed: bool
    verdict: str                     # 'PASS' | 'FAIL' | 'ABSTAIN'
    median_prob_fail: float
    per_frame_probs: list[float] = field(default_factory=list)
    frame_indices: list[int]     = field(default_factory=list)
    threshold: float             = DEFAULT_THRESHOLD
    n_crops: int                 = 0
    error: str | None            = None


def _load_phase3_model(model_path: str | Path, device: str):
    import torch
    from torchvision import models

    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 1)
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    net.load_state_dict(ckpt.get("state_dict", ckpt))
    net.eval()
    return net.to(device)


def _predict_crop(model, crop_bgr: np.ndarray, tf, device: str) -> float:
    """Return prob_fail for a single 480×640 BGR crop."""
    import torch
    from PIL import Image

    rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    x    = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze()
    return float(torch.sigmoid(logit).item())


def run_phase3(
    video_path: str | Path,
    T_star: int,
    model_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    n_frames: int = N_PHASE3_FRAMES,
    fps: float = FRAMES_PER_SECOND,
    device: str | None = None,
) -> Phase3Result:
    """
    Args:
        video_path  : path to video
        T_star      : frame index from Phase 1 (YOLO-confirmed splash)
        model_path  : path to models/v6a/best.pth
        threshold   : prob_fail > threshold → FAIL
        n_frames    : number of frames to sample (default 3, 1s apart)
        fps         : video FPS for frame spacing
        device      : 'cuda' | 'cpu' | None (auto)
    """
    import torch
    from torchvision import transforms

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _load_phase3_model(model_path, device)

    _tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    stride       = int(fps)   # 1 second apart
    frame_idxs   = [T_star + i * stride for i in range(n_frames)]

    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return Phase3Result(
            passed=False, verdict="ABSTAIN",
            median_prob_fail=0.0, error=f"cannot_open:{video_path}",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    probs: list[float]    = []
    valid_idxs: list[int] = []

    for fidx in frame_idxs:
        if fidx >= total_frames:
            log.debug("Frame %d beyond video length %d — skipping", fidx, total_frames)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        crop, info = detect_lcd(frame)
        if crop is None:
            log.debug("lcd_crop failed at frame %d", fidx)
            continue

        prob_fail = _predict_crop(model, crop, _tf, device)
        probs.append(prob_fail)
        valid_idxs.append(fidx)
        log.debug("Frame %d: prob_fail=%.4f", fidx, prob_fail)

    cap.release()

    if not probs:
        return Phase3Result(
            passed=False, verdict="ABSTAIN",
            median_prob_fail=0.0, frame_indices=frame_idxs,
            error="no_valid_crops_at_T_star",
        )

    median_pf = float(np.median(probs))
    passed    = median_pf <= threshold
    verdict   = "PASS" if passed else "FAIL"

    log.info(
        "Phase 3: %s  median_prob_fail=%.4f  frames=%s",
        verdict, median_pf, valid_idxs,
    )

    return Phase3Result(
        passed           = passed,
        verdict          = verdict,
        median_prob_fail = round(median_pf, 4),
        per_frame_probs  = [round(p, 4) for p in probs],
        frame_indices    = valid_idxs,
        threshold        = threshold,
        n_crops          = len(probs),
    )
