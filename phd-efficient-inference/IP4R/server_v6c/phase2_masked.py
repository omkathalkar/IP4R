"""Phase 2 — Masked EfficientNet Circuit/Ghost-Pixel Check.

Takes the frame at T* (from Phase 1), draws solid white rectangles over the 5
confirmed YOLO bboxes, then runs EfficientNet on the masked LCD crop.

If EfficientNet sees any residual lit content in the background (ghost pixels,
short-circuit artefacts), it returns FAIL. A clean background returns PASS.

Model: models/dts_p2v2_best.pth  (EfficientNet-B0, single sigmoid logit)
Output: prob_pass ∈ [0,1]. PASS if prob_pass >= threshold (default 0.5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.50


@dataclass
class Phase2Result:
    passed: bool
    verdict: str          # 'PASS' | 'FAIL' | 'ABSTAIN'
    prob_pass: float
    threshold: float
    n_masks_applied: int
    error: str | None = None


def _apply_white_masks(
    frame: np.ndarray,
    bboxes: dict[str, list[int]],
) -> np.ndarray:
    """Draw solid white rectangles over all confirmed YOLO bboxes."""
    masked = frame.copy()
    for cls_name, xyxy in bboxes.items():
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(masked, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return masked


def run_phase2(
    video_path: str | Path,
    T_star: int,
    bboxes: dict[str, list[int]],
    model_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    device: str | None = None,
) -> Phase2Result:
    """
    Args:
        video_path : path to video
        T_star     : frame index from Phase 1
        bboxes     : confirmed YOLO bboxes {class_name: [x1,y1,x2,y2]}
        model_path : path to dts_p2v2_best.pth
        threshold  : prob_pass >= threshold → PASS
        device     : 'cuda' | 'cpu' | None (auto)
    """
    import torch
    from torchvision import models, transforms
    from PIL import Image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, 1)
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    net.load_state_dict(ckpt.get("state_dict", ckpt))
    net.eval().to(device)

    _tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Read frame at T* ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, T_star)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return Phase2Result(
            passed=False, verdict="ABSTAIN", prob_pass=0.0,
            threshold=threshold, n_masks_applied=0,
            error=f"cannot_read_frame_{T_star}",
        )

    # ── Mask + crop ───────────────────────────────────────────────────────────
    masked_frame = _apply_white_masks(frame, bboxes)

    from ..server_v6a.lcd_crop import detect_lcd
    crop, _ = detect_lcd(masked_frame)
    if crop is None:
        return Phase2Result(
            passed=False, verdict="ABSTAIN", prob_pass=0.0,
            threshold=threshold, n_masks_applied=len(bboxes),
            error="lcd_crop_failed_on_masked_frame",
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    rgb   = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)
    x     = _tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = net(x).squeeze()
    prob_pass = float(torch.sigmoid(logit).item())

    passed  = prob_pass >= threshold
    verdict = "PASS" if passed else "FAIL"
    log.info("Phase 2: %s (prob_pass=%.3f, masks=%d)", verdict, prob_pass, len(bboxes))

    return Phase2Result(
        passed          = passed,
        verdict         = verdict,
        prob_pass       = round(prob_pass, 4),
        threshold       = threshold,
        n_masks_applied = len(bboxes),
    )
