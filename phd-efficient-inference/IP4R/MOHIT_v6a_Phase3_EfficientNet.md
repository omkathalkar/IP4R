# Phase 3 — EfficientNet Full-Screen Verification
### IP4R · Handoff Document for Mohit
**Date:** 2026-09-15  
**Branch:** `yolo-detection`  
**Author:** Om Kathalkar

---

## Context

After your YOLO pipeline (Phase 1) confirms the splash-screen checklist is complete and Phase 2 (masked EfficientNet) confirms the background is clean, we need a **Phase 3** that verifies the full LCD — all 21 icons and all digit zones — are correctly formed in the splash state.

This document describes the EfficientNet-B0 model we have trained for this purpose, how to load it, and exactly where to plug it into your pipeline.

---

## Where It Plugs In

```
Your pipeline                          │  What to add
───────────────────────────────────────┼─────────────────────────────────────
Phase 1: YOLO checklist complete ✓     │
  → T* confirmed (footer_digits fires) │
Phase 2: Masked EfficientNet PASS ✓    │
  → Background clean                   │
                                       │  ← INSERT PHASE 3 HERE
Phase 3: Full-screen EfficientNet      │  Extract crop at T*
  → All 21 icons verified              │  Run EfficientNet-B0
                                       │  PASS / FAIL
Final verdict                          │
```

---

## The Model

| Property | Value |
|---|---|
| Architecture | EfficientNet-B0 (torchvision) |
| Head | `Linear(1280 → 1)` — single logit, sigmoid → prob_pass |
| Weights | `~/src/IP4R/models/v6a/best.pth` |
| Input size | 224 × 224 RGB |
| Normalisation | ImageNet mean/std |
| Output | `prob_pass ∈ [0, 1]` — higher = more likely GOOD |
| Threshold | `prob_pass ≥ 0.5` → PASS |
| Training val AUC | 0.8077 |

### What it was trained on

| Session | Label | Videos |
|---|---|---|
| Jun-27-2026 Good (12-FPS) | GOOD | 50 |
| Jun-27-2026 Not Good (12-FPS) | NOT GOOD | 57 |
| Aug-28-2026 Good (12-FPS) | GOOD | 14 |
| Aug-28-2026 Not Good (12-FPS) | NOT GOOD | 50 |
| **Total crops** | | **171** (64 GOOD, 107 NOT GOOD) |

Each crop is an LCD perspective-corrected frame (480×640 BGR) captured at the splash state.

---

## Full Pipeline After YOLO

```
T* (from YOLO Phase 1)
  │
  ▼
Extract 3 frames:  frame[T*],  frame[T* + 12],  frame[T* + 24]
  (12 fps → every 12 frames = 1 second apart)
  │
  ▼
For each frame → detect_lcd() → 480×640 BGR crop
  (perspective-corrected to remove jig/background)
  │
  ▼
For each crop → EfficientNet-B0 → prob_pass
  │
  ▼
median(prob_pass across 3 frames)
  │
  ├── ≥ 0.5 → PASS ✓
  └──  < 0.5 → FAIL ✗  [optionally run ROI inspector to highlight which icon failed]
```

---

## Code — Drop-in Module

Save this as `phase3_efficientnet.py` in your project:

```python
"""
Phase 3 — EfficientNet Full-Screen Splash Verification
Plug this in after YOLO (Phase 1) and masked EfficientNet (Phase 2) both pass.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
from pathlib import Path


# ── Image transform (must match training) ────────────────────────────────────
_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ── Model loader ──────────────────────────────────────────────────────────────
def load_phase3_model(weights_path: str, device: str = "cpu"):
    """
    Load EfficientNet-B0 Phase 3 model.

    Args:
        weights_path: Path to best.pth on tangent server
                      e.g. '/home/om/src/IP4R/models/v6a/best.pth'
        device: 'cuda', 'cpu', or 'mps'
    Returns:
        model in eval mode
    """
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 1)

    ckpt = torch.load(weights_path, map_location=device)
    # handles both raw state_dict and {'state_dict': ...} checkpoints
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ── Single-crop inference ─────────────────────────────────────────────────────
def predict_crop(model, crop_bgr: np.ndarray, device: str = "cpu") -> float:
    """
    Run EfficientNet on one 480×640 BGR LCD crop.

    Returns:
        prob_pass (float, 0–1). Higher = more likely GOOD.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = _TRANSFORM(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x)[0, 0]
    return torch.sigmoid(logit).item()


# ── Phase 3 verdict ───────────────────────────────────────────────────────────
def phase3_verdict(
    model,
    crops: list,          # list of 480×640 BGR np.ndarray
    threshold: float = 0.5,
    device: str = "cpu",
) -> dict:
    """
    Run Phase 3 on a list of crops extracted around T*.

    Returns dict with:
        passed          (bool)
        verdict         ('PASS' or 'FAIL')
        median_prob_pass (float)
        per_frame_probs  (list of float)
    """
    probs = [predict_crop(model, c, device=device) for c in crops]
    median_prob = float(np.median(probs))
    passed = median_prob >= threshold

    return {
        "passed":            passed,
        "verdict":           "PASS" if passed else "FAIL",
        "median_prob_pass":  round(median_prob, 4),
        "per_frame_probs":   [round(p, 4) for p in probs],
        "threshold":         threshold,
        "n_frames":          len(crops),
    }
```

---

## Integration — How to Call It

```python
from phase3_efficientnet import load_phase3_model, phase3_verdict
from server_v3.lcd_crop import detect_lcd   # existing crop utility

# ── Load once at server startup ───────────────────────────────────────────────
PHASE3_MODEL = load_phase3_model(
    weights_path="/home/om/src/IP4R/models/v6a/best.pth",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# ── Inside your per-video inference function ──────────────────────────────────
def run_full_pipeline(video_frames: list, phase1_result: dict, phase2_result: dict):

    # Only run Phase 3 if Phase 1 and Phase 2 both passed
    if not phase1_result["complete"] or not phase2_result["passed"]:
        return {
            "verdict": "FAIL",
            "failed_at": "phase1" if not phase1_result["complete"] else "phase2",
        }

    # ── Extract 3 frames at T* (1 second apart) ───────────────────────────────
    T_star = phase1_result["T_star"]          # frame index (int)
    FPS = 12                                   # adjust if different
    frame_indices = [T_star, T_star + FPS, T_star + FPS * 2]

    raw_frames = [video_frames[i] for i in frame_indices
                  if i < len(video_frames)]

    # ── Perspective-crop each frame to 480×640 ────────────────────────────────
    crops = []
    for f in raw_frames:
        crop, info = detect_lcd(f)
        if crop is not None:
            crops.append(crop)

    if len(crops) == 0:
        return {"verdict": "ABSTAIN", "reason": "lcd_crop_failed_at_T_star"}

    # ── Phase 3 EfficientNet ──────────────────────────────────────────────────
    phase3 = phase3_verdict(PHASE3_MODEL, crops)

    return {
        "verdict":      phase3["verdict"],
        "phase1":       phase1_result,
        "phase2":       phase2_result,
        "phase3":       phase3,
    }
```

---

## Expected Output

### GOOD unit
```json
{
  "verdict": "PASS",
  "phase1": { "complete": true, "T_star": 180 },
  "phase2": { "passed": true, "prob_pass": 0.861 },
  "phase3": {
    "passed": true,
    "verdict": "PASS",
    "median_prob_pass": 0.892,
    "per_frame_probs": [0.861, 0.892, 0.923],
    "threshold": 0.5,
    "n_frames": 3
  }
}
```

### NOT GOOD unit (segment defect)
```json
{
  "verdict": "FAIL",
  "phase1": { "complete": false, "T_star": null,
              "checklist": { "middle_block": 0.32 } },
  "failed_at": "phase1"
}
```

### NOT GOOD unit (passes YOLO but fails full-screen check)
```json
{
  "verdict": "FAIL",
  "phase3": {
    "passed": false,
    "verdict": "FAIL",
    "median_prob_pass": 0.31,
    "per_frame_probs": [0.28, 0.31, 0.34],
    "threshold": 0.5,
    "n_frames": 3
  }
}
```

---

## Dependencies

```
torch >= 2.0
torchvision >= 0.15
opencv-python >= 4.8
numpy >= 1.24
```

Install:
```bash
pip install torch torchvision opencv-python numpy
```

---

## What the Model Checks

Phase 3 EfficientNet sees the **complete LCD crop** at the splash state and scores whether it looks like a correctly formed all-segments-on display. It catches:

| Defect type | How it catches it |
|---|---|
| Missing icon (Auto, Fan, Battery, Lock, etc.) | Holistic visual difference from GOOD training crops |
| Dim or low-contrast icon | Reduced activation in learned feature map |
| Smeared or blurred segment | Structural difference detected by EfficientNet features |
| Partial digit stroke missing | Visual pattern mismatch vs training GOOD |
| Ghost/stray pixel in icon area | Pixel density anomaly |

> **Note:** YOLO (Phase 1) already catches severely defective digit blocks (e.g. `88` → `78` = confidence drop). Phase 3 EfficientNet is the safety net for subtler defects that YOLO's 5-class detector does not cover — specifically the 16 static icon-strip elements.

---

## Quick Test on Tangent Server

Run this to verify the model loads and gives sensible output on the GOOD video from your demo:

```bash
cd ~/src/IP4R

python3 - <<'EOF'
import cv2, torch, numpy as np
from phase3_efficientnet import load_phase3_model, phase3_verdict
from server_v3.lcd_crop import detect_lcd

model = load_phase3_model("models/v6a/best.pth")

# Load 3 frames from the known-GOOD Aug-28 video at T*≈15s (frame 180 at 12fps)
cap = cv2.VideoCapture("/home/om/src/FDU Dataset/Aug-28-2026/12 FPS/GOOD/video@12FPS_20260828_011156.mp4")

crops = []
for t in [180, 192, 204]:          # T*=180, T*+12, T*+24
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ok, frame = cap.read()
    if ok:
        crop, _ = detect_lcd(frame)
        if crop is not None:
            crops.append(crop)
cap.release()

result = phase3_verdict(model, crops)
print(result)
# Expected: verdict=PASS, median_prob_pass >= 0.80
EOF
```

---

## Threshold Tuning Guide

The default threshold is **0.5**. If you observe false alarms or missed defects after testing on the locked eval sets, adjust as follows:

| Observation | Action |
|---|---|
| GOOD units scoring < 0.5 (false FAIL) | Lower threshold to 0.4 or increase to 3 frames |
| NOT GOOD units scoring > 0.5 (missed) | Raise threshold to 0.6 or retrain with more NOT GOOD crops |
| Consistent bias across a session | Session-specific normalisation or retrain with that session's GOOD data |

> **Do not tune on Jul-14 or Sep-09 eval sets.** These are locked. Tune only on Jun-27 / Aug-28 training data.

---

## Files on Tangent Server

| File | Path |
|---|---|
| Phase 1 YOLO weights | `~/src/IP4R/data/macro_dataset/runs/macro_test/weights/best.pt` |
| Phase 2 EfficientNet weights | `~/src/IP4R/models/dts_p2v2_best.pth` |
| **Phase 3 EfficientNet weights** | `~/src/IP4R/models/v6a/best.pth` |
| LCD crop utility | `~/src/IP4R/server_v3/lcd_crop.py` |
| Phase 3 module (this doc's code) | `~/src/IP4R/server_v6a/phase3_efficientnet.py` (to be created) |

---

*Questions → Om Kathalkar*
