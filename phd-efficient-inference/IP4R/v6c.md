# IP4R v6c — EfficientNet + YOLO Sandwich Pipeline

> **Author:** Om Kathalkar | **Contributor:** Mohit (external dev)  
> **Date:** 2026-09-15  
> **Status:** Architecture proposed. Forced_15s_Results validated on 2 Aug-28 videos (1 GOOD, 1 NOT GOOD).  
> **Supersedes:** v6a (timing-only EfficientNet). Extends Mohit's 3-phase proposal with EfficientNet in Phase 3.

---

## 0. Model Registry

| Model | Path (tangent server) | Purpose |
|---|---|---|
| **YOLO (Phase 1)** | `~/src/IP4R/data/macro_dataset/runs/macro_test/weights/best.pt` | 5-class segment detector: top_left_block, top_right_block, middle_block, signal_icon, footer_digits |
| **EfficientNet Phase 2** | `~/src/IP4R/models/dts_p2v2_best.pth` | Circuit / ghost-pixel check on YOLO-masked frame |
| **EfficientNet Phase 3** | `~/src/IP4R/models/v6a/best.pth` (candidate — needs eval) | Full 21-icon splash verification on YOLO-anchored crop |

Git branch for this work: **`yolo-detection`**

---

## 1. Background — Why v6a Alone Is Not Enough

The v6a approach (timing-based splash extraction → EfficientNet-B0 binary classifier) showed **good performance on unseen data** but had one structural weakness:

- **Timing is guessed**, not confirmed. It infers the splash window from video duration (e.g. "t=12–16s for ~19s videos"). This drifts across sessions and jig setups.
- **No structural verification.** A video where segments partially lit at t=15s would still produce a crop that EfficientNet scores — EfficientNet has to do all the heavy lifting.

Mohit's proposal fixes the timing problem with YOLO and adds a circuit-cleanliness check. The logical extension is to then use EfficientNet at the YOLO-confirmed timestamp for full-screen verification.

---

## 2. v6a EfficientNet — Summary

### Architecture
```
Video (MP4, 12 fps)
  │
  ▼ Session duration → infer splash window (e.g. t=12–16s)
  ▼ Score frames by global lit-pixel coverage → pick best splash crop
  ▼ EfficientNet-B0 (binary head) → prob_fail
  ▼ PASS if prob_fail < 0.5
```

### Training
| Dataset | Good | Not Good | Total crops |
|---------|------|----------|-------------|
| Jun-27-2026 | 50 vids | 57 vids | — |
| Aug-28-2026 | 14 vids | 50 vids | — |
| **Total harvested** | 64 | 107 | **171 crops** |

- Architecture: EfficientNet-B0, `Linear(1280 → 1)`, sigmoid output
- Training: 5 epochs head-only → 25 epochs full fine-tune, LR 1e-3 → 3e-5, batch 16, CUDA
- **Best val AUC: 0.8077 @ epoch 26** (~31s on RTX 3050)
- Model: `~/src/IP4R/models/v6a/best.pth`

### Eval Results

**Jul-14 (10 videos, 5 GOOD + 5 NOT GOOD) — 5/10 (50%) — UNUSABLE**

| Video | GT | Verdict | prob_fail |
|---|---|---|---|
| 141138 | GOOD | PASS ✓ | 0.313 |
| 141341 | NOT_GOOD | PASS ✗ | 0.333 |
| 141950 | NOT_GOOD | PASS ✗ | 0.324 |
| 142150 | GOOD | PASS ✓ | 0.292 |
| 142516 | NOT_GOOD | PASS ✗ | 0.336 |
| 142739 | GOOD | PASS ✓ | 0.285 |
| 143041 | GOOD | PASS ✓ | 0.385 |
| 143208 | GOOD | PASS ✓ | 0.289 |
| 143426 | NOT_GOOD | PASS ✗ | 0.255 |
| 143604 | NOT_GOOD | PASS ✗ | 0.462 |

All prob_fail < 0.5. GOOD/NOT_GOOD ranges overlap completely — no threshold separates them. Root cause: domain shift (model saw Jun-27 white-jig + Aug-28 dark-jig; Jul-14 jig looks different).

**Sep-09 (13 videos, all GOOD) — 9/13 (69%)**

| Video | GT | Verdict | prob_fail |
|---|---|---|---|
| 200811 | GOOD | FAIL ✗ | 0.593 |
| 201244 | GOOD | FAIL ✗ | 0.647 |
| 201351 | GOOD | FAIL ✗ | 0.535 |
| 201454 | GOOD | PASS ✓ | 0.439 |
| 201608 | GOOD | PASS ✓ | 0.467 |
| 201704 | GOOD | PASS ✓ | 0.418 |
| 201759 | GOOD | PASS ✓ | 0.452 |
| 201854 | GOOD | PASS ✓ | 0.411 |
| 201950 | GOOD | PASS ✓ | 0.443 |
| 202046 | GOOD | PASS ✓ | 0.455 |
| 202145 | GOOD | PASS ✓ | 0.406 |
| 202244 | GOOD | FAIL ✗ | 0.513 |
| 202345 | GOOD | PASS ✓ | 0.476 |

Note: raising threshold to 0.55 recovers 201351 + 202244 → 11/13, but 200811 + 201244 remain wrong.

**On Aug-28 unseen data: performing well** — this is the session the model trained on, so it generalises within-session.

### Root cause of failure
EfficientNet latched onto jig background / lighting characteristics rather than LCD segment patterns. Crops from a new session look visually different → prob_fail distribution shifts.

---

## 3. Mohit's 3-Phase Pipeline — DL + CV Sandwich

### Architecture overview

```
Video (MP4, 12 fps)
  │
  ▼ ─────────────────────────────────────────────────────
  │  PHASE 1: Dynamic Icon Tracking (YOLO)
  │  Run YOLO (best.pt) on every frame
  │  Maintain checklist of 5 moving segments:
  │    [ ] top_left_block   (left clock display: 10:00)
  │    [ ] top_right_block  (right clock display: 10:00)
  │    [ ] middle_block     (temperature: 88)
  │    [ ] signal_icon      (signal bars)
  │    [ ] footer_digits    (bottom strip: 88888)
  │  Each segment is confirmed once its YOLO confidence ≥ threshold
  │  Phase complete ONLY when all 5 are checked [x]
  │  → If checklist never completes: FAIL (segment never lit)
  │  → footer_digits confirmed = splash state reached = timestamp T*
  ▼ ─────────────────────────────────────────────────────
  │  PHASE 2: Circuit & Stray Pixel Check (EfficientNet Masking)
  │  At T*: take the confirmed YOLO bounding boxes
  │  Draw solid WHITE masks over those 5 valid icon regions
  │  → This hides the "good" segments from the classifier
  │  Feed white-masked image to EfficientNet (dts_p2v2_best.pth)
  │  EfficientNet now only sees the LCD BACKGROUND
  │  → PASS: background is clean (no ghost/stray pixels)
  │  → FAIL: background has residual lit regions = circuit short
  ▼ ─────────────────────────────────────────────────────
  │  PHASE 3: Full-Screen 21-Icon Verification
  │  Extract 3 stable frames from T* (t=15s, 16s, 17s)
  │  Run 21-ROI Inspector (rois.yaml) on each frame
  │  Check ALL 21 elements (Auto_Mode → Sleep_Mode)
  │  Coverage + Patch Classification per ROI
  │  → Any ROI below threshold: FAIL + draw red box on that icon
  ▼ ─────────────────────────────────────────────────────
  FINAL VERDICT
  GOOD: Phase 1 checklist complete AND Phase 2 PASS AND Phase 3 all 21 icons OK
  NOT GOOD: Failed at any phase
```

### YOLO checklist — 5 segments

| Segment | LCD Region | All-on appearance |
|---|---|---|
| `top_left_block` | Left clock display | `10:00` (all strokes lit) |
| `top_right_block` | Right clock display | `10:00` (all strokes lit) |
| `middle_block` | Temperature + mode area | `88` (all segments lit) |
| `signal_icon` | Signal bars | All 3 bars filled |
| `footer_digits` | Bottom 5-digit strip | `88888` |

YOLO confidence threshold (suggested): **0.5**. Below this, the checklist item stays unchecked.

### What YOLO failure looks like

From `debug_video@12FPS_20260828_014317` (NOT GOOD unit, Aug-28):
- `middle_block` displayed `78` instead of `88` (missing segments in the temperature digit)
- YOLO confidence for `middle_block` dropped to **0.32** (red box) — below threshold
- Global checklist: `[ ] middle_block: 0.32` — UNCHECKED → Phase 1 FAILS
- Phase 1 catches the defect purely through YOLO confidence degradation on a malformed digit

From `debug_video@12FPS_20260828_011156` (GOOD unit, Aug-28):
- All 5 segments detected at high confidence (0.87–0.91), all green boxes
- One frame showed `top_right_block 0.74` (yellow/red) but the checklist aggregates across frames → checked at 0.90 from a better frame
- Phase 1 passes, Phase 2 passes (EfficientNet 86.1% at t=15s, 75.7% at t=16s)

---

## 4. Proposed Modification — EfficientNet in Phase 3

### The problem with the current Phase 3

The 21-ROI Inspector requires:
1. Registration (ORB+RANSAC+ECC) of the extracted frame to a golden reference
2. Per-element coverage + SSIM thresholds calibrated from training data

Registration is the known fragile step — it fails on ~73% of dark-background frames (Sep-09 issue). Even when it works, the 21 coverage thresholds need per-session recalibration to avoid false alarms.

### The proposal

Replace or supplement Phase 3's 21-ROI inspector with **EfficientNet-B0 on the YOLO-anchored crop**:

```
Phase 3 (proposed):
  Frames extracted at T* (YOLO-confirmed splash timestamp)
  │
  ▼ LCD crop (perspective-corrected 480×640)
  ▼ EfficientNet-B0 (same model as Phase 2 or retrained)
  ▼ prob_fail over 3–4 frames → median → threshold
  ▼ PASS / FAIL
```

### Why this is better than standalone v6a

| | v6a (old) | v6c Phase 3 (proposed) |
|---|---|---|
| Frame selection | Timing guess from video duration | YOLO-confirmed T* (exact splash timestamp) |
| Frame quality | Best-coverage heuristic | YOLO anchor = frame already verified to have segments lit |
| Prior verification | None | Phase 1 confirmed checklist + Phase 2 confirmed clean background |
| Input consistency | Variable (depends on timing drift) | Consistent (YOLO always fires at same LCD state) |
| Domain shift risk | High (timing drift ≠ same frame content) | Lower (anchor is semantically consistent across sessions) |

### Why EfficientNet makes sense here

At T* (after Phase 1 and 2 pass), we know:
- All 5 major segment groups have been confirmed lit at high YOLO confidence
- The background circuit is clean (no ghost pixels)
- The LCD is in the splash state

The remaining question Phase 3 must answer: **are the 16 icon-strip elements and remaining static labels also correctly formed?** This is exactly what EfficientNet trained on splash-state crops can check — it sees the full LCD in its most information-rich state.

### Comparison of Phase 3 options

| Option | Pros | Cons |
|---|---|---|
| **21-ROI Inspector (current)** | Explainable (per-element verdict), no ML training | Requires registration (fragile on dark frames), 21 threshold calibrations |
| **EfficientNet only (proposed)** | Fast, no registration needed, robust to minor alignment shifts | Black box verdict, needs retrain on YOLO-anchored crops |
| **Both in sequence** | Catch what EfficientNet misses with ROI fallback | More complex, slower |
| **EfficientNet + YOLO diff-mask** | EfficientNet on the NON-masked region = only checks what YOLO didn't already verify | Novel, requires more engineering |

**Recommended for v6c:** Run EfficientNet as the primary Phase 3 check. Keep 21-ROI as an optional diagnostic layer (not in the FAIL gate) to localise which icon failed when EfficientNet says FAIL.

---

## 5. v6c Implementation Plan

### Phase 1 — YOLO (already implemented by Mohit)
- Model: `best.pt` (YOLO, 5-class: top_left_block, top_right_block, middle_block, signal_icon, footer_digits)
- Confidence threshold: 0.5
- Checklist logic: per-class running max confidence across all frames
- Completion: footer_digits confirmed → record T* (frame index + timestamp)
- Output: `{checklist: {...}, complete: bool, T_star: float, bboxes: {...}}`

### Phase 2 — EfficientNet on masked frame (already implemented by Mohit)
- Take the frame at T*
- Draw solid white rectangles over the 5 YOLO bounding boxes
- Crop to LCD region (480×640)
- Run `dts_p2v2_best.pth` → prob_pass
- PASS if prob_pass ≥ 0.5
- Output: `{passed: bool, prob_pass: float, masked_frame: path}`

### Phase 3 — EfficientNet on YOLO-anchored full crop (new)

**Goal:** Verify all 21 static icons on the splash frame are correctly formed.  
YOLO (Phase 1) confirmed the 5 major segment blocks. Phase 3 checks the remaining 16 icon-strip elements (Auto_Mode, Cool_Mode, Dry_Mode, Fan_Mode, Heat_Mode, IR_Transmission, Clock, Energy-Save, Lock, Turbo, ion, Battery, Light, H_Swing, V_Swing, Sleep_Mode) plus validates the 5 YOLO blocks at full resolution.

**Why EfficientNet here, not the 21-ROI inspector:**  
The 21-ROI inspector requires per-frame registration (ORB+RANSAC+ECC). Registration fails on ~73% of dark-jig frames (Sep-09 root cause). EfficientNet doesn't need registration — it sees the whole crop and learns what a correct splash looks like holistically.

```
YOLO Phase 1 output
  │
  ├─ T* (frame index of footer_digits confirmation)
  ├─ bboxes {top_left_block, top_right_block, middle_block, signal_icon, footer_digits}
  └─ checklist {all 5 confirmed = True}
           │
           ▼
  Step 3a — Extract 3 frames at T*, T*+12, T*+24  (1s apart at 12fps)
           │
           ▼
  Step 3b — LCD crop via detect_lcd() → 480×640 BGR
           (server_v3/lcd_crop.py — reuse existing)
           │
           ▼
  Step 3c — EfficientNet-B0 inference on each crop
           Model: ~/src/IP4R/models/v6a/best.pth  (first attempt)
           → prob_pass per frame
           │
           ▼
  Step 3d — Aggregate: median(prob_pass) across 3 frames
           PASS if median ≥ 0.5
           FAIL if median < 0.5
           │
           ▼
  Step 3e — On FAIL only: run 21-ROI inspector for localisation
           (diagnostic overlay with red boxes — NOT in FAIL gate)
```

**Step 3c — EfficientNet integration code sketch**

```python
import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
import numpy as np, cv2

TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_phase3_model(weights_path: str, device="cpu"):
    model = efficientnet_b0()
    model.classifier[1] = torch.nn.Linear(1280, 1)
    ckpt = torch.load(weights_path, map_location=device)
    # handle both raw state_dict and {'state_dict': ...} checkpoints
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)

def predict_crop(model, crop_bgr: np.ndarray, device="cpu") -> float:
    """Returns prob_pass (higher = more likely GOOD)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = TRANSFORM(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x)[0, 0]
    return torch.sigmoid(logit).item()   # prob_pass

def phase3_verdict(model, crops: list, threshold=0.5) -> dict:
    probs = [predict_crop(model, c) for c in crops]
    median_prob = float(np.median(probs))
    return {
        "passed": median_prob >= threshold,
        "verdict": "PASS" if median_prob >= threshold else "FAIL",
        "median_prob_pass": median_prob,
        "per_frame_probs": probs,
        "threshold": threshold,
    }
```

**Integration point in the pipeline** (`server_v3/worker.py` or new `server_v6c/pipeline.py`):

```python
# After Phase 1 (YOLO) and Phase 2 (masked EfficientNet) both pass:
if phase1_result["complete"] and phase2_result["passed"]:
    T_star = phase1_result["T_star"]          # frame index
    frames = extract_frames(video, [T_star, T_star+12, T_star+24])
    crops  = [detect_lcd(f) for f in frames]  # 480×640 each
    phase3 = phase3_verdict(phase3_model, crops)

    if not phase3["passed"]:
        # Optional: localise with 21-ROI inspector
        diagnostic = run_roi_inspector(crops[0])
        phase3["roi_diagnostic"] = diagnostic

final_verdict = (
    phase1_result["complete"]
    and phase2_result["passed"]
    and phase3["passed"]
)
```

### Retraining EfficientNet for Phase 3

Current `dts_p2v2_best.pth` was trained on Phase-2-gated crops (all-segments-on frames, detected by the old phase classifier). These are semantically the same as YOLO-anchored frames, so the model may transfer directly.

**To verify transfer:** run `dts_p2v2_best.pth` on the 3 frames extracted at T* from the GOOD video (011156). If prob_pass ≥ 0.85 on all 3, the model transfers. If not, harvest new crops via YOLO anchor and retrain.

**Retrain dataset (if needed):**

| Session | Source | Label | Notes |
|---|---|---|---|
| Jun-27-2026 Good | YOLO-anchored crops at T* | GOOD | Replace timing-guessed crops |
| Jun-27-2026 Not Good | YOLO-anchored crops at T* | NOT GOOD | |
| Aug-28-2026 Good | YOLO-anchored crops at T* | GOOD | |
| Aug-28-2026 Not Good | YOLO-anchored crops at T* | NOT GOOD | |

Do NOT use Jul-14 or Sep-09 for retraining (locked eval sets).

---

## 6. Evidence from Mohit's Forced_15s_Results

**Source:** `/Users/om_kathalkar/Downloads/Forced_15s_Results.zip`  
**Videos tested:** 2 Aug-28 videos (1 GOOD, 1 NOT GOOD)

### Video 011156 — GOOD unit

| Step | Image | Observation |
|---|---|---|
| Phase 1 YOLO | `01_yolo_detections_with_accuracy.jpg` | All 5 segments detected (green boxes). Checklist all [x]: middle_block 0.87, signal_icon 0.91, top_left 0.89, top_right 0.90, footer 0.91 |
| Phase 2 mask | `02_white_masked_screen.jpg` | 5 white rectangles cover all detected segments cleanly |
| Phase 2 EfficientNet t=15s | `03_stage2_cnn_frame_15s.jpg` | **PASS (86.1%)** — all 5 YOLO bboxes shown (green). Background clean |
| Phase 2 EfficientNet t=16s | `03_stage2_cnn_frame_16s.jpg` | **PASS (75.7%)** — consistent |
| Phase 2 EfficientNet t=17s | `03_stage2_cnn_frame_17s.jpg` | PASS |
| Phase 2 EfficientNet t=18s | `03_stage2_cnn_frame_18s.jpg` | PASS |
| Phase 3 crop (t=15s) | `04_stage2_cnn_crop_15s.jpg` | Clean LCD crop, full splash state visible |

**Overall: GOOD → PASS correctly**

### Video 014317 — NOT GOOD unit

| Step | Image | Observation |
|---|---|---|
| Phase 1 YOLO | `01_yolo_detections_with_accuracy.jpg` | `middle_block` detected at **0.32** (red box) — LOW CONFIDENCE. Display shows `78` instead of `88` (missing segments in temperature digit). Checklist: `[ ] middle_block: 0.32` UNCHECKED |
| Phase 1 result | Checklist | **INCOMPLETE → FAIL at Phase 1** |
| Phase 2 mask | `02_white_masked_screen.jpg` | Only 4 white masks (middle_block excluded — not confirmed) |
| Phase 2 EfficientNet t=15s | `03_stage2_cnn_frame_15s.jpg` | **PASS (56.8%)** — YOLO still shows all bboxes for annotation but middle_block confidence was low. Background appears clean |
| Phase 3 crop | `04_stage2_cnn_crop_15s.jpg` | LCD crop shows the defect region (middle_block area is visible as not masked) |

**Overall: NOT GOOD → caught by Phase 1 (YOLO confidence drop on `middle_block: 0.32`)**

**Key insight:** The defect (`78` instead of `88`) was NOT caught by Phase 2 EfficientNet (56.8% = PASS). It was caught purely by YOLO confidence degradation. This confirms Phase 2 EfficientNet is only for ghost/circuit defects, not segment-missing defects. Phase 3 EfficientNet on the full frame would be the right layer to catch segment-level defects that YOLO misses at borderline confidence.

---

## 7. Open Questions for Next Implementation Session

1. **Does `models/v6a/best.pth` transfer to YOLO-anchored Phase 3 crops?**
   - v6a was trained on timing-guessed crops; YOLO-anchored crops are semantically the same state but may differ in exact frame content
   - Test: on tangent server, run `phase3_verdict(model, crops)` on 3 frames from video 011156 at T* → expect prob_pass ≥ 0.85 on all 3
   - If score is poor: harvest new YOLO-anchored crops from Jun-27 + Aug-28 and retrain (crops already cached in `data/v6a_crops/` — fast re-harvest)

2. **YOLO model details (`macro_dataset/runs/macro_test/weights/best.pt`)**
   - Training dataset: `data/macro_dataset/` — need to confirm: number of annotated frames, per-class mAP, confidence threshold used during inference
   - Ask Mohit: what confidence threshold is currently applied per class? Is 0.5 global or per-class?
   - `middle_block 0.32` (defective unit) vs `middle_block 0.87` (GOOD) — margin is large enough for 0.5 threshold

3. **YOLO confidence threshold: 0.5 global or per-class?**
   - `top_right_block 0.74` appeared as a low-confidence (red) box in the GOOD video but the global checklist still showed it as confirmed at 0.90 from a better frame
   - Confirm: checklist uses the **per-class running max** across all frames, not single-frame confidence

4. **How many frames to extract at T* for Phase 3?**
   - 3 frames (T*, T*+12, T*+24 = 1s apart) is sufficient for median voting
   - If `footer_digits` is first detected very early in the splash, T*+24 may still be in the splash state — confirm video length headroom

5. **Sep-09 compatibility with YOLO?**
   - Sep-09 uses a black jig, shorter videos (~19s), Phase-B starts at t=0.0s
   - YOLO should be more robust than timing-based extraction (semantically driven, not time-based)
   - But need to test: does YOLO fire on Sep-09 dark-background frames? The lower contrast may reduce confidence scores

6. **Where is `best.pt` on the tangent server?**
   - Confirmed path: `~/src/IP4R/data/macro_dataset/runs/macro_test/weights/best.pt`
   - Branch: `yolo-detection`

---

## 8. Files Reference

| File | Location (tangent server) | Description |
|---|---|---|
| **`best.pt`** | `~/src/IP4R/data/macro_dataset/runs/macro_test/weights/best.pt` | YOLO Phase 1 model, 5-class: top_left_block, top_right_block, middle_block, signal_icon, footer_digits. Branch: `yolo-detection` |
| `dts_p2v2_best.pth` | `~/src/IP4R/models/dts_p2v2_best.pth` | EfficientNet-B0 Phase 2 (masked circuit check) |
| **`models/v6a/best.pth`** | `~/src/IP4R/models/v6a/best.pth` | EfficientNet Phase 3 candidate — first model to test on YOLO-anchored crops |
| `Forced_15s_Results.zip` | `~/Downloads/Forced_15s_Results.zip` (local Mac) | Mohit's 2-video demo (1 GOOD + 1 NOT GOOD, Aug-28-2026) |
| `server_v6a/` | `~/src/IP4R/server_v6a/` | v6a pipeline code — reuse `lcd_crop.py`, `frame_extract.py` |
| `server_v3/lcd_crop.py` | `~/src/IP4R/server_v3/lcd_crop.py` | LCD crop (perspective-correct to 480×640) — reuse for Phase 3 |
| `data/macro_dataset/` | `~/src/IP4R/data/macro_dataset/` | YOLO training dataset (annotations + images for 5 segment classes) |

---

*Document maintained by Om Kathalkar. Created 2026-09-15.*
