# v6a — Direction B Results & Next Steps

**Date:** 2026-09-13  
**Approach:** Timing-based splash frame extraction → EfficientNet-B0 binary classifier  
**Status:** Deployed as selectable option on production server (port 8083) — 50% Jul-14, 69% Sep-09; not recommended as primary method (use v4c DL or pure CV)

---

## What Was Built (`server_v6a/`)

| File | Role |
|---|---|
| `lcd_crop.py` | Perspective-corrects raw frame → 480×640 LCD crop |
| `frame_extract.py` | Session detection by duration, picks best splash frame from timing window |
| `model.py` | EfficientNet-B0 with binary head; `predict_crop()` → prob_fail |
| `dataset.py` | Harvests/caches best splash crop per training video |
| `train.py` | Two-phase: 5 epochs head-only → 25 epochs full fine-tune, saves best by val AUC |
| `pipeline.py` | `predict_video()` end-to-end; `V6aPipeline` wrapper |
| `eval.py` | Evaluation with fragment-based label matching; hardcoded Jul-14 labels |
| `labels_jul14.json` | Jul-14 ground-truth (both 30-FPS and inferred 12-FPS timestamps) |

---

## Training Run (2026-09-13)

**Dataset:**
- Jun-27-2026 Good 12-FPS: 50 videos
- Jun-27-2026 Not Good 12-FPS: 57 videos
- Aug-28-2026 Good: 14 videos
- Aug-28-2026 Not Good: 50 videos
- **Total crops harvested:** 171 (64 GOOD, 107 NOT_GOOD)
- **Split:** 136 train, 35 val

**Training:** 30 epochs (5 head-only + 25 full fine-tune), LR=1e-3 → 3e-5, batch=16, CUDA  
**Best val AUC:** 0.8077 @ epoch 26  
**Training time:** ~31 seconds on CUDA RTX 3050

**Model artefact:** `~/src/IP4R/models/v6a/best.pth` (tangent server)

---

## Eval Results

### Jul-14 (12-FPS, 10 videos, 5 GOOD + 5 NOT_GOOD) — **5/10 (50%)**

| Video stem | GT | Verdict | prob_fail | Score |
|---|---|---|---|---|
| 141138 | GOOD | PASS ✓ | 0.313 | 0.284 |
| 141341 | NOT_GOOD | PASS ✗ | 0.333 | 0.293 |
| 141950 | NOT_GOOD | PASS ✗ | 0.324 | 0.337 |
| 142150 | GOOD | PASS ✓ | 0.292 | 0.316 |
| 142516 | NOT_GOOD | PASS ✗ | 0.336 | 0.338 |
| 142739 | GOOD | PASS ✓ | 0.285 | 0.494 |
| 143041 | GOOD | PASS ✓ | 0.385 | 0.445 |
| 143208 | GOOD | PASS ✓ | 0.289 | 0.439 |
| 143426 | NOT_GOOD | PASS ✗ | 0.255 | 0.320 |
| 143604 | NOT_GOOD | PASS ✗ | 0.462 | 0.445 |

**Problem:** All 10 prob_fail values < 0.5. GOOD and NOT_GOOD overlap completely (GOOD: 0.285–0.385, NOT_GOOD: 0.255–0.462). No threshold adjustment can fix this.

### Sep-09 (13 videos, all GOOD) — **9/13 (69%)**

| Video stem | GT | Verdict | prob_fail |
|---|---|---|---|
| 200811 | GOOD | **FAIL ✗** | 0.593 |
| 201244 | GOOD | **FAIL ✗** | 0.647 |
| 201351 | GOOD | **FAIL ✗** | 0.535 |
| 201454 | GOOD | PASS ✓ | 0.439 |
| 201608 | GOOD | PASS ✓ | 0.467 |
| 201704 | GOOD | PASS ✓ | 0.418 |
| 201759 | GOOD | PASS ✓ | 0.452 |
| 201854 | GOOD | PASS ✓ | 0.411 |
| 201950 | GOOD | PASS ✓ | 0.443 |
| 202046 | GOOD | PASS ✓ | 0.455 |
| 202145 | GOOD | PASS ✓ | 0.406 |
| 202244 | GOOD | **FAIL ✗** | 0.513 |
| 202345 | GOOD | PASS ✓ | 0.476 |

**Note:** Raising threshold to 0.55 fixes 201351 and 202244 → 11/13. The other two (200811, 201244) stay wrong.

---

## Root Cause

**Domain shift.** The CNN trained on Jun-27 (white jig) + Aug-28 (dark jig) splash crops. Jul-14 crops look different — the model may have latched onto jig background texture or lighting characteristics rather than LCD segment patterns. The GOOD/NOT_GOOD prob_fail ranges overlap completely in Jul-14, meaning the signal the CNN learned doesn't transfer.

Session detection timings are correct:
- Jun-27: ~26s → [18, 22]s window ✓
- Aug-28: ~19.6s → [12, 16]s window ✓
- Jul-14: ~24.5s → [18, 22]s window ✓ (inferred correctly)
- Sep-09: ~19.6s → [12, 16]s window ✓

LCD detection worked on all videos (method: perspective_inner or perspective). Frame scoring (harmonic mean of digit and icon zone contrast after CLAHE) correctly identifies the splash window.

---

## Improvement Options (if returning to v6a)

### Option 1 — Stronger augmentation (quick, try first)
Replace `TRAIN_TRANSFORM` with more aggressive augmentation:
- `RandomGrayscale(p=0.3)` — forces model to learn structure, not color
- `RandomErasing(p=0.3, scale=(0.02, 0.2))` — simulate missing segments as occlusion
- Larger rotation: ±10°
- Stronger ColorJitter: brightness=0.5, contrast=0.5
- `GaussianBlur(kernel_size=3, p=0.3)`

Expected improvement: forces the model to learn LCD segment geometry rather than background appearance.

### Option 2 — Inspect crops (diagnose before training again)
```bash
# On server: look at what the extracted splash crops look like
ls ~/src/IP4R/data/v6a_crops/
# GOOD crops: jun27_good/, aug28_good/
# NOT GOOD crops: jun27_bad/, aug28_bad/
```
Open a few side-by-side (GOOD vs NOT_GOOD from Jun-27, and compare with the Jul-14 crops from the overlay dir) to understand what the defects look like and whether they're visible in the extracted frame.

### Option 3 — Per-ROI coverage (Direction C, no CNN)
Keep timing-based frame extraction but replace the CNN with deterministic per-element threshold checks:
- Register the best splash crop to the golden using `cv2.phaseCorrelate` (single-frame, no video loop)
- Compute `nc(e)` for each of the 21 Atlas ROIs
- Flag FAIL if any element is below `mean − 3σ` from the training GOOD set
- No neural network → no domain shift

This is essentially Tier A applied to a single timing-extracted frame. Much simpler, explicitly interpretable, and directly measures what's wrong.

### Option 4 — Use both frames: splash + main
Extract TWO frames per video: the splash (all-segments-on) and the main display frame (icons active). The main display shows the operating-mode icon (Cool_Mode, etc.) — if it doesn't light up correctly, it's a FAIL. More signal per video.

---

## Rerun Commands (on tangent server)

```bash
cd ~/src/IP4R

# Retrain (crops already cached — fast)
python3 -m server_v6a.train \
  --jun27-good   '/home/om/src/FDU Dataset/June-27-2026/Good DTS Videos FHD 1920x1080/12-FPS' \
  --jun27-bad    '/home/om/src/FDU Dataset/June-27-2026/Not Good DTS Videos FHD 1920x1080/12-FPS' \
  --aug28-good   '/home/om/src/FDU Dataset/Aug-28-2026/12 FPS/GOOD' \
  --aug28-bad    '/home/om/src/FDU Dataset/Aug-28-2026/12 FPS/NOT GOOD' \
  --cache-dir    data/v6a_crops \
  --out-dir      models/v6a \
  --epochs       30

# Eval Jul-14
python3 -m server_v6a.eval \
  --model   models/v6a/best.pth \
  --dataset '/home/om/src/FDU Dataset/July-14-2026/Unseen Test DTS Dataset/12-FPS' \
  --name    Jul-14 --out-dir data/v6a_results

# Eval Sep-09
python3 -m server_v6a.eval \
  --model   models/v6a/best.pth \
  --dataset '/home/om/src/FDU Dataset/Sep-09-2026' \
  --all-good --name Sep-09 --out-dir data/v6a_results
```
