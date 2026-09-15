# IP4R v6a — Fresh Start Briefing

> **Purpose:** Self-contained briefing for a new conversation/agent to design and implement a fresh QC model.  
> **Status:** Starting fresh — v6a supersedes all server_v2 / server_v3 / server_v4c / v08 work.  
> **Problem with current system:** New test data gives PASS for every video, including defective units. The v08 "fix" was a fragile side-effect of SCAN_STEP frame-index timing, not a principled solution.

---

## 1. The Task

Inspect AC-remote LCD panels from factory self-test videos and output **PASS / FAIL** per video.

The factory self-test sequence powers the remote and the LCD enters an **all-segments-on splash state** — every 7-segment digit shows `8`, every icon illuminates simultaneously. A **defective unit** will have one or more elements missing, dim, partially lit, smeared, or absent. The QC system must detect this from a short video clip.

The input is an **MP4 video** recorded by a fixed camera pointed at the remote on a fixed jig. The output is a single **PASS / FAIL** verdict per video.

---

## 2. Dataset Inventory

### 2.1 Training Data

| Session | FPS | Resolution | Duration | Good folder | Not Good folder |
|---------|-----|------------|----------|-------------|-----------------|
| **June-27-2026** | 12 | 1920×1080 | ~25 s | `good/` | `not_good/` |
| **August-28-2026** | 12 | 1920×1080 | ~18 s | `good/` | `not_good/` |

**June-27-2026** — white/light-coloured jig background. Videos are ~25 seconds long.  
**August-28-2026** — dark/black jig background. Remote position is the same as June-27, but the background behind the remote is different. Videos are ~18 seconds long.

Both sessions have clearly labelled `good/` and `not_good/` subfolders.

### 2.2 Test Data (Unseen — Locked, Never Train On)

| Session | Videos | Labels | Notes |
|---------|--------|--------|-------|
| **July-14-2026** | 10 | 5 GOOD + 5 NOT GOOD | Locked eval set |
| **September-09-2026** | 13 | All GOOD | Locked eval set |

These sets have confirmed ground-truth labels. They must **never** be used for training or threshold tuning — only for final evaluation.

---

## 3. Video Structure & Timing

The camera records the remote from power-on through the self-test sequence. The **splash frame** (all-segments-on) is the only frame that matters for QC.

### June-27-2026 video timeline

```
 0 s  ─── power on, display off / boot
~20 s ─── SPLASH frame appears  ← QC target window
~23 s ─── main display activates (shows all icons)
~25 s ─── end of recording
```

**Frame extraction target:** t = 18–22 s (grab frames around the splash window)

### August-28-2026 video timeline

```
 0 s  ─── power on, display off / boot
~14 s ─── SPLASH frame appears  ← QC target window
~15 s ─── splash ends
~17 s ─── main display activates (all icons lit)
~18 s ─── end of recording
```

**Frame extraction target:** t = 12–16 s (grab frames around the splash window)

> **Key insight:** The splash occurs at different absolute times in the two sessions, but the relative structure (splash → main → end) is the same. The simplest approach is to use timing-based frame extraction relative to video duration or to look for the characteristic high-coverage pattern in the correct time window.

---

## 4. The Splash Frame — What We're Looking For

```
┌──────────────────────────────────────┐
│  splash_frame — the QC target        │
│                                      │
│  • All 7-seg digits show "8"         │
│  • All icons lit simultaneously      │
│  • Occurs ≈ 14–15 s (Aug-28) or      │
│    ≈ 20 s (Jun-27) in GOOD videos    │
│                                      │
│  Defective unit:                     │
│  • One or more elements missing,     │
│    dim, partially lit, or smeared    │
└──────────────────────────────────────┘
```

Visible segments in a GOOD splash frame: `18:88 · 18:88 · 88 · |||  · 88888` (clocks, set-temp, signal bars, bottom strip)

---

## 5. The 21 Atlas ROI Elements

The LCD glass contains **21 named regions** that should all be lit in the splash state. These are the "Atlas" — the inspection vocabulary.

| # | Name | Category |
|---|------|----------|
| 1 | Auto_Mode | Icon |
| 2 | Cool_Mode | Icon |
| 3 | Dry_Mode | Icon |
| 4 | Fan_Mode | Icon |
| 5 | Heat_Mode | Icon |
| 6 | IR_Transmission | Icon |
| 7 | Timer_OFF | 7-seg display |
| 8 | Clock | Label/small element |
| 9 | Timer_ON | 7-seg display |
| 10 | Temperature | 7-seg display (set temp "88") |
| 11 | Fan_Speed | Icon / glyph |
| 12 | Energy_Save_Mode | Icon |
| 13 | Lock | Icon |
| 14 | Turbo | Icon/label |
| 15 | ion | Icon |
| 16 | Battery | Icon |
| 17 | Light | Icon |
| 18 | H_Swing | Icon |
| 19 | V_Swing | Icon |
| 20 | Foot_Display | 7-seg strip (bottom "88888") |
| 21 | Sleep_Mode | Icon |

In the current codebase the Atlas is expanded to **111 binary masks** (sub-regions of each of the 21 elements). The feature used for each mask element `e` is `nc(e)` — **normalised coverage**: the fraction of lit pixels in the registered sample frame within that mask region.

---

## 6. Current Pipeline (v08) — What Exists

### 6.1 High-level flow

```
Video (MP4, 12 fps)
  │
  ▼ sample every SCAN_STEP=3 frames
detect_lcd()          perspective-crop LCD to 480×640 BGR
  │
  ▼ per-frame
_classify_phase()     Phase-A (splash: digits lit, icons dark)
                      Phase-B (DTS:    digits lit, icons lit)
  │ Phase-B frames only
  ▼
register()            ORB+RANSAC → ECC (or CLAHE+PhaseCorr for dark bg)
                      warp sample frame onto golden reference
  │ aligned frame
  ▼
feature extraction    nc(e) for each of 111 Atlas masks → 365 features
  │
  ▼
LightGBM             trained on June-27 + August-28 data
                      outputs PASS/FAIL probability per frame
  │ collect per-frame probs
  ▼
verdict               median(p2_probs) >= τ_clf → PASS
                      fail_ratio >= 0.50 → FAIL
                      insufficient evidence → ABSTAIN
```

### 6.2 Registration strategy (register_v6 / v08)

- **Dark background (Aug-28 style):** CLAHE preprocessing → phase correlation (coarse) → ECC affine refinement.
- **Light background (Jun-27 / Jul-14 style):** ORB+RANSAC → ECC (original approach, unmodified).
- Background type detected by `frame_median < 63`.

### 6.3 Trained model artefacts (tangent server)

| File | Description |
|------|-------------|
| `lgb_model.txt` | LightGBM classifier (sandwich_20260907_140534) |
| `lr_model.pkl` / `lr_scaler.pkl` | Logistic regression baseline |
| `dts_p2v2_best.pth` | EfficientNet-B0 Phase-2 model (older layer) |

**Training data used:** June-27-2026 (good: 50 vids, not good: 57 vids) + August-28-2026 (good: 14 vids, not good: 50 vids). Total: ~9,400 rows in training parquet.

### 6.4 Current eval results (before v6a)

| Set | Result | Notes |
|-----|--------|-------|
| Jul-14 (10 vids) | 9/10 | Miss = known hardware defect unit |
| Sep-09 (13 GOOD vids) | 13/13 | **Fragile** — achieved via SCAN_STEP frame-timing side-effect |
| **New test data** | **0/N** effectively | Everything passes, including NOT GOOD units |

---

## 7. Why v08 Fails — Root Causes

### 7.1 The SCAN_STEP hack (fragile Sep-09 "fix")

v08 achieves Sep-09=13/13 because the register_v6 CLAHE change slightly shifted which frames get classified as phase_b. Those frames happened to fall on indices that are **not multiples of SCAN_STEP=3**, so the server skips them → zero phase_b frames detected → default verdict = PASS. This means the model is not actually running on those videos — it is defaulting to PASS due to no evidence. When NOT GOOD videos have phase_b frames that do fall on SCAN_STEP multiples (as the new test data does), they get scored and the fragile LightGBM fails.

### 7.2 Domain gap in training data

- The LightGBM was trained mainly on Jun-27 + Aug-28 data.
- New test sessions have different camera exposure, different lighting, or different LCD characteristics.
- The model never saw these distributions, so its nc(e) features are out-of-distribution → unreliable classification.

### 7.3 Per-frame registration on dim frames

In dark-background sessions, most Phase-B frames are dim (C_ref < 5). ORB finds no keypoints → registration fails → features are garbage → classifier noise. Even with CLAHE+PhaseCorr, registration fails on 73% of dark frames in some videos.

### 7.4 Imbalanced training

Good: 64 videos total. Not Good: ~107 videos total. The model may be biased toward FAIL on uncertain inputs, which now manifests as false PASS on out-of-distribution good-looking frames.

---

## 8. v6a Goals — What the Fresh Model Should Do

### 8.1 Core requirements

1. **Principled splash frame extraction** — use timing + simple brightness/coverage heuristics per dataset, not a fragile phase classifier calibrated on a specific session's exposure.
2. **Register once per video** — find the single best frame (highest coverage, sharpest), register it against the golden, apply that one transform to all candidate frames. Never fail 73% of frames.
3. **Robust to domain shift** — the approach must work on both Jun-27 (light bg) and Aug-28 (dark bg) without per-session parameter tuning.
4. **No reliance on SCAN_STEP timing coincidences** — must produce a genuine verdict from LCD content.
5. **Generalise to new test sessions** — the model must not just memorise Jun-27/Aug-28 distributions.

### 8.2 Dataset split for v6a (immutable)

| Session | Role |
|---------|------|
| June-27-2026 good + not_good | Training |
| August-28-2026 good + not_good | Training |
| July-14-2026 (10 vids, labelled) | **Locked test — never train on** |
| September-09-2026 (13 vids, labelled) | **Locked test — never train on** |

### 8.3 Suggested fresh directions (pick one or combine)

**Direction A — Timing-based frame grab + Tier A CV (no ML)**  
Extract 3–5 frames from the known splash window (e.g., t=18–22 s for Jun-27, t=12–16 s for Aug-28), register the brightest against a golden, compute per-ROI coverage for all 21 Atlas elements, threshold each. Pure CV, no training, no domain gap. Fragile if timing varies.

**Direction B — Timing-based frame grab + lightweight CNN on splash crop**  
Same as A for frame extraction + registration. Then pass the registered LCD crop through a small CNN (MobileNetV3 or EfficientNet-B0) fine-tuned to classify PASS/FAIL on splash crops. Training data: extract one best splash frame per training video → ~170 labelled crops → straightforward binary classification. Robust to lighting via ImageNet normalisation + augmentation. Much simpler than LightGBM on 365 features.

**Direction C — Register-once + per-element threshold (current Atlas, rewritten cleanly)**  
Keep the 21-element Atlas and nc(e) features but fix the pipeline: register once per video (best frame, not per-frame), extract per-element nc(e), apply a simple per-element threshold (learned from training good videos at mean−3σ). No LightGBM, no 365 features. Fails only if the element is genuinely dim/missing.

**Direction D — Optical flow / temporal feature over splash window**  
Detect the splash window by its temporal signature (sudden drop in total coverage as it ends), extract a temporal feature (e.g., mean and std of per-element nc(e) over the splash window), classify with LightGBM. More robust than single-frame.

### 8.4 What to avoid

- Do not re-use the current SCAN_STEP + phase-detect + per-frame-register pipeline — it is the root of all fragility.
- Do not add new per-session thresholds or flags without a clear principled rationale.
- Do not tune thresholds on the locked test sets (Jul-14, Sep-09).

---

## 9. File Layout (v6a will live here)

```
IP4R/
├── CLAUDE_v6a.md           ← this document
├── server_v6a/             ← new package (create fresh, do not copy server_v3)
│   ├── __init__.py
│   ├── frame_extract.py    extract best splash frame(s) from video by timing
│   ├── register.py         register-once (find best frame, warp all candidates)
│   ├── atlas.py            21-element ROI map + nc(e) feature extraction
│   ├── model.py            classifier (CNN or LightGBM — TBD after design)
│   ├── pipeline.py         end-to-end: video → verdict
│   └── eval.py             eval script: run on Jul-14 / Sep-09, print results
├── data/
│   ├── reference/
│   │   ├── golden_splash.jpg   ← best splash frame from a known-good unit
│   │   └── atlas_rois.yaml     ← 21-element normalised ROI coords
│   └── v6a_results/            ← eval output
└── notebooks/
    └── v6a_design.ipynb        ← exploratory / design work
```

---

## 10. Golden Reference for v6a

The golden reference frame for v6a should be:
- A single high-quality splash frame from a known-good unit
- Captured from **both** session types (or use a synthetic clean average)
- Stored as `data/reference/golden_splash.jpg`
- Resolution: 480×640 (post-LCD-crop perspective transform) or native 1920×1080 (pre-crop)

The ROI map (`atlas_rois.yaml`) defines normalised [0,1] coordinates for the 21 Atlas elements on the golden canvas. These coordinates are then used to locate elements in any registered sample frame.

---

## 11. Quick-start Eval Commands (target)

```bash
# After implementing v6a pipeline:
python server_v6a/eval.py \
    --dataset-july14 /path/to/July-14-2026/ \
    --dataset-sep09  /path/to/Sep-09-2026/ \
    --model          data/v6a_model.pkl    \
    --golden         data/reference/golden_splash.jpg

# Expected output:
# Jul-14: 9/10 (or better)
# Sep-09: 13/13 (genuine, not SCAN_STEP artifact)
```

---

*Document maintained by Om Kathalkar. Created 2026-09-13 for v6a fresh-start design.*
