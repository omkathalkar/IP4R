# IP4R — Solution Wiki & Deployment Strategy

> **Project:** IP4R — AC-Remote LCD Splash-Screen Quality Control  
> **Author:** Om Kathalkar  
> **Version:** v03  
> **Last updated:** July 2026  
> **Status:** Production-ready (Tier A) · Tier B optional

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Architecture](#3-solution-architecture)
4. [Technical Deep Dive](#4-technical-deep-dive)
   - 4.1 [Preprocessing](#41-preprocessing)
   - 4.2 [Registration](#42-registration)
   - 4.3 [Per-ROI Inspection (Tier A)](#43-per-roi-inspection-tier-a)
   - 4.4 [Deep Anomaly Detection (Tier B)](#44-deep-anomaly-detection-tier-b)
   - 4.5 [Decision & Reporting](#45-decision--reporting)
5. [ROI Coverage Map](#5-roi-coverage-map)
6. [Performance Metrics](#6-performance-metrics)
7. [Input Modes](#7-input-modes)
8. [Deployment Strategy](#8-deployment-strategy)
   - 8.1 [Option A — Python Direct Install](#81-option-a--python-direct-install)
   - 8.2 [Option B — Docker (Tier A)](#82-option-b--docker-tier-a)
   - 8.3 [Option C — Docker (Tier A+B)](#83-option-c--docker-tier-ab)
   - 8.4 [Deployment Comparison](#84-deployment-comparison)
9. [Configuration Reference](#9-configuration-reference)
10. [Operations Guide](#10-operations-guide)
11. [Roadmap](#11-roadmap)
12. [Troubleshooting](#12-troubleshooting)
13. [Repository Layout](#13-repository-layout)

---

## 1. Executive Summary

IP4R is a **training-free, CPU-real-time** visual quality-control system that inspects AC-remote LCD displays during the all-segments-on ("splash") self-test state. It compares every digit, icon, and label against a single golden-reference photograph and outputs a structured **PASS / FAIL verdict** with exact element-level localisation.

**Key numbers (v03):**

| Metric | Value |
|--------|-------|
| Detection rate | **92.7 %** |
| False-alarm rate | **0.42 %** |
| Calibration dataset | 3,351 good images |
| Elements inspected | 28 ROIs (icons, digits, labels) |
| Inference time | < 1 s per frame (CPU, MacBook Air M2) |
| Dependencies | numpy · opencv · scikit-image · PyYAML |
| GPU required | No (CPU-only; Apple MPS is a bonus for Tier B) |

---

## 2. Problem Statement

### What is being inspected

Every AC remote undergoes a factory self-test during which the LCD enters an **all-segments-on splash state** — every digit segment, icon, and printed label illuminates simultaneously. A defective unit will show one or more elements that are:

- **Missing or unlit** — dead segment, open circuit
- **Partially lit** — single segment stroke missing within a "8" digit
- **Low-contrast or dim** — manufacturing defect or poor contact
- **Blurred or smeared** — contamination, poor bonding

### Why classical vision, not a neural network

The expected appearance is **fully known in advance** — the golden reference image *is* the specification. This is the textbook setting for **golden-template optical inspection**, not open-set recognition. A registration-anchored classical pipeline is:

- Training-free (one reference photo is all that is needed)
- Fully explainable (every verdict traces to a named score)
- CPU-real-time (< 1 s per image on a laptop)
- Robust by design (two orthogonal checks per element)

A deep neural network would require hundreds of labelled defective samples, add GPU dependency, and produce opaque verdicts — none of which are justified when the target appearance is fixed.

Deep anomaly detection (Anomalib PatchCore) is retained as an **optional Tier B** for unforeseen defect types once a good-image bank is accumulated.

---

## 3. Solution Architecture

### Pipeline overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│   Single image · Folder · Live camera · Video file              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  1. PREPROCESS  │  grayscale · denoise · illumination-normalise
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  2. REGISTER    │  ORB (2000 feat) + RANSAC → ECC sub-pixel refine
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │    3. TIER A INSPECTION      │  per-ROI, training-free
              │  ┌───────────────────────┐  │
              │  │ Coverage ratio        │  │  "is it lit / present?"
              │  │ SSIM score            │  │  "is it shaped right / clean?"
              │  │ Diff-map density      │  │  "does structure match?"
              │  │ Segment occupancy     │  │  "which stroke is missing?"  (digits)
              │  │ CNN defect prob       │  │  "feature-level anomaly?"   (digits)
              │  └───────────────────────┘  │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  4. TIER B (optional, off)  │  Anomalib PatchCore / PaDiM
              │  train-on-good · no defects │  "unforeseen anomaly heatmap"
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  5. DECIDE &    │  PASS if all ROIs pass
                    │     REPORT      │  overlay PNG + JSON report
                    └─────────────────┘
```

### Two-tier design rationale

| Tier | Method | Needs | Covers |
|------|--------|-------|--------|
| **A — Deterministic** | Coverage + SSIM + segments | 1 golden image | Known defect types with exact localisation |
| **B — Deep (optional)** | PatchCore / PaDiM | 20–50 good images | Unforeseen or subtle anomalies; pixel heatmap |

Tier A runs always. Tier B activates when a good-bank is available and `tier_b.enabled: true` is set in config. The two verdicts are fused: Tier B can only add a FAIL, never override a Tier A FAIL to PASS.

---

## 4. Technical Deep Dive

### 4.1 Preprocessing

Every image (golden and sample) goes through the same three-step normalisation before any comparison:

1. **Grayscale conversion** — eliminates colour illumination differences; reduces data to what the LCD contrast check actually needs
2. **Denoising** — `cv2.fastNlMeansDenoising` suppresses sensor noise that would inflate SSIM penalty
3. **Illumination normalisation** — CLAHE (contrast-limited adaptive histogram equalisation) so that coverage and SSIM comparisons are robust to ambient lighting shifts between the golden capture and the sample

### 4.2 Registration

Registration is **the make-or-break step**. SSIM and diff-map are not shift-invariant — a 2-pixel translation can collapse an SSIM score even on a perfect unit. The pipeline uses a two-stage strategy:

```
Stage 1 — ORB + RANSAC  (coarse, geometric)
  • 2000 ORB keypoints detected on both images
  • Brute-force Hamming matcher with cross-check
  • findHomography (RANSAC, reproj threshold 5 px, min 12 inlier matches)
  → Handles translation, rotation, scale, minor perspective

Stage 2 — ECC refinement  (fine, sub-pixel)
  • findTransformECC (affine motion model, 100 iterations, ε = 1e-5)
  • Applied to the ORB-warped image, correcting residual misalignment
  → Sub-pixel accuracy, critical for SSIM validity

Fallback: if ORB returns < 12 inlier matches, ECC runs from scratch on
  the resized sample. If ECC also fails, sample is inspected unregistered
  and flagged with  fallback: true  in the report.
```

### 4.3 Per-ROI Inspection (Tier A)

After registration, the sample is scored independently in each of the 28 named ROIs. Five checks run in sequence; any single failure marks the ROI as FAIL.

#### Check 1 — Coverage ratio

```
coverage_sample  =  fraction of "lit" pixels in sample ROI patch
                    (adaptive local threshold, block=31, C=5)

coverage_golden  =  same on the golden ROI patch

PASS  iff  coverage_sample  >=  coverage_golden × coverage_ratio_min
```

`coverage_ratio_min` is **calibrated at 3σ below the mean** of 3,351 good-unit measurements (global default: 0.6749; per-ROI overrides in `rois.yaml`).

The adaptive threshold is **locally relative** — it is robust to global brightness drift and to the reflective-LCD polarity (`active_is_dark: true`: active segments are darker than the background).

#### Check 2 — SSIM score

```
SSIM computed with window size 7 on the registered pair (sample ROI vs golden ROI)

PASS  iff  SSIM  >=  ssim_min  (global 0.1945; per-ROI calibrated floor)
```

SSIM captures **structural similarity** — it penalises blur, smear, low contrast, and ghost segments even when coverage is within range. It is computed per-ROI after registration so that alignment error does not corrupt the score.

#### Check 3 — Diff-map density (optional, off by default)

```
diff  =  |golden_patch − sample_patch|  (absdiff, GaussianBlur k=3)
density  =  fraction of pixels with diff > 20

PASS  iff  density  <=  diff_max  (per-ROI override or global 0.99)
```

Catches structural differences that survive both coverage and SSIM — e.g., a single missing stroke in a digit.

#### Check 4 — Seven-segment occupancy (optional)

For single-digit ROIs whose aspect ratio suggests a seven-segment character, the patch is binarised and each of the 7 canonical segment sub-regions is tested for occupancy. In the all-on state all 7 must be above `on_coverage_min: 0.2`. For multi-digit strip ROIs (`n_digits > 1`), the strip is split per-digit and coverage is checked on each sub-digit.

#### Check 5 — CNN defect probability (digit ROIs)

A lightweight **MNIST-feature scorer** (or optionally a fine-tuned `DigitCNN`) computes a defect probability for each digit patch using cosine distance from a feature embedding of the golden patch.

```
defect_prob  =  1 − cosine_similarity(embed(sample), embed(golden))

PASS  iff  defect_prob  <  defect_threshold  (default 0.45)
```

The threshold 0.45 sits above the maximum observed good-bank score (0.42 on Rmt09/Rmt13 datasets), providing a clean separation.

### 4.4 Deep Anomaly Detection (Tier B)

Tier B is **off by default** and activates only when `tier_b.enabled: true` and a populated `data/good_bank/` exist.

**Model options:** PatchCore (default) · PaDiM

**Workflow:**
1. Train PatchCore on `data/good_bank/` images (coreset memory bank of CNN patch features)
2. At inference: compute nearest-neighbour distance from each patch to the bank → pixel-level anomaly heatmap
3. Threshold heatmap max/mean → anomaly score → FAIL if above `anomaly_threshold`

Tier B catches unforeseen defect types (new product variants, novel contamination patterns) that fall outside the Tier A deterministic checks. It requires no defective training samples.

### 4.5 Decision & Reporting

```python
passed = not any(roi.passed == False for roi in roi_results)
```

The decision is simple: **any ROI failure → image FAIL** (`fail_if_any_roi_fails: true` in config, configurable).

Every inspection writes two artefacts to `data/results/`:

| Artefact | Contents |
|----------|----------|
| `<stem>_overlay.jpg` | Original frame with colour-coded ROI boxes (green=pass, red=fail), verdict banner, fail-element labels |
| `<stem>_report.json` | Structured JSON: verdict, registration info, per-ROI scores (coverage, SSIM, diff, segment fails, CNN prob), fail reasons |

---

## 5. ROI Coverage Map

The system currently inspects **28 named elements** across three categories.

### Icons & Glyphs (16 elements)

| ROI name | Description |
|----------|-------------|
| `icon_display_frame` | Display border frame |
| `icon_snowflake` | Cool / snowflake mode icon |
| `icon_waterdrop` | Dry / water-drop mode icon |
| `icon_fan` | 3-blade fan icon |
| `icon_heat` | Heat / sunburst mode icon |
| `icon_wifi` | Wi-Fi connectivity icon |
| `glyph_pie` | Pie / half-moon glyph (set-temp area) |
| `glyph_auto_fan` | Auto-fan glyph |
| `glyph_signal_bars` | Signal strength bars |
| `icon_econo_e` | Circled-E (Econo mode) |
| `icon_lock` | Child-lock icon |
| `icon_ion` | Ion / plasma icon |
| `icon_battery` | Remote battery indicator frame |
| `icon_bulb` | Light-bulb icon |
| `glyph_sparkle` | Sparkle / clean glyph |
| `glyph_swing_right` | Swing / louver direction glyph |

### Seven-Segment Digits (5 ROIs)

| ROI name | Content | n_digits |
|----------|---------|---------|
| `clock_time_off` | `10:00` timer-off display | 4 |
| `clock_time_on` | `10:00` timer-on display | 4 |
| `digit_settemp_tens` | Set-temperature tens digit (`8`) | 1 |
| `digit_settemp_units` | Set-temperature units digit (`8`) | 1 |
| `digit_bottom_88888` | Bottom 5-digit strip (`88888`) | 5 |

### Labels (7 elements)

| ROI name | Printed text |
|----------|-------------|
| `label_clock` | `CLOCK` |
| `text_time_off` | `TIME OFF` |
| `text_time_on` | `TIME ON` |
| `label_celsius` | `°C` |
| `label_set_temp` | `SET TEMP` |
| `label_auto` | `AUTO` |
| `label_turbo` | `TURBO` |

> **Note:** ROI coordinates are stored in normalised [0, 1] space (`rois.yaml`) and survive resolution changes without re-authoring.

---

## 6. Performance Metrics

### Calibration basis

| Dataset | Images | Split |
|---------|--------|-------|
| Rmt 06, 08–13 | 3,351 | Good units (threshold calibration) |
| Fault images | Separate zips | Known-bad units (detection validation) |

### Detection performance (v03)

| Metric | Value | Calibration method |
|--------|-------|-------------------|
| Detection rate | **92.7 %** | Fraction of known-bad units correctly flagged |
| False-alarm rate | **0.42 %** | Fraction of known-good units incorrectly flagged |
| Coverage threshold | 0.6749 | Mean − 3σ from 3,351 good-unit measurements |
| SSIM threshold | 0.1945 | Mean − 3σ per-ROI calibrated floor |
| CNN threshold | 0.45 | Above Rmt09/Rmt13 good-bank max (0.42) |

### Inference latency (approximate)

| Platform | Mode | Time per image |
|----------|------|---------------|
| MacBook Air M2 (CPU) | Single image | ~0.4 s |
| MacBook Air M2 (CPU) | Batch (4 workers) | ~0.15 s per image |
| Live camera | Per frame (fast coverage) | ~30 ms |
| Live camera | Full inspection (on trigger) | ~0.4 s |

---

## 7. Input Modes

### Summary table

| Mode | Command | When to use |
|------|---------|-------------|
| Sanity check | `ip4r selfcheck` | First run, verifying install |
| Single image | `ip4r inspect photo.jpg` | One-off offline inspection |
| Folder batch | `ip4r inspect data/samples/` | Batch of saved images |
| Live camera | `ip4r camera` | Real-time production-line QC |
| Live camera (USB) | `ip4r camera --camera 1` | External USB camera |
| Video file | `python scripts/inspect_video.py video.mp4` | Recorded splash sequence |
| Parallel batch | `python scripts/inspect_batch.py data/samples/` | High-volume offline batch (4 workers) |

### Live camera workflow

```
Camera feed  →  fast coverage check every frame (≈30 ms)
                        ↓
              N consecutive frames all pass?  (default N=3)
                        ↓  YES
              Full inspection fires on trigger frame  (≈0.4 s)
                        ↓
              PASS / FAIL banner + ROI overlay displayed
              Result saved to  data/results/
                        ↓
              Press [r] to reset → scan next unit
```

**Camera window controls:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Force-inspect the current frame immediately |
| `r` | Reset — scan again for the next unit |
| `SPACE` | Save current raw frame to `data/samples/` |

---

## 8. Deployment Strategy

### 8.1 Option A — Python Direct Install

**Best for:** development workstations, testing laptops, small-scale line deployment.

#### Prerequisites

- Python 3.11 or newer
- pip
- A camera (built-in or USB) for live mode

#### Install steps

```bash
# 1. Extract the deployment zip
unzip ip4r_deployment_v03.zip
cd deployment

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

# 3. Install the package
pip install -e .

# 4. Verify
ip4r selfcheck
```

Expected output of selfcheck:
```
=== IP4R PASS : lcd_all_on.jpg ===
registration: {'method': 'orb_ecc', 'fallback': False}
  [ok ] icon_display_frame   cov=0.xx   ssim=0.xx
  ...
  -> 0/28 element(s) failed
```

#### Optional Tier B (heavy, pulls PyTorch)

```bash
pip install -e ".[dl]"
```

This adds PyTorch (CPU wheel) + Anomalib. Only needed if Tier B deep anomaly detection is enabled.

---

### 8.2 Option B — Docker (Tier A)

**Best for:** production-line deployment, containerised CI, systems where Python environment management is not desired.

**Image size:** ~500 MB  
**Dependencies inside container:** numpy · opencv-headless · scikit-image · PyYAML  
**GPU:** not required

#### Build

```bash
# From the deployment/ root
docker compose -f deploy/docker-compose.yml build tier-a
```

#### Run

```bash
# Sanity check
docker compose -f deploy/docker-compose.yml run --rm tier-a selfcheck

# Inspect a single image
docker compose -f deploy/docker-compose.yml run --rm tier-a inspect data/samples/photo.jpg

# Inspect an entire folder
docker compose -f deploy/docker-compose.yml run --rm tier-a inspect data/samples/

# Synthesise a defective sample and inspect it
docker compose -f deploy/docker-compose.yml run --rm tier-a synth --mode erase --out data/samples/defect_01.jpg
docker compose -f deploy/docker-compose.yml run --rm tier-a inspect data/samples/defect_01.jpg
```

#### Volume mounts (from docker-compose.yml)

```
Host path               Container path          Direction
─────────────────────   ─────────────────────   ─────────
data/samples/      →    /app/data/samples/      Input (put images here on host)
data/results/      ←    /app/data/results/      Output (overlays + JSON land here)
```

Files baked into the image at build time (not mounted):
- `config/default.yaml`
- `data/reference/lcd_all_on.jpg` (golden spec)
- `data/reference/rois.yaml`
- `models/digit_cnn.pt`

#### Always-on deployment (add to docker-compose.yml)

```yaml
services:
  tier-a:
    restart: unless-stopped
    # ...rest of existing config
```

---

### 8.3 Option C — Docker (Tier A+B)

**Best for:** production line where good-bank training is complete and Tier B is enabled.

**Image size:** ~3 GB  
**Additional dependencies:** PyTorch (CPU wheel) · Anomalib · Transformers

#### Build

```bash
docker compose -f deploy/docker-compose.yml build tier-ab
```

#### Enable Tier B

1. Populate `data/good_bank/` with 20–50 photos of known-good units
2. Create `config/tier_ab.yaml`:

```yaml
tier_b:
  enabled: true
  model: patchcore
  anomaly_threshold: null   # set after training
```

3. In `deploy/docker-compose.yml`, uncomment the config volume line under `tier-ab`:

```yaml
volumes:
  - ../data/samples:/app/data/samples
  - ../data/results:/app/data/results
  - ../data/good_bank:/app/data/good_bank
  - ../config/tier_ab.yaml:/app/config/default.yaml   # ← uncomment this
```

4. Rebuild and run:

```bash
docker compose -f deploy/docker-compose.yml build tier-ab
docker compose -f deploy/docker-compose.yml run --rm tier-ab inspect data/samples/
```

---

### 8.4 Deployment Comparison

| Criterion | Option A (Python) | Option B (Docker Tier A) | Option C (Docker Tier A+B) |
|-----------|:-----------------:|:------------------------:|:---------------------------:|
| Setup effort | Low | Medium | Medium–High |
| Image size | ~200 MB (venv) | ~500 MB | ~3 GB |
| GPU needed | No | No | No |
| Live camera support | Yes | Needs `--device` flag | Needs `--device` flag |
| Tier B support | Yes (`pip install -e ".[dl]"`) | No | Yes |
| Reproducibility | Depends on host Python | High | High |
| Production isolation | No | Yes | Yes |
| Recommended for | Dev / testing | Factory line (Tier A only) | Factory line (full system) |

> **Recommendation:** Use Option A for the testing team. Use Option B for production-line deployment once testing passes.

---

## 9. Configuration Reference

All thresholds live in `config/default.yaml`. No values are hard-coded in the pipeline.

```yaml
paths:
  reference_image: data/reference/lcd_all_on.jpg
  roi_map:         data/reference/rois.yaml
  samples_dir:     data/samples
  results_dir:     data/results

preprocess:
  to_gray: true
  denoise: true
  illumination_normalize: true

registration:
  enabled: true
  method: orb_ecc           # orb_ecc | orb_only | ecc_only | none
  orb_features: 2000
  ransac_reproj_thresh: 5.0
  min_matches: 12
  ecc_iterations: 100
  ecc_eps: 1.0e-05

tier_a:
  enabled: true
  roi_kinds: [digit]        # kinds to inspect: digit | icon | label (or all)
  coverage:
    enabled: true
    active_is_dark: true    # true = reflective LCD; false = emissive/LED
    coverage_ratio_min: 0.6749   # calibrated 3σ floor (3351 good images)
    adaptive_block: 31
    adaptive_C: 5
  ssim:
    enabled: true
    ssim_min: 0.1945         # calibrated 3σ floor
    win_size: 7
  diff:
    enabled: true            # set false to skip diff-map check
    max_density: 0.99
    blur_k: 3
    pixel_threshold: 20
  segments:
    enabled: true
    on_coverage_min: 0.2
    max_aspect_ratio: 1.5
  cnn:
    model_type: mnist_hf     # mnist_hf (no training) | digit_cnn (trained weights)
    model_path: models/digit_cnn.pt
    defect_threshold: 0.45

tier_b:
  enabled: false             # set true once good_bank is populated
  model: patchcore           # patchcore | padim
  good_images_dir: data/good_bank
  image_size: 256
  anomaly_threshold: null    # set after training on your good-bank

decision:
  fail_if_any_roi_fails: true

report:
  save_overlay: true
  save_json: true
  overlay_pass_color: [0, 180, 0]    # green
  overlay_fail_color: [0, 0, 230]    # red
```

### Per-ROI overrides (in rois.yaml)

Individual ROIs can override global thresholds:

```yaml
- name: digit_settemp_tens
  kind: digit
  ssim_min: -1.0          # disable SSIM for this ROI
  diff_max: 0.99          # relax diff check
  cov_ratio_min: 0.90     # stricter coverage (large digit)
```

Setting `ssim_min: -1.0` effectively disables the SSIM check for that ROI.

---

## 10. Operations Guide

### Adding a new ROI

```bash
ip4r roi-edit
```

Opens an interactive OpenCV window on the golden image. Draw a box, type the name, press `s` to save. ROIs are appended to `data/reference/rois.yaml` automatically.

Naming convention: `<kind>_<descriptor>` — e.g., `icon_snowflake`, `digit_settemp_tens`, `label_turbo`.

### Calibrating thresholds on good units

Once 20–50 good-unit photos are collected:

```bash
python scripts/calibrate.py data/good_bank/ --out config/calibrated.yaml
ip4r --config config/calibrated.yaml inspect data/samples/
```

The calibrator computes per-ROI `mean − k·σ` floors from the good-bank and writes them into `rois.yaml`. This is **the single biggest reliability improvement** available.

### Rebuilding the golden reference

If the camera position or lighting changes, recapture the golden:

1. Place a known-perfect unit and enter splash state
2. Capture several images and compute their mean: `python scripts/calibrate.py --make-golden`
3. Copy the output to `data/reference/lcd_all_on.jpg`
4. Re-run `ip4r roi-edit` to adjust any ROI coordinates that drifted

### Reading batch results

```bash
python scripts/inspect_batch.py data/samples/ --out data/results/run_01
python -c "
import json
d = json.load(open('data/results/run_01/summary.json'))
print(f'Total: {d[\"total\"]}  PASS: {d[\"passed\"]}  FAIL: {d[\"failed\"]}')
"
```

### Interpreting the JSON report

```json
{
  "image_path": "data/samples/unit_042.jpg",
  "passed": false,
  "n_rois": 28,
  "n_failed": 1,
  "registration": {
    "method": "orb_ecc",
    "fallback": false         ← true means registration failed; check lighting
  },
  "rois": [
    {
      "name": "digit_settemp_tens",
      "passed": false,
      "coverage": 0.12,
      "coverage_golden": 0.71,
      "coverage_pass": false,
      "ssim": 0.61,
      "ssim_pass": true,
      "reason": "coverage 0.12 < 0.60 (golden 0.71)"  ← human-readable cause
    }
  ]
}
```

---

## 11. Roadmap

| ID | Task | Priority | Status |
|----|------|----------|--------|
| T1 | Extend ROI map to all LCD elements (currently 28 of ~40+) | High | Open |
| T2 | Capture 20–50 good units; calibrate per-ROI thresholds to `mean − k·σ` | **Critical** | Open |
| T3 | Per-segment ROIs for all seven-seg digit ROIs; enable `segments.py` | Medium | Open |
| T4 | Harden registration: glare/reflection masks; ECC-only fallback improvement | Medium | Open |
| T5 | Wire Tier B: train Anomalib PatchCore on good-bank; fuse heatmap verdict | Low | Open |
| T6 | Batch mode CSV export; latency benchmark on production hardware | Low | Open |
| T7 | Rebuild golden as mean of several good captures (kills single-photo glare bias) | High | Open |

**The highest-leverage action is T2** — data-driven threshold calibration from real good units. The current 3σ floor was computed on the dataset available; expanding the calibration set tightens the operating window and directly improves both detection rate and false-alarm rate.

---

## 12. Troubleshooting

### FAILs on known-good units

**Symptom:** Good remote consistently fails 1–2 ROIs.  
**Likely cause:** Registration jitter — check the JSON report for `"fallback": true`.  
**Fix:** Ensure consistent lighting; avoid specular glare on the LCD; hold/mount the remote at a fixed distance and angle. If camera angle changed since golden capture, recapture the golden.

### Very low SSIM on all ROIs

**Symptom:** Every element shows `ssim_pass: false`, even on a good unit.  
**Likely cause:** Golden was captured at a different exposure or with different preprocessing settings.  
**Fix:** Recapture the golden under current conditions; alternatively, lower `ssim_min` temporarily and re-calibrate from good units (T2).

### Camera never auto-triggers

**Symptom:** Live camera shows `SCANNING…` indefinitely.  
**Fix:**  
1. Press `s` to force-inspect and check the coverage values in the status bar
2. If coverage values are near zero, the polarity may be wrong — set `active_is_dark: false` if your LCD has bright-on-dark segments
3. Check that the remote is in the splash state (all-on), not a normal display state
4. Reduce `--stable-frames` to 1 temporarily to see if inspection triggers

### Cannot open camera

```
RuntimeError: Cannot open camera index 0
```

Try: `ip4r camera --camera 1` or `--camera 2`. On macOS, camera 0 is usually the built-in FaceTime camera; USB cameras appear at higher indices.

### `fallback: true` in every report

ORB feature matching is failing — the sample and golden have too few shared features.  
**Causes:** motion blur, severe glare, very small LCD in frame, low contrast.  
**Fix:** Improve lighting; ensure the LCD fills at least 30% of the frame; reduce glare with diffuse lighting or a polarising filter.

### Wrong polarity (icons seem inverted)

If your LCD has **bright segments on a dark background** (emissive / LED panel):

```yaml
# config/default.yaml
tier_a:
  coverage:
    active_is_dark: false
```

---

## 13. Repository Layout

```
IP4R/
├── README.md                    Human quickstart
├── IP4R_WIKI.md                 ← this document
├── TESTING_GUIDE.md             Step-by-step test plan for testing team
├── CLAUDE.md                    Notes for Claude Code (internal)
├── pyproject.toml               Package definition + dependencies
│
├── config/
│   └── default.yaml             All thresholds and pipeline switches
│
├── data/
│   ├── reference/
│   │   ├── lcd_all_on.jpg       Golden reference image (the spec)
│   │   ├── lcd_all_on_original.jpg
│   │   └── rois.yaml            28-element ROI map (normalised coords)
│   ├── samples/                 Put input images here
│   ├── results/                 Overlay PNGs + JSON reports land here
│   └── good_bank/               For Tier B training (populate before enabling)
│
├── models/
│   └── digit_cnn.pt             Trained digit CNN weights (~762 KB)
│
├── scripts/
│   ├── inspect_camera.py        Live camera inspection (splash-triggered)
│   ├── inspect_video.py         Video file inspection (splash-triggered)
│   ├── inspect_batch.py         Parallel batch inspection (4 workers)
│   └── calibrate.py             Threshold calibration from good-bank images
│
├── src/ip4r/                    Python package
│   ├── cli.py                   ip4r CLI (selfcheck/inspect/camera/synth/roi-edit)
│   ├── pipeline.py              Inspector class — orchestrates 0→4
│   ├── preprocess.py            Grayscale · denoise · illumination normalise
│   ├── registration.py          ORB+RANSAC homography + ECC sub-pixel refinement
│   ├── inspect.py               Per-ROI coverage + SSIM + diff + segment + CNN
│   ├── segments.py              Seven-segment per-stroke occupancy
│   ├── report.py                Annotated overlay + JSON report writer
│   ├── roi.py                   ROI dataclasses + YAML load/save
│   ├── config.py                YAML config loader/validator
│   ├── synth.py                 Synthetic defect generator (erase/dim/blur/shift)
│   ├── digit_cnn.py             DigitCNN scorer (trained weights path)
│   ├── mnist_scorer.py          MNIST feature-similarity scorer (no training)
│   └── tools/roi_editor.py      Interactive OpenCV ROI authoring tool
│
├── deploy/
│   ├── Dockerfile.tier-a        Lightweight image (~500 MB, no PyTorch)
│   ├── Dockerfile.tier-ab       Full image (~3 GB, with PyTorch + Anomalib)
│   └── docker-compose.yml       Compose file for both tiers
│
├── deployment/                  Packaged release folder
├── ip4r_deployment_v03.zip      Shareable deployment archive (805 KB)
│
├── IP4R_dataset/                Raw dataset zips (Rmt 06–13 + fault images)
└── tests/
    └── test_pipeline.py         Pytest smoke tests
```

---

*This document is maintained by Om Kathalkar. For questions, contact via the CVIT lab or the IP4R project channel.*
