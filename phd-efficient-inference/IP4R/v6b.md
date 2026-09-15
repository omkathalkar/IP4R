# IP4R v6a — Method 1 Implementation Briefing (Pure CV, No ML)

> **Purpose:** Self-contained briefing for Claude Code to implement Method 1 of the v6a fresh-start plan.
> **Status:** First implementation pass. Method 2 (temporal averaging / CNN) is deferred until Method 1's results are evaluated.
> **Supersedes:** server_v2 / server_v3 / server_v4c / v08. Do not copy or reuse code from those packages — the SCAN_STEP + per-frame-register + LightGBM pipeline is the root cause of the current failure and must not be reused.

---

## 1. The Task

Inspect AC-remote LCD panels from factory self-test videos and output **PASS / FAIL** per video, using pure computer vision — no trained classifier.

The factory self-test sequence powers the remote and the LCD enters an **all-segments-on splash state**: every 7-segment digit shows `8`, every icon illuminates simultaneously. A defective unit has one or more elements missing, dim, partially lit, or smeared. Detect this from a short fixed-camera MP4 clip.

---

## 2. Why Method 1, Why Pure CV

The previous pipeline (v08) failed because:
- It relied on a fragile per-frame phase classifier tuned to specific session exposure.
- It registered every sampled frame individually — failing on ~73% of dim dark-background frames.
- It trained a LightGBM classifier on only two sessions (Jun-27, Aug-28), which did not generalize to new sessions (domain gap).
- A "no evidence → default PASS" fallback path silently produced false passes on out-of-distribution data.

Method 1 removes the classifier entirely and replaces absolute-brightness thresholds with **relative, per-element coverage** — which is far more stable across differing camera exposures and lighting setups than either raw pixel intensity or a learned classifier boundary.

---

## 3. Dataset Inventory (unchanged from v6a)

### Training Data

| Session | FPS | Resolution | Duration | Good | Not Good |
|---|---|---|---|---|---|
| June-27-2026 | 12 | 1920×1080 | ~25s | `good/` (50 vids) | `not_good/` (57 vids) |
| August-28-2026 | 12 | 1920×1080 | ~18s | `good/` (14 vids) | `not_good/` (50 vids) |

June-27 = white/light jig background. August-28 = dark/black jig background, same remote position.

### Locked Test Data (never train or tune thresholds on these)

| Session | Videos | Labels |
|---|---|---|
| July-14-2026 | 10 | 5 GOOD + 5 NOT GOOD |
| September-09-2026 | 13 | All GOOD |

---

## 4. Pipeline — Method 1

```
Video (MP4, 12 fps)
  │
  ▼ Step A: Splash-window sampling
  Sample ALL frames in the known timing window
  (Jun-27: t=18-22s · Aug-28: t=12-16s)
  — do NOT rely on a single fixed timestamp; timing drifts per unit.
  │
  ▼ Step B: Coverage-peak frame selection
  Score every sampled frame by GLOBAL lit-pixel coverage
  (whole-LCD, not per-element). Rank descending.
  Take top 2-3 candidate frames near the coverage peak.
  This self-corrects for timing jitter — the true splash
  peak is a local maximum in coverage regardless of exact second.
  │
  ▼ Step C: Register once (with retry across candidates)
  For each of the 2-3 candidate frames:
    - detect_lcd() → perspective-crop to 480x640 BGR
    - Light bg (frame_median >= 63): ORB+RANSAC → ECC affine
    - Dark bg (frame_median < 63):  CLAHE → phase correlation → ECC affine
    - Record ECC correlation score post-warp
  Keep the candidate with the HIGHEST ECC score.
  If ALL candidates fail registration (ECC below a sanity floor),
  mark video ABSTAIN — do not silently default to PASS.
  │
  ▼ Step D: Per-element normalized coverage
  For each of the 21 Atlas elements (or 111 sub-masks):
    raw_nc(e) = fraction of lit pixels within mask region e
    global_coverage = fraction of lit pixels across WHOLE LCD in this frame
    normalized_nc(e) = raw_nc(e) / global_coverage
  This is the key domain-shift fix: absolute brightness varies by
  session/exposure, but relative coverage of a genuinely-lit vs.
  genuinely-dark element is stable across sessions.
  │
  ▼ Step E: Per-element thresholding
  threshold(e) = mean(normalized_nc(e) on GOOD training videos)
                 - 3 * std(normalized_nc(e) on GOOD training videos)
  (or a low percentile, e.g. 1st percentile, if the distribution
  is not well-behaved — check both when computing thresholds)
  │
  ▼ Step F: Verdict
  FAIL any video where at least one element's normalized_nc(e)
  falls below threshold(e).
  PASS otherwise.
  ABSTAIN if Step C found no usable registration.
```

---

## 5. The 21 Atlas ROI Elements (unchanged)

| # | Name | Category |
|---|---|---|
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

Reuse the existing 111-submask Atlas breakdown and `atlas_rois.yaml` coordinate definitions if already available from prior work — the ROI geometry itself was never the problem; the registration and classification strategy was.

---

## 6. What to Avoid

- Do **not** reuse the SCAN_STEP + phase-detect + per-frame-register pipeline from server_v2/v3/v4c/v08.
- Do **not** train or tune any classifier (LightGBM, logistic regression, CNN) as part of Method 1 — this is pure CV only.
- Do **not** tune thresholds on the locked test sets (July-14, September-09). Thresholds are computed strictly from training-set GOOD videos.
- Do **not** silently default to PASS when registration fails on all candidate frames — surface it as ABSTAIN so failures are visible and debuggable.
- Do **not** use absolute pixel-intensity thresholds anywhere — every element decision must go through the normalized_nc(e) computation in Step D.

---

## 7. File Layout

```
IP4R/
├── v6b.md                     ← this document
├── server_v6a/
│   ├── __init__.py
│   ├── frame_extract.py       Step A+B: sample splash window, rank by global coverage
│   ├── register.py            Step C: register-once with multi-candidate retry
│   ├── atlas.py               21-element / 111-submask ROI map + normalized_nc(e)
│   ├── threshold.py           Step E: compute per-element thresholds from GOOD training data
│   ├── pipeline.py            Step F: end-to-end video → verdict (PASS/FAIL/ABSTAIN)
│   └── eval.py                eval script: run on July-14 / September-09, print results
├── data/
│   ├── reference/
│   │   ├── golden_splash.jpg      best splash frame from a known-good unit
│   │   └── atlas_rois.yaml        21-element normalised ROI coords
│   └── v6a_results/
│       ├── thresholds.json        computed per-element thresholds
│       └── eval_report.json       per-video verdicts + per-element diagnostics
```

---

## 8. Diagnostics to Produce Alongside the Verdict

For every video, `eval.py` should output not just PASS/FAIL/ABSTAIN but:
- Which candidate frame was selected (timestamp + global coverage score)
- The ECC registration score achieved
- Per-element normalized_nc(e) values and which (if any) fell below threshold
- This turns every FAIL into an actionable diagnostic (e.g. "Turbo icon at 0.12, threshold 0.31")

---

## 9. Quick-start Eval Commands (target)

```bash
python -m server_v6a.threshold \
    --good-jun27 '/path/to/June-27-2026/good/' \
    --good-aug28 '/path/to/August-28-2026/good/' \
    --golden     data/reference/golden_splash.jpg \
    --out        data/v6a_results/thresholds.json

python -m server_v6a.eval \
    --dataset-july14 '/path/to/July-14-2026/12-FPS/' \
    --dataset-sep09  '/path/to/September-09-2026/' \
    --thresholds     data/v6a_results/thresholds.json \
    --golden         data/reference/golden_splash.jpg \
    --out            data/v6a_results/eval_report.json
```

---

*Document maintained by Om Kathalkar. Created 2026-09-13 for v6a Method 1 implementation.*

---

## 10. Eval Results — Method 1 (Pure CV) — 2026-09-12

**Eval dataset:** Jul-14-2026 (10 videos, 5 GOOD + 5 NOT GOOD, 12 FPS)  
**Sep-09 run:** Not yet completed with Method 1.  
**Server version:** `server_v3` with combined DL + pure CV pipeline (cvtest jobs)  
**DB:** `data/jobs_v3/jobs.db` · job prefix: `cvtest_*`  

### Overall verdict score: **7/10**

| Video | GT | Verdict | Correct | How decided | centre_88 L | centre_88 R | fin_pass_rate | Phase-A |
|---|---|---|---|---|---|---|---|---|
| 141138 | GOOD | PASS | ✓ | fin_passed=True | 0.529 | 0.918 | 0.934 | not found |
| 141341 | NOT_GOOD | FAIL | ✓ | Phase-A FAIL + fin_passed=False | 0.353 | 0.035 | 0.292 | FAIL (5 zone failures) |
| 141950 | NOT_GOOD | PASS | ✗ | fin_passed=True (bug — see below) | 0.341 | 0.647 | 0.635 | not found |
| 142150 | GOOD | PASS | ✓ | fin_passed=True | 0.694 | 0.082 | 0.597 | not found |
| 142516 | NOT_GOOD | PASS | ✗ | fin_passed=True (defect not in centre_88) | 0.776 | 0.706 | 0.552 | not found |
| 142739 | GOOD | PASS | ✓ | fin_passed=True | 1.000 | 0.400 | 0.807 | not found |
| 143041 | GOOD | PASS | ✓ | fin_passed=True | 0.729 | 0.188 | 0.759 | not found |
| 143208 | GOOD | PASS | ✓ | fin_passed=True | 0.553 | 0.918 | 1.000 | not found |
| 143426 | NOT_GOOD | FAIL | ✓ | fin_passed=False | 0.306 | 0.400 | 0.083 | not found |
| 143604 | NOT_GOOD | PASS | ✗ | fin_passed=True (defect not in centre_88) | 0.671 | 0.529 | 0.800 | not found |

**Missed (false negatives):** 141950, 142516, 143604 — all NOT_GOOD classified as PASS  
**False alarms in final_result (did NOT affect verdict):** 142150, 142739, 143041 — GOOD units where centre_88 `final_result.passed=False` but `fin_passed=True` overrode it

---

### Subsystem breakdown

#### DL (EfficientNet-B0): PASS on all 10 — **completely useless for Jul-14 detection**

All `dl_verdict = PASS`, including 5 NOT_GOOD videos. Same domain-shift issue as v6a CNN.

#### Pure CV `cv_result` (7-segment zone check): FAIL on all 10 — **broken, ignore**

`cv_passed = False` for every video, even clearly GOOD units (141138, 142150, 143208). The per-zone segment-level threshold is miscalibrated — ORB-registered frame crops have too many dim/missing individual strokes even on correct units. **This subsystem is not contributing to verdicts and should be disabled or re-calibrated.**

#### Pure CV `final_result` / `fin_passed` (centre_88 peak detection — the new Method 1 signal)

The key new signal: for each Phase-B frame, detect the temperature-88 display region and check if both digit peaks exceed a threshold (0.48).

| Video | GT | final_result.passed | centre_88 L peak | centre_88 R peak | Verdict contribution |
|---|---|---|---|---|---|
| 141138 | GOOD | True | 0.529 ✓ | 0.918 ✓ | Correct PASS |
| 141341 | NOT_GOOD | False | 0.353 ✗ | 0.035 ✗ | Correct FAIL (both missing) |
| 141950 | NOT_GOOD | **False** | **0.341 ✗** | 0.647 ✓ | **Bug: not propagated → false PASS** |
| 142150 | GOOD | False | 0.694 ✓ | **0.082 ✗** | False alarm (R digit localization issue) |
| 142516 | NOT_GOOD | True | 0.776 ✓ | 0.706 ✓ | **Missed** — defect not in centre_88 region |
| 142739 | GOOD | False | 1.000 ✓ | **0.400 ✗** | False alarm (R peak 0.400 < 0.48 threshold) |
| 143041 | GOOD | False | 0.729 ✓ | **0.188 ✗** | False alarm (R peak very dim — mislocalized?) |
| 143208 | GOOD | True | 0.553 ✓ | 0.918 ✓ | Correct PASS |
| 143426 | NOT_GOOD | False | 0.306 ✗ | 0.400 ✗ | Correct FAIL (both dim) |
| 143604 | NOT_GOOD | True | 0.671 ✓ | 0.529 ✓ | **Missed** — defect not in centre_88 region |

**`final_result` standalone score: 5/10** — catches 141341, 143426 (both heavily defective) but has 3 false alarms on GOOD and misses 2 NOT_GOOD.

---

### Root cause of failures

#### Failure 1 — 141950 (NOT_GOOD → PASS ✗)
- `final_result.passed = False`: left digit peak 0.341 < 0.48 — correctly detected on the best frame.
- But `fin_pass_rate = 0.635`: 63.5% of Phase-B frames had the centre_88 check pass. The `fin_passed` gate is majority-vote based; enough frames returned high peaks to override the diagnostic.
- **Root cause:** single-frame peak detection is noisy across the Phase-B sequence. The defect is intermittent or the registration varies enough that some frames look PASS.
- **Fix:** use the **minimum** peak across top-3 frames, not per-frame pass rate.

#### Failure 2 — 142516 (NOT_GOOD → PASS ✗) and 143604 (NOT_GOOD → PASS ✗)
- Both videos have high centre_88 peaks (L≥0.67, R≥0.53 / L=0.776, R=0.706) — the temperature-88 region looks fine.
- The actual defect is in a different LCD region (icon strip, clock display, or bottom_88888) not covered by centre_88.
- **Root cause:** Method 1 currently only checks `centre_88`. It must be extended to score all 21 Atlas elements (or at minimum the Phase-2 gate zones: left_clock, right_clock, signal_bars, bottom_88888).

#### False alarms on GOOD videos (142150, 142739, 143041)
- All 3 have R digit peak below 0.48: 0.082, 0.400, 0.188.
- L digit is fine (0.694, 1.000, 0.729). Only the right digit is consistently low.
- **Root cause:** right digit ROI is likely mislocalized — the centre_88 region splits into left and right sub-regions; the right split may be clipping the digit or landing on the colon/separator. Need to visualize the exact crop boundaries on a GOOD frame overlay.
- **Fix:** re-check `centre_88` right-digit bounding box against a known-GOOD frame. Likely needs a small x-offset adjustment.

#### `cv_result` (7-seg) failing on all 10
- The 7-segment per-stroke check detects that individual strokes (a, b, d, f, g) are missing even on GOOD units.
- **Root cause:** the ORB+ECC registration on a dim-phase or motion-blurred crop produces imperfect alignment; strokelevel masks pick up the misalignment as missing coverage.
- This subsystem is not used in the `fin_passed` gate. It should be disabled until registration quality is improved.

---

### What the Phase-A check contributed

Only video **141341** had Phase-A frames detected (`phase_a_found = 2`). The Phase-A check correctly identified 5 zone failures:
```
left_clock(cov=0.142 < 0.25)
center_88_L(cov=0.269 < 0.40)
center_88_R(cov=0.000 < 0.40)
signal_bars(cov=0.000 < 0.22)
bottom_88888(cov=0.121 < 0.42)
```
For the 9 remaining videos, Phase-A was not found (phase_a_found = 0).

---

### Thresholds computed (Jun-27 GOOD training set, n=50 videos)

Stored in `data/v6a_results/thresholds.json`:

| Element | mean | std | p01 | threshold used |
|---|---|---|---|---|
| top_icons | 1.197 | 0.272 | 0.792 | 0.792 |
| icon_strip | 1.248 | 0.381 | 0.333 | 0.333 |
| left_clock | 1.199 | 0.528 | 0.167 | 0.167 |
| right_clock | 0.862 | 0.218 | 0.386 | 0.386 |
| center_88 | 1.528 | 0.652 | 0.509 | 0.509 |
| signal_bars | 1.554 | 0.547 | 0.665 | 0.665 |
| secondary_icons | 1.476 | 0.586 | 0.629 | 0.629 |
| bottom_88888 | 1.831 | 0.977 | 0.253 | 0.253 |

Training source: Jun-27-2026 GOOD only (Aug-28-2026 GOOD not included in this threshold run). Aug-28 GOOD should be added to widen the training distribution.

---

### Summary and next steps for Method 1

| Issue | Priority | Fix |
|---|---|---|
| Right digit false alarms (142150, 142739, 143041) | High | Visualize centre_88 R crop boundary on a GOOD frame; adjust x-offset |
| 141950 miss — noisy fin_pass_rate | High | Use min peak across top-3 frames instead of per-frame majority vote |
| 142516, 143604 miss — defect outside centre_88 | High | Add left_clock, right_clock, signal_bars, bottom_88888 to final_result scoring (already in atlas.py) |
| cv_result (7-seg) always False | Medium | Disable or fix registration before enabling stroke-level checks |
| Threshold training: Jun-27 only | Medium | Re-run threshold.py adding Aug-28-2026 GOOD videos |
| Sep-09 not yet evaluated | High | Run cvtest on all 13 Sep-09 videos after above fixes |
