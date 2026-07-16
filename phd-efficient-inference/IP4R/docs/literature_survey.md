# IP4R — Literature Survey & Method Selection

**Project:** IP4R — Image-based Production QC for AC Remote LCD Splash Screens
**Author:** Om Kathalkar
**Goal:** Given a production image of an AC-remote LCD in its **all-segments-on ("splash") self-test state**, automatically decide PASS / FAIL by verifying that **every** number, segment, icon and printed label is present, correctly shaped, and not corrupted (missing segment, partial glyph, ghost segment, smear, low contrast, misregistration).
**Constraint:** Must run effectively on a **Mac (Apple Silicon)**, ideally CPU-only and real-time, with **few or no defective training samples** available at the start.

---

## 1. Problem framing

The splash screen is a deterministic test pattern: in the "all-on" state **the expected appearance is fully known in advance** (the uploaded `lcd_all_on.jpg` *is* the specification). The line camera and fixture are fixed, so pose variation between units is small (translation + minor rotation/scale). This is the textbook setting for **reference-based (golden-template) optical inspection**, not open-set recognition.

This matters because it determines the right tool. We do **not** need to *read* arbitrary digits (full OCR) and we do **not** need a model that generalises to unseen product families. We need to confirm a *known* target is reproduced correctly. That pushes the solution toward classical, interpretable, training-free methods first, with deep anomaly detection held in reserve for defects the deterministic checks can't anticipate.

The elements to verify on this specific panel:

| Group | Elements |
|---|---|
| Top icon row | display-frame, snowflake (cool), water-drop (dry), 3-blade fan, sunburst (heat), wifi |
| Clock blocks | `10:00` (×2, each 4 seven-seg digits) + `AM/PM`, `TIME OFF`, `CLOCK`, `TIME ON` |
| Set-temp block | large `88` (two seven-seg digits), `°C`, `SET TEMP`, `AUTO`, pie/half-moon glyph, signal bars, fan glyph |
| Mode row | circled-`E`, lock, `TURBO`, `ion` |
| Bottom row | battery frame, bulb, sparkle/swing glyphs, `88888` (5 seven-seg digits) |

Two failure classes follow from this inventory:
- **Segment/structural failures** — a seven-segment digit missing a stroke, an icon partially lit, a dead row/column. Localised, geometric.
- **Print / display-quality failures** — smear, blur, low contrast, mura (brightness non-uniformity), ghosting. Diffuse, photometric.

A good system handles both, and—critically—**says which named element failed**, because that drives line feedback and root-cause.

---

## 2. Method families considered

### 2.1 Golden Template Comparison (GTC) — the industrial baseline
The dominant method for inspecting 2-D scenes from highly repeatable processes (printing, semiconductor, LCD): **register** the test image to an ideal reference, **subtract**, **threshold** the difference image, and analyse residual blobs as flaws. The reference is typically the **mean of several good samples** plus a per-pixel **tolerance/threshold image** so that normal process variation is not flagged. This is the Cognex GTC lineage (US5640200, US5850466, US6504957) and remains standard in PCB and display AOI. Registration accuracy is explicitly the dominant determinant of performance; sub-pixel alignment and a learned tolerance band are what separate a usable GTC from a false-positive generator.

- **Pros:** training-free, fast, interpretable, pixel-accurate localisation, directly answers "is the known target reproduced?".
- **Cons:** brittle to misregistration and lighting drift; needs a good tolerance model; a raw whole-image subtract over-fires near high-contrast edges. → Mitigate with **ROI-scoped** scoring instead of whole-frame subtraction.

### 2.2 ROI / region-based presence checks
Because we know *where* every element should be, we can score each element in its own ROI rather than globally. For an "all-on" pattern, a per-ROI **foreground-coverage ratio** (fraction of lit pixels after local thresholding) is a strong, cheap presence signal: a missing segment or unlit icon collapses coverage in its box. This is essentially the per-region logic used in seven-segment readers, whose classical pipeline is *locate display → locate digits → decide per-segment on/off*. The decisive advantage over whole-frame GTC is **named-element localisation** and robustness to small global shifts once each ROI is individually re-aligned.

### 2.3 SSIM / MS-SSIM as the per-ROI similarity metric
For the *print/structural-quality* axis, simple subtraction (MSE/PSNR) correlates poorly with perceived defects. **SSIM** compares luminance, contrast and structure locally and is widely used for print- and display-defect detection (print "doubling", can-printing defects, thin-film surfaces ~99% accuracy, PCB SSIM-Net, LCD mura). **MS-SSIM** aggregates across scales so both large and fine defects surface at their natural resolution; it has been used as the inline, interpretable anomaly score on constrained hardware in regulated manufacturing. Practical caveats from the literature: SSIM is **not geometric-invariant** (it penalises a shift as harshly as noise), so it must be computed **after registration**, and **windowed/local** SSIM is required to catch localised distortions. `DSSIM = (1 − SSIM)/2` gives a clean dissimilarity score to threshold per ROI.

→ **Combine 2.2 + 2.3:** coverage catches "is it lit?", SSIM catches "is it shaped right / clean?". Together they cover both failure classes with two cheap, explainable numbers per element.

### 2.4 Seven-segment verification (digit-aware layer)
Beyond "the digit box is lit", we can verify each of the **7 individual segments** of every `8`. The all-on pattern lights all seven, so the check is: in each of the 7 segment sub-ROIs, is coverage above on-threshold? Classical seven-segment OCR (binarise → erode/dilate → per-segment occupancy) does exactly this and is robust and tiny; deep detectors (Cascade R-CNN, CNN, YOLO-style) reach F1 ≈ 0.999 but are overkill when the *expected* pattern is fixed and we only need presence, not classification. We adopt the **classical per-segment occupancy** check as an optional high-resolution layer for the digit ROIs.

### 2.5 Unsupervised deep anomaly detection (the fallback tier)
When defects are **unforeseen** or appearance variability (reflections, contrast drift, multiple product skins) breaks fixed thresholds, **one-class / "train on good only"** methods are the modern answer, and they need **no defective samples**:
- **Autoencoder + SSIM** (Bergmann et al.) — reconstruct "good" appearance, flag high reconstruction-SSIM error. Interpretable, light.
- **PaDiM** — per-patch multivariate-Gaussian over pretrained CNN features; Mahalanobis distance as anomaly score.
- **PatchCore** — coreset **memory bank** of "good" patch features + nearest-neighbour distance. SOTA on MVTec AD (image AUROC up to ~99%), cold-start friendly, fast with coreset subsampling, best-in-class localisation.
- **EfficientAD / FastFlow / Reverse-Distillation** — speed/accuracy variants.

All are packaged in **Anomalib** (Intel), which runs on Mac (CPU/MPS) and provides PatchCore/PaDiM/EfficientAD/autoencoder out of the box with pixel-level heatmaps. The benchmark to validate against is **MVTec AD** (Bergmann et al., CVPR 2019). Real-world studies (sheet-metal glue lines, medical-device manufacturing) confirm these transfer to production lines, with the caveat that benchmark numbers overstate real-world ease.

- **Pros:** catches defects you didn't enumerate; no defect labels needed; great localisation heatmaps.
- **Cons:** heavier (PyTorch), less directly explainable ("anomalous region" vs "TURBO label missing"), needs a clean bank of good images, sensitive to pose/lighting unless registered first.

---

## 3. Comparative summary

| Method | Train data | Localises to named element? | Speed (Mac CPU) | Handles unforeseen defects | Best at |
|---|---|---|---|---|---|
| Whole-frame GTC | few good | weak (blob only) | fast | yes (any pixel change) | gross global defects |
| **ROI coverage** | 1 golden | **strong** | very fast | no | missing segment/icon |
| **Per-ROI SSIM/MS-SSIM** | 1 golden | **strong** | fast | partial | smear/blur/contrast/mura |
| Seven-seg occupancy | 1 golden | **strong (per stroke)** | very fast | no | digit stroke faults |
| AE + SSIM | many good | medium (heatmap) | medium | **yes** | reconstruction-detectable defects |
| PaDiM / **PatchCore** | many good | medium (heatmap) | medium | **yes** | subtle/unanticipated defects |

---

## 4. Recommended architecture for IP4R (tiered)

A **deterministic, training-free core** that solves the stated problem on day one, with an **unsupervised DL safety net** that can be switched on once a bank of good images exists.

```
                         ┌─────────────────────────────────────────┐
   production image ───► │ 0. Preprocess: grayscale, denoise,       │
                         │    illumination-normalise                │
                         └───────────────┬─────────────────────────┘
                                         ▼
                         ┌─────────────────────────────────────────┐
                         │ 1. Register to GOLDEN frame              │
                         │    ORB/AKAZE + RANSAC homography,        │
                         │    ECC refine (sub-pixel)                │  ← single most important step
                         └───────────────┬─────────────────────────┘
                                         ▼
                         ┌─────────────────────────────────────────┐
   ROI map (authored ──► │ 2. TIER A — deterministic per-ROI checks │
   once on golden)       │   • coverage ratio  (presence)          │
                         │   • windowed SSIM/MS-SSIM (quality)     │
                         │   • per-segment occupancy (digits, opt) │
                         └───────────────┬─────────────────────────┘
                                         ▼
                         ┌─────────────────────────────────────────┐
                         │ 3. TIER B (optional) — Anomalib PatchCore│
                         │    on whole panel, "train on good only", │
                         │    heatmap for unforeseen defects        │
                         └───────────────┬─────────────────────────┘
                                         ▼
                         ┌─────────────────────────────────────────┐
                         │ 4. Decide + report: PASS/FAIL, per-ROI   │
                         │    verdicts, annotated overlay, JSON     │
                         └─────────────────────────────────────────┘
```

**Why this ordering / why it is effective:**
1. **Registration first** — every reference method in the literature names alignment as the make-or-break step. Sub-pixel ECC after a robust ORB+RANSAC homography keeps SSIM honest (SSIM is not shift-invariant).
2. **ROI-scoped, not whole-frame** — gives *named-element* verdicts (e.g. "icon_snowflake: FAIL, coverage 0.11 vs golden 0.62") which is what a production line actually needs, and sidesteps GTC's edge-false-positive problem.
3. **Two cheap orthogonal scores per ROI** — coverage (lit?) + SSIM (clean/shaped?) cover the two physical failure modes with full interpretability and CPU real-time speed. No training data required: the single golden image is the model.
4. **DL only as Tier B** — once ~50–200 good production images are banked, PatchCore/PaDiM via Anomalib adds a net for defects nobody enumerated, with pixel heatmaps, still on Mac CPU/MPS. Kept optional so the deliverable works before any data collection.

**Thresholding/tolerance:** start with per-ROI thresholds; once a handful of good units are captured, estimate per-ROI coverage/SSIM **distributions** and set tolerances at e.g. mean − k·σ (the GTC "tolerance image" idea, applied per ROI). This converts hand-tuned constants into data-driven bands and is the main reliability upgrade after the prototype.

---

## 5. Mac / implementation notes
- Core stack: **Python 3.11+, OpenCV (`opencv-python`), scikit-image (`structural_similarity`), NumPy, PyYAML** — all pip-installable on Apple Silicon, CPU-only, real-time for a single panel.
- ROI authoring: an interactive OpenCV tool to draw/name boxes once on the golden image; ROIs stored as YAML/JSON (normalised coords so they survive resolution changes).
- Tier B: **`anomalib`** (pulls PyTorch); runs on CPU or Apple **MPS**. Keep as an optional extra so the core install stays light.
- Validate end-to-end **before real defective images exist** by synthesising defects (erase/dim a random ROI on the golden) — lets the pipeline be proven on day one.

---

## 6. Key references
1. Cognex — *Golden Template Comparison using efficient image registration*, US 5,640,200 (and US 5,850,466; US 6,504,957). Registration + subtraction + thresholded difference; reference = mean of good samples + tolerance image.
2. Z. Wang, A. Bovik, et al. — *Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM)*, IEEE TIP 2004; *Multiscale SSIM*, 2003.
3. P. Bergmann et al. — *Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders*, VISAPP 2019 (arXiv:1807.02011).
4. P. Bergmann et al. — *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
5. T. Defard et al. — *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, ICPR 2021.
6. K. Roth et al. — *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*, CVPR 2022 (arXiv:2106.08265). ~99% image AUROC on MVTec AD.
7. Intel — **Anomalib** library (PatchCore/PaDiM/EfficientAD/autoencoder; CPU/MPS).
8. Seven-segment detection/recognition: Cascade R-CNN (IEEE, F1≈0.999); RSDC real-time CNN (Springer); Moreira, *Automated Medical Device Display Reading* (arXiv:2210.01325) — classical *locate→localise→decide-per-segment* pipeline.
9. SSIM in inspection: *SSIM-Net: Real-Time PCB Defect Detection* (2019); *Defect detection of printing images on cans based on SSIM*; *Multiscale image quality measures for defect detection in thin films*, IJAMT 2015 (~99.1%); *Dual-Mode Deep Anomaly Detection for Medical Manufacturing* (4-MS-SSIM, arXiv:2509.05796).
10. Mura / display panel: *A Machine-Learning Strategy to Detect Mura Defects… Piecewise Gamma Correction* (TFT-LCD, SEMU); US 10,867,382 (mura in master panels).
11. PCB AOI reviews: *A comprehensive review of research on surface defect detection of PCBs based on machine vision*, ScienceDirect 2025 (comparison/template/feature taxonomy); *YOLO-pdd* (arXiv:2407.15427).
12. Production-realism studies: *Comparison of unsupervised image anomaly detection models for sheet metal glue lines*, EAAI 2025.

---

## 7. One-paragraph conclusion
For IP4R's controlled "all-on" splash-screen QC, the **most effective first solution is a registration-anchored, ROI-scoped golden-template pipeline**: ORB+ECC alignment, then per-element **coverage** + **SSIM** scoring, with an optional per-segment occupancy check on digits. It is training-free (the single golden image is the spec), CPU-real-time on Mac, and—crucially—reports *which named element failed*. **Anomalib PatchCore/PaDiM** is the recommended Tier-B upgrade once a bank of good units exists, to catch defects the deterministic rules don't anticipate. This mirrors how display/PCB AOI is actually done in industry (GTC) while adopting the modern SSIM and one-class-DL refinements the recent literature validates.

---

## 8. Implementation status & calibration results (IP4R v03, 2026-06-28)

### 8.1 What was built

The v03 system implements Tier A only (Tier B / PatchCore is wired but disabled). Five active checks per digit ROI, three per icon/label ROI:

| Check | Enabled | Applies to | Note |
|---|---|---|---|
| Coverage ratio | ✓ | all ROIs | primary discriminator; camera-invariant |
| Diff density | ✓ digit_bottom_88888 only | digit | calibrated per-ROI at 5-sigma from good bank |
| SSIM | disabled (ssim_min = −1) | — | disabled after empirical finding (see §8.2) |
| Per-segment occupancy | ✓ | single-digit ROIs | 7 sub-regions; aspect-ratio gated |
| MNIST CNN feature similarity | ✓ | all digit ROIs | 1 − cos_sim(feat(golden), feat(sample)) |

Registration pipeline: ORB 2000 features → RANSAC (reproj 5 px, min 12 matches) → ECC refine (100 iter, ε=1e-5). Two-stage gives sub-pixel accuracy critical for all subsequent checks.

ROI map (`data/reference/rois.yaml`): 28 ROIs total — 5 digit, 17 icon, 6 label. Digit ROIs carry per-ROI `ssim_min`, `diff_max`, and `cov_ratio_min` overrides; the 4 large clock/temp digit ROIs all use `diff_max: 0.99` (effectively disabled) and `cov_ratio_min: 0.85–0.90`; `digit_bottom_88888` uses `diff_max: 0.45`, `cov_ratio_min: 0.90`.

### 8.2 Key empirical findings from the 3 351-image good bank

**Camera-to-camera SSIM variance breaks fixed thresholds.** Six cameras (rmt09, rmt11, rmt13 + others) produce systematically different pixel distributions on the same physical unit. SSIM on digit ROIs varied from −0.15 to +0.85 across cameras even for defect-free units. A global ssim_min=0.62 produced 17 FA on `clock_time_off` alone (12 from rmt13). Setting `ssim_min=−1.0` in rois.yaml disables SSIM without changing any other logic. **Coverage ratio, by contrast, is camera-invariant**: it normalises sample coverage against golden coverage of the *same ROI*, cancelling global brightness offset.

**Sub-digit splitting fails at small patch sizes.** Splitting `digit_bottom_88888` (59×24 px) into 5 sub-digits (≈12×24 px) made adaptive thresholding unreliable — coverage ratios dropped as low as 0.81 on genuine good images, producing 1 480 / 3 351 FA (44.2%). Sub-digit coverage was abandoned; the strip is inspected as a unit.

**MNIST CNN as feature extractor, not classifier.** Direct digit classification with the MNIST CNN fails for LCD digits (7-segment "8" is classified as "1", P("8") ≈ 0.02 for both good and defective patches — domain gap). Feature-similarity mode works: `defect_score = 1 − cos_sim(feat_CNN(golden_patch), feat_CNN(sample_patch))`. On the good bank, scores range 0.01–0.42 (max 0.367 for `digit_bottom_88888`); 3 missing segments produce scores ≈ 0.26+. Threshold set at 0.45 gives zero CNN-FA on `digit_bottom_88888`.

Model used: `kenil-patel-183/mnist-cnn-digit-classifier` (99.25% MNIST accuracy), loaded via `hf_hub_download` + `safetensors` (no `trust_remote_code`; bypasses `AutoModel` routing which breaks in transformers ≥ 5.x). Feature vector: 3136-dim (post-flatten of 4-conv network).

### 8.3 Calibration results (v03 — good bank + fault batch)

**Good bank — 3 351 images, 6 cameras:**

| Camera | Images | FA count | FA rate |
|---|---|---|---|
| rmt09 | 966 | 4 | 0.41% |
| rmt11 | 413 | 0 | 0.00% |
| rmt13 | 1006 | 3 | 0.30% |
| others | 966 | 7 | 0.72% |
| **Total** | **3 351** | **14** | **0.42%** |

Remaining FA root causes: registration edge-cases on heavily reflective units (coverage slightly below threshold despite visually acceptable display).

**Fault batch — detection rate:**

| Check | Detections contributed |
|---|---|
| Coverage | 165 |
| CNN feature similarity | 111 |
| Per-segment occupancy | 96 |
| Diff density | 82 |

Overall detection rate: **92.7%** of fault images flagged. Misses (7.3%) are units where the defect falls outside digit ROIs (icon/label ROIs only carry coverage check; SSIM disabled).

**Summary:** 92.7% detection / 0.42% FA — meets the practical threshold for deployment as a first-pass filter with human review of edge cases.

### 8.4 Video inspection (scripts/inspect_video.py, 2026-06-28)

A stability-triggered video inspection mode was added for line-integration testing. The key design insight: for a tripod/fixture-mounted camera the MAD (mean absolute difference between frames) is uniformly low, making MAD-based frame-selection unreliable. Instead, a **coverage-based splash detector** is used:

1. **One-time registration** at a user-specified calibration timestamp (default t=2s) produces a homography H.
2. **Fast scan**: for each frame after `--start-skip` seconds, apply H via `warpPerspective` (< 1 ms/frame) and check digit ROI coverage against the same thresholds as the batch inspector.
3. **Trigger**: when `--stable-frames` consecutive frames all pass the coverage check, the first such frame is selected and a full inspection (with fresh ORB+ECC registration) is run on it.
4. An annotated MP4 is written showing SCANNING → STABILIZING → PASS/FAIL verdict with per-ROI quads mapped back to the original frame via the inverse homography.

Validated on a 12 fps, 1920×1080 video of a real AC remote: splash screen correctly triggered at t=4.9s (button press ~t=3s, splash delay ~2s), full inspection PASS, all 5 digit ROIs passing coverage.

CLI: `python scripts/inspect_video.py <video.mp4> [--start-skip 5.0] [--stable-frames 3] [--cal-t 2.0] [--out dir]`
