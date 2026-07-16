# CLAUDE.md — IP4R

> Read this first. It tells you (Claude Code) what IP4R is, how it's built, and what to do next.

## What this project is
**IP4R** inspects production photos of an **AC-remote LCD in its all-segments-on "splash" self-test state** and decides **PASS / FAIL** by verifying every digit, segment, icon and printed label is present, correctly shaped, and clean.

The golden reference is `data/reference/lcd_all_on.jpg` — that image *is* the spec. Every element that should be lit is lit in it.

The full rationale and method comparison is in **`docs/literature_survey.md`** — read it before changing the inspection logic.

## Core idea (the "why")
This is a **reference-based (golden-template)** inspection problem, not OCR and not open-set recognition. The expected appearance is fully known, so the effective approach is:
1. **Register** each sample to the golden frame (ORB+RANSAC homography → ECC sub-pixel refine). *Alignment is the make-or-break step.*
2. **Per-ROI deterministic checks** (Tier A, training-free, the single golden image is the model):
   - **coverage ratio** → "is this element lit / present?"
   - **windowed SSIM** → "is it shaped right / clean (no smear, blur, low contrast)?"
   - optional **per-segment occupancy** for seven-seg digits.
3. **Optional Tier B**: Anomalib PatchCore/PaDiM ("train on good only") for unforeseen defects — off by default, switch on once good images are banked.
4. **Report**: PASS/FAIL + per-ROI verdicts + annotated overlay + JSON.

## Repo layout
```
ip4r/
├─ CLAUDE.md                  ← you are here
├─ README.md                  ← human quickstart
├─ pyproject.toml             ← deps (core light; [dl] extra pulls torch/anomalib)
├─ config/default.yaml        ← paths + thresholds + which checks run
├─ data/
│  ├─ reference/lcd_all_on.jpg   ← golden
│  ├─ reference/rois.yaml        ← authored ROI map (create with the roi-editor)
│  ├─ samples/                   ← incoming production images
│  └─ results/                   ← overlays + JSON reports
├─ src/ip4r/
│  ├─ config.py        load/validate YAML config
│  ├─ preprocess.py    grayscale, denoise, illumination-normalise
│  ├─ registration.py  ORB+RANSAC homography + ECC refine
│  ├─ roi.py           ROI dataclasses + YAML load/save (normalised coords)
│  ├─ inspect.py       per-ROI coverage + SSIM scoring → verdicts
│  ├─ segments.py      seven-segment per-stroke occupancy (digit ROIs)
│  ├─ report.py        annotated overlay + JSON report
│  ├─ pipeline.py      orchestrates 0→4 for one image
│  ├─ synth.py         synthesise defective samples from the golden (for testing)
│  ├─ cli.py           `ip4r` command (selfcheck / inspect / roi-edit / synth)
│  └─ tools/roi_editor.py   interactive OpenCV ROI drawing tool
└─ tests/test_pipeline.py
```

## Conventions
- **Python 3.11+**, type hints, dataclasses, `pathlib`. Keep Tier A dependency-light (numpy, opencv-python, scikit-image, pyyaml). Torch/anomalib live behind the optional `[dl]` extra — never import them at module top level in core code.
- ROIs are stored in **normalised [0,1] coords** so they survive resolution changes. Names are snake_case and descriptive (`icon_snowflake`, `digit_settemp_tens`, `label_turbo`).
- Every inspection produces a structured `InspectionResult` (see `inspect.py`) — don't print verdicts ad hoc; go through `report.py`.
- All thresholds live in `config/default.yaml`, never hard-coded in logic.
- Mac target: CPU-only must always work. MPS is a bonus for Tier B only.

## How to run (sanity)
```bash
pip install -e .
# 1) prove the pipeline end-to-end with no real data: golden vs golden = all PASS
ip4r selfcheck
# 2) author ROIs once on the golden (draw boxes, type names, press s to save)
ip4r roi-edit
# 3) make a fake defective unit (erases/dims a random element) and inspect it
ip4r synth --out data/samples/defect_01.jpg
ip4r inspect data/samples/defect_01.jpg
```

## Roadmap / TODO (do these in order)
- [ ] **T1** Author the real ROI map on the golden via `ip4r roi-edit` (the shipped `rois.yaml` is a small starter set — extend to all elements in the survey's inventory table).
- [ ] **T2** Capture 20–50 *good* production units; calibrate per-ROI thresholds to `mean − k·σ` (replace hand-set constants). This is the biggest reliability win.
- [ ] **T3** Add per-segment ROIs for the seven-seg digits and enable `segments.py` occupancy checks.
- [ ] **T4** Harden registration: handle glare/reflection masks; fall back to ECC-only when feature matches are sparse.
- [ ] **T5** (optional) Wire Tier B: train Anomalib PatchCore on the good bank; fuse its heatmap verdict with Tier A.
- [ ] **T6** Batch mode + CSV/line-feedback export; latency benchmark on Mac.

## Gotchas
- SSIM is **not** shift-invariant — always score *after* registration. If SSIM is noisy, registration is the suspect, not the threshold.
- The reference photo includes the plastic bezel and background; the ROI map (and ideally a screen-region crop in registration) should restrict scoring to the LCD glass.
- A single golden photo carries that day's lighting/glare. For production, rebuild the golden as the **mean of several good captures** (GTC) once available.
