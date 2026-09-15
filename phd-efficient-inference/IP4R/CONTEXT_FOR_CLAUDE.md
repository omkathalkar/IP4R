# IP4R — Current State & New Approach Request

> Hand this file to Claude Code along with the codebase. It summarises everything tried so far,
> the two remaining failures, and asks for a fresh approach.

---

## What this project is

**IP4R** is a video-based QC system for AC-remote LCD splash-screen inspection (all-segments-on state).
A factory CM4 unit posts a short MP4 clip to a FastAPI server on `tangentthoughttech.com:8083`.
The server decides **PASS / FAIL / ABSTAIN** per video and returns the verdict within ~8 s (CPU).

The active production model is **server_v4c** — a *Sandwich LightGBM* classifier trained on 365
per-frame features derived from a 111-element Atlas mask registered to each frame.

**Two permanently locked eval sets:**
- **Jul-14-2026** — 10 videos (5 GOOD + 5 NOT GOOD) · target ≥ 9/10
- **Sep-09-2026** — 13 videos (all GOOD) · target 13/13

**Current score:** Jul-14 = **9/10**, Sep-09 = **11/13** — 2 remaining failures.

---

## LCD Frame Taxonomy

### `splash_frame` — the target detection frame (t ≈ 14–15 s)

The **`splash_frame`** is the canonical all-segments-on LCD state that occurs at approximately
**t = 14–15 s** in every GOOD video and again briefly at the end of the clip. This is the frame
the system must detect and score. A GOOD unit's `splash_frame` looks like:

```
┌─────────────────────────┐
│  18:88   18:88          │   ← all timer segments lit (8-shaped digits)
│                         │
│     88              ▌▌▌ │   ← temperature "88" + signal bar icon
│                         │
│        88888            │   ← foot display, all segments on
└─────────────────────────┘
```

- Every 7-segment digit shows `8` (all 7 strokes lit).
- All icons (WiFi/signal bars, snowflake, fan, heat, etc.) are illuminated simultaneously.
- Background is the LCD glass colour (grey-green in lighter jig, slightly darker in black jig).
- **This is Phase-B** in the pipeline's phase classifier vocabulary.

A `splash_frame` from the **Sep-09 black-jig setup** looks dimmer overall (lower ambient, PWM
aliasing → bimodal brightness) but the segment pattern is identical.

### Idle / Phase-A state

Between video start and t ≈ 14 s the remote LCD shows its **idle/normal operating state** —
digits display actual temperature/time values, only active-mode icons are lit, no all-segments-on
pattern. This is Phase-A. The phase classifier uses `_classify_phase()` to gate out Phase-A
frames before LightGBM scoring.

Reference photo of idle state: `data/reference/idle_remote.png`
(shows partial segments lit, non-uniform icon state — NOT a valid scoring frame).

---

## ROI Map — 21 Elements (full Atlas)

The 111-mask Atlas is built from these 21 named regions. The server computes `nc(e)` (normalised
coverage) for each mask element per registered frame; LightGBM trains on those 365 derived features.

| # | Name | Type |
|---|------|------|
| 1 | Auto_Mode | Icon |
| 2 | Cool_Mode | Icon |
| 3 | Dry_Mode | Icon |
| 4 | Fan_Mode | Icon |
| 5 | Heat_Mode | Icon |
| 6 | IR_Transmission | Icon |
| 7 | Timer_OFF | 7-seg + label |
| 8 | Clock | Icon |
| 9 | Timer_ON | 7-seg + label |
| 10 | Temperature | 7-seg (2-digit) |
| 11 | Fan_Speed | Icon (bars) |
| 12 | Energy-Save_Mode | Icon |
| 13 | Lock | Icon |
| 14 | Turbo | Label |
| 15 | ion | Label |
| 16 | Battery | Icon |
| 17 | Light | Icon |
| 18 | H_Swing | Icon |
| 19 | V_Swing | Icon |
| 20 | Foot_Display | 7-seg (5-digit) |
| 21 | Sleep_Mode | Icon |

Full annotated ROI overlay: `docs/example_overlay.png`

---

## New Samples — Sep-11-2026 Dataset

A new folder of test videos has been received from the client at the **same Google Drive link**
(Drive folder ID: `1Wrf4Qf-bOaYDhzI2dRoM2arNvZR4EW7O`, new subfolder: `Sep-11-2026/`).

**Status as of 2026-09-11:** Not yet downloaded to the tangent server.

**Next step (manual, on tangent server):**
```bash
# SSH into tangent server
ssh om@tangentthoughttech.com

# Create dataset directory
mkdir -p ~/src/FDU\ Dataset/Sep-11-2026/

# Download from Drive (use gdown or rclone with the folder ID)
gdown --folder 1Wrf4Qf-bOaYDhzI2dRoM2arNvZR4EW7O -O ~/src/FDU\ Dataset/ --remaining-ok
# or navigate to the Sep-11-2026 subfolder specifically
```

**Dataset policy for Sep-11-2026:**
- Labels unknown until reviewed — treat as **unlabeled eval** initially.
- Do NOT add to training data until labels are confirmed and a hold-out round is complete.
- If all GOOD: may serve as additional validation alongside Sep-09.
- If mixed GOOD/NOT GOOD: lock as a new eval set (never train).

---

## Everything we tried (chronological)

### Phase 1 — Baseline (server_v3 / EfficientNet-B0)
- EfficientNet-B0 trained on 6 507 Phase-2 crops (`dts_p2v2_best.pth`, val_acc = 0.9992).
- Worked well on Jul-14. Fell apart on Sep-09: different jig (black vs grey), shorter videos (~19 s vs ~25 s), Phase-A missing entirely.

### Phase 2 — Sandwich LightGBM (server_v4c)
- Replaced CNN with a LightGBM model trained on 365 handcrafted features from 111-mask Atlas per frame.
- Training data: Aug-28-2026 GOOD (14 vids, w=2.0), Aug-28-2026 NOT GOOD (13 of 50 kept, w=2.0), Jun-27-2026 (214 vids, w=1.0).
- τ_clf = 0.65 (F1-optimal 0.527 didn't transfer; manually swept).
- **Jul-14 result: 9/10** (video 142812 permanently fails — pre-existing hardware defect, not a model bug).
- **Sep-09 result at this point: 3/13** — catastrophically wrong.

### Phase 3 — Root cause diagnosis on Sep-09 (video 201351 instrumented)
Discovered:

```
Phase-B frames detected:  168
register() returned None: 122  ← 73% failure — ORB finds no keypoints on dark frames
C_ref < 5.0 (dim):         18
Valid frames:              28  (server saw 9 via SCAN_STEP=3)

C_ref distribution: bimodal — most frames near-dark, occasional bright spikes
```

**Why Sep-09 is hard vs Jul-14:**
- Black jig (frame_median ≈ 48–50) vs lighter jig (≈ 76–78).
- Phase-B starts at t = 0.0 s — no Phase-A visible (different protocol).
- Valid frames after full pipeline: 3–28 (vs 70–140 for Jul-14).
- Med_Cref: 7–16 (vs 20–45 for Jul-14).
- **Hypothesis: PWM backlight aliasing** — if backlight frequency isn't a clean multiple of 12 fps, alternating dark/bright frames explain the bimodal C_ref.

**Central architectural finding:** ORB + ECC registration is re-solved independently on every frame.
On dark frames, ORB finds no keypoints → `register()` returns None → frame is silently dropped.
The geometry is physically constant per video (fixed jig). Re-solving per frame is the bug.

### Phase 4 — lcd_crop_v5f + worker_v5c (deployed 2026-09-10)

**lcd_crop_v5f changes** (`server_v3/lcd_crop.py`):
1. Frame-level dark-background flag: `dark_bg = frame_median < 63` (Sep-09 ≈ 48, Jul-14 ≈ 77).
2. Pass 1B (threshold=185 inner contour search) **gated on `dark_bg=True`** — this stopped Jul-14 from getting the wrong crop path that caused the v06 regression.
3. Pass 3 morphology kernels **restored to v3 originals**: body k20, LCD mask k_c=12×12 (v4 had drifted to k30 + 8×8).
4. `_crop_is_valid` check restored in `_inner_contour_search` (was accidentally dropped).
5. `dark_bg` flag propagated in every returned info dict.

**worker_v5c changes** (`server_v3/worker.py`):
- `_classify_phase(gray, thresh, score_min, dark_bg=False)` — new parameter.
- Bright-LCD fallback (`bottom_80 < 0.10` → accept if `seg_cov ≥ 0.65 and icon_cov ≥ _PHASE_A_ICON_MAX`) **gated on `dark_bg=True`** only — preserves strict Jul-14 behavior.

**app.py** (`server_v4/sandwich/app.py`):
- `crop, _lcd_info = detect_lcd(frame)` → `dark_bg` passed into `_classify_phase`.

**Result after v5f + v5c:**
- Jul-14: 9/10 ✓ (recovered from v06 regression of 8/10)
- Sep-09: **11/13** (up from 3/13)

---

## Two remaining Sep-09 failures

### Failure 1 — video `201454`: ABSTAIN
- n_b = 2 phase_b frames reach LightGBM.
- `EVIDENCE_MIN = 1.5` requires n × (fail_ratio − 0.5) ≥ 1.5 → with only 2 frames this threshold is never met.
- Server defaults to ABSTAIN.
- Root cause: phase classifier is still not finding Phase-B frames in this video.
- This is a **phase detection / frame-count** problem, not a LightGBM problem.

### Failure 2 — video `202145`: FAIL
- n_b = 5 frames, gate = `classifier`, LightGBM outputs FAIL.
- LightGBM was trained on Jul-14 and Aug-28 brightness regimes. Sep-09's lit LCD looks different (different backlight exposure / jig reflectance).
- The features (`nc(e)`, `seg_cov`, `icon_cov`, etc.) computed on Sep-09 frames fall outside the training distribution → LightGBM misclassifies a legitimately GOOD video as FAIL.
- Root cause: **domain gap** in the feature distribution.

---

## What has NOT been tried yet (planned Fix 1–6 from wiki §15.4, only partial Fix 4 done)

| Fix | What | Status |
|-----|------|--------|
| Fix 1 | **Register-once-per-video**: use brightest N frames to solve homography once; warp all frames with that transform | ❌ Not implemented |
| Fix 2 | **Soft C_ref weight**: replace hard `C_MIN_ABS=5` gate with `w = C_ref/(C_ref+5)` | ❌ Not implemented |
| Fix 3 | **Wilson-score ABSTAIN**: replace `n × (fail_ratio − 0.5)` with Wilson CI on weighted fail_ratio | ❌ Not implemented |
| Fix 4 | **`_classify_phase` recalibration for no-Phase-A case** | Partial (v5c did dark_bg gating; full percentile-based threshold not done) |
| Fix 5 | **Synthetic Sep-09 FAIL injection** via Atlas mask darkening/brightening | ❌ Not implemented |
| Fix 6 | **Retrain** with Sep-09 GOOD + synthetic FAIL + re-sweep τ_clf | ❌ Not implemented |

---

## Current infrastructure

- **Tangent server:** `om@tangentthoughttech.com` (password saved)
- **Container:** `fqct_server_v4c`, port 8083
- **Source code is NOT volume-mounted** — update via `docker cp file.py fqct_server_v4c:/app/...`
- **Reload model:** `POST /api/reset`
- Key files on tangent:
  - `~/src/IP4R/server_v4/sandwich/app.py`
  - `~/src/IP4R/server_v3/lcd_crop.py`
  - `~/src/IP4R/server_v3/worker.py`
  - `~/src/IP4R/server_v4/runs/sandwich_20260909_081408/lgb_model.txt` (LightGBM, text format)

**Dataset policy (immutable):**
- Jul-14 (10 vids) and Sep-09 (13 vids) are **eval-only — NEVER train on them**.
- Synthetic Sep-09 FAIL (Fix 5) may be used for training once generated.

---

## What I need from you

The system currently scores Sep-09 = **11/13** and is stuck. The 2 remaining failures are:
1. `201454` — too few phase_b frames → ABSTAIN
2. `202145` — LightGBM domain gap → wrong FAIL

**Please propose a concrete new approach** to fix both, specifically addressing:

1. **For `201454` (ABSTAIN / too few phase_b frames):**
   - The phase classifier is the bottleneck. What changes to `_classify_phase` in `worker.py`
     would reliably detect Phase-B on a Sep-09 video where Phase-A is absent and the jig is black?
   - Should we implement Fix 1 (register-once-per-video) so dark frames don't get dropped before
     even reaching the phase classifier?
   - Or is there a simpler fallback: if `n_b < threshold` and `dark_bg=True`, default to PASS
     (given Sep-09 is all-GOOD, zero-evidence → PASS is the safe call)?

2. **For `202145` (LightGBM domain gap → FAIL):**
   - The feature distribution of Sep-09 bright frames differs from training. What is the best
     way to handle this without retraining (which requires synthetic FAIL data we don't have yet)?
   - Options: feature normalization per session, a second LightGBM trained only on high-Cref frames,
     calibrate τ_clf separately for dark_bg=True videos, or a domain-adaptation wrapper.
   - Or should we just go ahead with Fix 5 (synthetic FAIL injection) + Fix 6 (retrain)?

3. **Is there a simpler structural fix** we are missing — e.g., changing the ABSTAIN→PASS
   default when `dark_bg=True` and `n_b=0` (most Sep-09 videos currently show `no_b_frames`
   in the dashboard and default to PASS — so this pattern is already working for 10/13)?

Please look at the relevant code in `server_v3/worker.py`, `server_v3/lcd_crop.py`, and
`server_v4/sandwich/app.py` before proposing. Prioritize solutions that do NOT require retraining
(no labelled Sep-09 FAIL data exists yet). If retraining is unavoidable, outline what synthetic
data to generate and how to validate it without contaminating the Sep-09 eval set.
