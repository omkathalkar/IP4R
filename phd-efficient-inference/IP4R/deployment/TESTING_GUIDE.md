# IP4R — Testing Team Guide

**System:** AC-Remote LCD Splash-Screen Quality Check  
**Version:** v03  
**Contact:** Om Kathalkar

---

## What this system does

Point a camera (or provide a photo/video) at an AC remote in its **all-segments-on self-test
("splash") state**. IP4R compares every digit, icon, and label against the golden reference image
and outputs:

- **PASS** — every element is lit, correctly shaped, and clean
- **FAIL** — one or more elements are missing, dim, blurred, or malformed, with the specific
  failing element(s) named

Outputs for every inspection: annotated overlay image (PNG) + structured JSON report, both saved
to `data/results/`.

---

## Installation

### Requirements
- Python 3.11 or newer
- macOS, Linux, or Windows
- A USB or built-in camera (for live mode) **or** image/video files (for offline mode)

### Steps

```bash
# 1. Clone / copy the repo to your machine
cd IP4R

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install
pip install -e .

# 4. Verify install
ip4r --help
```

You should see the list of subcommands: `selfcheck`, `inspect`, `camera`, `synth`, `roi-edit`.

---

## Input modes

| Mode | Command | When to use |
|------|---------|-------------|
| **Sanity check** | `ip4r selfcheck` | First run — proves the pipeline works |
| **Single photo** | `ip4r inspect photo.jpg` | One image captured with phone/camera |
| **Folder of photos** | `ip4r inspect data/samples/` | Batch of saved images |
| **Live camera** | `ip4r camera` | Real-time QC on the production line |
| **Video file** | `python scripts/inspect_video.py video.mp4` | Recorded video of the splash sequence |
| **Batch with overlays** | `python scripts/inspect_batch.py data/samples/` | Parallel batch + per-image overlay |

---

## Test Plan — Step by Step

### Step 0 — Sanity check (no hardware needed)

```bash
ip4r selfcheck
```

**Expected output:**
```
=== IP4R PASS : lcd_all_on.jpg ===
registration: {'method': 'orb_ecc', 'fallback': False}
  [ok ] digit_settemp_tens   cov=0.xx   ssim=0.xx
  [ok ] ...
  -> 0/N element(s) failed
```

This confirms the pipeline is installed and working. If this fails, stop and report the error.

---

### Step 1 — Test with a synthetic defect (no hardware needed)

```bash
# Create a fake defective image (erases a random element)
ip4r synth --mode erase --out data/samples/defect_erase.jpg
ip4r inspect data/samples/defect_erase.jpg
```

**Expected output:** `FAIL` with one ROI flagged.

```bash
# Test other defect types
ip4r synth --mode dim   --out data/samples/defect_dim.jpg
ip4r synth --mode blur  --out data/samples/defect_blur.jpg
ip4r synth --mode shift --out data/samples/defect_shift.jpg

ip4r inspect data/samples/
```

**Expected output:** All 4 defective images → `FAIL`. Overlay images saved to `data/results/`.

---

### Step 2 — Test with a real photo

Capture a photo of the AC remote in splash state (all segments on) using a phone or camera.
Copy the file to `data/samples/`.

```bash
ip4r inspect data/samples/your_photo.jpg
```

**Expected output:** `PASS` if the remote is good. Check the overlay in `data/results/` — green
boxes = passing ROIs, red boxes = failing ROIs.

---

### Step 3 — Live camera test

Connect a USB camera or use the built-in webcam.

```bash
# Built-in camera (index 0)
ip4r camera

# External USB camera (try index 1 or 2)
ip4r camera --camera 1
```

**Window controls:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Force-inspect the current frame immediately |
| `r` | Reset — scan again after a result is shown |
| `SPACE` | Save the current raw frame to `data/samples/` |

**Workflow:**
1. The window shows `SCANNING…` with coverage values updating in real time.
2. Hold the AC remote steady in front of the camera in splash state.
3. When coverage passes the threshold for 3 consecutive frames the display changes to
   `STABILIZING… (1/3)`, `(2/3)`, `(3/3)`.
4. The full inspection fires automatically → `PASS` or `FAIL` banner appears.
5. Overlay + JSON report saved to `data/results/`.
6. Press `r` to reset and test the next unit.

**Tip — camera not found?** Try:
```bash
ip4r camera --camera 0    # built-in
ip4r camera --camera 1    # first USB
ip4r camera --camera 2    # second USB
```

**Tip — too many false triggers?** Increase `--stable-frames`:
```bash
ip4r camera --stable-frames 5
```

---

### Step 4 — Test with a video file

If you record the splash sequence as a video:

```bash
python scripts/inspect_video.py data/samples/remote_video.mp4
```

The script scans for the splash screen automatically and triggers inspection when found.
Outputs: `data/results/video_triggered/annotated.mp4` (annotated video) +
`triggered_frame.jpg`.

Options:
```bash
python scripts/inspect_video.py video.mp4 \
    --start-skip 5        # skip first 5 s (while picking up the remote)
    --stable-frames 3     # frames to confirm splash
    --out data/results/my_test
```

---

### Step 5 — Batch inspection

```bash
python scripts/inspect_batch.py data/samples/ --out data/results/batch_run
```

Processes all images in parallel (4 workers by default). Outputs:
- Per-image overlay: `<stem>_overlay.jpg`
- Per-image report: `<stem>_report.json`
- Summary: `summary.json` with pass/fail counts

```bash
# Show summary
python -c "import json; d=json.load(open('data/results/batch_run/summary.json')); print(d)"
```

---

## Reading the results

### Overlay image
- **Green box** = element passed (coverage and SSIM both OK)
- **Red box** = element failed (label shows which check failed)
- **Green banner** = overall PASS
- **Red banner** = overall FAIL + count of failing elements

### JSON report (`data/results/<name>_report.json`)

```json
{
  "image_path": "data/samples/photo.jpg",
  "passed": false,
  "registration": {"method": "orb_ecc", "fallback": false},
  "roi_results": [
    {
      "name": "digit_settemp_tens",
      "passed": false,
      "coverage": 0.12,
      "ssim": 0.45,
      "reason": "coverage below threshold"
    },
    ...
  ],
  "failed_rois": ["digit_settemp_tens"]
}
```

---

## What to verify during testing

| Check | How |
|-------|-----|
| Good unit → PASS | Inspect a known-good remote |
| Missing segment → FAIL | `ip4r synth --mode erase` |
| Dim segment → FAIL | `ip4r synth --mode dim` |
| Blurred LCD → FAIL | `ip4r synth --mode blur` |
| Correct ROI localisation | Check overlay boxes align with LCD elements |
| Camera auto-trigger | Live camera confirms splash in ≤ 3 s |
| Results saved | Check `data/results/` after each run |
| No crash on bad image | `ip4r inspect data/samples/` with a blurry/dark shot |

---

## Things that can go wrong

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FAIL` on a good unit | Registration failed (check terminal: `fallback: True`) | Hold remote steadier, ensure good lighting |
| Never triggers in camera mode | Remote not fully in frame, or splash not showing | Press `s` to force-inspect; check `--stable-frames` |
| Very slow in camera mode | Large frame resolution | Camera is auto-set to 1280×720; laptop cameras may still be slow |
| `FileNotFoundError: rois.yaml` | ROI map missing | Run `ip4r roi-edit` once to create it |
| `Cannot open camera index 0` | Wrong camera index | Try `--camera 1` or `--camera 2` |

---

## Docker (optional — no Python install needed)

```bash
# Build (from IP4R/ root)
docker compose -f deploy/docker-compose.yml build tier-a

# Sanity check
docker compose -f deploy/docker-compose.yml run --rm tier-a selfcheck

# Inspect a folder (files must be in IP4R/data/samples/)
docker compose -f deploy/docker-compose.yml run --rm tier-a inspect data/samples/

# Synth + inspect
docker compose -f deploy/docker-compose.yml run --rm tier-a synth --mode erase --out data/samples/defect_01.jpg
docker compose -f deploy/docker-compose.yml run --rm tier-a inspect data/samples/defect_01.jpg
```

> **Note:** Live camera mode (`ip4r camera`) requires passing the camera device to Docker
> (`--device /dev/video0`) and is easier to run directly with Python on the host.

---

## File locations summary

| What | Where |
|------|-------|
| Golden reference image | `data/reference/lcd_all_on.jpg` |
| ROI map | `data/reference/rois.yaml` |
| Input images | `data/samples/` |
| Output overlays + JSON | `data/results/` |
| Config (thresholds) | `config/default.yaml` |
