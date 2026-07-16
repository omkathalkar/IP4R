# IP4R — AC-Remote LCD Splash-Screen QC

**Version:** v03  
**Author:** Om Kathalkar  
**Detection rate:** 92.7 %  |  **False-alarm rate:** 0.42 %

---

## What it does

IP4R inspects a photo, video, or live camera feed of an AC remote in its
**all-segments-on self-test ("splash") state** and outputs **PASS / FAIL** by verifying
every digit, icon, and printed label against a known-good golden reference image.

It tells you **exactly which element failed** and saves an annotated overlay image + JSON
report for every inspection.

```
Input image / camera
       ↓
  Preprocess (denoise, normalise)
       ↓
  Register to golden (ORB+RANSAC → ECC sub-pixel)
       ↓
  Per-ROI checks
  ├─ Coverage ratio  →  "is this element lit / present?"
  └─ SSIM score      →  "is it correctly shaped / clean?"
       ↓
  PASS / FAIL + overlay + JSON report
```

---

## Folder layout

```
deployment/
├── README.md                 ← you are here
├── TESTING_GUIDE.md          ← step-by-step test plan for the testing team
├── pyproject.toml            ← Python package definition + dependencies
├── config/
│   └── default.yaml          ← all thresholds and pipeline switches
├── data/
│   ├── reference/
│   │   ├── lcd_all_on.jpg    ← golden reference image (the spec)
│   │   ├── rois.yaml         ← ROI map (27 elements defined)
│   │   └── lcd_all_on_original.jpg
│   ├── samples/              ← put input images here
│   └── results/              ← overlay PNGs + JSON reports land here
├── models/
│   └── digit_cnn.pt          ← trained digit CNN weights
├── scripts/
│   ├── inspect_camera.py     ← live camera inspection
│   ├── inspect_video.py      ← video file inspection
│   ├── inspect_batch.py      ← parallel batch inspection
│   └── calibrate.py          ← threshold calibration tool
├── src/ip4r/                 ← Python package source
└── deploy/
    ├── Dockerfile.tier-a     ← lightweight Docker image (~500 MB)
    ├── Dockerfile.tier-ab    ← + PyTorch + Anomalib (~3 GB)
    └── docker-compose.yml
```

---

## Installation

### Option A — Python (recommended for testing)

**Requirements:** Python 3.11+, pip

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install
pip install -e .

# Verify
ip4r --help
```

### Option B — Docker (recommended for production line)

```bash
# Build the lightweight Tier-A image (no GPU needed)
docker compose -f deploy/docker-compose.yml build tier-a

# Sanity check
docker compose -f deploy/docker-compose.yml run --rm tier-a selfcheck
```

---

## Quick start

### 1. Sanity check (no camera / images needed)

```bash
ip4r selfcheck
```

Inspects the golden reference against itself — should be all-PASS. Confirms the install works.

### 2. Test with a synthetic defect

```bash
ip4r synth --mode erase --out data/samples/defect_01.jpg
ip4r inspect data/samples/defect_01.jpg
```

Creates a fake defective image, then inspects it. Should output `FAIL` with the erased element named.

Other defect types: `--mode dim`, `--mode blur`, `--mode shift`.

### 3. Inspect a real photo

Copy a photo of the remote in splash state into `data/samples/`, then:

```bash
ip4r inspect data/samples/your_photo.jpg
```

Or inspect an entire folder:

```bash
ip4r inspect data/samples/
```

### 4. Live camera

```bash
ip4r camera                    # built-in webcam (index 0)
ip4r camera --camera 1         # USB camera
ip4r camera --stable-frames 5  # stricter splash confirmation
```

**Camera window controls:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Force-inspect the current frame now |
| `r` | Reset and scan for the next unit |
| `SPACE` | Save the current raw frame to `data/samples/` |

The system watches the live feed, waits for the splash screen to appear and hold steady for 3
consecutive frames, then fires the full inspection automatically.

### 5. Video file

```bash
python scripts/inspect_video.py path/to/splash_video.mp4
```

Scans the video for the splash screen, triggers inspection automatically, and saves:
- `data/results/video_triggered/annotated.mp4` — annotated video
- `data/results/video_triggered/triggered_frame.jpg` — the inspected frame

### 6. Parallel batch

```bash
python scripts/inspect_batch.py data/samples/ --out data/results/batch_run
```

Processes all images in parallel (4 workers) and writes per-image overlays + a `summary.json`.

---

## Input modes at a glance

| Mode | Command | Use case |
|------|---------|----------|
| Sanity check | `ip4r selfcheck` | Verify install |
| Single photo | `ip4r inspect photo.jpg` | Quick one-off check |
| Folder batch | `ip4r inspect data/samples/` | Offline batch |
| **Live camera** | `ip4r camera` | Production line real-time QC |
| Video file | `python scripts/inspect_video.py video.mp4` | Recorded splash sequence |
| Parallel batch | `python scripts/inspect_batch.py data/samples/` | High-volume offline batch |

---

## Reading results

### Annotated overlay

Saved to `data/results/<image_name>_overlay.jpg` (or `camera_<timestamp>_overlay.jpg`).

- **Green box** = element passed
- **Red box** = element failed (hover text shows which check failed)
- **Green banner** = overall PASS
- **Red banner** = overall FAIL, with count and names of failing elements

### JSON report

Saved alongside every overlay:

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
    }
  ],
  "failed_rois": ["digit_settemp_tens"]
}
```

---

## Configuration

All thresholds are in `config/default.yaml`. Key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tier_a.coverage.coverage_ratio_min` | 0.6749 | Minimum pixel coverage fraction for a lit element |
| `tier_a.ssim.ssim_min` | 0.1945 | Minimum structural similarity score |
| `tier_a.coverage.active_is_dark` | `true` | Reflective LCD (dark segments on light background) |
| `tier_b.enabled` | `false` | Deep anomaly detection (requires good-bank training) |

Pass a custom config file: `ip4r --config my_config.yaml inspect photo.jpg`

---

## Docker deployment

### Tier A — lightweight (production line)

```bash
docker compose -f deploy/docker-compose.yml build tier-a
docker compose -f deploy/docker-compose.yml run --rm tier-a selfcheck
docker compose -f deploy/docker-compose.yml run --rm tier-a inspect data/samples/photo.jpg
docker compose -f deploy/docker-compose.yml run --rm tier-a inspect data/samples/
```

Input images go into `data/samples/` (bind-mounted from host).  
Outputs land in `data/results/` on the host.

### Tier A+B — with deep anomaly detection

Requires a populated `data/good_bank/` and enabling Tier B in config. See `TESTING_GUIDE.md`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FAIL` on a good unit | Registration jitter / poor lighting | Better lighting; ensure remote fills frame |
| Camera never triggers | Splash not fully visible | Press `s` to force-inspect; check frame with `SPACE` |
| `Cannot open camera 0` | Wrong index | Try `--camera 1` or `--camera 2` |
| All ROIs failing | Wrong polarity config | Check `active_is_dark` in `config/default.yaml` |
| Slow on laptop | High-res camera | Camera is capped at 1280×720 automatically |

---

## Testing team

See **`TESTING_GUIDE.md`** for a full step-by-step test plan covering all input modes, expected
outputs, and what to verify.
