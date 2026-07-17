# FQCT Server — User Guide

> **System:** FQCT Inference Server · LCD Defect Detection · AC Remote Production QC  
> **Model:** EfficientNet-B0 (`dts_p2v2_best.pth`)  
> **Version:** 2.0  
> **Last updated:** July 2026

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Accessing the Dashboard](#2-accessing-the-dashboard)
3. [Submitting a Video for Inspection](#3-submitting-a-video-for-inspection)
4. [Reading the Result](#4-reading-the-result)
5. [The Overlay Image](#5-the-overlay-image)
6. [Recent Jobs Table](#6-recent-jobs-table)
7. [Server Statistics](#7-server-statistics)
8. [Adjusting Configuration](#8-adjusting-configuration)
9. [Using the API (Automated / CM4 Units)](#9-using-the-api-automated--cm4-units)
10. [Troubleshooting](#10-troubleshooting)
11. [Quick Reference](#11-quick-reference)

---

## 1. What This System Does

The FQCT Inference Server inspects AC remote LCD displays during the factory DTS (Display Test Sequence). A short video of the LCD in its **all-segments-on state** is submitted to the server; the server analyses it using a trained deep learning model and returns a **PASS** or **FAIL** verdict within approximately **5–10 seconds**.

The system runs on a central GPU machine on the factory LAN and can handle up to **10 simultaneous inspection jobs** — one per test unit.

---

## 2. Accessing the Dashboard

Open a web browser and go to:

```
http://tangentthoughttech.com:8080
```

> If you are on the same LAN as the server, use the server's local IP address instead:
> ```
> http://<SERVER_LAN_IP>:8080
> ```

You will see the **FQCT Inference Server** dashboard with three sections:

| Section | What it shows |
|---------|---------------|
| **Submit Video for Inspection** | Upload form for manual video testing |
| **Recent Jobs** | History of all past inspection results |
| **Config** | Live configuration values (for admin use) |

---

## 3. Submitting a Video for Inspection

### Steps

1. Open the dashboard at `http://tangentthoughttech.com:8080`
2. In the **Submit Video for Inspection** box:
   - **Job ID** — leave blank to auto-generate, or type a meaningful name (e.g. `unit03_batch01`)
   - **Video** — click **Choose File** and select your `.mp4` video file
3. Click **Submit**

The button will show *Uploading…* then *Queued — waiting for result…* while the server processes the video. A progress bar advances automatically.

4. After **5–10 seconds** the result appears directly below the form.

### What video format to use

| Property | Requirement |
|----------|-------------|
| Format | MP4 (`.mp4`) or AVI (`.avi`) |
| Frame rate | 5 FPS or 12 FPS (both supported) |
| Resolution | Any (1080p recommended) |
| Content | Must contain the LCD splash state (all segments on) |
| Duration | 10–30 seconds is ideal |

> The video should capture the **DTS all-segments-on phase** — the moment when every digit, icon, and segment on the LCD is illuminated simultaneously.

---

## 4. Reading the Result

After processing you will see:

```
PASS
Job: job_1784322273633_ghs8 | P2 prob: 94.0% | P2 frames: 49/60 | Inference: 4596ms
```

### Verdict

| Verdict | Meaning |
|---------|---------|
| **PASS** (green) | LCD passed inspection — all segments detected above threshold |
| **FAIL** (red) | LCD failed inspection — one or more segments below threshold |

### Metrics explained

| Field | Meaning |
|-------|---------|
| **P2 prob** | Confidence score from the model. Higher = more confident PASS. Values above 50% → PASS. Example: 94.0% means very high confidence. |
| **P2 frames** | How many frames were analysed out of the total sampled. Example: `49/60` means 49 frames qualified as Phase-2 (all-segments-on) out of 60 frames checked. |
| **Inference** | Time taken to process the video (milliseconds). Typically 4,000–9,000 ms. |

### Segment table

Below the verdict you will see a table of the five LCD zones that were checked:

| Segment | What it covers |
|---------|----------------|
| `left_clock` | Left clock time display (TIME OFF digits) |
| `right_clock` | Right clock time display (TIME ON digits) |
| `center_88` | Large centre temperature digits (88) |
| `signal_bars` | Signal / fan-speed bar indicators |
| `bottom_88888` | Bottom five-digit strip (88888) |

Each row shows:

| Column | Meaning |
|--------|---------|
| **Coverage** | Fraction of the zone that is lit (0.0 – 1.0). A perfect all-on display is typically 0.35 – 0.85 depending on zone. |
| **Min** | Minimum coverage required for that zone to pass. |
| **Status** | **OK** (green) if coverage ≥ min, **FAIL** (red) if below. |

Example result from the demo:

| Segment | Coverage | Min | Status |
|---------|----------|-----|--------|
| left_clock | 0.399 | 0.38 | OK |
| right_clock | 0.308 | 0.25 | OK |
| center_88 | 0.677 | 0.60 | OK |
| signal_bars | 0.760 | 0.35 | OK |
| bottom_88888 | 0.834 | 0.55 | OK |

All five zones passed → overall verdict **PASS**.

---

## 5. The Overlay Image

Directly below the segment table, an **annotated image** of the best Phase-2 frame from the video is displayed. This is the clearest frame the model selected for inspection.

```
┌─────────────────────────────────────┐
│  DTS v2: PASS                       │  ← Verdict banner (green=PASS, red=FAIL)
│  conf=93.9%  P2=60fr (49ok/11fail)  │  ← Confidence and frame count
│                                     │
│   [LCD image with coloured boxes]   │  ← Actual LCD frame from the video
│                                     │
│  Boxes:  green = segment passed     │
│          red   = segment failed     │
└─────────────────────────────────────┘
```

The overlay tells you **exactly which part of the LCD was checked** and whether it passed. A red box on any zone indicates that zone had insufficient segment coverage in the video.

You can also open the overlay image in full size by clicking the **overlay** link in the Recent Jobs table.

---

## 6. Recent Jobs Table

The dashboard automatically refreshes every **5 seconds** and shows the last 30 jobs.

| Column | Meaning |
|--------|---------|
| **Job ID** | Unique identifier for this inspection run |
| **Status** | `pending` → `processing` → `done` or `error` |
| **Verdict** | PASS (green) / FAIL (red) once done |
| **P2 prob** | Model confidence (higher = more certain PASS) |
| **P2 frames** | Frames passing Phase-2 gate / total sampled |
| **Inference (ms)** | Processing time |
| **Created** | Date and time the job was submitted |
| **Overlay** | Link to open the annotated LCD frame image |
| **Error** | Error message if the job failed (e.g. `no_phase2_frame_detected`) |

### Status meanings

| Status | Colour | Meaning |
|--------|--------|---------|
| `pending` | white | Job is in the queue, waiting for a free worker |
| `processing` | orange | Inference is running now |
| `done` | green | Inference complete — verdict is available |
| `error` | red | Something went wrong — check the Error column |

---

## 7. Server Statistics

At the top of the page, five live counters show the current server state:

| Card | Meaning |
|------|---------|
| **Pending** | Jobs queued but not yet started |
| **Processing** | Jobs currently running |
| **Done (all time)** | Total completed inspections since server started |
| **Errors** | Total failed jobs |
| **Avg inference (ms)** | Average processing time over the last 100 jobs |

A healthy server at idle shows **0 Pending / 0 Processing**.

---

## 8. Adjusting Configuration

At the bottom of the dashboard is the **Config** panel showing the live server settings:

```json
{
  "model": {
    "path": "models/dts_p2v2_best.pth",
    "prob_threshold": 0.5
  },
  "video": {
    "frame_step": 5,
    "dark_thresh": 110
  },
  "phase2": {
    "digit_score_min": 0.35,
    "icon_max_cov": 0.12
  },
  "server": {
    "max_workers": 10,
    "retention_days": 30
  }
}
```

To change a value **without restarting the server**, use the update fields at the top of the Config section:

1. Type the key in dot-notation, e.g. `model.prob_threshold`
2. Type the new value, e.g. `0.6`
3. Click **Update**

The change takes effect immediately for all new jobs.

### Key settings to know

| Key | Default | Effect of changing |
|-----|---------|-------------------|
| `model.prob_threshold` | `0.5` | Lower → more lenient (more PASS). Raise → stricter (more FAIL). |
| `video.frame_step` | `5` | Lower = more frames analysed = slower but more accurate. |
| `video.dark_thresh` | `110` | Adjust if lighting conditions change (pixel brightness cutoff). |
| `phase2.digit_score_min` | `0.35` | Min segment fill to qualify a frame as Phase-2. |

> **Admin note:** Changes are written to `config/fqct_server.yaml` on disk and persist across server restarts.

---

## 9. Using the API (Automated / CM4 Units)

The server exposes a simple HTTP API for automated test units.

### Submit a video (fire-and-forget)

```bash
curl -X POST http://<SERVER_IP>:8080/inspect_queue \
     -F "video=@/path/to/dts_test.mp4" \
     -F "job_id=unit03_$(date +%s)"
```

Response:
```json
{"job_id": "unit03_1752000000", "status": "queued"}
```

### Poll for result (after ~20 s IR test window)

```bash
curl "http://<SERVER_IP>:8080/inference_result?job_id=unit03_1752000000"
```

Response when done:
```json
{
  "job_id": "unit03_1752000000",
  "status": "done",
  "result": {
    "passed": true,
    "verdict": "PASS",
    "median_p2_prob": 0.9395,
    "p2_frames": 49,
    "p2_pass": 49,
    "p2_fail": 0,
    "inference_ms": 4596.0,
    "roi_results": [...]
  }
}
```

Use `result.passed` (`true` / `false`) for your automated go/no-go decision.

### Get the overlay image

```bash
curl "http://<SERVER_IP>:8080/jobs/unit03_1752000000/overlay" -o overlay.jpg
```

### All endpoints

| Method | URL | Purpose |
|--------|-----|---------|
| `POST` | `/inspect_queue` | Submit video for inspection |
| `GET` | `/inference_result?job_id=<id>` | Get result for a job |
| `GET` | `/jobs/<id>/overlay` | Download overlay JPEG |
| `GET` | `/health` | Server liveness check |
| `GET` | `/status` | Queue depth and stats |
| `GET` | `/api/jobs?limit=50` | Recent job list (JSON) |
| `GET` | `/api/config` | Current config (JSON) |
| `POST` | `/api/config` | Update a config key |
| `GET` | `/` | Web dashboard |

---

## 10. Troubleshooting

### "no_phase2_frame_detected" error

The server could not find any frame in the video where all LCD segments are fully on.

**Causes and fixes:**

| Cause | Fix |
|-------|-----|
| Video was recorded before the DTS splash started | Start recording 2–3 seconds before the splash state begins |
| Video is too short | Ensure the recording covers at least 5 seconds of the all-on state |
| Camera angle/lighting causing LCD reflections | Reposition the camera or adjust lighting to reduce glare |
| `phase2.digit_score_min` threshold too strict | Lower it slightly via Config panel (e.g. `0.28`) |

### Job stuck in "processing" for more than 60 seconds

The server may be overloaded or a previous job crashed the worker.

1. Check the server statistics — if **Processing** count is at 10, all workers are busy. Wait.
2. If the count is stuck for more than 2 minutes, contact the system admin to restart the server.

### FAIL result on a unit that looks visually fine

1. Open the **overlay** image for that job — identify which segment zone is red.
2. Check whether the camera was properly positioned during recording (all LCD segments visible).
3. If coverage values are close to the minimum threshold, the `model.prob_threshold` may need adjustment — contact the admin.
4. Retry with a fresh recording if the original video had motion blur or poor focus.

### Dashboard shows no jobs / page won't load

The server may be down. Try:

```bash
# Check server health
curl http://<SERVER_IP>:8080/health
```

If this fails, contact the system admin to restart the server:

```bash
# Admin: SSH to server and restart
ssh om@tangentthoughttech.com
kill $(lsof -ti:8080) 2>/dev/null; sleep 1
cd /home/om/src/fqct_server
nohup python3 -m uvicorn server.app:app --host 0.0.0.0 --port 8080 \
    --config config/fqct_server.yaml > server.log 2>&1 &
```

### Cannot reach the dashboard from my machine

If accessing from **outside the factory LAN**, use an SSH tunnel:

```bash
# Run this on your local machine — then open http://localhost:8080
sshpass -p 'useme123' ssh -N -L 8080:localhost:8080 om@tangentthoughttech.com
```

---

## 11. Quick Reference

### Dashboard URL

```
http://tangentthoughttech.com:8080
```

### Typical workflow

```
1. Open dashboard
2. Enter a Job ID (or leave blank)
3. Click "Choose File" → select the DTS video
4. Click "Submit"
5. Wait 5–10 seconds
6. Read verdict: PASS (green) or FAIL (red)
7. Check overlay image to see exactly which zone failed (if FAIL)
```

### Result interpretation at a glance

| P2 prob | P2 frames ratio | What it means |
|---------|-----------------|---------------|
| > 90 % | > 70 % | Strong PASS — all segments clearly on |
| 60–90 % | 50–70 % | Confident PASS — minor variation in some frames |
| 50–60 % | Any | Marginal PASS — re-test recommended |
| < 50 % | Any | **FAIL** — segment(s) missing or dim |

### Demo result (from live test)

The screenshot below was captured from a live 12 FPS test video (`video@12FPS_20260714_143041.mp4`):

- **Verdict:** PASS
- **P2 probability:** 94.0 %  
- **P2 frames:** 49 qualified out of 60 sampled
- **Inference time:** 4,596 ms (~4.6 seconds)
- All five segment zones passed with coverage well above minimums
- The overlay showed the LCD clearly with all segments lit (DTS v2: PASS banner, green ROI boxes)

This represents a healthy, correctly manufactured unit.

### Contact / Support

For issues with the server or model, contact:

> **Om Kathalkar** — CVIT Lab, IIIT Hyderabad  
> Project: IP4R / FQCT  
> Repository: `~/phd-efficient-inference/IP4R/`  
> Server: `om@tangentthoughttech.com` port 8080

---

*This guide covers FQCT Server v2.0 (EfficientNet-B0, Phase-2-gated). Last updated July 2026.*
