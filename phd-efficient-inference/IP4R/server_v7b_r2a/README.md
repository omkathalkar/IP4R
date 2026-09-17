# IP4R v7b-r2a — AC Remote LCD Inspection Server

**Version:** v7b-r2a  
**Organisation:** Tangent Thought Technologies (T3)  
**API spec:** Inference API Documentation V1  
**Default port:** 8084

---

## What This Model Does

IP4R inspects AC-remote LCD units during the factory splash self-test (all-segments-on state).  
A 12 fps video of the LCD is submitted; the server runs a 3-phase YOLO pipeline and returns one of three verdicts:

| Verdict | API value | Meaning |
|---------|-----------|---------|
| PASS | `non-defective` | All 15 required icons confirmed in ≥2 consecutive frames at ≥70% confidence |
| ABSTAIN | `abstain` | All missing icons were seen in at least 1 frame but never in 2 consecutive frames — borderline/dim LCD, send to human re-inspection |
| FAIL | `defective` | At least one required icon was **never** detected — icon genuinely absent, confident defect |

### Pipeline Phases

```
Phase 1  (0 – 14.9 s)  5-class YOLO (macro model) detects digit-region bounding boxes
Phase 2  (15.0 s)       Element YOLO on masked frame — anomaly flag only (diagnostic)
Phase 3  (16.0 s+)      Element YOLO every frame (batch=8); 21-class checklist
                         Icon confirmed only if it appears in ≥2 consecutive frames at ≥0.70 conf
                         Icons that appear in 1 frame but not 2 consecutive → ABSTAIN
                         Icons never seen → FAIL
```

### Required Icons (15)
`Auto_Mode, Battery, Cool_Mode, Dry_Mode, Fan_Mode, Fan_Speed, Foot_Display, H_Swing, Light, Lock, Temperature, Timer_OFF, Timer_ON, Turbo, V_Swing`

### Eval Results (Sep-15-2026 dataset, 14 GOOD + 49 NOT_GOOD)

| Metric | Value |
|--------|-------|
| GOOD pass rate (excl ABSTAIN) | 10/12 = **83%** |
| NOT_GOOD catch rate (excl ABSTAIN) | 34/45 = **76%** |
| Sent to human review (ABSTAIN) | 6 videos (2 GOOD + 4 NOT_GOOD) |

---

## Directory Structure

```
server_v7b_r2a/
├── README.md          ← this file
├── requirements.txt
├── __init__.py
├── pipeline.py        ← core 3-phase pipeline + ABSTAIN logic
├── eval.py            ← batch evaluator (offline use)
└── app.py             ← FastAPI server (API v1 spec)
```

### Model Files Required (not included — obtain from T3 server)

Place these two YOLO `.pt` files relative to the **project root** (parent of `server_v7b_r2a/`):

```
data/macro_dataset/runs/macro_test/weights/best.pt   ← Phase 1 macro model (5-class)
runs/phase2_elem_v1/weights/best.pt                  ← Phase 3 element model (21-class, mAP50=0.97)
```

Paths are configured at the top of `app.py` (`P1_MODEL`, `ELEM_MODEL`). Update if needed.

---

## Requirements

Python 3.10 or 3.11 recommended.

```
fastapi>=0.110.0
uvicorn>=0.29.0
ultralytics>=8.1.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Running the Server

```bash
# From the project root (parent of server_v7b_r2a/)
cd /path/to/IP4R
source /path/to/venv/bin/activate

# Start server on port 8084
python -m uvicorn server_v7b_r2a.app:app --host 0.0.0.0 --port 8084

# Or directly
python -m server_v7b_r2a.app
```

Keep alive with nohup:
```bash
nohup python -m uvicorn server_v7b_r2a.app:app --host 0.0.0.0 --port 8084 > server.log 2>&1 &
```

---

## API Reference

Base URL: `http://<host>:8084/api/v1/inference`

All responses follow the standard envelope:
```json
{ "success": true, "status": "...", "message": "...", "data": {} }
```

---

### POST /api/v1/inference/submit-job

Submit a video for inspection. Non-blocking — returns `job_id` immediately.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | file | Yes | Video file (.mp4/.avi/.mov/.mkv) |
| `job_id` | string | No | Caller-supplied ID. Auto-generated if omitted. |
| `defect` | string | No | Caller hint: `defective` \| `non-defective`. Defaults to `unclassified`. |

**Example:**
```bash
curl -X POST "http://<host>:8084/api/v1/inference/submit-job" \
  -F "video=@/path/to/video.mp4" \
  -F "job_id=JOB-001" \
  -F "defect=non-defective"
```

**Success response:**
```json
{
  "success": true,
  "status": "job_scheduled",
  "message": "Video received and inference job scheduled successfully",
  "data": {
    "job_id": "JOB-001",
    "file_name": "JOB-001_20260917T120000Z_non_defective.mp4",
    "submitted_at": "2026-09-17T12:00:00Z"
  }
}
```

---

### GET /api/v1/inference/result

Poll the result of a submitted job.

| Parameter | Type | Required |
|-----------|------|----------|
| `job_id` | string | Yes |

**Example:**
```bash
curl "http://<host>:8084/api/v1/inference/result?job_id=JOB-001"
```

**Completed response:**
```json
{
  "success": true,
  "status": "completed",
  "message": "Inference completed successfully",
  "data": {
    "job_id": "JOB-001",
    "submitted_at": "2026-09-17T12:00:00Z",
    "completed_at": "2026-09-17T12:01:55Z",
    "input_defect_flag": "non-defective",
    "verdict": "non-defective",
    "confidence": 0.847,
    "proof": {
      "frame_index": 210,
      "timestamp_in_video": "00:00:17.500",
      "image_url": "http://<host>:8084/proofs/JOB-001/frame_210.jpg",
      "bounding_box": null,
      "defect_type": "all_present"
    },
    "detail": {
      "p1_ok": false,
      "p2_flag": true,
      "p3_thr": 0.7,
      "icons_confirmed": ["Auto_Mode", "Battery", "..."],
      "icons_flickered": [],
      "icons_absent": []
    }
  }
}
```

**Verdict values:** `non-defective` | `abstain` | `defective`  
**Defect type values:** `all_present` | `icon_flicker` | `icon_absence`

**While processing:**
```json
{ "success": true, "status": "processing", "message": "Job is still in progress", "data": { "job_id": "JOB-001" } }
```

---

### GET /api/v1/inference/status

Returns the most recent N job rows.

| Parameter | Type | Required | Range |
|-----------|------|----------|-------|
| `rows` | integer | Yes | 1–100 |

**Example:**
```bash
curl "http://<host>:8084/api/v1/inference/status?rows=10"
```

**Response:**
```json
{
  "success": true,
  "status": "ok",
  "message": "Fetched latest 10 job rows",
  "data": {
    "count": 10,
    "jobs": [
      { "job_id": "JOB-001", "state": "completed", "verdict": "non-defective", "submitted_at": "..." }
    ]
  }
}
```

---

### GET /proofs/{job_id}/{filename}

Serves the annotated proof frame JPEG referenced in the result response.

---

### GET /health

```json
{ "status": "ok", "version": "v7b-r2a", "pending": 0, "processing": 0 }
```

---

## Web Dashboard

Open `http://<host>:8084/` in a browser for the visual inspection UI with drag-and-drop upload, live verdict display, icon breakdown, proof frame preview, and job history.

---

## Notes for Nagesh

- The server runs as a plain Python process (uvicorn), not Docker. Run it inside a `screen` or `nohup` session.
- Proof images are stored in `/tmp/v7b_r2a_proofs/{job_id}/` — this is ephemeral. For persistent storage, change `PROOF_DIR` in `app.py` to a permanent path.
- Uploaded videos are deleted after processing. Set `UPLOAD_DIR` to a persistent path if archival is needed.
- `MAX_WORKERS = 2` — processes up to 2 videos in parallel. Each video takes ~2 minutes on CPU.
- The `P1_MODEL` and `ELEM_MODEL` paths at the top of `app.py` must point to the actual `.pt` files on the deployment machine.
