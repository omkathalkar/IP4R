# IP4R v6a — EfficientNet-B0 LCD Inspection API

**Port:** 8083 · **Model:** EfficientNet-B0 · **Model version:** v6a  
**API Base URL:** `http://<host>:8083/api/v1/inference`  
**GUI dashboard:** `http://<host>:8083/`

---

## Setup & Run (on the server)

### Step 1 — Clone the repo

```bash
git clone -b main_deploy https://github.com/omkathalkar/IP4R.git
cd IP4R
```

### Step 2 — Place model weight

```bash
mkdir -p models/v6a
# copy best.pth into models/v6a/
```

### Step 3 — Build and run

```bash
docker build -f Dockerfile.v6a -t ip4r-v6a:latest .

docker run -d --name ip4r_v6a \
  -p 8083:8083 \
  -v $(pwd)/models/v6a/best.pth:/models/v6a_best.pth:ro \
  -v ip4r_v6a_proofs:/tmp/v6a_proofs \
  -e V6A_MODEL=/models/v6a_best.pth \
  -e PUBLIC_HOST=http://<your-host>:8083 \
  --restart unless-stopped \
  ip4r-v6a:latest
```

### Step 4 — Verify

```bash
curl http://localhost:8083/health
# {"status":"ok","version":"v6a","model":"EfficientNet-B0","pending":0,"processing":0}
```

### Manage the container

```bash
docker ps --filter name=ip4r_v6a         # status
docker logs ip4r_v6a -f                  # live logs
docker restart ip4r_v6a                  # restart
docker stop ip4r_v6a && docker rm ip4r_v6a  # remove
```

---

## API Reference

### POST /api/v1/inference/submit-job

Submit a video. Non-blocking — returns `job_id` immediately.

**Content-Type:** `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `video` | Yes | `.mp4` / `.avi` / `.mov` / `.mkv` |
| `job_id` | No | Your ID string. Auto-generated if omitted. |
| `defect` | No | `"defective"` or `"non-defective"`. Defaults to `"unclassified"`. |

```bash
curl -X POST "http://tangentthoughttech.com:8083/api/v1/inference/submit-job" \
  -F "video=@/path/to/video.mp4" \
  -F "job_id=JOB-001" \
  -F "defect=non-defective"
```

**Response:**
```json
{
  "success": true,
  "status": "job_scheduled",
  "message": "Video received and inference job scheduled successfully",
  "data": {
    "job_id": "JOB-001",
    "file_name": "JOB-001_20260918T090000Z_non-defective.mp4",
    "submitted_at": "2026-09-18T09:00:00Z"
  }
}
```

---

### GET /api/v1/inference/result?job_id=JOB-001

Poll every 3–5 s until `status == "completed"`. Processing takes ~1–3 s.

```bash
curl "http://tangentthoughttech.com:8083/api/v1/inference/result?job_id=JOB-001"
```

**While processing:**
```json
{ "success": true, "status": "processing", "data": { "job_id": "JOB-001", "submitted_at": "..." } }
```

**Completed — key fields are `verdict` and `proof.image_url`:**
```json
{
  "success": true,
  "status": "completed",
  "data": {
    "job_id": "JOB-001",
    "submitted_at": "2026-09-18T09:00:00Z",
    "completed_at": "2026-09-18T09:00:02Z",
    "input_defect_flag": "non-defective",
    "verdict": "pass",
    "confidence": 0.813,
    "proof": {
      "frame_index": null,
      "timestamp_in_video": "00:00:19.60",
      "image_url": "http://tangentthoughttech.com:8083/proofs/JOB-001/splash.jpg",
      "bounding_box": { "x": 0, "y": 0, "width": 480, "height": 640 },
      "defect_type": "all_present"
    },
    "detail": {
      "model": "EfficientNet-B0",
      "prob_fail": 0.187,
      "inference_ms": 480.7,
      "frame_info": { "t_sec": 19.6, "frame_idx": 100 }
    }
  }
}
```

> **Note:** `proof.frame_index` is `null` — v6a classifies the best splash frame as a whole crop, not individual icon detections. `proof.bounding_box` covers the full LCD crop region.

---

### Verdict values

| `verdict` | Meaning |
|-----------|---------|
| `"pass"` | EfficientNet confidence the LCD is good exceeds threshold (`p_fail < 0.40`) |
| `"fail"` | Defect predicted with high confidence (`p_fail > 0.60`) |
| `"abstain"` | Confidence in uncertain zone (`p_fail 0.40–0.60`) — send to human re-inspection |

---

### GET /api/v1/inference/status?rows=10

```bash
curl "http://tangentthoughttech.com:8083/api/v1/inference/status?rows=10"
```

```json
{
  "success": true,
  "status": "ok",
  "message": "Fetched latest 2 job rows",
  "data": {
    "count": 2,
    "jobs": [
      { "job_id": "JOB-002", "state": "processing", "verdict": null,   "submitted_at": "..." },
      { "job_id": "JOB-001", "state": "completed",  "verdict": "pass", "submitted_at": "..." }
    ]
  }
}
```

---

### GET /health

```json
{ "status": "ok", "version": "v6a", "model": "EfficientNet-B0", "pending": 0, "processing": 0 }
```

---

## Integration flow

```
POST /submit-job  →  save job_id

loop every 3–5 s:
  GET /result?job_id=<id>
    "processing"  →  keep polling
    "completed"   →  read data.verdict + data.proof.image_url  ✓
    "job_failed"  →  read data.error_detail
    success=false →  read status field for error code
```

---

## Error responses

| status | success | When |
|--------|---------|------|
| `job_scheduled` | true | Video accepted |
| `job_id_conflict` | false | Supplied `job_id` already exists |
| `invalid_parameter` | false | Bad `defect` value or invalid `rows` |
| `missing_file` | false | No video in request |
| `job_not_found` | false | Unknown `job_id` |
| `job_failed` | false | Inference error — check `data.error_detail` |

---

## How it differs from v7b-r2a

| | v6a (this) | v7b-r2a |
|--|-----------|---------|
| Port | 8083 | 8084 |
| Model | EfficientNet-B0 (CNN) | 3-phase YOLO |
| Speed | ~1–3 s | ~20–35 s |
| `proof.frame_index` | `null` (whole-frame classifier) | integer |
| `verdict` basis | `p_fail` probability threshold | per-icon checklist (15 icons) |
| ABSTAIN zone | `p_fail` 0.40–0.60 | icon glimpsed but not confirmed |
