# IP4R v7b-r2a — AC Remote LCD Inspection API

**API Base URL:** `http://<host>/api/v1/inference`  
**Default port:** 8084 · **Model version:** v7b-r2a

---

## Setup & Run (on the server)

### Step 1 — Clone the repo

```bash
git clone -b main_deploy https://github.com/omkathalkar/IP4R.git
cd IP4R/phd-efficient-inference/IP4R
```

### Step 2 — Install dependencies

```bash
pip install -r server_v7b_r2a/requirements.txt
```

### Step 3 — Set model paths

```bash
export P1_MODEL=/path/to/p1_best.pt
export ELEM_MODEL=/path/to/elem_best.pt
```

### Step 4 — Start the server

```bash
nohup python -m uvicorn server_v7b_r2a.app:app --host 0.0.0.0 --port 8084 > server.log 2>&1 &
echo "Server PID: $!"
```

### Step 5 — Verify

```bash
curl http://localhost:8084/health
```

Expected:
```json
{ "status": "ok", "version": "v7b-r2a", "pending": 0, "processing": 0 }
```

---

### Other commands

```bash
# View logs
tail -f server.log

# Stop server
pkill -f server_v7b_r2a

# Restart
pkill -f server_v7b_r2a
nohup python -m uvicorn server_v7b_r2a.app:app --host 0.0.0.0 --port 8084 > server.log 2>&1 &
```

---

### Nginx reverse proxy

Proxy port 80 → 8084. Add this block to your nginx config and reload:

```nginx
server {
    listen 80;
    server_name <your-domain>;

    client_max_body_size 512M;

    location / {
        proxy_pass         http://127.0.0.1:8084;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

```bash
nginx -s reload
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
curl -X POST "http://<host>/api/v1/inference/submit-job" \
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
    "file_name": "JOB-001_20260918T080000Z_non_defective.mp4",
    "submitted_at": "2026-09-18T08:00:00Z"
  }
}
```

---

### GET /api/v1/inference/result?job_id=JOB-001

Poll every 3–5 s until `status == "completed"`. Processing takes ~90–120 s.

```bash
curl "http://<host>/api/v1/inference/result?job_id=JOB-001"
```

**While processing:**
```json
{ "success": true, "status": "processing", "data": { "job_id": "JOB-001" } }
```

**Completed — the two key fields are `verdict` and `proof.image_url`:**
```json
{
  "success": true,
  "status": "completed",
  "data": {
    "job_id": "JOB-001",
    "submitted_at": "2026-09-18T08:00:00Z",
    "completed_at": "2026-09-18T08:02:05Z",
    "input_defect_flag": "non-defective",
    "verdict": "pass",
    "confidence": 0.847,
    "proof": {
      "frame_index": 210,
      "timestamp_in_video": "00:00:17.500",
      "image_url": "http://<host>/proofs/JOB-001/frame_210.jpg",
      "bounding_box": { "x": 145, "y": 62, "width": 80, "height": 44 },
      "defect_type": "all_present"
    },
    "detail": {
      "p1_ok": true,
      "p2_flag": false,
      "p3_thr": 0.7,
      "icons_confirmed": ["Auto_Mode", "Battery", "Cool_Mode", "Dry_Mode", "Fan_Mode",
                          "Fan_Speed", "Foot_Display", "H_Swing", "Light", "Lock",
                          "Temperature", "Timer_OFF", "Timer_ON", "Turbo", "V_Swing"],
      "icons_flickered": [],
      "icons_absent": []
    }
  }
}
```

---

### Verdict values

| `verdict` | Meaning |
|-----------|---------|
| `"pass"` | All 15 required icons confirmed |
| `"fail"` | At least one icon was never detected — confident defect |
| `"abstain"` | Icons seen but not stable — send to human re-inspection |

---

### GET /api/v1/inference/status?rows=10

```bash
curl "http://<host>/api/v1/inference/status?rows=10"
```

```json
{
  "success": true,
  "status": "ok",
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
{ "status": "ok", "version": "v7b-r2a", "pending": 0, "processing": 0 }
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
