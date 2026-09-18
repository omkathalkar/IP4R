# IP4R — AC Remote LCD Inspection API

Two models in production, both Dockerized with `--restart unless-stopped`:

| Model | Port | URL | Description |
|-------|------|-----|-------------|
| **v7b-r2a** (YOLO) | 8084 | `http://tangentthoughttech.com:8084` | Primary — 3-phase YOLO pipeline, PASS/ABSTAIN/FAIL |
| **v6a** (EfficientNet-B0) | 8083 | `http://tangentthoughttech.com:8083` | Secondary — single-frame CNN classifier |

Both servers implement the same **Inference API V1** spec (endpoints, envelope, verdicts).

---

## Quick health check

```bash
curl http://tangentthoughttech.com:8084/health
# {"status":"ok","version":"v7b-r2a","pending":0,"processing":0}

curl http://tangentthoughttech.com:8083/health
# {"status":"ok","version":"v6a","model":"EfficientNet-B0","pending":0,"processing":0}
```

---

## Server-side setup (v7b-r2a on port 8084)

> One-time deployment on the production server. Docker must be installed.

### Step 1 — Clone the repo

```bash
git clone -b main_deploy https://github.com/omkathalkar/IP4R.git
cd IP4R
```

### Step 2 — Place model weights

```bash
mkdir -p models
# copy your weights files into models/
# p1_best.pt  → Phase 1 macro YOLO
# elem_best.pt → Phase 2/3 element YOLO
```

### Step 3 — Build and run

```bash
docker build -f Dockerfile.v7b_r2a -t ip4r-v7b-r2a:latest .

docker run -d --name ip4r_v7b_r2a \
  -p 8084:8084 \
  -v $(pwd)/models/p1_best.pt:/models/p1_best.pt:ro \
  -v $(pwd)/models/elem_best.pt:/models/elem_best.pt:ro \
  -v ip4r_v7b_proofs:/tmp/v7b_r2a_proofs \
  -v ip4r_v7b_uploads:/tmp/v7b_r2a_uploads \
  -e P1_MODEL=/models/p1_best.pt \
  -e ELEM_MODEL=/models/elem_best.pt \
  -e PUBLIC_HOST=http://<your-host>:8084 \
  --restart unless-stopped \
  ip4r-v7b-r2a:latest
```

### Step 4 — Verify

```bash
curl http://localhost:8084/health
# {"status":"ok","version":"v7b-r2a","pending":0,"processing":0}
```

### Manage the container

```bash
docker ps --filter name=ip4r_v7b_r2a        # status
docker logs ip4r_v7b_r2a -f                  # live logs
docker restart ip4r_v7b_r2a                  # restart
docker stop ip4r_v7b_r2a && docker rm ip4r_v7b_r2a  # remove
```

### Nginx reverse proxy (optional)

Proxy port 80 → 8084:

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

**Base URL:** `http://<host>:8084/api/v1/inference`  
**GUI dashboard:** `http://<host>:8084/`

### POST /api/v1/inference/submit-job

Submit a video. Non-blocking — returns `job_id` immediately.

**Content-Type:** `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `video` | Yes | `.mp4` / `.avi` / `.mov` / `.mkv` |
| `job_id` | No | Your ID string. Auto-generated if omitted. |
| `defect` | No | `"defective"` or `"non-defective"`. Defaults to `"unclassified"`. |

```bash
curl -X POST "http://tangentthoughttech.com:8084/api/v1/inference/submit-job" \
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

Poll every 3–5 s until `status == "completed"`. Typical processing time: **20–35 s**.

```bash
curl "http://tangentthoughttech.com:8084/api/v1/inference/result?job_id=JOB-001"
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
    "completed_at": "2026-09-18T08:00:25Z",
    "input_defect_flag": "non-defective",
    "verdict": "pass",
    "confidence": 0.847,
    "proof": {
      "frame_index": 210,
      "timestamp_in_video": "00:00:17.500",
      "image_url": "http://tangentthoughttech.com:8084/proofs/JOB-001/frame_210.jpg",
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
curl "http://tangentthoughttech.com:8084/api/v1/inference/status?rows=10"
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

---

## v6a EfficientNet-B0 (port 8083)

Same API V1 spec, different model. Replace `:8084` with `:8083` in all URLs above.

### Build and run

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

```bash
curl http://localhost:8083/health
# {"status":"ok","version":"v6a","model":"EfficientNet-B0","pending":0,"processing":0}
```
