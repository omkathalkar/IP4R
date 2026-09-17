# IP4R v7b-r2a — AC Remote LCD Inspection API

**API Base URL:** `http://<host>/api/v1/inference`  
**Default port:** 8084  
**Model version:** v7b-r2a

---

## Deployment (Docker)

### 1. Prerequisites

- Docker + Docker Compose installed
- Two YOLO model files (obtain from T3):
  - `p1_best.pt` — Phase 1 macro model (5-class)
  - `elem_best.pt` — Phase 3 element model (21-class)

---

### 2. Get the code

```bash
git clone https://github.com/omkathalkar/IP4R.git
cd IP4R/phd-efficient-inference/IP4R
```

---

### 3. Place model files

```bash
mkdir -p models
cp /path/to/p1_best.pt   models/p1_best.pt
cp /path/to/elem_best.pt models/elem_best.pt
```

The directory should look like:
```
phd-efficient-inference/IP4R/
├── Dockerfile.v7b_r2a
├── docker-compose.v7b_r2a.yml
├── models/
│   ├── p1_best.pt
│   └── elem_best.pt
└── server_v7b_r2a/
    ├── app.py
    ├── pipeline.py
    └── ...
```

---

### 4. Build and run

```bash
docker-compose -f docker-compose.v7b_r2a.yml up -d --build
```

---

### 5. Verify it's running

```bash
curl http://localhost:8084/health
```

Expected response:
```json
{ "status": "ok", "version": "v7b-r2a", "pending": 0, "processing": 0 }
```

---

### 6. Nginx reverse proxy

Nginx sits in front of the container and proxies port 80 → 8084. Add this server block to your nginx config:

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

Then reload nginx:
```bash
nginx -s reload
# or if using Docker nginx:
docker exec <nginx-container> nginx -s reload
```

---

### Other useful commands

```bash
# View logs
docker logs -f ip4r-v7b-r2a

# Stop
docker-compose -f docker-compose.v7b_r2a.yml down

# Restart
docker-compose -f docker-compose.v7b_r2a.yml restart

# Rebuild after code change
docker-compose -f docker-compose.v7b_r2a.yml up -d --build
```

---

## API Reference

### POST /api/v1/inference/submit-job

Submit a video. Returns immediately with a `job_id`. Non-blocking.

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

Poll for result. Processing takes ~90–120 s. Poll every 3–5 s.

**While processing:**
```json
{
  "success": true,
  "status": "processing",
  "data": { "job_id": "JOB-001", "submitted_at": "2026-09-18T08:00:00Z" }
}
```

**When done — key fields: `verdict` and `proof.image_url`:**
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

| `defect_type` | Verdict | Meaning |
|---------------|---------|---------|
| `all_present` | pass | Every icon confirmed |
| `icon_flicker` | abstain | Icons glimpsed but not 2 consecutive frames |
| `icon_absence` | fail | One or more icons never detected |

---

### GET /api/v1/inference/status?rows=10

Recent jobs (active + completed).

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

### GET /proofs/{job_id}/{filename}

Serves the annotated proof frame JPEG. The full URL is in `proof.image_url` — fetch it directly.

---

### GET /health

```json
{ "status": "ok", "version": "v7b-r2a", "pending": 0, "processing": 0 }
```

---

## Standard response envelope

Every response:
```json
{ "success": true|false, "status": "<code>", "message": "<text>", "data": {} }
```

---

## Recommended integration flow

```
POST /submit-job  →  save job_id

loop every 3–5 s:
  GET /result?job_id=<id>
    "processing"  → keep polling
    "completed"   → read data.verdict + data.proof.image_url  ✓
    "job_failed"  → read data.error_detail
    success=false → read status field for error code
```

---

## Web UI

Open `http://<host>/` in a browser — drag-and-drop upload, live verdict, icon breakdown, proof frame preview.

---

## Source

Server code: [`phd-efficient-inference/IP4R/server_v7b_r2a/`](phd-efficient-inference/IP4R/server_v7b_r2a/)  
Docker files: [`Dockerfile.v7b_r2a`](phd-efficient-inference/IP4R/Dockerfile.v7b_r2a) · [`docker-compose.v7b_r2a.yml`](phd-efficient-inference/IP4R/docker-compose.v7b_r2a.yml)
