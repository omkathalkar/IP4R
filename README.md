# IP4R v7b-r2a — AC Remote LCD Inspection API

**Live server:** `http://ip4r-v7b.tangentthoughttech.com`  
**Port:** 8084 · **API version:** v1 · **Model version:** v7b-r2a

---

## Quick Start

```bash
# Step 1 — Submit a video (non-blocking, returns immediately)
curl -X POST "http://ip4r-v7b.tangentthoughttech.com/api/v1/inference/submit-job" \
  -F "video=@/path/to/video.mp4" \
  -F "job_id=JOB-001" \
  -F "defect=non-defective"

# Step 2 — Poll for result (poll every 3–5 s until status == "completed")
curl "http://ip4r-v7b.tangentthoughttech.com/api/v1/inference/result?job_id=JOB-001"
```

Processing takes **~90–120 seconds** per video on CPU.

---

## Verdict Values

| `verdict` | Meaning |
|-----------|---------|
| `"non-defective"` | All required icons confirmed — unit passes |
| `"defective"` | At least one required icon was never detected — confident defect |
| `"abstain"` | Icons were seen but inconsistent — send to human re-inspection |

---

## API Endpoints

### POST /api/v1/inference/submit-job

**Content-Type:** `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `video` | Yes | Video file (.mp4 / .avi / .mov / .mkv) |
| `job_id` | No | Your own ID string. Auto-generated if omitted. |
| `defect` | No | `"defective"` or `"non-defective"`. Defaults to `"unclassified"`. |

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

| Error `status` | Cause |
|----------------|-------|
| `job_id_conflict` | `job_id` already exists |
| `invalid_parameter` | `defect` value not in allowed set |
| `missing_file` | No video file in request |

---

### GET /api/v1/inference/result?job_id=JOB-001

**While processing:**
```json
{
  "success": true,
  "status": "processing",
  "message": "Job is still in progress",
  "data": {
    "job_id": "JOB-001",
    "submitted_at": "2026-09-17T12:00:00Z"
  }
}
```

**When completed — key fields are `verdict` and `proof.image_url`:**
```json
{
  "success": true,
  "status": "completed",
  "message": "Inference completed successfully",
  "data": {
    "job_id": "JOB-001",
    "submitted_at": "2026-09-17T12:00:00Z",
    "completed_at": "2026-09-17T12:02:05Z",
    "input_defect_flag": "non-defective",

    "verdict": "non-defective",
    "confidence": 0.847,

    "proof": {
      "frame_index": 210,
      "timestamp_in_video": "00:00:17.500",
      "image_url": "http://ip4r-v7b.tangentthoughttech.com/proofs/JOB-001/frame_210.jpg",
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

| `defect_type` | Verdict | Meaning |
|---------------|---------|---------|
| `all_present` | non-defective | Every icon confirmed |
| `icon_flicker` | abstain | Icons seen but not stable across frames |
| `icon_absence` | defective | One or more icons never detected |

| Error `status` | Cause |
|----------------|-------|
| `job_not_found` | Unknown `job_id` |
| `job_failed` | Pipeline error; see `data.error_detail` |

---

### GET /api/v1/inference/status?rows=10

Returns the most recent N jobs (active + completed). Useful for dashboards.

```json
{
  "success": true,
  "status": "ok",
  "message": "Fetched latest 10 job rows",
  "data": {
    "count": 2,
    "jobs": [
      { "job_id": "JOB-002", "state": "processing", "verdict": null,            "submitted_at": "2026-09-17T12:02:00Z" },
      { "job_id": "JOB-001", "state": "completed",  "verdict": "non-defective", "submitted_at": "2026-09-17T12:00:00Z" }
    ]
  }
}
```

### GET /proofs/{job_id}/{filename}

Serves the annotated proof frame JPEG. The full URL is already in `proof.image_url` — fetch it directly.

### GET /health

```json
{ "status": "ok", "version": "v7b-r2a", "pending": 0, "processing": 0 }
```

---

## Standard Response Envelope

Every response follows:
```json
{ "success": true|false, "status": "<code>", "message": "<text>", "data": {} }
```

---

## Recommended Integration Flow

```
POST /submit-job
  → save job_id

loop every 3–5 s:
  GET /result?job_id=<id>
  → status "processing"  → keep polling
  → status "completed"   → use data.verdict + data.proof.image_url  ✓
  → status "job_failed"  → handle data.error_detail
  → success false        → handle data.status
```

---

## Web UI

Open `http://ip4r-v7b.tangentthoughttech.com/` in a browser — drag-and-drop upload, live verdict, icon breakdown, proof frame preview.

---

## Source

Full pipeline source: [`phd-efficient-inference/IP4R/server_v7b_r2a/`](phd-efficient-inference/IP4R/server_v7b_r2a/)
