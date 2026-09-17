# IP4R v7b-r2a — AC Remote LCD Inspection API

**Version:** v7b-r2a  
**Base URL:** `http://ip4r-v7b.tangentthoughttech.com`  
**Port:** 8084  
**Status:** Live

---

## Quick Start for Integration

Submit a video, get back a `verdict` and a `proof image URL`. That's the core loop.

```bash
# 1. Submit video → get job_id
curl -X POST "http://ip4r-v7b.tangentthoughttech.com/api/v1/inference/submit-job" \
  -F "video=@/path/to/video.mp4" \
  -F "job_id=JOB-001" \
  -F "defect=non-defective"

# 2. Poll until completed → read verdict + proof link
curl "http://ip4r-v7b.tangentthoughttech.com/api/v1/inference/result?job_id=JOB-001"
```

A video takes ~90–120 seconds to process. Poll every 3–5 seconds until `status` is `"completed"`.

---

## Verdict Values

| `verdict` | Meaning |
|-----------|---------|
| `"non-defective"` | All required icons confirmed — unit is good |
| `"defective"` | At least one required icon was never detected — confident defect |
| `"abstain"` | Icons were glimpsed but inconsistent — send to human re-inspection |

---

## API Reference

### POST /api/v1/inference/submit-job

Submit a video for inspection. Returns immediately with a `job_id`.

**Content-Type:** `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `video` | Yes | Video file (.mp4 / .avi / .mov / .mkv) |
| `job_id` | No | Your own ID string. Auto-generated if omitted. |
| `defect` | No | Hint: `"defective"` or `"non-defective"`. Defaults to `"unclassified"`. |

**Response:**
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

**Error cases:**

| `status` | Cause |
|----------|-------|
| `job_id_conflict` | A job with that `job_id` already exists |
| `invalid_parameter` | `defect` value not in allowed set |
| `missing_file` | No video file in request |

---

### GET /api/v1/inference/result

Poll for the result of a submitted job.

| Parameter | Required |
|-----------|----------|
| `job_id` | Yes |

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

**When completed — the two key fields are `verdict` and `proof.image_url`:**
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

**`defect_type` values:**

| `defect_type` | Verdict | Meaning |
|---------------|---------|---------|
| `all_present` | non-defective | Every icon confirmed |
| `icon_flicker` | abstain | Icons glimpsed but not stable |
| `icon_absence` | defective | One or more icons never detected |

**Error cases:**

| `status` | Cause |
|----------|-------|
| `job_not_found` | Unknown `job_id` |
| `job_failed` | Pipeline error during processing |

---

### GET /api/v1/inference/status

Returns the most recent N job rows (active + completed). Useful for a dashboard or health check.

| Parameter | Required | Range |
|-----------|----------|-------|
| `rows` | Yes | 1–100 |

```bash
curl "http://ip4r-v7b.tangentthoughttech.com/api/v1/inference/status?rows=10"
```

**Response:**
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

---

### GET /proofs/{job_id}/{filename}

Serves the annotated proof frame JPEG referenced in the result response.

The `image_url` in the result already gives the full URL — just fetch it directly.

---

### GET /health

```json
{ "status": "ok", "version": "v7b-r2a", "pending": 0, "processing": 0 }
```

---

## Standard Response Envelope

Every response follows this structure:

```json
{
  "success": true | false,
  "status": "<machine-readable code>",
  "message": "<human-readable description>",
  "data": {}
}
```

---

## Recommended Integration Flow

```
POST /submit-job
  → store job_id
  → start polling loop (every 3–5 s)

GET /result?job_id=<id>
  → status == "processing"  → keep polling
  → status == "completed"   → read data.verdict + data.proof.image_url
  → status == "job_failed"  → handle error (data.error_detail)
  → success == false        → handle error (status field)
```

---

## Web Dashboard

Open `http://ip4r-v7b.tangentthoughttech.com/` in a browser for a visual UI with drag-and-drop upload, live verdict display, icon breakdown, and proof frame preview.

---

## Model — What Is Being Inspected

IP4R inspects AC-remote LCD units during the factory splash self-test (all-segments-on state). A 12 fps video is submitted; the server runs a 3-phase YOLO pipeline checking for 15 required icons.

**Required icons (15):**  
`Auto_Mode, Battery, Cool_Mode, Dry_Mode, Fan_Mode, Fan_Speed, Foot_Display, H_Swing, Light, Lock, Temperature, Timer_OFF, Timer_ON, Turbo, V_Swing`

**Eval results (Sep-15 dataset, 63 videos):**

| Metric | Result |
|--------|--------|
| GOOD pass rate (excl. abstain) | 10/12 = **83%** |
| NOT_GOOD catch rate (excl. abstain) | 34/45 = **76%** |
| Sent to human review (abstain) | 6 videos |

---

## Notes

- Processing time: ~90–120 seconds per video on CPU
- Proof images are stored at `/tmp/v7b_r2a_proofs/{job_id}/` on the server (ephemeral — reboot clears them)
- Uploaded videos are deleted after processing
- Max 2 videos processed in parallel (`MAX_WORKERS = 2`)
