"""
IP4R v7b-r2a Inference Server
Implements Inference API Documentation V1 (Tangent Thought Technologies)

Base URL : /api/v1/inference
Endpoints:
  POST /api/v1/inference/submit-job     Upload video, schedule job
  GET  /api/v1/inference/result         Poll job verdict + proof
  GET  /api/v1/inference/status         Recent job summary (last N rows)
  GET  /proofs/{job_id}/{filename}      Serve proof frame JPEG
  GET  /health                          Health check (internal)
  GET  /                                Web dashboard UI
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from server_v7b_r2a.pipeline import (
    P1_CLASSES, P1_CONF_THR, P1_FALLBACK_ENABLED, P1_SAMPLE_STEP,
    P2_CONF_THR, P2_FLAG_ENABLED, P3_CONF_THR, P3_START_SEC,
    P1_END_SEC, P1_SKIP_SEC, P2_SEC,
    ELEMENT_NAMES, REQUIRED_ICONS,
    ADAPTIVE_THR_ENABLED, BATCH_SIZE, BRIGHTNESS_NORM_MODE,
    _load, _normalize_frame, _adaptive_thr, _p1_fallback_boxes,
    _infer_batch, _apply_temporal_glimpse, _three_way_verdict,
)

# ── Config ────────────────────────────────────────────────────────────────────
PORT        = 8084
UPLOAD_DIR  = Path("/tmp/v7b_r2a_uploads")
PROOF_DIR   = Path("/tmp/v7b_r2a_proofs")
MAX_WORKERS = 2
MAX_HISTORY = 100

P1_MODEL   = str(_ROOT / "data/macro_dataset/runs/macro_test/weights/best.pt")
ELEM_MODEL = str(_ROOT / "runs/phase2_elem_v1/weights/best.pt")

VALID_DEFECT = {"defective", "non-defective"}

for _d in (UPLOAD_DIR, PROOF_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Job store ─────────────────────────────────────────────────────────────────
_lock    = threading.Lock()
_jobs: dict[str, dict] = {}
_history: deque = deque(maxlen=MAX_HISTORY)
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

app = FastAPI(title="IP4R v7b-r2a Inference Server", version="1.0.0")


# ── Timestamp helper — ISO 8601 with Z suffix, no microseconds ────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Standard envelope helpers ─────────────────────────────────────────────────

def _ok(status: str, message: str, data: dict) -> JSONResponse:
    return JSONResponse({"success": True,  "status": status,  "message": message, "data": data})

def _err(status: str, message: str, data: dict = {}, code: int = 400) -> JSONResponse:
    return JSONResponse({"success": False, "status": status, "message": message, "data": data},
                        status_code=code)


# ── Verdict mapping ───────────────────────────────────────────────────────────
# Internal → API spec

def _api_verdict(verdict: str) -> str:
    return {"PASS": "non-defective", "FAIL": "defective", "ABSTAIN": "abstain"}.get(verdict, "unknown")


def _confidence(verdict: str, checklist: dict, fail_icons: list) -> float:
    if verdict == "PASS":
        vals = [v for n, v in checklist.items() if n in REQUIRED_ICONS and v is not None]
        return round(float(np.mean(vals)), 3) if vals else 1.0
    if verdict == "FAIL":
        return round(len(fail_icons) / max(len(REQUIRED_ICONS), 1), 3)
    return 0.5  # ABSTAIN


# ── Pipeline + proof frame extraction ────────────────────────────────────────

def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _run_and_capture(video_path: str, job_id: str) -> dict:
    """
    Full v7b-r2a pipeline. Returns result dict including proof frame path.
    """
    vp = Path(video_path)
    cap = cv2.VideoCapture(str(vp))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 12.0
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    f_p1_start = int(P1_SKIP_SEC  * fps)
    f_p1_end   = min(int(P1_END_SEC   * fps), n_frames - 1)
    f_p2       = min(int(P2_SEC       * fps), n_frames - 1)
    f_p3_start = min(int(P3_START_SEC * fps), n_frames - 1)

    # Phase 1
    m1 = _load(P1_MODEL)
    p1_best: dict[str, tuple[float, list]] = {}
    cap = cv2.VideoCapture(str(vp))
    fi = f_p1_start
    while fi <= f_p1_end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frm = cap.read()
        if ok:
            preds = m1.predict(frm, conf=P1_CONF_THR, imgsz=640, verbose=False)
            if preds and len(preds[0].boxes):
                for i in range(len(preds[0].boxes)):
                    cid = int(preds[0].boxes.cls[i].item())
                    if cid >= len(P1_CLASSES):
                        continue
                    nm = P1_CLASSES[cid]
                    cf = float(preds[0].boxes.conf[i].item())
                    bx = preds[0].boxes.xyxy[i].cpu().numpy().tolist()
                    if nm not in p1_best or cf > p1_best[nm][0]:
                        p1_best[nm] = (cf, bx)
        fi += P1_SAMPLE_STEP
    cap.release()

    p1_ok = len(p1_best) == len(P1_CLASSES)
    p1_fallback_used = False
    mask_boxes: list[list[int]] = [list(map(int, bx)) for _, bx in p1_best.values()]
    if P1_FALLBACK_ENABLED and not p1_ok:
        p1_fallback_used = True
        fallback = _p1_fallback_boxes(W, H)
        detected = set(p1_best.keys())
        for i, cls_name in enumerate(P1_CLASSES):
            if cls_name not in detected:
                mask_boxes.append(fallback[i])

    # Phase 2
    m2 = _load(ELEM_MODEL)
    anomalous: list[str] = []
    cap = cv2.VideoCapture(str(vp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_p2)
    ok2, p2_frm = cap.read()
    cap.release()
    if ok2:
        masked = p2_frm.copy()
        PAD = 6
        for bx in mask_boxes:
            x1, y1, x2, y2 = bx
            masked[max(0, y1 - PAD):y2 + PAD, max(0, x1 - PAD):x2 + PAD] = 255
        masked_norm = _normalize_frame(masked)
        preds = m2.predict(masked_norm, conf=P2_CONF_THR, imgsz=640, verbose=False)
        if preds and len(preds[0].boxes):
            for i in range(len(preds[0].boxes)):
                cid = int(preds[0].boxes.cls[i].item())
                if cid < len(ELEMENT_NAMES):
                    anomalous.append(ELEMENT_NAMES[cid])
    p2_flag = P2_FLAG_ENABLED and any(n in REQUIRED_ICONS for n in anomalous)

    # Phase 3 — collect + batch infer
    p3_raw:  list[np.ndarray] = []
    p3_norm: list[np.ndarray] = []
    p3_idx:  list[int]        = []
    cap = cv2.VideoCapture(str(vp))
    for fi in range(f_p3_start, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok3, frm = cap.read()
        if not ok3:
            continue
        p3_raw.append(frm)
        p3_norm.append(_normalize_frame(frm))
        p3_idx.append(fi)
    cap.release()

    p3_thr      = _adaptive_thr(p3_raw)
    frame_dets  = _infer_batch(m2, p3_norm, p3_idx, conf=p3_thr)
    checklist, glimpsed = _apply_temporal_glimpse(frame_dets, p3_idx, p3_thr)
    verdict, abstain_icons, fail_icons = _three_way_verdict(checklist, glimpsed)

    # Proof frame: frame with the most confirmed icon detections
    best_fi = p3_idx[0] if p3_idx else f_p3_start
    best_count = -1
    for fi in p3_idx:
        count = sum(1 for nm, cf, _ in frame_dets.get(fi, [])
                    if cf >= p3_thr and nm in REQUIRED_ICONS)
        if count > best_count:
            best_count = count
            best_fi = fi

    proof_path = None
    proof_raw_idx = p3_idx.index(best_fi) if best_fi in p3_idx else 0
    if p3_raw:
        proof_frame = p3_raw[proof_raw_idx].copy()
        # Draw detections on proof frame
        for nm, cf, bx in frame_dets.get(best_fi, []):
            if cf >= p3_thr:
                x1, y1, x2, y2 = [int(v) for v in bx]
                color = (0, 220, 0) if nm in REQUIRED_ICONS else (200, 200, 0)
                cv2.rectangle(proof_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(proof_frame, f"{nm} {cf:.2f}",
                            (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        # Verdict banner
        vc = (0, 200, 0) if verdict == "PASS" else (0, 140, 255) if verdict == "ABSTAIN" else (0, 0, 220)
        cv2.rectangle(proof_frame, (0, 0), (W, 36), (0, 0, 0), -1)
        cv2.putText(proof_frame,
                    f"IP4R v7b-r2a | {verdict} | {len([n for n in REQUIRED_ICONS if checklist[n]])}/{len(REQUIRED_ICONS)} icons | thr={p3_thr:.2f}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, vc, 1, cv2.LINE_AA)

        proof_dir = PROOF_DIR / job_id
        proof_dir.mkdir(parents=True, exist_ok=True)
        proof_filename = f"frame_{best_fi}.jpg"
        proof_path = str(proof_dir / proof_filename)
        cv2.imwrite(proof_path, proof_frame)

    # Bounding box of the highest-confidence required-icon detection on the proof frame
    proof_bbox = None
    for nm, cf, bx in sorted(frame_dets.get(best_fi, []), key=lambda x: -x[1]):
        if nm in REQUIRED_ICONS:
            x1, y1, x2, y2 = [int(v) for v in bx]
            proof_bbox = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
            break

    found = [n for n in REQUIRED_ICONS if checklist[n] is not None]
    conf  = _confidence(verdict, checklist, fail_icons)
    t_sec = best_fi / fps

    return {
        "verdict":         verdict,
        "api_verdict":     _api_verdict(verdict),
        "confidence":      conf,
        "p1_ok":           p1_ok,
        "p1_fallback":     p1_fallback_used,
        "p2_flag":         p2_flag,
        "p2_anomalous":    anomalous,
        "p3_thr":          round(p3_thr, 3),
        "p3_found":        found,
        "p3_abstain_icons":abstain_icons,
        "p3_fail_icons":   fail_icons,
        "p3_checklist":    checklist,
        "p3_glimpsed":     glimpsed,
        "proof_frame_idx": best_fi,
        "proof_frame_ts":  _ts(t_sec),
        "proof_path":      proof_path,
        "proof_bbox":      proof_bbox,
        "defect_type":     ("icon_absence" if verdict == "FAIL"
                            else "icon_flicker" if verdict == "ABSTAIN"
                            else "all_present"),
    }


# ── Background worker ─────────────────────────────────────────────────────────

def _process(job_id: str, video_path: str, defect_hint: str):
    with _lock:
        _jobs[job_id]["state"] = "processing"
        _jobs[job_id]["started_at"] = _utcnow()

    try:
        result = _run_and_capture(video_path, job_id)
        with _lock:
            _jobs[job_id].update({
                "state":        "completed",
                "result":       result,
                "completed_at": _utcnow(),
                "error":        None,
            })
    except Exception as e:
        with _lock:
            _jobs[job_id].update({
                "state":        "job_failed",
                "result":       None,
                "completed_at": _utcnow(),
                "error":        str(e),
            })
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass
        with _lock:
            _history.appendleft({**_jobs[job_id], "job_id": job_id})


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/v1/inference/submit-job")
async def submit_job(
    video:  UploadFile = File(...),
    job_id: str        = Form(default=""),
    defect: str        = Form(default="unclassified"),
):
    # Validate defect
    if defect not in VALID_DEFECT and defect != "unclassified":
        return _err("invalid_parameter",
                    "Field 'defect' must be one of: defective, non-defective")

    # Validate file
    if not video.filename:
        return _err("missing_file", "No video file found in the request")
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        return _err("missing_file", "Upload must be a video file (.mp4/.avi/.mov/.mkv)")

    # Resolve job ID
    jid = job_id.strip() or f"JOB-{uuid.uuid4().hex[:6].upper()}"
    with _lock:
        if jid in _jobs:
            return _err("job_id_conflict",
                        "A job with the provided job_id already exists",
                        {"job_id": jid})

    # Save file using naming convention: <JOB-ID>_<timestamp>_<defect>.<ext>
    ts_str      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    defect_tag  = defect.replace("-", "_")
    filename    = f"{jid}_{ts_str}_{defect_tag}{ext}"
    video_path  = UPLOAD_DIR / filename
    with open(video_path, "wb") as f:
        f.write(await video.read())

    submitted_at = _utcnow()
    with _lock:
        _jobs[jid] = {
            "state":        "queued",
            "filename":     video.filename,
            "stored_name":  filename,
            "defect_hint":  defect,
            "submitted_at": submitted_at,
            "started_at":   None,
            "completed_at": None,
            "result":       None,
            "error":        None,
        }

    _executor.submit(_process, jid, str(video_path), defect)

    return _ok("job_scheduled",
               "Video received and inference job scheduled successfully",
               {
                   "job_id":       jid,
                   "file_name":    filename,
                   "submitted_at": submitted_at,
               })


@app.get("/api/v1/inference/result")
async def get_result(job_id: str = Query(...)):
    with _lock:
        job = _jobs.get(job_id)

    if job is None:
        return _err("job_not_found", "No job found for the given job_id",
                    {}, code=404)

    state = job["state"]

    if state in ("queued", "processing"):
        return _ok("processing", "Job is still in progress", {
            "job_id":       job_id,
            "submitted_at": job["submitted_at"],
        })

    if state == "job_failed":
        return _err("job_failed",
                    "Inference failed while processing the video",
                    {"job_id": job_id, "error_detail": job.get("error", "")},
                    code=500)

    # completed
    r    = job["result"]
    host = "http://ip4r-v7b.tangentthoughttech.com"
    proof_filename = f"frame_{r['proof_frame_idx']}.jpg"
    image_url = f"{host}/proofs/{job_id}/{proof_filename}"

    return _ok("completed", "Inference completed successfully", {
        "job_id":            job_id,
        "submitted_at":      job["submitted_at"],
        "completed_at":      job["completed_at"],
        "input_defect_flag": job["defect_hint"],
        "verdict":           r["api_verdict"],
        "confidence":        r["confidence"],
        "proof": {
            "frame_index":        r["proof_frame_idx"],
            "timestamp_in_video": r["proof_frame_ts"],
            "image_url":          image_url,
            "bounding_box":       r.get("proof_bbox"),
            "defect_type":        r["defect_type"],
        },
        "detail": {
            "p1_ok":           r["p1_ok"],
            "p2_flag":         r["p2_flag"],
            "p3_thr":          r["p3_thr"],
            "icons_confirmed": r["p3_found"],
            "icons_flickered": r["p3_abstain_icons"],
            "icons_absent":    r["p3_fail_icons"],
        },
    })


@app.get("/api/v1/inference/status")
async def get_status(rows: int = Query(...)):
    if rows < 1 or rows > 100:
        return _err("invalid_parameter",
                    "Parameter 'rows' must be an integer between 1 and 100")
    with _lock:
        # Active jobs (queued / processing) shown first, then completed history
        active = [
            {"job_id": jid, **job}
            for jid, job in _jobs.items()
            if job["state"] in ("queued", "processing")
        ]
        history = list(_history)

    combined = active + history
    recent = combined[:rows]

    jobs_out = []
    for j in recent:
        jobs_out.append({
            "job_id":       j.get("job_id"),
            "state":        j.get("state"),
            "verdict":      _api_verdict(j["result"]["verdict"]) if j.get("result") else None,
            "submitted_at": j.get("submitted_at"),
        })

    return _ok("ok", f"Fetched latest {len(jobs_out)} job rows", {
        "count": len(jobs_out),
        "jobs":  jobs_out,
    })


@app.get("/proofs/{job_id}/{filename}")
async def serve_proof(job_id: str, filename: str):
    path = PROOF_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "Proof image not found")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/health")
async def health():
    with _lock:
        pending    = sum(1 for j in _jobs.values() if j["state"] == "queued")
        processing = sum(1 for j in _jobs.values() if j["state"] == "processing")
    return {"status": "ok", "version": "v7b-r2a", "pending": pending, "processing": processing}


# ── Dashboard UI ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return HTMLResponse(content=_UI)


_UI = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>IP4R v7b-r2a — Inspection Server</title>
<style>
  :root {
    --bg:#0d1117; --surface:#161b22; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e;
    --pass:#3fb950; --abstain:#d29922; --fail:#f85149; --accent:#58a6ff;
    --radius:10px;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}

  .topbar{display:flex;align-items:center;gap:12px;padding:14px 24px;background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:10}
  .topbar h1{font-size:1.1rem;font-weight:600;letter-spacing:.5px}
  .badge{font-size:.7rem;font-weight:700;padding:3px 10px;border-radius:20px;background:#21262d;color:var(--accent);border:1px solid var(--border);letter-spacing:.4px}
  .topbar .meta{margin-left:auto;font-size:.78rem;color:var(--muted)}

  .main{display:flex;gap:20px;padding:24px;max-width:1400px;margin:0 auto}
  .left{flex:1;min-width:0;display:flex;flex-direction:column;gap:20px}
  .right{width:360px;flex-shrink:0;display:flex;flex-direction:column;gap:20px}

  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px}
  .card h2{font-size:.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px}

  .stats{display:flex;gap:12px}
  .stat{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px;text-align:center}
  .stat .n{font-size:1.8rem;font-weight:700}
  .stat .l{font-size:.72rem;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.6px}
  .stat.pass .n{color:var(--pass)} .stat.abstain .n{color:var(--abstain)} .stat.fail .n{color:var(--fail)} .stat.pend .n{color:var(--accent)}

  .drop-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:36px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s}
  .drop-zone:hover,.drop-zone.drag{border-color:var(--accent);background:rgba(88,166,255,.05)}
  .drop-zone .icon{font-size:2.4rem;margin-bottom:10px}
  .drop-zone p{color:var(--muted);font-size:.85rem}
  .drop-zone strong{color:var(--accent)}
  #file-input{display:none}

  .form-row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
  select,input[type=text]{background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:8px 12px;font-size:.85rem}
  select{cursor:pointer} input[type=text]{flex:1;min-width:140px}
  select:focus,input:focus{outline:none;border-color:var(--accent)}
  .btn{background:var(--accent);color:#0d1117;border:none;border-radius:6px;padding:8px 20px;font-weight:700;font-size:.85rem;cursor:pointer;transition:opacity .15s}
  .btn:hover{opacity:.85} .btn:disabled{opacity:.4;cursor:not-allowed}
  #selected-file{font-size:.8rem;color:var(--muted);margin-top:8px;min-height:18px}

  .result-box{border-radius:var(--radius);padding:18px 20px;display:none;border:2px solid transparent}
  .result-box.pass{border-color:var(--pass);background:rgba(63,185,80,.06);display:block}
  .result-box.abstain{border-color:var(--abstain);background:rgba(210,153,34,.06);display:block}
  .result-box.fail{border-color:var(--fail);background:rgba(248,81,73,.06);display:block}
  .result-box.waiting{border-color:var(--border);background:var(--surface);display:block}
  .verdict-label{font-size:1.5rem;font-weight:800;letter-spacing:1px}
  .verdict-label.pass{color:var(--pass)} .verdict-label.abstain{color:var(--abstain)} .verdict-label.fail{color:var(--fail)}
  .result-meta{font-size:.78rem;color:var(--muted);margin-top:4px}

  .icon-lists{margin-top:14px;display:flex;flex-direction:column;gap:8px}
  .icon-group label{font-size:.7rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;display:block;margin-bottom:4px}
  .icon-chips{display:flex;flex-wrap:wrap;gap:5px}
  .chip{font-size:.71rem;padding:2px 9px;border-radius:20px;border:1px solid var(--border);background:#21262d;color:var(--muted)}
  .chip.found{color:var(--pass);border-color:var(--pass)}
  .chip.abstain{color:var(--abstain);border-color:var(--abstain)}
  .chip.fail{color:var(--fail);border-color:var(--fail)}

  .proof-img{margin-top:14px;border-radius:8px;overflow:hidden;border:1px solid var(--border)}
  .proof-img img{width:100%;display:block}

  .spinner{display:inline-block;width:14px;height:14px;border:2px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .progress-bar{height:4px;background:var(--border);border-radius:2px;margin-top:14px;overflow:hidden}
  .progress-fill{height:100%;background:var(--accent);border-radius:2px;width:0%;transition:width .4s}

  .history-table{width:100%;border-collapse:collapse;font-size:.78rem}
  .history-table th{color:var(--muted);text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);font-weight:600}
  .history-table td{padding:7px 8px;border-bottom:1px solid #21262d}
  .history-table tr:last-child td{border-bottom:none}
  .vbadge{font-size:.68rem;font-weight:700;padding:2px 8px;border-radius:10px}
  .vbadge.non-defective{background:rgba(63,185,80,.15);color:var(--pass)}
  .vbadge.abstain{background:rgba(210,153,34,.15);color:var(--abstain)}
  .vbadge.defective{background:rgba(248,81,73,.15);color:var(--fail)}
  .vbadge.processing,.vbadge.queued{background:rgba(88,166,255,.1);color:var(--accent)}
  .empty-state{text-align:center;color:var(--muted);padding:28px 0;font-size:.82rem}

  .cfg-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;font-size:.78rem}
  .cfg-grid dt{color:var(--muted)} .cfg-grid dd{color:var(--text);font-weight:600}

  .legend{display:flex;flex-direction:column;gap:10px;font-size:.8rem}
  .legend-item{display:flex;gap:10px;align-items:flex-start}
  .legend-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:4px}
  .legend-dot.pass{background:var(--pass)} .legend-dot.abstain{background:var(--abstain)} .legend-dot.fail{background:var(--fail)}
  .legend-item p{color:var(--muted);font-size:.75rem;margin-top:2px}

  @media(max-width:900px){.main{flex-direction:column}.right{width:100%}.stats{flex-wrap:wrap}}
</style>
</head>
<body>

<div class="topbar">
  <h1>IP4R</h1>
  <span class="badge">v7b-r2a</span>
  <span class="badge" style="color:#d29922">ABSTAIN-aware</span>
  <div class="meta" id="topinfo">Loading…</div>
</div>

<div class="main">
  <div class="left">

    <div class="stats">
      <div class="stat pass">   <div class="n" id="c-pass">—</div><div class="l">Non-Defective</div></div>
      <div class="stat abstain"><div class="n" id="c-abs">—</div> <div class="l">Abstain</div></div>
      <div class="stat fail">   <div class="n" id="c-fail">—</div><div class="l">Defective</div></div>
      <div class="stat pend">   <div class="n" id="c-pend">—</div><div class="l">Queue</div></div>
    </div>

    <div class="card">
      <h2>Submit Inspection Job</h2>
      <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
        <div class="icon">🎬</div>
        <p>Drag &amp; drop a video here or <strong>click to browse</strong></p>
        <p style="margin-top:6px;font-size:.75rem">.mp4 · .avi · .mov · .mkv</p>
      </div>
      <input type="file" id="file-input" accept=".mp4,.avi,.mov,.mkv"/>
      <div id="selected-file"></div>
      <div class="form-row">
        <input type="text" id="job-id" placeholder="Job ID (auto if blank)" style="max-width:220px"/>
        <select id="defect-hint">
          <option value="unclassified">Defect hint: unclassified</option>
          <option value="defective">defective</option>
          <option value="non-defective">non-defective</option>
        </select>
        <button class="btn" id="submit-btn" onclick="submitJob()" disabled>Inspect</button>
      </div>
    </div>

    <div class="card" id="result-card" style="display:none">
      <h2>Latest Result</h2>
      <div class="result-box waiting" id="result-box">
        <span class="spinner"></span> Waiting…
        <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>
      </div>
    </div>

    <div class="card">
      <h2>Recent Jobs</h2>
      <div id="history-body"><div class="empty-state">No jobs yet</div></div>
    </div>

  </div>

  <div class="right">
    <div class="card">
      <h2>Verdict Guide</h2>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot pass"></div><div>
          <strong>Non-Defective (PASS)</strong>
          <p>All 15 required icons confirmed in ≥2 consecutive frames at ≥70% confidence.</p>
        </div></div>
        <div class="legend-item"><div class="legend-dot abstain"></div><div>
          <strong>Abstain</strong>
          <p>Every missing icon was glimpsed (≥1 frame) but never held 2 consecutive frames. Borderline — send to human re-inspection.</p>
        </div></div>
        <div class="legend-item"><div class="legend-dot fail"></div><div>
          <strong>Defective (FAIL)</strong>
          <p>At least one required icon was never detected above threshold. Icon genuinely absent — confident defect.</p>
        </div></div>
      </div>
    </div>

    <div class="card">
      <h2>Pipeline Config</h2>
      <dl class="cfg-grid">
        <dt>Version</dt>        <dd>v7b-r2a</dd>
        <dt>Phases</dt>         <dd>P1 + P2 + P3</dd>
        <dt>P3 threshold</dt>   <dd>70%</dd>
        <dt>Consecutive</dt>    <dd>≥ 2 frames</dd>
        <dt>ABSTAIN</dt>        <dd>ON (flicker)</dd>
        <dt>Brightness norm</dt><dd>none</dd>
        <dt>P1 fallback</dt>    <dd>ON</dd>
        <dt>Batch size</dt>     <dd>8</dd>
        <dt>Required icons</dt> <dd>15</dd>
        <dt>API version</dt>    <dd>v1</dd>
      </dl>
    </div>
  </div>
</div>

<script>
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const submitBtn  = document.getElementById('submit-btn');
const selectedEl = document.getElementById('selected-file');
let selectedFile = null;

fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag'); selectFile(e.dataTransfer.files[0]); });

function selectFile(f) {
  if (!f) return;
  selectedFile = f;
  selectedEl.textContent = `${f.name}  (${(f.size/1e6).toFixed(1)} MB)`;
  submitBtn.disabled = false;
}

let pollTimer = null;
async function submitJob() {
  if (!selectedFile) return;
  submitBtn.disabled = true;
  const fd = new FormData();
  fd.append('video', selectedFile);
  const jid = document.getElementById('job-id').value.trim();
  const defect = document.getElementById('defect-hint').value;
  if (jid) fd.append('job_id', jid);
  fd.append('defect', defect);

  let data;
  try {
    const r = await fetch('/api/v1/inference/submit-job', {method:'POST', body:fd});
    const env = await r.json();
    if (!env.success) { alert(env.message); submitBtn.disabled=false; return; }
    data = env.data;
  } catch(e) { alert('Upload failed: '+e); submitBtn.disabled=false; return; }

  showWaiting(data.job_id);
  startPoll(data.job_id);
}

function startPoll(jid) {
  if (pollTimer) clearInterval(pollTimer);
  let prog = 0;
  pollTimer = setInterval(async () => {
    prog = Math.min(prog+3, 90);
    const pf = document.getElementById('prog-fill');
    if (pf) pf.style.width = prog+'%';
    const r = await fetch('/api/v1/inference/result?job_id='+jid);
    const env = await r.json();
    const s = env.data?.state || env.status;
    if (s === 'completed' || env.status === 'completed') {
      clearInterval(pollTimer);
      if (pf) pf.style.width = '100%';
      showResult(env.data || {}, jid);
      refreshHistory();
      submitBtn.disabled = false;
    } else if (env.status === 'job_failed') {
      clearInterval(pollTimer);
      alert('Job failed: ' + (env.data?.error_detail || ''));
      submitBtn.disabled = false;
    }
  }, 2500);
}

function showWaiting(jid) {
  document.getElementById('result-card').style.display = 'block';
  document.getElementById('result-box').className = 'result-box waiting';
  document.getElementById('result-box').innerHTML =
    `<span class="spinner"></span> Processing job <code>${jid}</code>…
     <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>`;
}

function showResult(d, jid) {
  const v = (d.verdict || 'unknown');
  const cls = v === 'non-defective' ? 'pass' : v === 'abstain' ? 'abstain' : 'fail';
  const emoji = v === 'non-defective' ? '✅' : v === 'abstain' ? '⚠️' : '❌';
  const label = v === 'non-defective' ? 'NON-DEFECTIVE' : v.toUpperCase();
  const conf  = d.confidence != null ? ` — confidence ${(d.confidence*100).toFixed(1)}%` : '';

  const det = d.detail || {};
  const found   = det.icons_confirmed || [];
  const flicker = det.icons_flickered  || [];
  const absent  = det.icons_absent     || [];

  let iconHTML = `<div class="icon-group"><label>Confirmed (${found.length}/15)</label>
    <div class="icon-chips">${found.map(n=>`<span class="chip found">${n}</span>`).join('')||'<span style="color:var(--muted);font-size:.75rem">none</span>'}</div></div>`;
  if (flicker.length) iconHTML += `<div class="icon-group"><label>Glimpsed — not consecutive (${flicker.length})</label>
    <div class="icon-chips">${flicker.map(n=>`<span class="chip abstain">${n}</span>`).join('')}</div></div>`;
  if (absent.length) iconHTML += `<div class="icon-group"><label>Never seen — absent (${absent.length})</label>
    <div class="icon-chips">${absent.map(n=>`<span class="chip fail">${n}</span>`).join('')}</div></div>`;

  let proofHTML = '';
  if (d.proof?.image_url) {
    proofHTML = `<div class="proof-img"><img src="${d.proof.image_url}" alt="proof frame" loading="lazy"/></div>
    <div style="font-size:.72rem;color:var(--muted);margin-top:6px">Frame ${d.proof.frame_index} · ${d.proof.timestamp_in_video} · ${d.proof.defect_type}</div>`;
  }

  const meta = [jid, det.p1_ok!=null?(det.p1_ok?'P1✓':'P1✗'):'', det.p2_flag?'P2⚠':'', `thr=${det.p3_thr}`].filter(Boolean).join(' · ');

  document.getElementById('result-box').className = `result-box ${cls}`;
  document.getElementById('result-box').innerHTML = `
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:2rem">${emoji}</span>
      <div>
        <div class="verdict-label ${cls}">${label}${conf}</div>
        <div class="result-meta">${meta}</div>
      </div>
    </div>
    <div class="icon-lists">${iconHTML}</div>
    ${proofHTML}`;
}

async function refreshHistory() {
  try {
    const r = await fetch('/api/v1/inference/status?rows=50');
    const env = await r.json();
    if (!env.success) return;
    const jobs = env.data.jobs;
    let p=0, a=0, f=0;
    jobs.forEach(j => {
      if (j.verdict==='non-defective') p++;
      else if (j.verdict==='abstain') a++;
      else if (j.verdict==='defective') f++;
    });
    document.getElementById('c-pass').textContent = p;
    document.getElementById('c-abs').textContent  = a;
    document.getElementById('c-fail').textContent = f;
    if (!jobs.length) { document.getElementById('history-body').innerHTML='<div class="empty-state">No jobs yet</div>'; return; }
    const rows = jobs.map(j => {
      const v = j.verdict || j.state || '—';
      const t = j.submitted_at ? new Date(j.submitted_at).toLocaleTimeString() : '—';
      return `<tr>
        <td style="color:var(--muted)">${t}</td>
        <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${j.job_id||'—'}</td>
        <td><span class="vbadge ${v}">${v}</span></td>
        <td style="color:var(--muted)">${j.state||'—'}</td>
      </tr>`;
    }).join('');
    document.getElementById('history-body').innerHTML =
      `<table class="history-table"><thead><tr><th>Time</th><th>Job ID</th><th>Verdict</th><th>State</th></tr></thead><tbody>${rows}</tbody></table>`;
  } catch(_) {}
}

async function refreshHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('topinfo').textContent =
      `v7b-r2a · queue=${d.pending} · processing=${d.processing}`;
    document.getElementById('c-pend').textContent = d.pending + d.processing;
  } catch(_) {}
}

refreshHealth(); refreshHistory();
setInterval(refreshHealth, 5000);
setInterval(refreshHistory, 8000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
