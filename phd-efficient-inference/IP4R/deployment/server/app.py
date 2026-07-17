"""FQCT Inference Server — FastAPI application.

Endpoints (all on the LAN address, e.g. http://192.168.X.XXX:8000/):

  POST /inspect_queue        Upload DTS video + job_id.  Fire-and-forget.
  GET  /inference_result     Poll by job_id.  Returns verdict when done.
  GET  /health               Liveness check.
  GET  /status               Queue depth, avg inference time.
  GET  /                     Web admin dashboard.
  GET  /api/jobs             Recent job history (for dashboard).
  GET  /api/config           Current config YAML values.
  POST /api/config           Update a config key (hot-reload, no restart).

Designed for 10 concurrent FQCT units.  Each video (~18s, 12 FPS, 1080p)
is processed in < 4s on the centralized server, well within the 20s IR-test
window the FQCT spec allocates for server inference.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── repo path setup ───────────────────────────────────────────────────────────
_SERVER_DIR = Path(__file__).parent
_REPO_ROOT  = _SERVER_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ip4r.config import Config
from ip4r.pipeline import Inspector

from .job_store import JobStore
from .worker import process_video

# ── configuration ─────────────────────────────────────────────────────────────
_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"
_JOBS_DIR    = _REPO_ROOT / "data" / "jobs"
_MAX_WORKERS = 10   # one per FQCT unit

# ── singletons ────────────────────────────────────────────────────────────────
store    = JobStore(_JOBS_DIR / "jobs.db")
_cfg     = Config.load(_CONFIG_PATH)
_inspector = Inspector(_cfg)
_executor  = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
_cfg_lock  = threading.Lock()

# ── cleanup scheduler (runs daily) ───────────────────────────────────────────
def _daily_cleanup():
    while True:
        time.sleep(86400)
        n = store.purge_expired()
        if n:
            print(f"[cleanup] purged {n} expired jobs")

threading.Thread(target=_daily_cleanup, daemon=True).start()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="IP4R FQCT Inference Server",
    description="LCD defect detection for AC remote production line",
    version="1.0.0",
)

_static = _SERVER_DIR / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


# ── helpers ───────────────────────────────────────────────────────────────────
def _run_job(job_id: str, video_path: str) -> None:
    """Called in thread pool. Updates job store on completion/error."""
    store.mark_processing(job_id)
    job_dir = _JOBS_DIR / job_id
    try:
        with _cfg_lock:
            cfg       = _cfg
            inspector = _inspector
        result = process_video(job_id, video_path, job_dir, cfg, inspector)
        store.mark_done(job_id, result, result.get("inference_ms", 0))
    except Exception as exc:
        store.mark_error(job_id, str(exc))


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/inspect_queue", summary="Upload DTS video for LCD defect inference")
async def inspect_queue(
    job_id: str = Form(..., description="Unique job identifier from the FQCT unit"),
    video: UploadFile = File(..., description="DTS video file (MP4/AVI, 1080p @ 12 FPS)"),
):
    """
    Fire-and-forget: saves the video and queues the inference job.
    Returns immediately — do not wait for the inference result here.
    Poll GET /inference_result once the IR test is complete (~20s later).
    """
    job_dir   = _JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "video.mp4"

    content = await video.read()
    with open(video_path, "wb") as f:
        f.write(content)

    store.create_job(job_id, str(video_path))
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_job, job_id, str(video_path))

    return {"job_id": job_id, "status": "queued"}


@app.get("/inference_result", summary="Retrieve LCD verdict for a completed job")
async def inference_result(job_id: str):
    """
    Returns the inspection verdict once the server has finished processing.
    If the job is still pending/processing, returns status='processing'.
    The FQCT unit should call this after the IR test completes (~20s after POST).
    """
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/health", summary="Server liveness check")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/status", summary="Queue depth and server statistics")
async def status():
    s = store.get_status()
    s["server"] = "ip4r-fqct-v1"
    s["workers"] = _MAX_WORKERS
    return s


@app.get("/api/jobs", summary="Recent job history (for dashboard)")
async def api_jobs(limit: int = 50):
    return store.recent_jobs(limit=limit)


@app.get("/api/config", summary="Current active configuration")
async def api_config():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@app.post("/api/config", summary="Update a config key (hot-reload without restart)")
async def api_config_update(payload: dict):
    """
    Payload: { "key": "tier_a.ssim.ssim_min", "value": 0.20 }
    Dot-notation keys traverse nested YAML dicts.
    Changes take effect on the next job; no server restart required.
    """
    key   = payload.get("key")
    value = payload.get("value")
    if not key:
        raise HTTPException(400, "Missing 'key'")

    with _cfg_lock:
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        parts = key.split(".")
        node = raw
        for p in parts[:-1]:
            if p not in node:
                raise HTTPException(400, f"Key path '{key}' not found in config")
            node = node[p]
        node[parts[-1]] = value
        with open(_CONFIG_PATH, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        _cfg._data = raw

    return {"updated": key, "value": value}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    html = (_static / "index.html").read_text() if (_static / "index.html").exists() else _FALLBACK_HTML
    return HTMLResponse(content=html)


# ── fallback dashboard (no static files needed) ───────────────────────────────
_FALLBACK_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>IP4R FQCT Server</title>
<style>
  body{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;margin:0;padding:24px}
  h1{color:#4fc3f7;margin-bottom:4px}sub{color:#888}
  .cards{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}
  .card{background:#1e2230;border-radius:10px;padding:20px 28px;min-width:140px}
  .card .n{font-size:2.4rem;font-weight:700;color:#4fc3f7}
  .card .l{color:#888;font-size:.85rem;margin-top:4px}
  table{width:100%;border-collapse:collapse;background:#1e2230;border-radius:10px;overflow:hidden}
  th{background:#252a3a;padding:10px 14px;text-align:left;color:#aaa;font-size:.8rem}
  td{padding:9px 14px;border-top:1px solid #2a2f3f;font-size:.85rem}
  .pass{color:#4caf50}.fail{color:#ef5350}.proc{color:#ff9800}.err{color:#f44336}
  .refresh{margin:8px 0;color:#888;font-size:.8rem}
  input,button{font-family:inherit;padding:6px 12px;border-radius:6px;border:1px solid #444}
  input{background:#252a3a;color:#e0e0e0;width:300px}
  button{background:#1976d2;color:#fff;border:none;cursor:pointer;margin-left:8px}
  #cfg{background:#252a3a;border-radius:8px;padding:16px;font-size:.8rem;
       font-family:monospace;white-space:pre;overflow:auto;max-height:320px;margin-top:12px}
</style></head>
<body>
<h1>IP4R FQCT Inference Server</h1>
<sub>LCD defect detection · AC remote production QC</sub>

<div class="cards" id="cards">
  <div class="card"><div class="n" id="pending">…</div><div class="l">Pending</div></div>
  <div class="card"><div class="n" id="processing">…</div><div class="l">Processing</div></div>
  <div class="card"><div class="n" id="done">…</div><div class="l">Done (all time)</div></div>
  <div class="card"><div class="n" id="error">…</div><div class="l">Errors</div></div>
  <div class="card"><div class="n" id="avg">…</div><div class="l">Avg inference (ms)</div></div>
</div>

<h2 style="margin-bottom:4px">Recent Jobs</h2>
<div class="refresh" id="refresh"></div>
<table>
  <thead><tr><th>Job ID</th><th>Status</th><th>Inference (ms)</th><th>Created</th><th>Error</th></tr></thead>
  <tbody id="jobs"></tbody>
</table>

<h2 style="margin:24px 0 8px">Config</h2>
<div style="display:flex;gap:8px;align-items:center">
  <input id="ckey" placeholder="e.g. tier_a.ssim.ssim_min" />
  <input id="cval" placeholder="value" style="width:120px" />
  <button onclick="updateConfig()">Update</button>
  <span id="cmsg" style="color:#888;font-size:.85rem"></span>
</div>
<div id="cfg">Loading…</div>

<script>
async function load() {
  const [s, j, c] = await Promise.all([
    fetch('/status').then(r=>r.json()),
    fetch('/api/jobs?limit=30').then(r=>r.json()),
    fetch('/api/config').then(r=>r.json()),
  ]);
  document.getElementById('pending').textContent    = s.pending;
  document.getElementById('processing').textContent = s.processing;
  document.getElementById('done').textContent       = s.done;
  document.getElementById('error').textContent      = s.error;
  document.getElementById('avg').textContent        = s.avg_inference_ms ?? '—';
  document.getElementById('refresh').textContent    = 'Last updated: ' + new Date().toLocaleTimeString();

  const tbody = document.getElementById('jobs');
  tbody.innerHTML = j.map(r => {
    const cls = r.status==='done'?'pass':r.status==='error'?'err':r.status==='processing'?'proc':'';
    const ts  = new Date(r.created_at*1000).toLocaleString();
    return `<tr>
      <td>${r.job_id}</td>
      <td class="${cls}">${r.status}</td>
      <td>${r.inference_ms != null ? r.inference_ms.toFixed(0) : '—'}</td>
      <td>${ts}</td>
      <td style="color:#ef5350;font-size:.75rem">${r.error||''}</td>
    </tr>`;
  }).join('');

  document.getElementById('cfg').textContent = JSON.stringify(c, null, 2);
}

async function updateConfig() {
  const key = document.getElementById('ckey').value.trim();
  const raw = document.getElementById('cval').value.trim();
  const value = isNaN(raw) ? raw : Number(raw);
  const r = await fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({key, value})});
  const d = await r.json();
  document.getElementById('cmsg').textContent = r.ok ? `✓ ${d.updated} = ${d.value}` : `✗ ${JSON.stringify(d)}`;
  load();
}

load();
setInterval(load, 5000);
</script>
</body></html>"""
