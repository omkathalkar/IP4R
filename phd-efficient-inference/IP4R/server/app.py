"""FQCT Inference Server — FastAPI application.

Endpoints (all on the LAN address, e.g. http://192.168.X.XXX:8000/):

  POST /inspect_queue        Upload DTS video + job_id.  Fire-and-forget.
  GET  /inference_result     Poll by job_id.  Returns verdict when done.
  GET  /health               Liveness check.
  GET  /status               Queue depth, avg inference time.
  GET  /                     Web admin dashboard.
  GET  /api/jobs             Recent job history (for dashboard).
  GET  /api/config           Current config values.
  POST /api/config           Update a config key (hot-reload, no restart).

Inference engine: EfficientNet-B0 Phase-2-gated DTS inspector (dts_p2v2_best.pth).
Designed for 10 concurrent FQCT units. Each video (~18s, 12 FPS, 1080p)
is processed in < 4s on the centralized server.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .job_store import JobStore
from .worker import load_model, process_video

# ── paths ─────────────────────────────────────────────────────────────────────
_SERVER_DIR  = Path(__file__).parent
_REPO_ROOT   = _SERVER_DIR.parent
_CONFIG_PATH = _REPO_ROOT / "config" / "fqct_server.yaml"
_JOBS_DIR    = _REPO_ROOT / "data" / "jobs"

# ── singletons ────────────────────────────────────────────────────────────────
store      = JobStore(_JOBS_DIR / "jobs.db")
_cfg_lock  = threading.Lock()

with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

_model_path = _REPO_ROOT / _cfg["model"]["path"]
_model, _device = load_model(_model_path)
_max_workers    = int(_cfg.get("server", {}).get("max_workers", 10))
_executor       = ThreadPoolExecutor(max_workers=_max_workers)


# ── daily cleanup ─────────────────────────────────────────────────────────────
def _daily_cleanup():
    while True:
        time.sleep(86400)
        n = store.purge_expired()
        if n:
            print(f"[cleanup] purged {n} expired jobs")

threading.Thread(target=_daily_cleanup, daemon=True).start()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="FQCT Inference Server",
    description="LCD defect detection for AC remote production line (EfficientNet-B0)",
    version="2.0.0",
)

_static = _SERVER_DIR / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


# ── worker helper ─────────────────────────────────────────────────────────────
def _run_job(job_id: str, video_path: str) -> None:
    store.mark_processing(job_id)
    job_dir = _JOBS_DIR / job_id
    try:
        with _cfg_lock:
            cfg = dict(_cfg)
        result = process_video(job_id, video_path, job_dir, _model, _device, cfg)
        store.mark_done(job_id, result, result.get("inference_ms", 0))
    except Exception as exc:
        store.mark_error(job_id, str(exc))


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.post("/inspect_queue", summary="Upload DTS video for LCD defect inference")
async def inspect_queue(
    job_id: str = Form(..., description="Unique job identifier from the FQCT unit"),
    video: UploadFile = File(..., description="DTS video file (MP4/AVI, 1080p @ 12 FPS)"),
):
    """Fire-and-forget: saves the video and queues the inference job."""
    job_dir    = _JOBS_DIR / job_id
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
    s["server"]  = "fqct-v2"
    s["workers"] = _max_workers
    s["model"]   = str(_model_path.name)
    return s


@app.get("/jobs/{job_id}/overlay", summary="Overlay JPEG for a completed job")
async def job_overlay(job_id: str):
    overlay = _JOBS_DIR / job_id / "overlay.jpg"
    if not overlay.exists():
        raise HTTPException(status_code=404, detail="Overlay not ready")
    return FileResponse(str(overlay), media_type="image/jpeg")


@app.get("/api/jobs", summary="Recent job history")
async def api_jobs(limit: int = 50):
    return store.recent_jobs(limit=limit)


@app.get("/api/config", summary="Current active configuration")
async def api_config():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@app.post("/api/config", summary="Update a config key (hot-reload without restart)")
async def api_config_update(payload: dict):
    """
    Payload: { "key": "model.prob_threshold", "value": 0.6 }
    Dot-notation keys traverse nested YAML dicts.
    """
    key   = payload.get("key")
    value = payload.get("value")
    if not key:
        raise HTTPException(400, "Missing 'key'")

    with _cfg_lock:
        with open(_CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node:
                raise HTTPException(400, f"Key path '{key}' not found in config")
            node = node[p]
        node[parts[-1]] = value
        with open(_CONFIG_PATH, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        _cfg.clear()
        _cfg.update(raw)

    return {"updated": key, "value": value}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    html_path = _static / "index.html"
    html = html_path.read_text() if html_path.exists() else _FALLBACK_HTML
    return HTMLResponse(content=html)


# ── fallback dashboard ────────────────────────────────────────────────────────
_FALLBACK_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>FQCT Server</title>
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
  input[type=text]{background:#252a3a;color:#e0e0e0;width:260px}
  button{background:#1976d2;color:#fff;border:none;cursor:pointer;margin-left:8px}
  button:disabled{background:#444;cursor:not-allowed}
  #cfg{background:#252a3a;border-radius:8px;padding:16px;font-size:.8rem;
       font-family:monospace;white-space:pre;overflow:auto;max-height:260px;margin-top:12px}
  .upload-box{background:#1e2230;border-radius:10px;padding:24px 28px;margin:24px 0;max-width:640px}
  .upload-box h2{margin:0 0 16px;color:#4fc3f7;font-size:1.1rem}
  .upload-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  .upload-row label{color:#aaa;font-size:.85rem;width:70px}
  input[type=file]{background:#252a3a;color:#e0e0e0;border:1px solid #444;
                   border-radius:6px;padding:5px 10px;font-family:inherit;flex:1}
  #result-box{margin-top:16px;background:#252a3a;border-radius:8px;padding:14px 18px;
              display:none;font-size:.88rem}
  #result-box .verdict{font-size:1.5rem;font-weight:700;margin-bottom:6px}
  #result-box .meta{color:#aaa;font-size:.78rem;margin-bottom:10px}
  #result-box table{width:100%;font-size:.78rem;border-collapse:collapse}
  #result-box td{padding:3px 8px;border-top:1px solid #333}
  progress{width:100%;height:6px;border-radius:3px;accent-color:#4fc3f7;margin-top:8px}
</style></head>
<body>
<h1>FQCT Inference Server</h1>
<sub>LCD defect detection · AC remote production QC · EfficientNet-B0</sub>

<div class="upload-box">
  <h2>Submit Video for Inspection</h2>
  <div class="upload-row">
    <label>Job ID</label>
    <input type="text" id="ujob" placeholder="auto-generated if empty" style="flex:1" />
  </div>
  <div class="upload-row">
    <label>Video</label>
    <input type="file" id="uvid" accept="video/*" />
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <button id="ubtn" onclick="submitVideo()">Submit</button>
    <span id="umsg" style="color:#888;font-size:.85rem"></span>
  </div>
  <progress id="uprog" value="0" max="100" style="display:none"></progress>
  <div id="result-box">
    <div class="verdict" id="r-verdict"></div>
    <div class="meta" id="r-meta"></div>
    <img id="r-overlay" src="" alt="overlay"
         style="display:none;width:100%;max-width:480px;border-radius:8px;margin:12px 0;border:1px solid #333" />
    <table id="r-rois"><tr><th>Segment</th><th>Coverage</th><th>Min</th><th>Status</th></tr></table>
  </div>
</div>

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
  <thead><tr>
    <th>Job ID</th><th>Status</th><th>Verdict</th>
    <th>P2 prob</th><th>P2 frames</th><th>Inference (ms)</th><th>Created</th><th>Overlay</th><th>Error</th>
  </tr></thead>
  <tbody id="jobs"></tbody>
</table>

<h2 style="margin:24px 0 8px">Config</h2>
<div style="display:flex;gap:8px;align-items:center">
  <input id="ckey" placeholder="e.g. model.prob_threshold" />
  <input id="cval" placeholder="value" style="width:100px"/>
  <button onclick="updateConfig()">Update</button>
  <span id="cmsg" style="color:#888;font-size:.85rem"></span>
</div>
<div id="cfg">Loading…</div>

<script>
function genJobId() {
  return 'job_' + Date.now() + '_' + Math.random().toString(36).slice(2,6);
}

async function submitVideo() {
  const fileInput = document.getElementById('uvid');
  const jobInput  = document.getElementById('ujob');
  const btn       = document.getElementById('ubtn');
  const msg       = document.getElementById('umsg');
  const prog      = document.getElementById('uprog');
  const rbox      = document.getElementById('result-box');

  if (!fileInput.files.length) { msg.textContent = 'Please select a video file.'; return; }

  const job_id = jobInput.value.trim() || genJobId();
  jobInput.value = job_id;

  const fd = new FormData();
  fd.append('job_id', job_id);
  fd.append('video', fileInput.files[0]);

  btn.disabled = true;
  rbox.style.display = 'none';
  prog.style.display = 'block';
  prog.value = 10;
  msg.textContent = 'Uploading…';

  try {
    const r = await fetch('/inspect_queue', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await r.text());
    prog.value = 40;
    msg.textContent = 'Queued — waiting for result…';

    // Poll until done
    let attempts = 0;
    while (attempts++ < 60) {
      await new Promise(res => setTimeout(res, 2000));
      prog.value = Math.min(40 + attempts * 2, 90);
      const pr = await fetch('/inference_result?job_id=' + encodeURIComponent(job_id));
      const data = await pr.json();
      if (data.status === 'done') {
        prog.value = 100;
        showResult(data);
        msg.textContent = 'Done in ' + (data.inference_ms ? data.inference_ms.toFixed(0) + 'ms' : '?');
        load();
        break;
      } else if (data.status === 'error') {
        msg.textContent = 'Error: ' + (data.error || 'unknown');
        prog.style.display = 'none';
        break;
      }
    }
  } catch(e) {
    msg.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
}

function showResult(data) {
  const res = data.result || {};
  const rbox = document.getElementById('result-box');
  rbox.style.display = 'block';

  const verd = res.verdict || (data.status === 'error' ? 'ERROR' : '—');
  const vEl  = document.getElementById('r-verdict');
  vEl.textContent = verd;
  vEl.style.color = verd==='PASS' ? '#4caf50' : verd==='FAIL' ? '#ef5350' : '#ff9800';

  document.getElementById('r-meta').textContent =
    `Job: ${data.job_id}  |  ` +
    `P2 prob: ${res.median_p2_prob != null ? (res.median_p2_prob*100).toFixed(1)+'%' : '—'}  |  ` +
    `P2 frames: ${res.p2_pass ?? '—'}/${res.p2_frames ?? '—'}  |  ` +
    `Inference: ${res.inference_ms != null ? res.inference_ms.toFixed(0)+'ms' : '—'}`;

  const ovrEl = document.getElementById('r-overlay');
  ovrEl.src = `/jobs/${encodeURIComponent(data.job_id)}/overlay?t=` + Date.now();
  ovrEl.style.display = 'block';

  const tbody = document.getElementById('r-rois');
  const rows  = (res.roi_results || []).filter(r => r.name !== 'icon_strip');
  tbody.innerHTML = '<tr><th>Segment</th><th>Coverage</th><th>Min</th><th>Status</th></tr>' +
    rows.map(r => `<tr>
      <td>${r.name}</td>
      <td>${r.coverage}</td>
      <td>${r.min_cov}</td>
      <td style="color:${r.passed?'#4caf50':'#ef5350'}">${r.passed?'OK':'FAIL'}</td>
    </tr>`).join('');
}

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
    const res = r.result || {};
    const verdict = res.verdict ?? '—';
    const vcls = verdict==='PASS'?'pass':verdict==='FAIL'?'fail':'';
    const prob = res.median_p2_prob != null ? (res.median_p2_prob*100).toFixed(1)+'%' : '—';
    const p2   = res.p2_frames != null ? `${res.p2_pass}/${res.p2_frames}` : '—';
    const ts   = new Date(r.created_at*1000).toLocaleString();
    const ovr = r.status==='done'
      ? `<a href="/jobs/${encodeURIComponent(r.job_id)}/overlay" target="_blank"
             style="color:#4fc3f7;text-decoration:none">⬜ overlay</a>` : '—';
    return `<tr>
      <td style="font-size:.75rem">${r.job_id}</td>
      <td class="${cls}">${r.status}</td>
      <td class="${vcls}">${verdict}</td>
      <td>${prob}</td>
      <td>${p2}</td>
      <td>${r.inference_ms != null ? r.inference_ms.toFixed(0) : '—'}</td>
      <td>${ts}</td>
      <td>${ovr}</td>
      <td style="color:#ef5350;font-size:.75rem">${r.error||''}</td>
    </tr>`;
  }).join('');

  document.getElementById('cfg').textContent = JSON.stringify(c, null, 2);
}

async function updateConfig() {
  const key = document.getElementById('ckey').value.trim();
  const raw = document.getElementById('cval').value.trim();
  const value = isNaN(raw) ? raw : Number(raw);
  const r = await fetch('/api/config', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({key, value})});
  const d = await r.json();
  document.getElementById('cmsg').textContent = r.ok ? `✓ ${d.updated} = ${d.value}` : `✗ ${JSON.stringify(d)}`;
  load();
}

load();
setInterval(load, 5000);
</script>
</body></html>"""
