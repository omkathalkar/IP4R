"""
IP4R v7b-r2a Inspection Server — Port 8084

Endpoints:
  POST /inspect          Upload video → {job_id}
  GET  /result/{job_id} Poll result → {verdict, details}
  GET  /health
  GET  /                 Web UI
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
from pathlib import Path

import cv2

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from server_v7b_r2a.pipeline import (
    run_video,
    P3_CONF_THR, P3_MIN_CONSECUTIVE, ABSTAIN_ENABLED,
    BRIGHTNESS_NORM_MODE, REQUIRED_ICONS,
)

# ── Config ────────────────────────────────────────────────────────────────────
PORT        = 8084
UPLOAD_DIR  = Path("/tmp/v7b_r2a_uploads")
RESULT_DIR  = Path("/tmp/v7b_r2a_results")
MAX_WORKERS = 2
MAX_HISTORY = 100

P1_MODEL   = str(_ROOT / "data/macro_dataset/runs/macro_test/weights/best.pt")
ELEM_MODEL = str(_ROOT / "runs/phase2_elem_v1/weights/best.pt")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Job store ─────────────────────────────────────────────────────────────────
_lock    = threading.Lock()
_jobs: dict[str, dict] = {}
_history: deque = deque(maxlen=MAX_HISTORY)
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

app = FastAPI(title="IP4R v7b-r2a", version="7b-r2a")


def _process(job_id: str, video_path: str):
    with _lock:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["started_at"] = time.time()

    out_dir = str(RESULT_DIR / job_id)
    try:
        verdict = run_video(video_path, P1_MODEL, ELEM_MODEL, out_dir)
        result_json = Path(out_dir) / f"debug_{Path(video_path).stem}" / "result.json"
        details = {}
        if result_json.exists():
            with open(result_json) as f:
                details = json.load(f)
        with _lock:
            _jobs[job_id].update({
                "status":     "done",
                "verdict":    verdict,
                "details":    details,
                "finished_at": time.time(),
                "error":      None,
            })
    except Exception as e:
        with _lock:
            _jobs[job_id].update({
                "status":     "error",
                "verdict":    "ERROR",
                "error":      str(e),
                "finished_at": time.time(),
            })
    finally:
        try:
            os.remove(video_path)
        except Exception:
            pass
        with _lock:
            _history.appendleft({**_jobs[job_id], "job_id": job_id})


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    with _lock:
        pending    = sum(1 for j in _jobs.values() if j["status"] == "queued")
        processing = sum(1 for j in _jobs.values() if j["status"] == "processing")
    return {"status": "ok", "pending": pending, "processing": processing}


@app.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    job_id: str = Form(default=""),
):
    if not file.filename or not file.filename.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):
        raise HTTPException(400, "Upload must be a video file (.mp4/.avi/.mov/.mkv)")

    jid = job_id.strip() or str(uuid.uuid4())[:12]
    video_path = UPLOAD_DIR / f"{jid}_{file.filename}"
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    with _lock:
        _jobs[jid] = {
            "status":      "queued",
            "filename":    file.filename,
            "queued_at":   time.time(),
            "started_at":  None,
            "finished_at": None,
            "verdict":     None,
            "details":     {},
            "error":       None,
        }

    _executor.submit(_process, jid, str(video_path))
    return JSONResponse({"job_id": jid, "status": "queued"})


@app.get("/result/{job_id}")
async def result(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job_id '{job_id}' not found")

    resp = {
        "job_id":   job_id,
        "status":   job["status"],
        "verdict":  job.get("verdict"),
        "filename": job.get("filename"),
        "error":    job.get("error"),
    }
    if job["status"] == "done":
        d = job.get("details", {})
        resp["p1_ok"]           = d.get("p1_ok")
        resp["p1_fallback"]     = d.get("p1_fallback_used")
        resp["p2_flag"]         = d.get("p2_flag")
        resp["p3_thr"]          = d.get("p3_adaptive_thr")
        resp["p3_found"]        = d.get("p3_found", [])
        resp["p3_abstain_icons"]= d.get("p3_abstain_icons", [])
        resp["p3_fail_icons"]   = d.get("p3_fail_icons", [])
        elapsed = (job["finished_at"] or 0) - (job["started_at"] or 0)
        resp["elapsed_s"] = round(elapsed, 1)

    return JSONResponse(resp)


@app.get("/api/history")
async def history():
    with _lock:
        h = list(_history)
    return JSONResponse(h)


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
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --pass:     #3fb950;
    --abstain:  #d29922;
    --fail:     #f85149;
    --accent:   #58a6ff;
    --radius:   10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  /* ── top bar ── */
  .topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10;
  }
  .topbar h1 { font-size: 1.1rem; font-weight: 600; letter-spacing: .5px; }
  .badge {
    font-size: 0.7rem; font-weight: 700; padding: 3px 10px;
    border-radius: 20px; background: #21262d; color: var(--accent);
    border: 1px solid var(--border); letter-spacing: .4px;
  }
  .topbar .meta { margin-left: auto; font-size: 0.78rem; color: var(--muted); }

  /* ── layout ── */
  .main { display: flex; gap: 20px; padding: 24px; max-width: 1400px; margin: 0 auto; }
  .left  { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 20px; }
  .right { width: 360px; flex-shrink: 0; display: flex; flex-direction: column; gap: 20px; }

  /* ── cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
  }
  .card h2 { font-size: 0.78rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; margin-bottom: 14px; }

  /* ── stat strip ── */
  .stats { display: flex; gap: 12px; }
  .stat { flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px; text-align: center; }
  .stat .n { font-size: 1.8rem; font-weight: 700; }
  .stat .l { font-size: 0.72rem; color: var(--muted); margin-top: 2px; text-transform: uppercase; letter-spacing: .6px; }
  .stat.pass .n { color: var(--pass); }
  .stat.abstain .n { color: var(--abstain); }
  .stat.fail .n { color: var(--fail); }
  .stat.pend .n { color: var(--accent); }

  /* ── upload area ── */
  .drop-zone {
    border: 2px dashed var(--border); border-radius: var(--radius);
    padding: 36px; text-align: center; cursor: pointer;
    transition: border-color .2s, background .2s;
  }
  .drop-zone:hover, .drop-zone.drag { border-color: var(--accent); background: rgba(88,166,255,.05); }
  .drop-zone .icon { font-size: 2.4rem; margin-bottom: 10px; }
  .drop-zone p { color: var(--muted); font-size: 0.85rem; }
  .drop-zone strong { color: var(--accent); }
  #file-input { display: none; }

  /* ── form ── */
  .form-row { display: flex; gap: 10px; align-items: center; margin-top: 12px; flex-wrap: wrap; }
  input[type=text] {
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text); padding: 8px 12px; font-size: 0.85rem; flex: 1; min-width: 160px;
  }
  input[type=text]:focus { outline: none; border-color: var(--accent); }
  .btn {
    background: var(--accent); color: #0d1117; border: none; border-radius: 6px;
    padding: 8px 20px; font-weight: 700; font-size: 0.85rem; cursor: pointer;
    transition: opacity .15s;
  }
  .btn:hover { opacity: .85; }
  .btn:disabled { opacity: .4; cursor: not-allowed; }
  .btn.secondary { background: var(--surface); color: var(--text); border: 1px solid var(--border); font-weight: 500; }
  #selected-file { font-size: 0.8rem; color: var(--muted); margin-top: 8px; min-height: 18px; }

  /* ── result panel ── */
  .result-box {
    border-radius: var(--radius); padding: 18px 20px; display: none;
    border: 2px solid transparent;
  }
  .result-box.pass    { border-color: var(--pass);    background: rgba(63,185,80,.06); }
  .result-box.abstain { border-color: var(--abstain); background: rgba(210,153,34,.06); }
  .result-box.fail    { border-color: var(--fail);    background: rgba(248,81,73,.06); }
  .result-box.waiting { border-color: var(--border);  background: var(--surface); display: block; }
  .verdict-label { font-size: 1.6rem; font-weight: 800; letter-spacing: 1px; }
  .verdict-label.pass    { color: var(--pass); }
  .verdict-label.abstain { color: var(--abstain); }
  .verdict-label.fail    { color: var(--fail); }
  .result-meta { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }
  .icon-lists { margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }
  .icon-group label { font-size: 0.7rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; display: block; margin-bottom: 4px; }
  .icon-chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip {
    font-size: 0.71rem; padding: 2px 9px; border-radius: 20px;
    border: 1px solid var(--border); background: #21262d; color: var(--muted);
  }
  .chip.found   { color: var(--pass);    border-color: var(--pass); }
  .chip.abstain { color: var(--abstain); border-color: var(--abstain); }
  .chip.fail    { color: var(--fail);    border-color: var(--fail); }
  .progress-bar { height: 4px; background: var(--border); border-radius: 2px; margin-top: 14px; overflow: hidden; }
  .progress-fill { height: 100%; background: var(--accent); border-radius: 2px; width: 0%; transition: width .4s; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── history table ── */
  .history-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
  .history-table th { color: var(--muted); text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); font-weight: 600; }
  .history-table td { padding: 7px 8px; border-bottom: 1px solid #21262d; }
  .history-table tr:last-child td { border-bottom: none; }
  .vbadge { font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; }
  .vbadge.PASS    { background: rgba(63,185,80,.15);  color: var(--pass); }
  .vbadge.ABSTAIN { background: rgba(210,153,34,.15); color: var(--abstain); }
  .vbadge.FAIL    { background: rgba(248,81,73,.15);  color: var(--fail); }
  .vbadge.ERROR   { background: rgba(248,81,73,.1);   color: var(--fail); }
  .vbadge.queued, .vbadge.processing { background: rgba(88,166,255,.1); color: var(--accent); }
  .empty-state { text-align: center; color: var(--muted); padding: 28px 0; font-size: 0.82rem; }

  /* ── config panel ── */
  .cfg-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; font-size: 0.78rem; }
  .cfg-grid dt { color: var(--muted); }
  .cfg-grid dd { color: var(--text); font-weight: 600; }

  /* ── legend ── */
  .legend { display: flex; flex-direction: column; gap: 8px; font-size: 0.8rem; }
  .legend-item { display: flex; gap: 10px; align-items: flex-start; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }
  .legend-dot.pass    { background: var(--pass); }
  .legend-dot.abstain { background: var(--abstain); }
  .legend-dot.fail    { background: var(--fail); }
  .legend-item p { color: var(--muted); font-size: 0.75rem; margin-top: 2px; }

  @media (max-width: 900px) {
    .main { flex-direction: column; }
    .right { width: 100%; }
    .stats { flex-wrap: wrap; }
  }
</style>
</head>
<body>

<div class="topbar">
  <h1>IP4R</h1>
  <span class="badge">v7b-r2a</span>
  <span class="badge" style="color:#d29922">ABSTAIN-aware</span>
  <div class="meta" id="topinfo">Port 8084 · Loading…</div>
</div>

<div class="main">

  <!-- LEFT COLUMN -->
  <div class="left">

    <!-- stat strip -->
    <div class="stats">
      <div class="stat pass">   <div class="n" id="c-pass">—</div><div class="l">PASS</div></div>
      <div class="stat abstain"><div class="n" id="c-abs">—</div> <div class="l">ABSTAIN</div></div>
      <div class="stat fail">   <div class="n" id="c-fail">—</div><div class="l">FAIL</div></div>
      <div class="stat pend">   <div class="n" id="c-pend">—</div><div class="l">Queue</div></div>
    </div>

    <!-- upload card -->
    <div class="card">
      <h2>Inspect Video</h2>
      <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
        <div class="icon">🎬</div>
        <p>Drag &amp; drop a video here or <strong>click to browse</strong></p>
        <p style="margin-top:6px;font-size:0.75rem">.mp4 · .avi · .mov · .mkv</p>
      </div>
      <input type="file" id="file-input" accept=".mp4,.avi,.mov,.mkv"/>
      <div id="selected-file"></div>
      <div class="form-row">
        <input type="text" id="job-id" placeholder="Job ID (auto if blank)"/>
        <button class="btn" id="submit-btn" onclick="submitJob()" disabled>Inspect</button>
      </div>
    </div>

    <!-- active result -->
    <div class="card" id="result-card" style="display:none">
      <h2>Latest Result</h2>
      <div class="result-box waiting" id="result-box">
        <span class="spinner"></span> Waiting for result…
        <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>
      </div>
    </div>

    <!-- history -->
    <div class="card">
      <h2>Recent Jobs</h2>
      <div id="history-body">
        <div class="empty-state">No jobs yet</div>
      </div>
    </div>

  </div>

  <!-- RIGHT COLUMN -->
  <div class="right">

    <!-- legend -->
    <div class="card">
      <h2>Verdict Guide</h2>
      <div class="legend">
        <div class="legend-item">
          <div class="legend-dot pass"></div>
          <div>
            <strong>PASS</strong>
            <p>All 15 required icons confirmed in ≥2 consecutive frames above 70% confidence.</p>
          </div>
        </div>
        <div class="legend-item">
          <div class="legend-dot abstain"></div>
          <div>
            <strong>ABSTAIN</strong>
            <p>Every missing icon was glimpsed (≥1 frame above thr) but never held 2 consecutive frames. LCD may be dim/flickering. Send to human re-inspection.</p>
          </div>
        </div>
        <div class="legend-item">
          <div class="legend-dot fail"></div>
          <div>
            <strong>FAIL</strong>
            <p>At least one required icon was never detected above threshold. Icon is genuinely absent — confident defect.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- config -->
    <div class="card">
      <h2>Pipeline Config</h2>
      <dl class="cfg-grid">
        <dt>Version</dt>       <dd>v7b-r2a</dd>
        <dt>Phases</dt>        <dd>P1 + P2 + P3</dd>
        <dt>P3 threshold</dt>  <dd>70%</dd>
        <dt>Consecutive</dt>   <dd>≥ 2 frames</dd>
        <dt>ABSTAIN</dt>       <dd>ON (flicker)</dd>
        <dt>Brightness norm</dt><dd>none</dd>
        <dt>Adaptive thr</dt>  <dd>OFF</dd>
        <dt>P1 fallback</dt>   <dd>ON</dd>
        <dt>Batch size</dt>    <dd>8</dd>
        <dt>Required icons</dt><dd>15</dd>
        <dt>Port</dt>          <dd>8084</dd>
      </dl>
    </div>

  </div>
</div>

<script>
// ── file selection ─────────────────────────────────────────────────────────
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const submitBtn  = document.getElementById('submit-btn');
const selectedEl = document.getElementById('selected-file');
let selectedFile = null;

fileInput.addEventListener('change', () => { selectFile(fileInput.files[0]); });

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  selectFile(e.dataTransfer.files[0]);
});

function selectFile(f) {
  if (!f) return;
  selectedFile = f;
  selectedEl.textContent = `Selected: ${f.name}  (${(f.size/1e6).toFixed(1)} MB)`;
  submitBtn.disabled = false;
}

// ── submit ─────────────────────────────────────────────────────────────────
let pollTimer = null;

async function submitJob() {
  if (!selectedFile) return;
  submitBtn.disabled = true;

  const fd = new FormData();
  fd.append('file', selectedFile);
  const jid = document.getElementById('job-id').value.trim();
  if (jid) fd.append('job_id', jid);

  let data;
  try {
    const r = await fetch('/inspect', { method: 'POST', body: fd });
    data = await r.json();
  } catch(e) {
    alert('Upload failed: ' + e);
    submitBtn.disabled = false;
    return;
  }

  showResult('waiting', data.job_id, null, null);
  startPoll(data.job_id);
}

// ── polling ────────────────────────────────────────────────────────────────
function startPoll(jid) {
  if (pollTimer) clearInterval(pollTimer);
  let prog = 0;
  pollTimer = setInterval(async () => {
    prog = Math.min(prog + 4, 90);
    document.getElementById('prog-fill').style.width = prog + '%';

    const r = await fetch('/result/' + jid);
    if (!r.ok) return;
    const d = await r.json();

    if (d.status === 'done' || d.status === 'error') {
      clearInterval(pollTimer);
      document.getElementById('prog-fill').style.width = '100%';
      showResult(d.status === 'error' ? 'fail' : d.verdict.toLowerCase(), jid, d, null);
      refreshHistory();
      submitBtn.disabled = false;
    }
  }, 2000);
}

// ── result display ─────────────────────────────────────────────────────────
function showResult(state, jid, d, _) {
  const card = document.getElementById('result-card');
  const box  = document.getElementById('result-box');
  card.style.display = 'block';

  if (state === 'waiting') {
    box.className = 'result-box waiting';
    box.innerHTML = `<span class="spinner"></span> Processing job <code>${jid}</code>…
      <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>`;
    return;
  }

  const v = (d.verdict || 'ERROR').toUpperCase();
  const cls = v === 'PASS' ? 'pass' : v === 'ABSTAIN' ? 'abstain' : 'fail';
  const emoji = v === 'PASS' ? '✅' : v === 'ABSTAIN' ? '⚠️' : '❌';

  let descHTML = '';
  if (v === 'ABSTAIN') {
    descHTML = `<div style="margin-top:8px;font-size:0.8rem;color:var(--abstain)">
      All missing icons were glimpsed but didn't hold 2 consecutive frames.
      Recommend human re-inspection.
    </div>`;
  } else if (v === 'FAIL' && d.p3_fail_icons?.length) {
    descHTML = `<div style="margin-top:8px;font-size:0.8rem;color:var(--fail)">
      Icon(s) genuinely absent — confident defect detected.
    </div>`;
  }

  let iconHTML = '';
  const found   = d.p3_found        || [];
  const abstain = d.p3_abstain_icons|| [];
  const failed  = d.p3_fail_icons   || [];
  const total   = 15;

  iconHTML += `<div class="icon-group">
    <label>Confirmed (${found.length}/${total})</label>
    <div class="icon-chips">${found.map(n => `<span class="chip found">${n}</span>`).join('') || '<span style="color:var(--muted);font-size:0.75rem">none</span>'}</div>
  </div>`;

  if (abstain.length) {
    iconHTML += `<div class="icon-group">
      <label>Glimpsed — not consecutive (${abstain.length})</label>
      <div class="icon-chips">${abstain.map(n => `<span class="chip abstain">${n}</span>`).join('')}</div>
    </div>`;
  }
  if (failed.length) {
    iconHTML += `<div class="icon-group">
      <label>Never seen — absent (${failed.length})</label>
      <div class="icon-chips">${failed.map(n => `<span class="chip fail">${n}</span>`).join('')}</div>
    </div>`;
  }

  const meta = [
    d.filename,
    d.elapsed_s != null ? `${d.elapsed_s}s` : '',
    d.p3_thr    != null ? `thr=${d.p3_thr}` : '',
    d.p1_ok != null ? (d.p1_ok ? 'P1✓' : `P1✗${d.p1_fallback ? '+fb' : ''}`) : '',
    d.p2_flag   ? 'P2⚠' : '',
  ].filter(Boolean).join(' · ');

  box.className = `result-box ${cls}`;
  box.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:2rem">${emoji}</span>
      <div>
        <div class="verdict-label ${cls}">${v}</div>
        <div class="result-meta">${meta}</div>
      </div>
    </div>
    ${descHTML}
    <div class="icon-lists">${iconHTML}</div>
  `;
}

// ── history ────────────────────────────────────────────────────────────────
let passCount = 0, absCount = 0, failCount = 0;

async function refreshHistory() {
  const r = await fetch('/api/history');
  if (!r.ok) return;
  const jobs = await r.json();

  passCount = absCount = failCount = 0;
  jobs.forEach(j => {
    if (j.verdict === 'PASS')    passCount++;
    else if (j.verdict === 'ABSTAIN') absCount++;
    else if (j.verdict === 'FAIL')    failCount++;
  });
  document.getElementById('c-pass').textContent = passCount;
  document.getElementById('c-abs').textContent  = absCount;
  document.getElementById('c-fail').textContent = failCount;

  if (!jobs.length) {
    document.getElementById('history-body').innerHTML = '<div class="empty-state">No jobs yet</div>';
    return;
  }

  const rows = jobs.map(j => {
    const v = j.verdict || j.status || '—';
    const t = j.finished_at
      ? new Date(j.finished_at * 1000).toLocaleTimeString()
      : '…';
    const elapsed = (j.finished_at && j.started_at)
      ? `${(j.finished_at - j.started_at).toFixed(1)}s` : '';
    const detail = (j.details?.p3_fail_icons || []).slice(0,3).join(', ');
    return `<tr>
      <td style="color:var(--muted)">${t}</td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${j.filename||''}">${j.filename||j.job_id||'—'}</td>
      <td><span class="vbadge ${v}">${v}</span></td>
      <td style="color:var(--muted)">${elapsed}</td>
      <td style="color:var(--fail);font-size:0.72rem">${detail}</td>
    </tr>`;
  }).join('');

  document.getElementById('history-body').innerHTML = `
    <table class="history-table">
      <thead><tr>
        <th>Time</th><th>File</th><th>Verdict</th><th>Duration</th><th>Missing</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ── health ticker ──────────────────────────────────────────────────────────
async function refreshHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('topinfo').textContent =
      `Port 8084  ·  queue=${d.pending}  processing=${d.processing}`;
    document.getElementById('c-pend').textContent = d.pending + d.processing;
  } catch(_) {}
}

// ── boot ───────────────────────────────────────────────────────────────────
refreshHealth();
refreshHistory();
setInterval(refreshHealth, 5000);
setInterval(refreshHistory, 8000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
