"""
IP4R v6a — EfficientNet-B0 LCD Inspection Server
Inference API V1 (Tangent Thought Technologies)

Port : 8083
Base : /api/v1/inference
"""
from __future__ import annotations

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

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from server_v6a.pipeline_cnn import V6aPipeline, save_crop_overlay
from server_v6a.frame_extract import extract_best_splash_frame

# ── Config ────────────────────────────────────────────────────────────────────
PORT        = 8083
UPLOAD_DIR  = Path("/tmp/v6a_uploads")
PROOF_DIR   = Path("/tmp/v6a_proofs")
MAX_WORKERS = 2
MAX_HISTORY = 100

V6A_MODEL   = os.environ.get("V6A_MODEL", str(_ROOT / "models" / "v6a" / "best.pth"))
VALID_DEFECT = {"defective", "non-defective"}

for _d in (UPLOAD_DIR, PROOF_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Load model once at startup ────────────────────────────────────────────────
print(f"Loading v6a EfficientNet model: {V6A_MODEL}")
_pipeline = V6aPipeline(V6A_MODEL, use_abstain=True)
print("Model ready.")

# ── Job store ─────────────────────────────────────────────────────────────────
_lock     = threading.Lock()
_jobs: dict[str, dict] = {}
_history: deque        = deque(maxlen=MAX_HISTORY)
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

app = FastAPI(title="IP4R v6a Inference Server", version="1.0.0")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _ok(status, message, data={}):
    return JSONResponse({"success": True,  "status": status,  "message": message, "data": data})

def _err(status, message, data={}, code=400):
    return JSONResponse({"success": False, "status": status, "message": message, "data": data},
                        status_code=code)

def _api_verdict(v: str) -> str:
    return {"PASS": "pass", "FAIL": "fail", "ABSTAIN": "abstain"}.get(v, "unknown")


# ── Inference worker ──────────────────────────────────────────────────────────

def _process(job_id: str, video_path: str):
    with _lock:
        _jobs[job_id]["state"]      = "processing"
        _jobs[job_id]["started_at"] = _utcnow()

    try:
        result = _run(video_path, job_id)
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


def _run(video_path: str, job_id: str) -> dict:
    vp = Path(video_path)
    res = _pipeline.predict(vp)

    # Extract splash crop for proof image
    crop, frame_info = extract_best_splash_frame(vp)
    proof_path = None
    if crop is not None:
        proof_dir = PROOF_DIR / job_id
        proof_dir.mkdir(parents=True, exist_ok=True)
        proof_path = str(proof_dir / "splash.jpg")
        save_crop_overlay(crop, res, proof_path)

    fi = res.get("frame_info", {})
    t_sec = fi.get("t_sec", 0.0) or 0.0
    h = int(t_sec // 3600)
    m = int((t_sec % 3600) // 60)
    s = t_sec % 60
    ts = f"{h:02d}:{m:02d}:{s:06.3f}"

    # Bounding box: full splash crop (v6a classifies the whole frame)
    H = crop.shape[0] if crop is not None else 0
    W = crop.shape[1] if crop is not None else 0
    bbox = {"x": 0, "y": 0, "width": W, "height": H} if crop is not None else None

    return {
        "verdict":      res["verdict"],
        "api_verdict":  _api_verdict(res["verdict"]),
        "prob_fail":    res.get("prob_fail"),
        "confidence":   round(1.0 - res["prob_fail"], 3) if res.get("prob_fail") is not None else 0.5,
        "frame_info":   fi,
        "proof_ts":     ts,
        "proof_path":   proof_path,
        "bbox":         bbox,
        "inference_ms": res.get("inference_ms"),
        "defect_type":  "all_present" if res["verdict"] == "PASS" else
                        "uncertain"   if res["verdict"] == "ABSTAIN" else "defect_detected",
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/inference/submit-job")
async def submit_job(
    video:  UploadFile = File(...),
    job_id: str        = Form(default=""),
    defect: str        = Form(default="unclassified"),
):
    if defect not in VALID_DEFECT and defect != "unclassified":
        return _err("invalid_parameter", "Field 'defect' must be one of: defective, non-defective")
    if not video.filename:
        return _err("missing_file", "No video file found in the request")
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        return _err("missing_file", "Upload must be a video file (.mp4/.avi/.mov/.mkv)")

    jid = job_id.strip() or f"JOB-{uuid.uuid4().hex[:6].upper()}"
    with _lock:
        if jid in _jobs:
            return _err("job_id_conflict", "A job with the provided job_id already exists",
                        {"job_id": jid})

    ts_str   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{jid}_{ts_str}_{defect.replace('-','_')}{ext}"
    vpath    = UPLOAD_DIR / filename
    with open(vpath, "wb") as f:
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

    _executor.submit(_process, jid, str(vpath))

    return _ok("job_scheduled", "Video received and inference job scheduled successfully", {
        "job_id":       jid,
        "file_name":    filename,
        "submitted_at": submitted_at,
    })


@app.get("/api/v1/inference/result")
async def get_result(job_id: str = Query(...)):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return _err("job_not_found", "No job found for the given job_id", {}, code=404)

    state = job["state"]
    if state in ("queued", "processing"):
        return _ok("processing", "Job is still in progress", {
            "job_id": job_id, "submitted_at": job["submitted_at"],
        })
    if state == "job_failed":
        return _err("job_failed", "Inference failed while processing the video",
                    {"job_id": job_id, "error_detail": job.get("error", "")}, code=500)

    r    = job["result"]
    host = os.environ.get("PUBLIC_HOST", "http://tangentthoughttech.com:8083")
    image_url = f"{host}/proofs/{job_id}/splash.jpg" if r.get("proof_path") else None

    return _ok("completed", "Inference completed successfully", {
        "job_id":            job_id,
        "submitted_at":      job["submitted_at"],
        "completed_at":      job["completed_at"],
        "input_defect_flag": job["defect_hint"],
        "verdict":           r["api_verdict"],
        "confidence":        r["confidence"],
        "proof": {
            "frame_index":        None,
            "timestamp_in_video": r["proof_ts"],
            "image_url":          image_url,
            "bounding_box":       r["bbox"],
            "defect_type":        r["defect_type"],
        },
        "detail": {
            "model":        "EfficientNet-B0",
            "prob_fail":    r["prob_fail"],
            "inference_ms": r["inference_ms"],
            "frame_info":   r["frame_info"],
        },
    })


@app.get("/api/v1/inference/status")
async def get_status(rows: int = Query(...)):
    if rows < 1 or rows > 100:
        return _err("invalid_parameter", "Parameter 'rows' must be an integer between 1 and 100")
    with _lock:
        active = [{"job_id": jid, **job}
                  for jid, job in _jobs.items()
                  if job["state"] in ("queued", "processing")]
        history = list(_history)
    combined = (active + history)[:rows]
    jobs_out = [{
        "job_id":       j.get("job_id"),
        "state":        j.get("state"),
        "verdict":      _api_verdict(j["result"]["verdict"]) if j.get("result") else None,
        "submitted_at": j.get("submitted_at"),
    } for j in combined]
    return _ok("ok", f"Fetched latest {len(jobs_out)} job rows",
               {"count": len(jobs_out), "jobs": jobs_out})


@app.get("/proofs/{job_id}/{filename}")
async def serve_proof(job_id: str, filename: str):
    from fastapi import HTTPException
    path = PROOF_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "Proof image not found")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/health")
async def health():
    with _lock:
        pending    = sum(1 for j in _jobs.values() if j["state"] == "queued")
        processing = sum(1 for j in _jobs.values() if j["state"] == "processing")
    return {"status": "ok", "version": "v6a", "model": "EfficientNet-B0",
            "pending": pending, "processing": processing}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return HTMLResponse(_UI)


_UI = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>IP4R v6a — EfficientNet Inspection</title>
<style>
  :root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;
        --pass:#3fb950;--abstain:#d29922;--fail:#f85149;--accent:#58a6ff;--radius:10px}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
  .topbar{display:flex;align-items:center;gap:12px;padding:14px 24px;background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:10}
  .topbar h1{font-size:1.1rem;font-weight:600}
  .badge{font-size:.7rem;font-weight:700;padding:3px 10px;border-radius:20px;background:#21262d;color:var(--accent);border:1px solid var(--border)}
  .meta{margin-left:auto;font-size:.78rem;color:var(--muted)}
  .main{display:flex;gap:20px;padding:24px;max-width:1200px;margin:0 auto}
  .left{flex:1;min-width:0;display:flex;flex-direction:column;gap:20px}
  .right{width:320px;flex-shrink:0;display:flex;flex-direction:column;gap:20px}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px}
  .card h2{font-size:.78rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px}
  .stats{display:flex;gap:12px}
  .stat{flex:1;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:14px;text-align:center}
  .stat .n{font-size:1.8rem;font-weight:700} .stat .l{font-size:.72rem;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.6px}
  .stat.pass .n{color:var(--pass)} .stat.abstain .n{color:var(--abstain)} .stat.fail .n{color:var(--fail)} .stat.pend .n{color:var(--accent)}
  .drop-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:36px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s}
  .drop-zone:hover,.drop-zone.drag{border-color:var(--accent);background:rgba(88,166,255,.05)}
  .drop-zone .icon{font-size:2.4rem;margin-bottom:10px}
  .drop-zone p{color:var(--muted);font-size:.85rem} .drop-zone strong{color:var(--accent)}
  #file-input{display:none}
  .form-row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
  select,input[type=text]{background:var(--bg);border:1px solid var(--border);border-radius:6px;color:var(--text);padding:8px 12px;font-size:.85rem}
  input[type=text]{flex:1;min-width:140px}
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
  .vbadge.pass{background:rgba(63,185,80,.15);color:var(--pass)}
  .vbadge.abstain{background:rgba(210,153,34,.15);color:var(--abstain)}
  .vbadge.fail{background:rgba(248,81,73,.15);color:var(--fail)}
  .vbadge.processing,.vbadge.queued{background:rgba(88,166,255,.1);color:var(--accent)}
  .empty-state{text-align:center;color:var(--muted);padding:28px 0;font-size:.82rem}
  .legend{display:flex;flex-direction:column;gap:10px;font-size:.8rem}
  .legend-item{display:flex;gap:10px;align-items:flex-start}
  .legend-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:4px}
  .legend-dot.pass{background:var(--pass)} .legend-dot.abstain{background:var(--abstain)} .legend-dot.fail{background:var(--fail)}
  .legend-item p{color:var(--muted);font-size:.75rem;margin-top:2px}
  .cfg-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;font-size:.78rem}
  .cfg-grid dt{color:var(--muted)} .cfg-grid dd{color:var(--text);font-weight:600}
  @media(max-width:860px){.main{flex-direction:column}.right{width:100%}.stats{flex-wrap:wrap}}
</style>
</head>
<body>
<div class="topbar">
  <h1>IP4R</h1>
  <span class="badge">v6a</span>
  <span class="badge" style="color:#d29922">EfficientNet-B0</span>
  <div class="meta" id="topinfo">Loading…</div>
</div>
<div class="main">
  <div class="left">
    <div class="stats">
      <div class="stat pass">   <div class="n" id="c-pass">—</div><div class="l">Pass</div></div>
      <div class="stat abstain"><div class="n" id="c-abs">—</div> <div class="l">Abstain</div></div>
      <div class="stat fail">   <div class="n" id="c-fail">—</div><div class="l">Fail</div></div>
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
          <option value="non-defective">non-defective</option>
          <option value="defective">defective</option>
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
          <strong>PASS</strong>
          <p>EfficientNet confidence that the LCD is good exceeds threshold. All segments appear correctly lit.</p>
        </div></div>
        <div class="legend-item"><div class="legend-dot abstain"></div><div>
          <strong>Abstain</strong>
          <p>Model confidence is in the uncertain zone (p_fail 0.40–0.60). Send to human re-inspection.</p>
        </div></div>
        <div class="legend-item"><div class="legend-dot fail"></div><div>
          <strong>FAIL</strong>
          <p>EfficientNet predicts defect with high confidence (p_fail &gt; 0.60).</p>
        </div></div>
      </div>
    </div>
    <div class="card">
      <h2>Model Config</h2>
      <dl class="cfg-grid">
        <dt>Version</dt>      <dd>v6a</dd>
        <dt>Model</dt>        <dd>EfficientNet-B0</dd>
        <dt>Input</dt>        <dd>480×640 crop</dd>
        <dt>FAIL thr</dt>     <dd>p_fail &gt; 0.60</dd>
        <dt>ABSTAIN zone</dt> <dd>0.40 – 0.60</dd>
        <dt>Candidates</dt>   <dd>6 frames</dd>
        <dt>Port</dt>         <dd>8083</dd>
        <dt>API</dt>          <dd>v1</dd>
      </dl>
    </div>
  </div>
</div>
<script>
const dropZone=document.getElementById('drop-zone'),fileInput=document.getElementById('file-input'),submitBtn=document.getElementById('submit-btn'),selectedEl=document.getElementById('selected-file');
let selectedFile=null,pollTimer=null;
fileInput.addEventListener('change',()=>selectFile(fileInput.files[0]));
dropZone.addEventListener('dragover',e=>{e.preventDefault();dropZone.classList.add('drag')});
dropZone.addEventListener('dragleave',()=>dropZone.classList.remove('drag'));
dropZone.addEventListener('drop',e=>{e.preventDefault();dropZone.classList.remove('drag');selectFile(e.dataTransfer.files[0])});
function selectFile(f){if(!f)return;selectedFile=f;selectedEl.textContent=`${f.name}  (${(f.size/1e6).toFixed(1)} MB)`;submitBtn.disabled=false}
async function submitJob(){
  if(!selectedFile)return;submitBtn.disabled=true;
  const fd=new FormData();fd.append('video',selectedFile);
  const jid=document.getElementById('job-id').value.trim();
  if(jid)fd.append('job_id',jid);fd.append('defect',document.getElementById('defect-hint').value);
  let data;
  try{const r=await fetch('/api/v1/inference/submit-job',{method:'POST',body:fd});const env=await r.json();if(!env.success){alert(env.message);submitBtn.disabled=false;return}data=env.data}
  catch(e){alert('Upload failed: '+e);submitBtn.disabled=false;return}
  showWaiting(data.job_id);startPoll(data.job_id)
}
function startPoll(jid){
  if(pollTimer)clearInterval(pollTimer);let prog=0;
  pollTimer=setInterval(async()=>{
    prog=Math.min(prog+4,90);const pf=document.getElementById('prog-fill');if(pf)pf.style.width=prog+'%';
    const r=await fetch('/api/v1/inference/result?job_id='+jid);const env=await r.json();
    if(env.status==='completed'){clearInterval(pollTimer);if(pf)pf.style.width='100%';showResult(env.data,jid);refreshHistory();submitBtn.disabled=false}
    else if(!env.success&&env.status!=='processing'){clearInterval(pollTimer);alert('Job failed: '+(env.data?.error_detail||''));submitBtn.disabled=false}
  },2500)
}
function showWaiting(jid){
  document.getElementById('result-card').style.display='block';
  document.getElementById('result-box').className='result-box waiting';
  document.getElementById('result-box').innerHTML=`<span class="spinner"></span> Processing <code>${jid}</code>…<div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>`
}
function showResult(d,jid){
  const v=d.verdict||'unknown';const cls=v==='pass'?'pass':v==='abstain'?'abstain':'fail';
  const emoji=v==='pass'?'✅':v==='abstain'?'⚠️':'❌';
  const det=d.detail||{};const pf=det.prob_fail!=null?` — p_fail=${det.prob_fail.toFixed(3)}`:'';
  const conf=d.confidence!=null?`  confidence ${(d.confidence*100).toFixed(1)}%`:'';
  let proofHTML='';
  if(d.proof?.image_url)proofHTML=`<div class="proof-img"><img src="${d.proof.image_url}" alt="splash frame" loading="lazy"/></div><div style="font-size:.72rem;color:var(--muted);margin-top:6px">t=${d.proof.timestamp_in_video} · ${d.proof.defect_type}</div>`;
  document.getElementById('result-box').className=`result-box ${cls}`;
  document.getElementById('result-box').innerHTML=`
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:2rem">${emoji}</span>
      <div><div class="verdict-label ${cls}">${v.toUpperCase()}${conf}</div>
      <div class="result-meta">${jid}${pf} · EfficientNet-B0</div></div>
    </div>${proofHTML}`
}
async function refreshHistory(){
  try{const r=await fetch('/api/v1/inference/status?rows=50');const env=await r.json();if(!env.success)return;
  const jobs=env.data.jobs;let p=0,a=0,f=0;
  jobs.forEach(j=>{if(j.verdict==='pass')p++;else if(j.verdict==='abstain')a++;else if(j.verdict==='fail')f++});
  document.getElementById('c-pass').textContent=p;document.getElementById('c-abs').textContent=a;document.getElementById('c-fail').textContent=f;
  if(!jobs.length){document.getElementById('history-body').innerHTML='<div class="empty-state">No jobs yet</div>';return}
  const rows=jobs.map(j=>{const v=j.verdict||j.state||'—';const t=j.submitted_at?new Date(j.submitted_at).toLocaleTimeString():'—';
    return`<tr><td style="color:var(--muted)">${t}</td><td style="max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${j.job_id||'—'}</td><td><span class="vbadge ${v}">${v}</span></td><td style="color:var(--muted)">${j.state||'—'}</td></tr>`
  }).join('');
  document.getElementById('history-body').innerHTML=`<table class="history-table"><thead><tr><th>Time</th><th>Job ID</th><th>Verdict</th><th>State</th></tr></thead><tbody>${rows}</tbody></table>`}catch(_){}
}
async function refreshHealth(){
  try{const r=await fetch('/health');const d=await r.json();
  document.getElementById('topinfo').textContent=`v6a EfficientNet-B0 · queue=${d.pending} · processing=${d.processing}`;
  document.getElementById('c-pend').textContent=d.pending+d.processing}catch(_){}
}
refreshHealth();refreshHistory();setInterval(refreshHealth,5000);setInterval(refreshHistory,8000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
