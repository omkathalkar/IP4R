#!/usr/bin/env python3
"""
server_v7b — Mohit 3-Phase Pipeline (improvements over v7a)

v7b changes vs v7a:
  Ph1: CLAHE brightness normalisation applied to every Elem-YOLO input frame.
  Ph2: Per-video adaptive P3 threshold — scales down for dim LCDs, floored at 0.40.
  Ph3: Temporal consistency — icon only confirmed after N consecutive frames above thr.
  Ph4: Phase-2 anomaly flag (p2_flag) — logged, not yet a hard FAIL.
  Ph6: P1 fallback mask — fixed region mask when P1 YOLO misses any class.
  Ph7: Batch Phase-3 inference — all Phase-3 frames collected then run in batches.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Timing ────────────────────────────────────────────────────────────────────
P1_SKIP_SEC  = 3.0
P1_END_SEC   = 14.9
P2_SEC       = 15.0
P3_START_SEC = 16.0

# ── Base thresholds ───────────────────────────────────────────────────────────
P1_CONF_THR    = 0.30
P2_CONF_THR    = 0.25
P3_CONF_THR    = 0.70
P1_SAMPLE_STEP = 12

# ── v7b Ph1: Brightness normalisation ─────────────────────────────────────────
# Applied to every frame before it is passed to Elem YOLO (Phase 2 & 3).
BRIGHTNESS_NORM_MODE = "none"   # "none" | "clahe" | "gamma"
# "clahe" was tested and found to boost dim-but-present NOT_GOOD icons above
# the 0.70 threshold, collapsing NOT_GOOD catch from 71% → 22% on Sep-15.
CLAHE_CLIP_LIMIT     = 2.0
CLAHE_TILE_SIZE      = (8, 8)
GAMMA_VALUE          = 1.5

# ── v7b Ph2: Per-video adaptive threshold ──────────────────────────────────────
# Scales P3_CONF_THR down proportionally when mean Phase-3 brightness is below
# BRIGHTNESS_BASELINE. Floored at P3_CONF_THR_MIN to avoid trivial false-PASSes.
ADAPTIVE_THR_ENABLED = False
# Disabled: Sep-15 brightness ~51 across ALL videos (both GOOD and NOT_GOOD).
# Lowering threshold uniformly lets dim-but-present NOT_GOOD icons pass,
# collapsing NOT_GOOD catch from 71% → 22%. Discrimination depends on the
# brightness difference between GOOD icons and dim/absent NOT_GOOD icons.
BRIGHTNESS_BASELINE  = 120.0   # kept for reference; unused while disabled
P3_CONF_THR_MIN      = 0.40   # floor if adaptive is re-enabled in future

# ── v7b Ph3: Temporal consistency ─────────────────────────────────────────────
# An icon only counts as "found" once it clears the adaptive threshold in this
# many *consecutive* frames. Resets to 0 on any frame where it falls below.
P3_MIN_CONSECUTIVE = 1
# Tested at 2: caught 3 more NOT_GOOD on Sep-15 but created 2 new GOOD false
# alarms. Root cause unclear (label quality vs. genuine LCD flicker). Reverted
# to 1 (equivalent to v7a checklist logic) until ground truth is confirmed.

# ── v7b Ph4: Phase-2 anomaly flag ─────────────────────────────────────────────
P2_FLAG_ENABLED = True  # compute p2_flag; not yet wired as hard FAIL

# ── v7b Ph6: P1 fallback mask ─────────────────────────────────────────────────
# When P1 YOLO misses one or more classes, cover typical layout regions so
# Phase-2 masking still erases the digit/block areas.
P1_FALLBACK_ENABLED = True

# ── v7b Ph7: Batch Phase-3 inference ──────────────────────────────────────────
# Frames per YOLO batch; set to 0 to disable and fall back to per-frame mode.
BATCH_SIZE = 8

# Internal: YOLO is run at this low confidence to capture detections that might
# exceed the adaptive threshold (computed afterwards). Filtered in post-processing.
_YOLO_MIN_CONF = 0.20

# ── Class lists ───────────────────────────────────────────────────────────────
P1_CLASSES = [
    "footer_digits", "top_left_block", "top_right_block",
    "middle_block", "signal_icon",
]

ELEMENT_NAMES = [
    "Auto_Mode", "Battery", "Clock", "Cool_Mode", "Dry_Mode",
    "Energy-Save_Mode", "Fan_Mode", "Fan_Speed", "Foot_Display",
    "H_Swing", "Heat_Mode", "IR_Transmission", "Light", "Lock",
    "Sleep_Mode", "Temperature", "Timer_OFF", "Timer_ON",
    "Turbo", "V_Swing", "ion",
]

_EXCLUDED_ICONS = {
    "Sleep_Mode",
    "Clock",
    "Energy-Save_Mode",
    "Heat_Mode",
    "IR_Transmission",
    "ion",
}
REQUIRED_ICONS = [n for n in ELEMENT_NAMES if n not in _EXCLUDED_ICONS]

# ── Palette (BGR) ─────────────────────────────────────────────────────────────
GREEN  = (0, 220, 0)
RED    = (0, 0, 220)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
YELLOW = (0, 220, 220)
CYAN   = (220, 220, 0)
ORANGE = (0, 140, 255)

_CACHE: dict = {}


def _load(path: str):
    if path not in _CACHE:
        from ultralytics import YOLO
        _CACHE[path] = YOLO(path)
    return _CACHE[path]


# ── v7b helpers ───────────────────────────────────────────────────────────────

def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Apply brightness normalisation according to BRIGHTNESS_NORM_MODE."""
    if BRIGHTNESS_NORM_MODE == "clahe":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if BRIGHTNESS_NORM_MODE == "gamma":
        inv_g = 1.0 / GAMMA_VALUE
        lut = np.array([((i / 255.0) ** inv_g) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(frame, lut)
    return frame  # "none"


def _adaptive_thr(raw_frames: list[np.ndarray]) -> float:
    """Return the effective P3 threshold, scaled down for dim videos."""
    if not ADAPTIVE_THR_ENABLED or not raw_frames:
        return P3_CONF_THR
    brightness = float(np.mean([
        np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in raw_frames
    ]))
    if brightness >= BRIGHTNESS_BASELINE:
        return P3_CONF_THR
    scale = brightness / BRIGHTNESS_BASELINE
    return float(np.clip(P3_CONF_THR * scale, P3_CONF_THR_MIN, P3_CONF_THR))


def _p1_fallback_boxes(W: int, H: int) -> list[list[int]]:
    """Approximate bounding boxes for each P1_CLASSES region based on typical layout."""
    return [
        [0,            int(H * 0.85), W,            H           ],  # footer_digits
        [0,            0,             int(W * 0.25), int(H * 0.15)],  # top_left_block
        [int(W * 0.75), 0,            W,             int(H * 0.15)],  # top_right_block
        [int(W * 0.30), int(H * 0.40), int(W * 0.70), int(H * 0.60)],  # middle_block
        [int(W * 0.85), 0,            W,             int(H * 0.10)],  # signal_icon
    ]


def _infer_batch(
    m2,
    frames: list[np.ndarray],
    indices: list[int],
    conf: float = P3_CONF_THR,
) -> dict[int, list[tuple[str, float, list]]]:
    """Run Elem YOLO in batches; return {frame_idx: [(name, conf, box), ...]}."""
    out: dict[int, list] = {}
    bs = BATCH_SIZE if BATCH_SIZE > 0 else len(frames)
    for start in range(0, len(frames), bs):
        batch_frm = frames[start:start + bs]
        batch_idx = indices[start:start + bs]
        preds = m2.predict(batch_frm, conf=conf, imgsz=640, verbose=False)
        for pred, fi in zip(preds, batch_idx):
            dets: list[tuple[str, float, list]] = []
            if pred and len(pred.boxes):
                for i in range(len(pred.boxes)):
                    cid = int(pred.boxes.cls[i].item())
                    if cid < len(ELEMENT_NAMES):
                        dets.append((
                            ELEMENT_NAMES[cid],
                            float(pred.boxes.conf[i].item()),
                            pred.boxes.xyxy[i].cpu().numpy().tolist(),
                        ))
            out[fi] = dets
    return out


def _apply_temporal(
    frame_dets: dict[int, list],
    frame_order: list[int],
    thr: float,
) -> dict[str, float | None]:
    """Apply threshold + temporal consistency; return per-icon best-conf checklist."""
    checklist: dict[str, float | None] = {n: None for n in ELEMENT_NAMES}
    consecutive: dict[str, int]   = {n: 0   for n in ELEMENT_NAMES}
    streak_best: dict[str, float] = {n: 0.0 for n in ELEMENT_NAMES}

    for fi in frame_order:
        frame_best: dict[str, float] = {}
        for nm, cf, _ in frame_dets.get(fi, []):
            if cf >= thr and (nm not in frame_best or cf > frame_best[nm]):
                frame_best[nm] = cf

        for icon in ELEMENT_NAMES:
            if icon in frame_best:
                consecutive[icon] += 1
                streak_best[icon] = max(streak_best[icon], frame_best[icon])
                if consecutive[icon] >= P3_MIN_CONSECUTIVE:
                    if checklist[icon] is None or streak_best[icon] > checklist[icon]:
                        checklist[icon] = streak_best[icon]
            else:
                consecutive[icon] = 0
                streak_best[icon] = 0.0

    return checklist


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _txt(frame, text: str, pos: tuple, color=WHITE, scale=0.50, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = int(pos[0]), int(pos[1])
    cv2.rectangle(frame, (x - 2, y - th - 3), (x + tw + 2, y + bl + 1), BLACK, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _box(frame, x1, y1, x2, y2, label: str, conf: float, color):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    _txt(frame, f"{label} {conf:.2f}", (x1, y1 - 5), color, scale=0.45)


def _banner(frame, text: str, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), BLACK, -1)
    cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.80,
                color, 2, cv2.LINE_AA)


def _checklist(frame, items: dict, title: str, x: int, y: int,
               scale_title=0.52, scale_item=0.42):
    _txt(frame, title, (x, y), YELLOW, scale=scale_title)
    for i, (name, conf) in enumerate(items.items()):
        iy = y + 20 * (i + 1)
        if conf is not None:
            _txt(frame, f"[x] {name}: {conf:.2f}", (x, iy), GREEN, scale=scale_item)
        else:
            _txt(frame, f"[ ] {name}", (x, iy), RED, scale=scale_item)


def _thumbnail(frame, thumb_bgr, size: int = 210):
    th = cv2.resize(thumb_bgr, (size, int(size * thumb_bgr.shape[0] / thumb_bgr.shape[1])))
    h_t = th.shape[0]
    frame[0:h_t, 0:size] = th
    cv2.rectangle(frame, (0, 0), (size, h_t), GREEN, 2)


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_video(
    video_path: str,
    p1_model: str,
    elem_model: str,
    out_dir: str,
) -> str:
    vp = Path(video_path)
    od = Path(out_dir)
    dd = od / f"debug_{vp.stem}"
    dd.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(vp))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 12.0
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    f_p1_start = int(P1_SKIP_SEC  * fps)
    f_p1_end   = min(int(P1_END_SEC   * fps), n_frames - 1)
    f_p2       = min(int(P2_SEC        * fps), n_frames - 1)
    f_p3_start = min(int(P3_START_SEC * fps), n_frames - 1)

    print(f"\n{'='*70}")
    print(f"  {vp.name}  ({n_frames}f @ {fps:.0f}fps = {n_frames/fps:.1f}s)")
    print(f"  P1: {f_p1_start}-{f_p1_end}  P2: {f_p2}  P3: {f_p3_start}-{n_frames-1}")
    print(f"  norm={BRIGHTNESS_NORM_MODE}  adaptive={ADAPTIVE_THR_ENABLED}"
          f"  consec={P3_MIN_CONSECUTIVE}  batch={BATCH_SIZE}")
    print(f"{'='*70}")

    # ── PHASE 1 ───────────────────────────────────────────────────────────────
    m1 = _load(p1_model)
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
    p1_checklist = {c: (p1_best[c][0] if c in p1_best else None) for c in P1_CLASSES}
    p1_fallback_used = False

    # Ph6: build the set of mask boxes; add fallback regions for any missed class
    mask_boxes: list[list[int]] = [list(map(int, bx)) for _, bx in p1_best.values()]
    if P1_FALLBACK_ENABLED and not p1_ok:
        p1_fallback_used = True
        fallback = _p1_fallback_boxes(W, H)
        detected = set(p1_best.keys())
        for i, cls_name in enumerate(P1_CLASSES):
            if cls_name not in detected:
                mask_boxes.append(fallback[i])
        print(f"  Ph6 fallback: added {len(mask_boxes) - len(p1_best)} fixed mask(s)")

    print("Phase 1 results:")
    for c, v in p1_checklist.items():
        mark = "[x]" if v else "[ ]"
        print(f"  {mark} {c}: {v:.2f}" if v else f"  {mark} {c}")

    # debug image 01
    cap = cv2.VideoCapture(str(vp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_p1_end)
    _, dbg1 = cap.read()
    cap.release()
    if dbg1 is not None:
        d = dbg1.copy()
        for nm, (cf, bx) in p1_best.items():
            _box(d, *bx, nm, cf, GREEN if cf >= 0.70 else RED)
        _checklist(d, p1_checklist, "Phase 1 Checklist:", W - 285, 30)
        _banner(d, f"PHASE 1 — Digit Detection (end @ {P1_END_SEC}s)", GREEN)
        cv2.imwrite(str(dd / "01_phase1_detections.jpg"), d)

    # ── PHASE 2 ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(vp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_p2)
    _, p2_raw = cap.read()
    cap.release()

    anomalous: list[tuple[str, float, list]] = []
    p2_masked_frame = None

    if p2_raw is not None:
        masked = p2_raw.copy()
        PAD = 6
        for bx in mask_boxes:
            x1, y1, x2, y2 = bx
            cv2.rectangle(masked,
                          (max(0, x1 - PAD), max(0, y1 - PAD)),
                          (x2 + PAD, y2 + PAD), WHITE, -1)
        p2_masked_frame = masked.copy()
        cv2.imwrite(str(dd / "02_phase2_masked.jpg"), masked)

        m2 = _load(elem_model)
        masked_norm = _normalize_frame(masked)  # Ph1: normalise before inference
        preds = m2.predict(masked_norm, conf=P2_CONF_THR, imgsz=640, verbose=False)
        if preds and len(preds[0].boxes):
            for i in range(len(preds[0].boxes)):
                cid = int(preds[0].boxes.cls[i].item())
                if cid < len(ELEMENT_NAMES):
                    anomalous.append((
                        ELEMENT_NAMES[cid],
                        float(preds[0].boxes.conf[i].item()),
                        preds[0].boxes.xyxy[i].cpu().numpy().tolist(),
                    ))
        if anomalous:
            print(f"  Phase 2 ANOMALOUS: {[(n, f'{c:.2f}') for n, c, _ in anomalous]}")
        else:
            print("  Phase 2: No anomalous icons detected")

    # Ph4: flag if any REQUIRED_ICONS appear in Phase-2 anomalous
    p2_flag = P2_FLAG_ENABLED and any(n in REQUIRED_ICONS for n, _, _ in anomalous)
    if p2_flag:
        print("  ** p2_flag=True: required icon detected in masked Phase-2 frame")

    # ── PHASE 3: collect frames → adaptive thr → batch infer → temporal ───────
    m2 = _load(elem_model)

    p3_raw_frames:  list[np.ndarray] = []
    p3_norm_frames: list[np.ndarray] = []
    p3_frame_idx:   list[int]        = []

    cap = cv2.VideoCapture(str(vp))
    for fi in range(f_p3_start, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frm = cap.read()
        if not ok:
            continue
        p3_raw_frames.append(frm)
        p3_norm_frames.append(_normalize_frame(frm))  # Ph1
        p3_frame_idx.append(fi)
    cap.release()

    # Ph2: adaptive threshold
    p3_thr = _adaptive_thr(p3_raw_frames)
    print(f"  Phase 3 threshold: {p3_thr:.3f} "
          f"(base={P3_CONF_THR}, adaptive={'ON' if ADAPTIVE_THR_ENABLED else 'OFF'})")

    # Ph7: batch inference
    p3_frame_dets = _infer_batch(m2, p3_norm_frames, p3_frame_idx, conf=p3_thr)

    # Ph3: temporal consistency
    p3_checklist = _apply_temporal(p3_frame_dets, p3_frame_idx, p3_thr)

    required_found   = [n for n in REQUIRED_ICONS if p3_checklist[n] is not None]
    required_missing = [n for n in REQUIRED_ICONS if p3_checklist[n] is None]
    verdict = "PASS" if not required_missing else "FAIL"

    print(f"\nPhase 3: {len(required_found)}/{len(REQUIRED_ICONS)} required icons found"
          f"  (thr={p3_thr:.2f}, consec≥{P3_MIN_CONSECUTIVE})")
    if required_missing:
        print(f"  Missing: {required_missing}")
    print(f"  Verdict: {verdict}")

    # Debug images for Phase 3 at key seconds
    p3_snap_secs   = {int(s) for s in [16, 17, 18, 19] if int(s * fps) < n_frames}
    p3_saved_secs: set[int] = set()
    running_check: dict[str, float | None] = {n: None for n in ELEMENT_NAMES}

    for fi in p3_frame_idx:
        t_sec = int(fi / fps)
        for nm, cf, _ in p3_frame_dets.get(fi, []):
            if cf >= p3_thr and (running_check[nm] is None or cf > running_check[nm]):
                running_check[nm] = cf
        if t_sec in p3_snap_secs and t_sec not in p3_saved_secs:
            raw_idx = p3_frame_idx.index(fi)
            snap = p3_raw_frames[raw_idx].copy()
            for nm, cf, bx in p3_frame_dets.get(fi, []):
                if cf >= p3_thr:
                    _box(snap, *bx, nm, cf, GREEN)
            nf = sum(1 for v in running_check.values() if v is not None)
            _checklist(snap, running_check, "Phase 3 Icon Checklist:", W - 300, 30)
            _banner(snap,
                    f"PHASE 3 — [{t_sec}s]  {nf}/{len(ELEMENT_NAMES)} detected"
                    f"  thr={p3_thr:.2f}", CYAN)
            if p2_masked_frame is not None:
                _thumbnail(snap, p2_masked_frame)
            idx = len(p3_saved_secs) + 3
            cv2.imwrite(str(dd / f"{idx:02d}_phase3_{t_sec}s.jpg"), snap)
            p3_saved_secs.add(t_sec)

    # ── Annotated result video ─────────────────────────────────────────────────
    out_mp4 = od / f"{vp.stem}_result.mp4"
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W, H))

    running_p3: dict[str, float | None] = {n: None for n in ELEMENT_NAMES}

    cap = cv2.VideoCapture(str(vp))
    fi = 0
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        f = frm.copy()

        if fi < f_p1_start:
            _banner(f, f"Waiting...  {fi/fps:.1f}s  (Phase 1 starts at {P1_SKIP_SEC:.0f}s)", WHITE)

        elif fi <= f_p1_end:
            for nm, (cf, bx) in p1_best.items():
                _box(f, *bx, nm, cf, GREEN if cf >= 0.70 else RED)
            _checklist(f, p1_checklist, "Phase 1 Checklist:", W - 285, 30)
            nf = sum(1 for v in p1_checklist.values() if v is not None)
            _banner(f, f"PHASE 1 — Digit Detection  {fi/fps:.1f}s  {nf}/{len(P1_CLASSES)}", GREEN)

        elif fi == f_p2:
            if p2_masked_frame is not None:
                f = p2_masked_frame.copy()
            for nm, cf, bx in anomalous:
                _box(f, *bx, nm, cf, ORANGE)
            tag = (f"Anomalous: {', '.join(n for n,_,_ in anomalous)}"
                   if anomalous else "No anomalous icons")
            _banner(f, f"PHASE 2 — Anomaly @ 15s  |  {tag}", ORANGE if anomalous else GREEN)

        elif fi < f_p3_start:
            _banner(f, f"Transition...  {fi/fps:.1f}s  (Phase 3 starts at {P3_START_SEC:.0f}s)", YELLOW)

        else:
            for nm, cf, bx in p3_frame_dets.get(fi, []):
                if cf >= p3_thr:
                    if running_p3[nm] is None or cf > running_p3[nm]:
                        running_p3[nm] = cf
                    _box(f, *bx, nm, cf, GREEN)
            nf = sum(1 for v in running_p3.values() if v is not None)
            _checklist(f, running_p3, "Phase 3 Icon Checklist:", W - 300, 30)
            _banner(f,
                    f"PHASE 3 — {fi/fps:.1f}s  {nf}/{len(ELEMENT_NAMES)} detected"
                    f"  thr={p3_thr:.2f}", CYAN)
            if p2_masked_frame is not None:
                _thumbnail(f, p2_masked_frame)

        if fi == n_frames - 1:
            vc = GREEN if verdict == "PASS" else RED
            cv2.rectangle(f, (W // 2 - 220, H // 2 - 50),
                          (W // 2 + 220, H // 2 + 50), BLACK, -1)
            cv2.rectangle(f, (W // 2 - 220, H // 2 - 50),
                          (W // 2 + 220, H // 2 + 50), vc, 3)
            cv2.putText(f, f"VERDICT: {verdict}",
                        (W // 2 - 190, H // 2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, vc, 3, cv2.LINE_AA)
            miss_str = f"Missing: {', '.join(required_missing)}" if required_missing else ""
            if miss_str:
                _txt(f, miss_str, (W // 2 - 190, H // 2 + 50), RED, scale=0.50)

        writer.write(f)
        fi += 1

    cap.release()
    writer.release()
    print(f"  Result video: {out_mp4}")

    summary = {
        "video":             vp.name,
        "verdict":           verdict,
        "p1_ok":             p1_ok,
        "p1_fallback_used":  p1_fallback_used,
        "p1_detections":     {k: {"conf": v[0], "box": v[1]} for k, v in p1_best.items()},
        "p2_anomalous_icons": [{"name": n, "conf": c} for n, c, _ in anomalous],
        "p2_flag":           p2_flag,
        "p3_adaptive_thr":   p3_thr,
        "p3_checklist":      p3_checklist,
        "required_found":    required_found,
        "required_missing":  required_missing,
    }
    with open(dd / "result.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return verdict


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    P1_MODEL   = "/home/om/src/IP4R/data/macro_dataset/runs/macro_test/weights/best.pt"
    ELEM_MODEL = "/home/om/src/IP4R/runs/phase2_elem_v1/weights/best.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--out",        default="/tmp/v7b_results")
    parser.add_argument("--p1-model",   default=P1_MODEL)
    parser.add_argument("--elem-model", default=ELEM_MODEL)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for vpath in args.videos:
        v = run_video(vpath, args.p1_model, args.elem_model, args.out)
        print(f"\n  >> {Path(vpath).name}: {v}\n")
