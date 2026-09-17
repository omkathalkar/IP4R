#!/usr/bin/env python3
"""
server_v7b_r2a — v7b-r2 + ABSTAIN verdict

Changes vs v7b:
  consec=2       : temporal consistency requires 2 consecutive frames (same as v7b-r2)
  ABSTAIN logic  : icon seen in ≥1 frame above thr but never 2 consecutive → ABSTAIN
                   icon never seen above thr at all                         → FAIL
                   all required icons confirmed (consec met)                → PASS

  ABSTAIN accounts for borderline GOOD units where dim icons flicker across
  the threshold rather than being genuinely absent.
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

# ── Ph1: Brightness normalisation ─────────────────────────────────────────────
BRIGHTNESS_NORM_MODE = "none"   # "none" | "clahe" | "gamma"
CLAHE_CLIP_LIMIT     = 2.0
CLAHE_TILE_SIZE      = (8, 8)
GAMMA_VALUE          = 1.5

# ── Ph2: Adaptive threshold ───────────────────────────────────────────────────
ADAPTIVE_THR_ENABLED = False
BRIGHTNESS_BASELINE  = 120.0
P3_CONF_THR_MIN      = 0.40

# ── Ph3: Temporal consistency — 2 consecutive frames ─────────────────────────
P3_MIN_CONSECUTIVE = 2

# ── ABSTAIN: flicker-based three-way verdict ──────────────────────────────────
# PASS    — all required icons confirmed (consec≥2 met)
# ABSTAIN — some required icons not confirmed, but ALL of them were glimpsed
#           (detected in ≥1 frame above thr, just not 2 consecutive)
#           → dim LCD / borderline unit; send to human review
# FAIL    — at least one required icon never seen above thr at all
#           → icon genuinely absent; confident defect
ABSTAIN_ENABLED = True

# ── Ph4: Phase-2 anomaly flag ─────────────────────────────────────────────────
P2_FLAG_ENABLED = True

# ── Ph6: P1 fallback mask ─────────────────────────────────────────────────────
P1_FALLBACK_ENABLED = True

# ── Ph7: Batch Phase-3 inference ──────────────────────────────────────────────
BATCH_SIZE = 8

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
PURPLE = (200, 0, 200)

_CACHE: dict = {}


def _load(path: str):
    if path not in _CACHE:
        from ultralytics import YOLO
        _CACHE[path] = YOLO(path)
    return _CACHE[path]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_frame(frame: np.ndarray) -> np.ndarray:
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
    return frame


def _adaptive_thr(raw_frames: list[np.ndarray]) -> float:
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
    return [
        [0,             int(H * 0.85), W,             H            ],  # footer_digits
        [0,             0,             int(W * 0.25), int(H * 0.15)],  # top_left_block
        [int(W * 0.75), 0,             W,             int(H * 0.15)],  # top_right_block
        [int(W * 0.30), int(H * 0.40), int(W * 0.70), int(H * 0.60)],  # middle_block
        [int(W * 0.85), 0,             W,             int(H * 0.10)],  # signal_icon
    ]


def _infer_batch(
    m2,
    frames: list[np.ndarray],
    indices: list[int],
    conf: float = P3_CONF_THR,
) -> dict[int, list[tuple[str, float, list]]]:
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


def _apply_temporal_glimpse(
    frame_dets: dict[int, list],
    frame_order: list[int],
    thr: float,
) -> tuple[dict[str, float | None], dict[str, bool]]:
    """
    Apply consec≥P3_MIN_CONSECUTIVE temporal filter.

    Returns
    -------
    checklist : {icon: best_conf}  — None if consec threshold not met
    glimpsed  : {icon: bool}       — True if icon appeared ≥1 frame above thr
    """
    checklist:   dict[str, float | None] = {n: None  for n in ELEMENT_NAMES}
    glimpsed:    dict[str, bool]         = {n: False for n in ELEMENT_NAMES}
    consecutive: dict[str, int]          = {n: 0     for n in ELEMENT_NAMES}
    streak_best: dict[str, float]        = {n: 0.0   for n in ELEMENT_NAMES}

    for fi in frame_order:
        frame_best: dict[str, float] = {}
        for nm, cf, _ in frame_dets.get(fi, []):
            if cf >= thr and (nm not in frame_best or cf > frame_best[nm]):
                frame_best[nm] = cf

        for icon in ELEMENT_NAMES:
            if icon in frame_best:
                glimpsed[icon] = True
                consecutive[icon] += 1
                streak_best[icon] = max(streak_best[icon], frame_best[icon])
                if consecutive[icon] >= P3_MIN_CONSECUTIVE:
                    if checklist[icon] is None or streak_best[icon] > checklist[icon]:
                        checklist[icon] = streak_best[icon]
            else:
                consecutive[icon] = 0
                streak_best[icon] = 0.0

    return checklist, glimpsed


# kept for imports from eval.py that mirror v7b interface
def _apply_temporal(
    frame_dets: dict[int, list],
    frame_order: list[int],
    thr: float,
) -> dict[str, float | None]:
    checklist, _ = _apply_temporal_glimpse(frame_dets, frame_order, thr)
    return checklist


def _three_way_verdict(
    checklist: dict[str, float | None],
    glimpsed:  dict[str, bool],
) -> tuple[str, list[str], list[str]]:
    """
    Returns (verdict, abstain_icons, fail_icons).

    verdict      — "PASS" | "ABSTAIN" | "FAIL"
    abstain_icons — required icons glimpsed but not confirmed (consec not met)
    fail_icons    — required icons never seen above threshold
    """
    missing = [n for n in REQUIRED_ICONS if checklist[n] is None]
    if not missing:
        return "PASS", [], []

    if not ABSTAIN_ENABLED:
        return "FAIL", [], missing

    never_seen  = [n for n in missing if not glimpsed[n]]
    seen_not_consec = [n for n in missing if glimpsed[n]]

    if never_seen:
        # At least one icon is genuinely absent — confident defect
        return "FAIL", seen_not_consec, never_seen

    # All missing icons were glimpsed — borderline, send to review
    return "ABSTAIN", seen_not_consec, []


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _txt(frame, text, pos, color=WHITE, scale=0.50, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = int(pos[0]), int(pos[1])
    cv2.rectangle(frame, (x - 2, y - th - 3), (x + tw + 2, y + bl + 1), BLACK, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _box(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    _txt(frame, f"{label} {conf:.2f}", (x1, y1 - 5), color, scale=0.45)


def _banner(frame, text, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), BLACK, -1)
    cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.80,
                color, 2, cv2.LINE_AA)


def _checklist_three(frame, checklist, glimpsed, title, x, y,
                     scale_title=0.52, scale_item=0.42):
    _txt(frame, title, (x, y), YELLOW, scale=scale_title)
    for i, name in enumerate(REQUIRED_ICONS):
        iy = y + 20 * (i + 1)
        conf = checklist.get(name)
        if conf is not None:
            _txt(frame, f"[✓] {name}: {conf:.2f}", (x, iy), GREEN, scale=scale_item)
        elif glimpsed.get(name):
            _txt(frame, f"[~] {name} (glimpsed)", (x, iy), PURPLE, scale=scale_item)
        else:
            _txt(frame, f"[ ] {name}", (x, iy), RED, scale=scale_item)


def _thumbnail(frame, thumb_bgr, size=210):
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
    print(f"  norm={BRIGHTNESS_NORM_MODE}  adaptive={ADAPTIVE_THR_ENABLED}"
          f"  consec={P3_MIN_CONSECUTIVE}  abstain={ABSTAIN_ENABLED}  batch={BATCH_SIZE}")
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

    mask_boxes: list[list[int]] = [list(map(int, bx)) for _, bx in p1_best.values()]
    if P1_FALLBACK_ENABLED and not p1_ok:
        p1_fallback_used = True
        fallback = _p1_fallback_boxes(W, H)
        detected = set(p1_best.keys())
        for i, cls_name in enumerate(P1_CLASSES):
            if cls_name not in detected:
                mask_boxes.append(fallback[i])

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
        masked_norm = _normalize_frame(masked)
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

    p2_flag = P2_FLAG_ENABLED and any(n in REQUIRED_ICONS for n, _, _ in anomalous)

    # ── PHASE 3 ───────────────────────────────────────────────────────────────
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
        p3_norm_frames.append(_normalize_frame(frm))
        p3_frame_idx.append(fi)
    cap.release()

    p3_thr = _adaptive_thr(p3_raw_frames)
    p3_frame_dets = _infer_batch(m2, p3_norm_frames, p3_frame_idx, conf=p3_thr)
    p3_checklist, p3_glimpsed = _apply_temporal_glimpse(p3_frame_dets, p3_frame_idx, p3_thr)

    verdict, abstain_icons, fail_icons = _three_way_verdict(p3_checklist, p3_glimpsed)

    required_found = [n for n in REQUIRED_ICONS if p3_checklist[n] is not None]

    print(f"\nPhase 3: {len(required_found)}/{len(REQUIRED_ICONS)} confirmed"
          f"  (thr={p3_thr:.2f}, consec≥{P3_MIN_CONSECUTIVE})")
    if abstain_icons:
        print(f"  Glimpsed (not consec): {abstain_icons}")
    if fail_icons:
        print(f"  Never seen: {fail_icons}")
    print(f"  Verdict: {verdict}")

    # ── Annotated result video ─────────────────────────────────────────────────
    out_mp4 = od / f"{vp.stem}_result.mp4"
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W, H))
    vc = GREEN if verdict == "PASS" else (PURPLE if verdict == "ABSTAIN" else RED)

    running_p3: dict[str, float | None] = {n: None for n in ELEMENT_NAMES}
    running_glimpsed: dict[str, bool]   = {n: False for n in ELEMENT_NAMES}

    cap = cv2.VideoCapture(str(vp))
    fi = 0
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        f = frm.copy()

        if fi < f_p1_start:
            _banner(f, f"Waiting...  {fi/fps:.1f}s", WHITE)

        elif fi <= f_p1_end:
            for nm, (cf, bx) in p1_best.items():
                _box(f, *bx, nm, cf, GREEN if cf >= 0.70 else RED)
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
            _banner(f, f"Transition...  {fi/fps:.1f}s", YELLOW)

        else:
            for nm, cf, bx in p3_frame_dets.get(fi, []):
                if cf >= p3_thr:
                    running_glimpsed[nm] = True
                    if running_p3[nm] is None or cf > running_p3[nm]:
                        running_p3[nm] = cf
                    _box(f, *bx, nm, cf, GREEN)
            nf = sum(1 for v in running_p3.values() if v is not None)
            _checklist_three(f, running_p3, running_glimpsed,
                             "Phase 3 Checklist:", W - 300, 30)
            _banner(f,
                    f"PHASE 3 — {fi/fps:.1f}s  {nf}/{len(REQUIRED_ICONS)} confirmed"
                    f"  thr={p3_thr:.2f}", CYAN)
            if p2_masked_frame is not None:
                _thumbnail(f, p2_masked_frame)

        if fi == n_frames - 1:
            cv2.rectangle(f, (W // 2 - 240, H // 2 - 55),
                          (W // 2 + 240, H // 2 + 70), BLACK, -1)
            cv2.rectangle(f, (W // 2 - 240, H // 2 - 55),
                          (W // 2 + 240, H // 2 + 70), vc, 3)
            cv2.putText(f, f"VERDICT: {verdict}",
                        (W // 2 - 200, H // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, vc, 3, cv2.LINE_AA)
            if abstain_icons:
                _txt(f, f"Glimpsed: {', '.join(abstain_icons)}",
                     (W // 2 - 200, H // 2 + 38), PURPLE, scale=0.44)
            if fail_icons:
                _txt(f, f"Absent: {', '.join(fail_icons)}",
                     (W // 2 - 200, H // 2 + 58), RED, scale=0.44)

        writer.write(f)
        fi += 1

    cap.release()
    writer.release()

    summary = {
        "video":             vp.name,
        "verdict":           verdict,
        "p1_ok":             p1_ok,
        "p1_fallback_used":  p1_fallback_used,
        "p2_flag":           p2_flag,
        "p3_adaptive_thr":   p3_thr,
        "p3_found":          required_found,
        "p3_abstain_icons":  abstain_icons,
        "p3_fail_icons":     fail_icons,
        "p3_checklist":      p3_checklist,
        "p3_glimpsed":       p3_glimpsed,
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
    parser.add_argument("--out",        default="/tmp/v7b_r2a_results")
    parser.add_argument("--p1-model",   default=P1_MODEL)
    parser.add_argument("--elem-model", default=ELEM_MODEL)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for vpath in args.videos:
        v = run_video(vpath, args.p1_model, args.elem_model, args.out)
        print(f"\n  >> {Path(vpath).name}: {v}\n")
