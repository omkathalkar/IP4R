#!/usr/bin/env python3
"""
Phase 0 Diagnostic — raw confidence extraction for failing videos.

Runs Elem YOLO at very low confidence (0.10) across the Phase-3 window
of specified videos and reports per-icon max/mean confidence, frames seen,
and adaptive-threshold prediction. Saves 1-fps frame snapshots.

Usage:
    python server_v7b/phase0_diag.py \
        --videos  '/path/to/video1.mp4' '/path/to/video2.mp4' \
        --elem-model runs/phase2_elem_v1/weights/best.pt \
        --out data/v7_results/phase0_diagnosis.md
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

P3_START_SEC     = 16.0
REQUIRED_ICONS   = [
    "Auto_Mode", "Battery", "Cool_Mode", "Dry_Mode",
    "Fan_Mode", "Fan_Speed", "Foot_Display", "H_Swing",
    "Light", "Lock", "Temperature", "Timer_OFF",
    "Timer_ON", "Turbo", "V_Swing",
]
ELEMENT_NAMES = [
    "Auto_Mode", "Battery", "Clock", "Cool_Mode", "Dry_Mode",
    "Energy-Save_Mode", "Fan_Mode", "Fan_Speed", "Foot_Display",
    "H_Swing", "Heat_Mode", "IR_Transmission", "Light", "Lock",
    "Sleep_Mode", "Temperature", "Timer_OFF", "Timer_ON",
    "Turbo", "V_Swing", "ion",
]

DIAG_CONF        = 0.10   # run YOLO at this low threshold to see all detections
BRIGHTNESS_BASE  = 120.0
P3_CONF_THR      = 0.70
P3_CONF_THR_MIN  = 0.40

_CACHE: dict = {}

def _load(path: str):
    if path not in _CACHE:
        from ultralytics import YOLO
        _CACHE[path] = YOLO(path)
    return _CACHE[path]


def analyse_video(vp: Path, elem_model: str, snap_dir: Path) -> dict:
    cap = cv2.VideoCapture(str(vp))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 12.0
    cap.release()

    f_p3_start = min(int(P3_START_SEC * fps), n_frames - 1)
    m2 = _load(elem_model)

    # per-icon tracking (raw, no threshold)
    icon_confs:  dict[str, list[float]] = defaultdict(list)  # all conf values seen
    icon_frames: dict[str, int]         = defaultdict(int)   # frames icon appeared
    raw_brightness: list[float]         = []
    frame_snapshots_saved = 0

    cap = cv2.VideoCapture(str(vp))
    for fi in range(f_p3_start, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frm = cap.read()
        if not ok:
            continue

        gray_mean = float(np.mean(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)))
        raw_brightness.append(gray_mean)

        preds = m2.predict(frm, conf=DIAG_CONF, imgsz=640, verbose=False)
        seen_this_frame: set[str] = set()
        if preds and len(preds[0].boxes):
            for i in range(len(preds[0].boxes)):
                cid = int(preds[0].boxes.cls[i].item())
                if cid < len(ELEMENT_NAMES):
                    nm = ELEMENT_NAMES[cid]
                    cf = float(preds[0].boxes.conf[i].item())
                    icon_confs[nm].append(cf)
                    seen_this_frame.add(nm)
        for nm in seen_this_frame:
            icon_frames[nm] += 1

        # Save one frame per second (at exact second boundaries)
        t_sec = fi / fps
        t_sec_int = int(t_sec)
        if abs(t_sec - t_sec_int) < (1 / fps / 2) and t_sec_int >= int(P3_START_SEC):
            fname = snap_dir / f"{vp.stem}_t{t_sec_int:02d}s.jpg"
            cv2.imwrite(str(fname), frm)
            frame_snapshots_saved += 1

    cap.release()

    mean_brightness = float(np.mean(raw_brightness)) if raw_brightness else 0.0
    scale = mean_brightness / BRIGHTNESS_BASE if mean_brightness < BRIGHTNESS_BASE else 1.0
    adaptive_thr = float(np.clip(P3_CONF_THR * scale, P3_CONF_THR_MIN, P3_CONF_THR))

    # Categorise each required icon
    per_icon = {}
    for icon in REQUIRED_ICONS:
        confs = icon_confs.get(icon, [])
        max_cf = max(confs) if confs else 0.0
        mean_cf = float(np.mean(confs)) if confs else 0.0
        frames_seen = icon_frames.get(icon, 0)

        if max_cf >= P3_CONF_THR:
            category = "ok-above-0.70"
        elif max_cf >= adaptive_thr:
            category = "under-confident-but-lit"   # CLAHE + adaptive thr will help
        elif max_cf >= DIAG_CONF:
            category = "very-low-conf"             # needs retraining or different approach
        else:
            category = "not-detected"              # either absent or total miss

        per_icon[icon] = {
            "max_conf":    round(max_cf, 3),
            "mean_conf":   round(mean_cf, 3),
            "frames_seen": frames_seen,
            "category":    category,
        }

    return {
        "video":            vp.name,
        "n_p3_frames":      len(raw_brightness),
        "mean_brightness":  round(mean_brightness, 1),
        "adaptive_thr":     round(adaptive_thr, 3),
        "snapshots_saved":  frame_snapshots_saved,
        "per_icon":         per_icon,
    }


def write_report(results: list[dict], out_path: Path):
    lines = [
        "# Phase 0 Diagnostic Report",
        "",
        "Raw Elem-YOLO confidences (threshold=0.10) across Phase-3 window.",
        "Categories:",
        "- `ok-above-0.70` — detected above fixed threshold; no problem",
        "- `under-confident-but-lit` — above adaptive threshold but below 0.70"
        " → CLAHE + adaptive thr should fix",
        "- `very-low-conf` — detected but max conf < adaptive thr"
        " → may need retraining",
        "- `not-detected` — never detected even at 0.10 → hardware/capture issue"
        " or complete model miss",
        "",
    ]

    for r in results:
        lines += [
            f"## {r['video']}",
            "",
            f"- Phase-3 frames: {r['n_p3_frames']}",
            f"- Mean brightness: {r['mean_brightness']:.1f} "
            f"(baseline={BRIGHTNESS_BASE:.0f})",
            f"- Adaptive threshold: {r['adaptive_thr']:.3f} "
            f"(base=0.70, floor=0.40)",
            f"- Frame snapshots saved: {r['snapshots_saved']}",
            "",
            "| Icon | Max Conf | Mean Conf | Frames | Category |",
            "|------|----------|-----------|--------|----------|",
        ]
        for icon, d in r["per_icon"].items():
            cat_marker = {
                "ok-above-0.70":          "✓",
                "under-confident-but-lit": "~",
                "very-low-conf":          "?",
                "not-detected":           "✗",
            }.get(d["category"], "")
            lines.append(
                f"| {icon} | {d['max_conf']:.3f} | {d['mean_conf']:.3f}"
                f" | {d['frames_seen']} | {cat_marker} {d['category']} |"
            )

        # Summary verdict
        categories = [d["category"] for d in r["per_icon"].values()]
        n_lit   = sum(1 for c in categories if c in {"ok-above-0.70", "under-confident-but-lit"})
        n_miss  = sum(1 for c in categories if c == "not-detected")
        n_lowcf = sum(1 for c in categories if c == "very-low-conf")

        lines += [
            "",
            f"**Summary**: {n_lit} icons lit (visible), "
            f"{n_lowcf} very-low-conf, {n_miss} not detected.",
        ]
        if n_miss == 0 and n_lowcf == 0:
            lines.append("→ CLAHE + adaptive threshold should resolve all failures.")
        elif n_miss > 0:
            lines.append(f"→ {n_miss} icons genuinely absent or hardware issue "
                         "— threshold tuning cannot fix these.")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Report saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--elem-model",
                        default="runs/phase2_elem_v1/weights/best.pt")
    parser.add_argument("--out",
                        default="data/v7_results/phase0_diagnosis.md")
    args = parser.parse_args()

    out_path = Path(args.out)
    snap_dir = out_path.parent / "phase0_snaps"
    snap_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 0 Diagnostic — {len(args.videos)} video(s)")
    print(f"YOLO conf floor: {DIAG_CONF}  |  snaps → {snap_dir}")
    print()

    all_results = []
    raw_data    = []
    for vpath in args.videos:
        vp = Path(vpath)
        if not vp.exists():
            print(f"  [SKIP] not found: {vpath}")
            continue
        print(f"  Analysing {vp.name} ...")
        r = analyse_video(vp, args.elem_model, snap_dir)
        all_results.append(r)
        raw_data.append(r)
        print(f"    brightness={r['mean_brightness']:.1f}  "
              f"adaptive_thr={r['adaptive_thr']:.3f}  "
              f"snaps={r['snapshots_saved']}")

    write_report(all_results, out_path)

    # also save raw JSON for further analysis
    raw_json = out_path.with_suffix(".json")
    with open(raw_json, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw JSON    → {raw_json}")


if __name__ == "__main__":
    main()
