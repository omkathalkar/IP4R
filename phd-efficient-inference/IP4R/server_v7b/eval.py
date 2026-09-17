#!/usr/bin/env python3
"""
server_v7b eval — batch evaluation for the v7b 3-phase pipeline.

Usage:
    python -m server_v7b.eval \\
        --good-dir    '/path/to/GOOD' \\
        --notgood-dir '/path/to/NOT GOOD' \\
        --p1-model    data/macro_dataset/runs/macro_test/weights/best.pt \\
        --elem-model  runs/phase2_elem_v1/weights/best.pt \\
        --out         data/v7_results/eval_v7b_Sep15.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from server_v7b.pipeline import (
    P1_SKIP_SEC, P1_END_SEC, P2_SEC, P3_START_SEC,
    P1_CONF_THR, P2_CONF_THR, P3_CONF_THR, P1_SAMPLE_STEP,
    P1_CLASSES, ELEMENT_NAMES, REQUIRED_ICONS,
    BRIGHTNESS_NORM_MODE, ADAPTIVE_THR_ENABLED, BRIGHTNESS_BASELINE,
    P3_CONF_THR_MIN, P3_MIN_CONSECUTIVE, P2_FLAG_ENABLED,
    P1_FALLBACK_ENABLED, BATCH_SIZE,
    _load, _normalize_frame, _adaptive_thr, _p1_fallback_boxes,
    _infer_batch, _apply_temporal,
)

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def run_one(video_path: str, p1_model: str, elem_model: str) -> dict:
    vp = Path(video_path)

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

    # ── Phase 1 ───────────────────────────────────────────────────────────────
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
    p1_fallback_used = False

    # Ph6: fallback mask boxes
    mask_boxes: list[list[int]] = [list(map(int, bx)) for _, bx in p1_best.values()]
    if P1_FALLBACK_ENABLED and not p1_ok:
        p1_fallback_used = True
        fallback = _p1_fallback_boxes(W, H)
        detected = set(p1_best.keys())
        for i, cls_name in enumerate(P1_CLASSES):
            if cls_name not in detected:
                mask_boxes.append(fallback[i])

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    m2 = _load(elem_model)
    anomalous: list[str] = []

    cap = cv2.VideoCapture(str(vp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_p2)
    ok, p2_frm = cap.read()
    cap.release()

    if ok:
        masked = p2_frm.copy()
        PAD = 6
        for bx in mask_boxes:
            x1, y1, x2, y2 = bx
            masked[max(0, y1 - PAD):y2 + PAD, max(0, x1 - PAD):x2 + PAD] = 255
        masked_norm = _normalize_frame(masked)  # Ph1
        preds = m2.predict(masked_norm, conf=P2_CONF_THR, imgsz=640, verbose=False)
        if preds and len(preds[0].boxes):
            for i in range(len(preds[0].boxes)):
                cid = int(preds[0].boxes.cls[i].item())
                if cid < len(ELEMENT_NAMES):
                    anomalous.append(ELEMENT_NAMES[cid])

    p2_flag = P2_FLAG_ENABLED and any(n in REQUIRED_ICONS for n in anomalous)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
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

    # Ph2: adaptive threshold (from raw brightness, before CLAHE)
    p3_thr = _adaptive_thr(p3_raw_frames)

    # Ph7: batch inference; Ph3: temporal consistency
    p3_frame_dets = _infer_batch(m2, p3_norm_frames, p3_frame_idx, conf=p3_thr)
    p3_checklist  = _apply_temporal(p3_frame_dets, p3_frame_idx, p3_thr)

    required_missing = [n for n in REQUIRED_ICONS if p3_checklist[n] is None]
    verdict = "PASS" if not required_missing else "FAIL"

    return {
        "name":             vp.name,
        "verdict":          verdict,
        "p1_ok":            p1_ok,
        "p1_fallback_used": p1_fallback_used,
        "p1_found":         list(p1_best.keys()),
        "p2_anomalous":     anomalous,
        "p2_flag":          p2_flag,
        "p3_adaptive_thr":  round(p3_thr, 3),
        "p3_found":         [n for n in REQUIRED_ICONS if p3_checklist[n] is not None],
        "p3_missing":       required_missing,
        "p3_checklist":     p3_checklist,
    }


def main():
    parser = argparse.ArgumentParser(description="server_v7b end-to-end eval")
    parser.add_argument("--good-dir",    required=True)
    parser.add_argument("--notgood-dir", default=None)
    parser.add_argument("--p1-model",
                        default="data/macro_dataset/runs/macro_test/weights/best.pt")
    parser.add_argument("--elem-model",
                        default="runs/phase2_elem_v1/weights/best.pt")
    parser.add_argument("--out",
                        default="data/v7_results/eval_v7b.json")
    args = parser.parse_args()

    good_videos    = sorted(p for p in Path(args.good_dir).iterdir()
                            if p.suffix.lower() in VIDEO_EXT)
    notgood_videos = sorted(p for p in Path(args.notgood_dir).iterdir()
                            if p.suffix.lower() in VIDEO_EXT) \
                     if args.notgood_dir else []

    W = 108
    print()
    print("─" * W)
    print(f"  server_v7b  —  Mohit 3-Phase Pipeline")
    print(f"  P1 model  : {Path(args.p1_model).name}")
    print(f"  Elem model: {Path(args.elem_model).name}")
    print(f"  norm={BRIGHTNESS_NORM_MODE}  adaptive={'ON' if ADAPTIVE_THR_ENABLED else 'OFF'}"
          f"  consec≥{P3_MIN_CONSECUTIVE}  p2flag={'ON' if P2_FLAG_ENABLED else 'OFF'}"
          f"  p1fallback={'ON' if P1_FALLBACK_ENABLED else 'OFF'}  batch={BATCH_SIZE}")
    print(f"  P3 base thr={P3_CONF_THR:.0%}  adaptive floor={P3_CONF_THR_MIN:.0%}"
          f"  required={len(REQUIRED_ICONS)} icons")
    print("─" * W)
    fmt = "  {:<46} {:>9}  {:>4}  {:>5}  {:>6}  {:>5}  {:>3}"
    print(fmt.format("Video", "GT", "P1", "P2flg", "P3", "thr", "OK"))
    print("─" * W)

    results = []
    good_correct = notgood_correct = 0

    for vp in good_videos:
        r = run_one(str(vp), args.p1_model, args.elem_model)
        r["gt"] = "GOOD"
        ok = r["verdict"] == "PASS"
        if ok:
            good_correct += 1
        results.append(r)
        p3_str = f"{len(r['p3_found'])}/{len(REQUIRED_ICONS)}"
        print(fmt.format(
            vp.name[:45], "GOOD",
            "✓" if r["p1_ok"] else "✗",
            "!" if r["p2_flag"] else "-",
            p3_str,
            f"{r['p3_adaptive_thr']:.2f}",
            "✓" if ok else "✗",
        ))
        if not ok:
            print(f"    [FAIL] missing: {r['p3_missing']}")

    print("─" * W)

    for vp in notgood_videos:
        r = run_one(str(vp), args.p1_model, args.elem_model)
        r["gt"] = "NOT_GOOD"
        ok = r["verdict"] == "FAIL"
        if ok:
            notgood_correct += 1
        results.append(r)
        p3_str = f"{len(r['p3_found'])}/{len(REQUIRED_ICONS)}"
        print(fmt.format(
            vp.name[:45], "NOT_GOOD",
            "✓" if r["p1_ok"] else "✗",
            "!" if r["p2_flag"] else "-",
            p3_str,
            f"{r['p3_adaptive_thr']:.2f}",
            "✓" if ok else "✗",
        ))
        if ok:
            print(f"    [CAUGHT] missing: {r['p3_missing']}")

    print("─" * W)
    print(f"\n  GOOD  pass rate : {good_correct}/{len(good_videos)}"
          f"  ({100*good_correct/max(1,len(good_videos)):.0f}%)")
    print(f"  NOT_GOOD catch  : {notgood_correct}/{len(notgood_videos)}"
          f"  ({100*notgood_correct/max(1,len(notgood_videos)):.0f}%)")
    print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "version":         "v7b",
            "good_correct":    good_correct,
            "good_total":      len(good_videos),
            "notgood_correct": notgood_correct,
            "notgood_total":   len(notgood_videos),
            "config": {
                "norm_mode":         BRIGHTNESS_NORM_MODE,
                "adaptive_thr":      ADAPTIVE_THR_ENABLED,
                "brightness_baseline": BRIGHTNESS_BASELINE,
                "p3_conf_thr":       P3_CONF_THR,
                "p3_conf_thr_min":   P3_CONF_THR_MIN,
                "p3_min_consecutive": P3_MIN_CONSECUTIVE,
                "p2_flag_enabled":   P2_FLAG_ENABLED,
                "p1_fallback":       P1_FALLBACK_ENABLED,
                "batch_size":        BATCH_SIZE,
            },
            "results": results,
        }, f, indent=2)
    print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
