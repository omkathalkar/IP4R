#!/usr/bin/env python3
"""
server_v7b_r2a eval — batch evaluation with PASS / ABSTAIN / FAIL verdicts.

Usage:
    python -m server_v7b_r2a.eval \\
        --good-dir    '/path/to/GOOD' \\
        --notgood-dir '/path/to/NOT GOOD' \\
        --p1-model    data/macro_dataset/runs/macro_test/weights/best.pt \\
        --elem-model  runs/phase2_elem_v1/weights/best.pt \\
        --out         data/v7_results/eval_v7b_r2a_Sep15.json

Verdict semantics
-----------------
PASS    — all required icons confirmed (consec≥2 met)
ABSTAIN — some icons not confirmed but all of them were glimpsed (≥1 frame above thr)
          → borderline unit; recommend human re-inspection
FAIL    — at least one required icon never seen above threshold
          → icon genuinely absent; confident defect

Scoring
-------
  GOOD  ground truth : PASS=✓  ABSTAIN=?  FAIL=✗ (false alarm)
  NOT_GOOD ground truth : FAIL=✓  ABSTAIN=?  PASS=✗ (miss)
  ABSTAIN always shown separately — not counted as right or wrong in pass/fail rates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from server_v7b_r2a.pipeline import (
    P1_SKIP_SEC, P1_END_SEC, P2_SEC, P3_START_SEC,
    P1_CONF_THR, P2_CONF_THR, P3_CONF_THR, P1_SAMPLE_STEP,
    P1_CLASSES, ELEMENT_NAMES, REQUIRED_ICONS,
    BRIGHTNESS_NORM_MODE, ADAPTIVE_THR_ENABLED, BRIGHTNESS_BASELINE,
    P3_CONF_THR_MIN, P3_MIN_CONSECUTIVE, P2_FLAG_ENABLED,
    P1_FALLBACK_ENABLED, BATCH_SIZE, ABSTAIN_ENABLED,
    _load, _normalize_frame, _adaptive_thr, _p1_fallback_boxes,
    _infer_batch, _apply_temporal_glimpse, _three_way_verdict,
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
        masked_norm = _normalize_frame(masked)
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

    p3_thr = _adaptive_thr(p3_raw_frames)
    p3_frame_dets = _infer_batch(m2, p3_norm_frames, p3_frame_idx, conf=p3_thr)
    p3_checklist, p3_glimpsed = _apply_temporal_glimpse(p3_frame_dets, p3_frame_idx, p3_thr)

    verdict, abstain_icons, fail_icons = _three_way_verdict(p3_checklist, p3_glimpsed)

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
        "p3_abstain_icons": abstain_icons,
        "p3_fail_icons":    fail_icons,
        "p3_checklist":     p3_checklist,
        "p3_glimpsed":      p3_glimpsed,
    }


def main():
    parser = argparse.ArgumentParser(description="server_v7b_r2a — PASS/ABSTAIN/FAIL eval")
    parser.add_argument("--good-dir",    required=True)
    parser.add_argument("--notgood-dir", default=None)
    parser.add_argument("--p1-model",
                        default="data/macro_dataset/runs/macro_test/weights/best.pt")
    parser.add_argument("--elem-model",
                        default="runs/phase2_elem_v1/weights/best.pt")
    parser.add_argument("--out",
                        default="data/v7_results/eval_v7b_r2a.json")
    args = parser.parse_args()

    good_videos    = sorted(p for p in Path(args.good_dir).iterdir()
                            if p.suffix.lower() in VIDEO_EXT)
    notgood_videos = sorted(p for p in Path(args.notgood_dir).iterdir()
                            if p.suffix.lower() in VIDEO_EXT) \
                     if args.notgood_dir else []

    W = 112
    print()
    print("─" * W)
    print(f"  server_v7b_r2a  —  consec≥{P3_MIN_CONSECUTIVE} + ABSTAIN (flicker-based)")
    print(f"  P1 model  : {Path(args.p1_model).name}")
    print(f"  Elem model: {Path(args.elem_model).name}")
    print(f"  norm={BRIGHTNESS_NORM_MODE}  adaptive={'ON' if ADAPTIVE_THR_ENABLED else 'OFF'}"
          f"  consec≥{P3_MIN_CONSECUTIVE}  abstain={'ON' if ABSTAIN_ENABLED else 'OFF'}"
          f"  batch={BATCH_SIZE}")
    print(f"  P3 thr={P3_CONF_THR:.0%}  required={len(REQUIRED_ICONS)} icons")
    print("─" * W)
    fmt = "  {:<46} {:>9}  {:>4}  {:>5}  {:>7}  {:>5}  {:>7}"
    print(fmt.format("Video", "GT", "P1", "P2flg", "P3", "thr", "Verdict"))
    print("─" * W)

    results = []
    good_pass = good_abstain = good_fail = 0
    ng_fail   = ng_abstain   = ng_pass   = 0

    for vp in good_videos:
        r = run_one(str(vp), args.p1_model, args.elem_model)
        r["gt"] = "GOOD"
        v = r["verdict"]
        if v == "PASS":    good_pass    += 1
        elif v == "ABSTAIN": good_abstain += 1
        else:              good_fail    += 1
        results.append(r)
        p3_str = f"{len(r['p3_found'])}/{len(REQUIRED_ICONS)}"
        print(fmt.format(
            vp.name[:45], "GOOD",
            "✓" if r["p1_ok"] else "✗",
            "!" if r["p2_flag"] else "-",
            p3_str,
            f"{r['p3_adaptive_thr']:.2f}",
            v,
        ))
        if v == "ABSTAIN":
            print(f"    [ABSTAIN] glimpsed-not-consec: {r['p3_abstain_icons']}")
        elif v == "FAIL":
            print(f"    [FAIL]    never seen: {r['p3_fail_icons']}"
                  + (f"  glimpsed: {r['p3_abstain_icons']}" if r['p3_abstain_icons'] else ""))

    print("─" * W)

    for vp in notgood_videos:
        r = run_one(str(vp), args.p1_model, args.elem_model)
        r["gt"] = "NOT_GOOD"
        v = r["verdict"]
        if v == "FAIL":    ng_fail    += 1
        elif v == "ABSTAIN": ng_abstain += 1
        else:              ng_pass    += 1
        results.append(r)
        p3_str = f"{len(r['p3_found'])}/{len(REQUIRED_ICONS)}"
        print(fmt.format(
            vp.name[:45], "NOT_GOOD",
            "✓" if r["p1_ok"] else "✗",
            "!" if r["p2_flag"] else "-",
            p3_str,
            f"{r['p3_adaptive_thr']:.2f}",
            v,
        ))
        if v == "FAIL":
            print(f"    [CAUGHT]  never seen: {r['p3_fail_icons']}"
                  + (f"  glimpsed: {r['p3_abstain_icons']}" if r['p3_abstain_icons'] else ""))
        elif v == "ABSTAIN":
            print(f"    [ABSTAIN] glimpsed-not-consec: {r['p3_abstain_icons']}")

    print("─" * W)

    n_good = len(good_videos)
    n_ng   = len(notgood_videos)
    print()
    print(f"  GOOD ({n_good}):     PASS={good_pass}  ABSTAIN={good_abstain}  FAIL={good_fail}")
    print(f"  NOT_GOOD ({n_ng}):  FAIL={ng_fail}  ABSTAIN={ng_abstain}  PASS={ng_pass}")
    print()
    print(f"  GOOD pass rate    (excl ABSTAIN): "
          f"{good_pass}/{n_good - good_abstain}"
          f"  ({100*good_pass/max(1, n_good - good_abstain):.0f}%)"
          f"  [+{good_abstain} sent to review]")
    print(f"  NOT_GOOD catch    (excl ABSTAIN): "
          f"{ng_fail}/{n_ng - ng_abstain}"
          f"  ({100*ng_fail/max(1, n_ng - ng_abstain):.0f}%)"
          f"  [+{ng_abstain} sent to review]")
    print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "version":        "v7b_r2a",
            "good_pass":      good_pass,
            "good_abstain":   good_abstain,
            "good_fail":      good_fail,
            "good_total":     n_good,
            "ng_fail":        ng_fail,
            "ng_abstain":     ng_abstain,
            "ng_pass":        ng_pass,
            "ng_total":       n_ng,
            "config": {
                "norm_mode":          BRIGHTNESS_NORM_MODE,
                "adaptive_thr":       ADAPTIVE_THR_ENABLED,
                "p3_conf_thr":        P3_CONF_THR,
                "p3_conf_thr_min":    P3_CONF_THR_MIN,
                "p3_min_consecutive": P3_MIN_CONSECUTIVE,
                "abstain_enabled":    ABSTAIN_ENABLED,
                "p2_flag_enabled":    P2_FLAG_ENABLED,
                "p1_fallback":        P1_FALLBACK_ENABLED,
                "batch_size":         BATCH_SIZE,
            },
            "results": results,
        }, f, indent=2)
    print(f"  Saved → {args.out}")


if __name__ == "__main__":
    main()
