"""FQCT inference worker — EfficientNet-B0 Phase-2-gated DTS video inspector.

Pipeline per video:
  1. Sample frames at frame_step interval
  2. detect_lcd() — perspective-correct the LCD panel from each frame
  3. phase2_score() — gate: only frames where icon strip is blank AND
     digit zones are filled qualify as Phase 2 (all-segments-on state)
  4. predict_crop() — EfficientNet-B0 scores each Phase 2 frame (prob PASS)
  5. Verdict: median P2 prob >= prob_threshold → PASS, else FAIL
  6. Save best-frame overlay JPEG to job_dir
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from .lcd_crop import detect_lcd

# ── ROI definitions (coords on 480×640 LCD crop) ─────────────────────────────
_SEG_ROIS = [
    ("left_clock",    140, 205,  70, 195, 0.38),
    ("right_clock",   140, 205, 235, 380, 0.25),
    ("center_88",     215, 305,  70, 190, 0.60),
    ("signal_bars",   250, 305, 310, 390, 0.35),
    ("bottom_88888",  405, 455, 175, 395, 0.55),
]
_ICON_Y1, _ICON_Y2, _ICON_X1, _ICON_X2 = 90, 140, 85, 400

_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model loader (called once at startup) ────────────────────────────────────
def load_model(model_path: str | Path) -> tuple:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 1)
    ckpt = torch.load(str(model_path), map_location=device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval().to(device)
    print(f"[FQCT] EfficientNet-B0 loaded from {model_path} on {device}")
    return m, device


# ── Per-frame helpers ─────────────────────────────────────────────────────────
def _phase2_score(gray: np.ndarray, dark_thresh: int, digit_score_min: float
                  ) -> tuple[bool, float, float]:
    icon_cov    = float((gray[_ICON_Y1:_ICON_Y2, _ICON_X1:_ICON_X2] < dark_thresh).mean())
    total_dark  = float((gray < dark_thresh).mean())
    covs        = [float((gray[y1:y2, x1:x2] < dark_thresh).mean())
                   for (_, y1, y2, x1, x2, _) in _SEG_ROIS]
    digit_score = float(np.mean(covs))
    is_p2       = (digit_score > digit_score_min) and (total_dark > 0.15)
    return is_p2, digit_score, icon_cov


def _predict(model, device, crop_bgr: np.ndarray) -> float:
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    x   = _TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x).squeeze().item()
    return float(torch.sigmoid(torch.tensor(logit)).item())


def _analyse_rois(crop_bgr: np.ndarray, dark_thresh: int) -> list[dict]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    results = []
    for (name, y1, y2, x1, x2, min_cov) in _SEG_ROIS:
        cov = float((gray[y1:y2, x1:x2] < dark_thresh).mean())
        results.append({"name": name, "coverage": round(cov, 3),
                        "passed": cov >= min_cov, "min_cov": min_cov})
    icon_cov = float((gray[_ICON_Y1:_ICON_Y2, _ICON_X1:_ICON_X2] < dark_thresh).mean())
    results.append({"name": "icon_strip", "coverage": round(icon_cov, 3),
                    "passed": True, "min_cov": 0.0})
    return results


def _draw_overlay(crop_bgr: np.ndarray, verdict: str, prob: float,
                  roi_results: list[dict], p2_total: int,
                  p2_pass: int, p2_fail: int) -> np.ndarray:
    img = crop_bgr.copy()
    H, W = img.shape[:2]
    GREEN, RED, WHITE, BLACK = (40, 200, 40), (30, 30, 220), (255, 255, 255), (0, 0, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for r in roi_results:
        if r["name"] == "icon_strip":
            continue
        x1, y1, x2, y2 = (_SEG_ROIS[[s[0] for s in _SEG_ROIS].index(r["name"])][3],
                           _SEG_ROIS[[s[0] for s in _SEG_ROIS].index(r["name"])][1],
                           _SEG_ROIS[[s[0] for s in _SEG_ROIS].index(r["name"])][4],
                           _SEG_ROIS[[s[0] for s in _SEG_ROIS].index(r["name"])][2])
        color = GREEN if r["passed"] else RED
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 if r["passed"] else 2)
        cv2.putText(img, f"{r['name']}={r['coverage']:.2f}",
                    (x1 + 2, max(y1 + 12, 14)), FONT, 0.24, color, 1, cv2.LINE_AA)

    v_color = GREEN if verdict == "PASS" else RED
    cv2.rectangle(img, (0, 0), (W, 36), BLACK, -1)
    cv2.putText(img, f"DTS v2: {verdict}", (6, 25), FONT, 0.72, v_color, 2, cv2.LINE_AA)
    cv2.putText(img, f"conf={100*prob:.1f}%  P2={p2_total}fr({p2_pass}ok/{p2_fail}fail)",
                (6, 34), FONT, 0.28, WHITE, 1, cv2.LINE_AA)

    failed = [r for r in roi_results if not r["passed"] and r["name"] != "icon_strip"]
    if failed:
        ph = 16 + 14 * len(failed)
        cv2.rectangle(img, (0, H - ph), (W, H), (20, 20, 20), -1)
        cv2.putText(img, f"Failed segments ({len(failed)}):", (6, H - ph + 12),
                    FONT, 0.32, RED, 1, cv2.LINE_AA)
        for i, r in enumerate(failed):
            cv2.putText(img, f"  {r['name']}  cov={r['coverage']:.3f} < {r['min_cov']:.2f}",
                        (6, H - ph + 14 + 14 * (i + 1)), FONT, 0.30, RED, 1, cv2.LINE_AA)
    return img


# ── Main entry point called by the server worker thread ──────────────────────
def process_video(
    job_id: str,
    video_path: str,
    job_dir: Path,
    model,
    device,
    cfg: dict,
) -> dict:
    """Run Phase-2-gated EfficientNet-B0 inference on a DTS video.

    Returns a result dict suitable for storage in the job store.
    Raises RuntimeError if no Phase 2 frame is found.
    """
    t0 = time.perf_counter()

    frame_step      = int(cfg.get("video", {}).get("frame_step", 5))
    dark_thresh     = int(cfg.get("video", {}).get("dark_thresh", 110))
    digit_score_min = float(cfg.get("phase2", {}).get("digit_score_min", 0.35))
    prob_threshold  = float(cfg.get("model", {}).get("prob_threshold", 0.5))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 12.0
    frame_indices = list(range(0, total_frames, frame_step))

    p2_frames: list[tuple[float, float, np.ndarray]] = []  # (digit_score, prob, crop)
    total_cropped = 0

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        crop, _ = detect_lcd(frame)
        if crop is None:
            continue
        total_cropped += 1

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        is_p2, d_score, _ = _phase2_score(gray, dark_thresh, digit_score_min)
        if is_p2:
            prob = _predict(model, device, crop)
            p2_frames.append((d_score, prob, crop))

    cap.release()

    if not p2_frames:
        raise RuntimeError("no_phase2_frame_detected")

    # Sort by digit score descending; best frame = clearest Phase 2
    p2_frames.sort(key=lambda x: x[0], reverse=True)
    best_d_score, best_prob, best_crop = p2_frames[0]

    p2_probs = [p for (_, p, _) in p2_frames]
    p2_pass  = sum(1 for p in p2_probs if p >= prob_threshold)
    p2_fail  = len(p2_probs) - p2_pass
    med_prob = float(np.median(p2_probs))
    passed   = med_prob >= prob_threshold
    verdict  = "PASS" if passed else "FAIL"

    t_ms = (time.perf_counter() - t0) * 1000

    # Save artefacts
    job_dir.mkdir(parents=True, exist_ok=True)
    roi_results = _analyse_rois(best_crop, dark_thresh)
    overlay     = _draw_overlay(best_crop, verdict, med_prob, roi_results,
                                len(p2_frames), p2_pass, p2_fail)
    cv2.imwrite(str(job_dir / "best_p2_frame.jpg"), best_crop)
    cv2.imwrite(str(job_dir / "overlay.jpg"), overlay)

    return {
        "job_id":          job_id,
        "passed":          passed,
        "verdict":         verdict,
        "median_p2_prob":  round(med_prob, 4),
        "best_frame_prob": round(best_prob, 4),
        "p2_frames":       len(p2_frames),
        "p2_pass":         p2_pass,
        "p2_fail":         p2_fail,
        "total_cropped":   total_cropped,
        "prob_threshold":  prob_threshold,
        "roi_results":     roi_results,
        "inference_ms":    round(t_ms, 1),
    }
