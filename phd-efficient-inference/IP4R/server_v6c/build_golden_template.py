"""Build the golden template for Phase 2 content checks.

Samples N confirmed-GOOD videos, runs detect_lcd() on the T* frame of each,
and produces:
  1. Binary Otsu masks (backward-compat with phase2_template.py IoU check)
  2. Grayscale median ROI images for SSIM-based check (phase2_ssim.py)
  3. Per-ROI SSIM thresholds calibrated at the 1st-percentile of GOOD scores

Usage:
    python -m server_v6c.build_golden_template \\
        --dataset '/home/om/src/FDU Dataset/Sep-15-2026/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --n-samples 14 \\
        --out models/golden_template.npz

Output (.npz keys):
    roi_<name>      : uint8 binary Otsu mask  (H×W)
    gray_<name>     : uint8 grayscale median  (H×W)  ← SSIM golden
    ssim_thr_<name> : float32 1-D array [threshold]  ← 1st-pct of GOOD SSIM
    ssim_dist_<name>: float32 1-D array [scores…]    ← full calibration dist
    meta_n          : number of samples used
    meta_domain     : dataset name tag
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

ROI_ATLAS: list[tuple] = [
    ("top_icons",         0,  90,  85, 400),
    ("icon_strip",       90, 140,  85, 400),
    ("left_clock",      140, 205,  70, 195),
    ("right_clock",     140, 205, 235, 380),
    ("center_88",       215, 305,  70, 190),
    ("signal_bars",     250, 305, 310, 390),
    ("secondary_icons", 305, 405,  45, 435),
    ("bottom_88888",    405, 455, 175, 395),
]

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
SSIM_WIN   = 7
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


# ---------------------------------------------------------------------------
# SSIM (same implementation as phase2_ssim.py — no external deps)
# ---------------------------------------------------------------------------

def _ssim(a: np.ndarray, b: np.ndarray, win: int = SSIM_WIN) -> float:
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    k = np.ones((win, win), np.float32) / (win * win)

    def filt(x: np.ndarray) -> np.ndarray:
        return cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

    mu1, mu2 = filt(a), filt(b)
    sig1  = filt(a * a) - mu1 * mu1
    sig2  = filt(b * b) - mu2 * mu2
    sig12 = filt(a * b) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1 + sig2 + C2))
    p = win // 2
    return float(ssim_map[p:-p, p:-p].mean())


def _roi_ssim_vs_golden(gray_crop: np.ndarray, golden_gray: np.ndarray) -> float:
    live_c = _clahe.apply(gray_crop)
    gold_c = _clahe.apply(golden_gray)
    if live_c.shape != gold_c.shape:
        gold_c = cv2.resize(gold_c, (live_c.shape[1], live_c.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    win = min(SSIM_WIN, live_c.shape[0] - 2, live_c.shape[1] - 2)
    win = max(3, win if win % 2 == 1 else win - 1)
    return _ssim(live_c, gold_c, win=win)


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def otsu_binarize(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def process_video(
    video_path: Path,
    yolo_model_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Return (binary_masks, gray_crops) for the T* frame, or None if Phase 1 fails."""
    from .yolo_phase1 import run_yolo_phase1

    try:
        from server_v6a.lcd_crop import detect_lcd
    except ImportError:
        from ..server_v6a.lcd_crop import detect_lcd

    p1 = run_yolo_phase1(video_path, yolo_model_path)
    if not p1.complete or p1.T_star is None:
        log.warning("  Phase1 FAIL on %s — skipping", video_path.name)
        return None

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, p1.T_star)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        log.warning("  Cannot read frame %d from %s", p1.T_star, video_path.name)
        return None

    crop, _ = detect_lcd(frame)
    if crop is None:
        log.warning("  detect_lcd() failed at T*=%d for %s", p1.T_star, video_path.name)
        return None

    binary: dict[str, np.ndarray] = {}
    gray:   dict[str, np.ndarray] = {}
    for name, y1, y2, x1, x2 in ROI_ATLAS:
        roi_bgr = crop[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            continue
        binary[name] = otsu_binarize(roi_bgr)
        gray[name]   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    log.info("  OK  %s  T*=%d (%.1fs)", video_path.name, p1.T_star, p1.T_star_sec)
    return binary, gray


# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------

def build_golden_template(
    dataset_dir:     Path,
    yolo_model_path: Path,
    n_samples:       int  = 14,
    out_path:        Path = Path("models/golden_template.npz"),
    domain:          str  = "Sep-15",
    ssim_pct:        float = 1.0,   # percentile of GOOD SSIM to use as threshold
) -> Path:
    videos = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        raise RuntimeError(f"No video files found in {dataset_dir}")

    # Leakage guard: Sep-15 is the locked eval set — calibration must use Jul-14+Sep-09 ONLY
    if any(s in str(dataset_dir) for s in ("Sep-15", "Sep15", "Sep_15")):
        log.warning("=" * 70)
        log.warning("LEAKAGE WARNING: dataset_dir='%s' looks like the Sep-15 eval set!", dataset_dir)
        log.warning("Building a template from Sep-15 GOOD and then evaluating on Sep-15 GOOD")
        log.warning("produces inflated, non-generalisable numbers (train/eval overlap).")
        log.warning("Correct calibration data: Jul-14 GOOD (5 vids) + Sep-09 GOOD (13 vids).")
        log.warning("=" * 70)
    if any(s in domain for s in ("Sep-15", "Sep15", "Sep_15")):
        log.warning("LEAKAGE WARNING: domain='%s' contains 'Sep-15' (locked eval set).", domain)

    log.info("Building golden template from %d/%d videos in %s",
             min(n_samples, len(videos)), len(videos), dataset_dir)

    # Accumulators
    binary_stacks: dict[str, list[np.ndarray]] = {name: [] for name, *_ in ROI_ATLAS}
    gray_stacks:   dict[str, list[np.ndarray]] = {name: [] for name, *_ in ROI_ATLAS}
    used = 0

    for vid in videos[:n_samples]:
        log.info("Processing %s …", vid.name)
        result = process_video(vid, yolo_model_path)
        if result is None:
            continue
        bin_masks, gray_crops = result
        for name in [n for n, *_ in ROI_ATLAS]:
            if name in bin_masks:
                binary_stacks[name].append(bin_masks[name])
            if name in gray_crops:
                gray_stacks[name].append(gray_crops[name])
        used += 1

    if used == 0:
        raise RuntimeError("No valid samples collected — cannot build template")

    log.info("Collected %d samples. Computing median masks …", used)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict = {}
    golden_gray: dict[str, np.ndarray] = {}

    # ── Binary Otsu (backward-compat) ─────────────────────────────────────
    for name, stack in binary_stacks.items():
        if not stack:
            log.warning("  ROI %s: no binary samples — skipping", name)
            continue
        arr = np.stack(stack, axis=0).astype(np.float32)
        median_mask = (np.median(arr, axis=0) >= 0.5).astype(np.uint8)
        save_dict[f"roi_{name}"] = median_mask
        log.info("  [binary] ROI %-20s  shape=%s  active=%.3f",
                 name, median_mask.shape, median_mask.mean())

    # ── Grayscale median (for SSIM) ────────────────────────────────────────
    for name, stack in gray_stacks.items():
        if not stack:
            log.warning("  ROI %s: no grayscale samples — skipping", name)
            continue
        arr = np.stack(stack, axis=0).astype(np.float32)
        median_gray = np.median(arr, axis=0).astype(np.uint8)
        save_dict[f"gray_{name}"] = median_gray
        golden_gray[name] = median_gray

    # ── SSIM calibration: compute per-ROI SSIM of each training video vs golden ──
    log.info("Calibrating per-ROI SSIM thresholds (1st-pct of %d GOOD samples)…", used)
    ssim_scores_per_roi: dict[str, list[float]] = {name: [] for name, *_ in ROI_ATLAS}

    for name, stack in gray_stacks.items():
        golden = golden_gray.get(name)
        if golden is None or not stack:
            continue
        for crop_gray in stack:
            score = _roi_ssim_vs_golden(crop_gray, golden)
            ssim_scores_per_roi[name].append(score)

    sep = "-" * 60
    log.info(sep)
    log.info("  %-20s  %6s  %6s  %6s  %6s  n", "ROI", "thr", "min", "mean", "max")
    log.info(sep)
    for name, *_ in ROI_ATLAS:
        scores = ssim_scores_per_roi[name]
        if not scores:
            continue
        thr  = float(np.percentile(scores, ssim_pct))
        save_dict[f"ssim_thr_{name}"]  = np.array([thr],    dtype=np.float32)
        save_dict[f"ssim_dist_{name}"] = np.array(scores,   dtype=np.float32)
        log.info("  %-20s  %.4f  %.4f  %.4f  %.4f  %d",
                 name, thr, min(scores), float(np.mean(scores)), max(scores), len(scores))
    log.info(sep)

    save_dict["meta_n"]      = np.array([used])
    save_dict["meta_domain"] = np.array([domain], dtype=object)

    np.savez(str(out_path), **save_dict)
    log.info("Golden template saved → %s  (%d samples, %d keys)",
             out_path, used, len(save_dict) - 2)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build Phase 2 golden template (binary + SSIM)")
    ap.add_argument("--dataset",     required=True, help="Folder of GOOD MP4 videos")
    ap.add_argument("--yolo-model",  required=True, help="Path to YOLO best.pt")
    ap.add_argument("--n-samples",   type=int,   default=14)
    ap.add_argument("--out",         default="models/golden_template.npz")
    ap.add_argument("--domain",      default="Sep-15")
    ap.add_argument("--ssim-pct",    type=float, default=1.0,
                    help="Percentile of GOOD SSIM to use as threshold (default 1.0 = 1st-pct)")
    args = ap.parse_args()

    build_golden_template(
        dataset_dir     = Path(args.dataset),
        yolo_model_path = Path(args.yolo_model),
        n_samples       = args.n_samples,
        out_path        = Path(args.out),
        domain          = args.domain,
        ssim_pct        = args.ssim_pct,
    )


if __name__ == "__main__":
    main()
