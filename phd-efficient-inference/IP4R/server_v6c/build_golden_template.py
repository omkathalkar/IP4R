"""Build the golden template for Phase 2 content check.

Samples N confirmed-GOOD videos, runs detect_lcd() on the T* frame of each,
Otsu-binarizes each ROI, and median-stacks across all samples.
The resulting per-ROI binary masks are the "ground truth" pattern that a
GOOD unit should produce — used by phase2_template.py for IoU comparison.

Usage:
    python -m server_v6c.build_golden_template \\
        --dataset '/home/om/src/FDU Dataset/Sep-15-2026/GOOD' \\
        --yolo-model data/macro_dataset/runs/macro_test/weights/best.pt \\
        --n-samples 12 \\
        --out models/golden_template.npz

Output (.npz keys):
    roi_<name>   : uint8 binary mask (H×W) for that ROI, median across samples
    meta_n       : number of samples used
    meta_domain  : dataset name tag
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

# ROI atlas — same coordinates as phase2_roi.py (480×640 detect_lcd() crop)
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


def otsu_binarize(crop_bgr: np.ndarray) -> np.ndarray:
    """Local Otsu threshold on the crop — robust to overall brightness shifts."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def process_video(video_path: Path, yolo_model_path: Path) -> dict[str, np.ndarray] | None:
    """Return per-ROI Otsu masks for the T* frame, or None if Phase 1 fails."""
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

    masks: dict[str, np.ndarray] = {}
    for name, y1, y2, x1, x2 in ROI_ATLAS:
        roi_crop = crop[y1:y2, x1:x2]
        if roi_crop.size == 0:
            continue
        masks[name] = otsu_binarize(roi_crop)

    log.info("  OK  %s  T*=%d (%.1fs)", video_path.name, p1.T_star, p1.T_star_sec)
    return masks


def build_golden_template(
    dataset_dir: Path,
    yolo_model_path: Path,
    n_samples: int = 12,
    out_path: Path = Path("models/golden_template.npz"),
    domain: str = "Sep-15",
) -> Path:
    videos = sorted(p for p in dataset_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        raise RuntimeError(f"No video files found in {dataset_dir}")

    log.info("Building golden template from %d/%d videos in %s",
             min(n_samples, len(videos)), len(videos), dataset_dir)

    roi_stacks: dict[str, list[np.ndarray]] = {name: [] for name, *_ in ROI_ATLAS}
    used = 0

    for vid in videos[:n_samples]:
        log.info("Processing %s …", vid.name)
        masks = process_video(vid, yolo_model_path)
        if masks is None:
            continue
        for name, stack in roi_stacks.items():
            if name in masks:
                stack.append(masks[name])
        used += 1

    if used == 0:
        raise RuntimeError("No valid samples collected — cannot build template")

    log.info("Collected %d samples. Computing median masks …", used)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, np.ndarray] = {}
    for name, stack in roi_stacks.items():
        if not stack:
            log.warning("  ROI %s: no samples — skipping", name)
            continue
        arr = np.stack(stack, axis=0).astype(np.float32)
        # Median across samples, then threshold at 0.5 (majority vote)
        median_mask = (np.median(arr, axis=0) >= 0.5).astype(np.uint8)
        save_dict[f"roi_{name}"] = median_mask
        log.info("  ROI %-20s  shape=%s  active_frac=%.3f",
                 name, median_mask.shape, median_mask.mean())

    save_dict["meta_n"]      = np.array([used])
    save_dict["meta_domain"] = np.array([domain], dtype=object)

    np.savez(str(out_path), **save_dict)
    log.info("Golden template saved → %s  (%d samples, %d ROIs)", out_path, used, len(save_dict) - 2)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Phase 2 golden template")
    ap.add_argument("--dataset",     required=True, help="Folder of GOOD MP4 videos")
    ap.add_argument("--yolo-model",  required=True, help="Path to YOLO best.pt")
    ap.add_argument("--n-samples",   type=int, default=12)
    ap.add_argument("--out",         default="models/golden_template.npz")
    ap.add_argument("--domain",      default="Sep-15")
    args = ap.parse_args()

    build_golden_template(
        dataset_dir    = Path(args.dataset),
        yolo_model_path= Path(args.yolo_model),
        n_samples      = args.n_samples,
        out_path       = Path(args.out),
        domain         = args.domain,
    )


if __name__ == "__main__":
    main()
