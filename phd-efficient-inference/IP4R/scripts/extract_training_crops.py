"""
Phase 1 -- Training Data Extractor
===================================
Extracts individual ROI crops from video(s) for training the digit-presence detector.

For each frame it:
  1. Detects the remote (green-box contour)
  2. Aligns the frame to the golden reference using ORB + ECC
  3. Crops each of the target ROIs to a standard 64x64 patch
  4. Measures coverage to auto-label: on / off / uncertain
  5. Saves organised under data/training/<roi_name>/on|off|uncertain/

Usage
-----
  Single video:
    python scripts/extract_training_crops.py video.mp4

  Folder of videos (processes all .mp4/.avi/.mov):
    python scripts/extract_training_crops.py "D:/Videos/GOOD"

  Limit to specific ROIs only:
    python scripts/extract_training_crops.py video.mp4 --rois timer_off temperature_tens

  Sample every N-th frame (default=3, i.e. every 3rd frame):
    python scripts/extract_training_crops.py video.mp4 --every 6

Options
-------
  --out          Output root directory (default: data/training)
  --rois         Space-separated ROI names to extract (default: all 7 digit/bar ROIs)
  --every        Sample every N-th frame (default: 3)
  --cov-on       Coverage >= this -> label "on"  (default: 0.10)
  --cov-off      Coverage <= this -> label "off" (default: 0.02)
  --patch-size   Output crop size in pixels (default: 64)
  --cal-t        Calibration frame time in seconds (default: 2.0)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# -- project root on sys.path ------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ip4r.config import Config
from ip4r.pipeline import Inspector
from ip4r.preprocess import preprocess
from ip4r.registration import register
from ip4r.inspect import _coverage


# ---------------------------------------------------------------------------
# Default ROIs targeted for training (the Stage-1 digit/bar elements)
# ---------------------------------------------------------------------------
DEFAULT_DIGIT_ROIS = [
    "timer_off",
    "clock",
    "timer_on",
    "temperature_tens",
    "temperature_units",
    "fan_speed_bars",
    "foot_display",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_proportional_rois(frame, inspector):
    """Map ROIs from golden to frame using proportional bbox scaling."""
    bbox = inspector._get_remote_bbox(frame)
    if not bbox:
        return None
        
    vx, vy, vw_box, vh_box = bbox
    
    gh, gw = inspector.golden_proc.shape[:2]
    gx, gy, gw_box, gh_box = 124, 5, 368, 475
    
    mapped_rois = {}
    for roi in inspector.rois:
        abs_x = roi.x * gw
        abs_y = roi.y * gh
        abs_w = roi.w * gw
        abs_h = roi.h * gh
        
        frac_x = (abs_x - gx) / gw_box
        frac_y = (abs_y - gy) / gh_box
        frac_w = abs_w / gw_box
        frac_h = abs_h / gh_box
        
        final_x = int(vx + frac_x * vw_box)
        final_y = int(vy + frac_y * vh_box)
        final_w = int(frac_w * vw_box)
        final_h = int(frac_h * vh_box)
        
        mapped_rois[roi.name] = (final_x, final_y, final_w, final_h)
        
    return mapped_rois

def _crop_roi(frame_proc, mapped_rois, roi_name, patch_size):
    """Crop ROI from the preprocessed frame."""
    if roi_name not in mapped_rois:
        return None
        
    x, y, w, h = mapped_rois[roi_name]
    crop = frame_proc[y:y+h, x:x+w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_AREA)


def _auto_label(patch, cov_on, cov_off):
    """
    Auto-label a crop based on adaptive-threshold coverage.
    Returns 'on', 'off', or 'uncertain'.
    """
    cov = _coverage(patch, block=15, c=8, active_is_dark=True)
    if cov >= cov_on:
        return "on"
    elif cov <= cov_off:
        return "off"
    return "uncertain"


# ---------------------------------------------------------------------------
# Per-video extractor
# ---------------------------------------------------------------------------

def extract_from_video(video_path, inspector, cfg, target_rois,
                       out_root, every, cov_on, cov_off, patch_size):
    """
    Extract crops from a single video.
    Returns counts dict: {roi_name: {on: N, off: N, uncertain: N}}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name}")
        return {}

    fps   = cap.get(cv2.CAP_PROP_FPS) or 12.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem  = video_path.stem

    print(f"  {video_path.name}  ({total} frames @ {fps:.1f}fps)", flush=True)

    # Build output dirs
    for roi in target_rois:
        for lbl in ("on", "off", "uncertain"):
            (out_root / roi.name / lbl).mkdir(parents=True, exist_ok=True)

    counts = {r.name: {"on": 0, "off": 0, "uncertain": 0} for r in target_rois}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fi = 0
    align_failures = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fi % every != 0:
            fi += 1
            continue

        # Align via proportional mapping
        mapped_rois = get_proportional_rois(frame, inspector)
        if mapped_rois is None:
            align_failures += 1
            fi += 1
            continue

        t_sec = fi / fps
        proc = preprocess(frame, cfg)

        # Crop each ROI
        for roi in target_rois:
            patch = _crop_roi(proc, mapped_rois, roi.name, patch_size)
            if patch is None:
                continue
            label = _auto_label(patch, cov_on, cov_off)
            fname = f"{stem}_f{fi:05d}_t{t_sec:.1f}s.jpg"
            cv2.imwrite(str(out_root / roi.name / label / fname), patch)
            counts[roi.name][label] += 1

        fi += 1

    cap.release()

    if align_failures:
        print(f"    {align_failures} frames skipped (alignment failed)")

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract per-ROI training crops from video(s)."
    )
    ap.add_argument("input",
                    help="Video file OR folder of videos")
    ap.add_argument("--out",        default="data/training",
                    help="Output root dir (default: data/training)")
    ap.add_argument("--rois",       nargs="+", default=None,
                    help="ROI names to extract. Default: all 7 digit/bar ROIs")
    ap.add_argument("--every",      type=int,   default=3,
                    help="Sample every N-th frame (default: 3)")
    ap.add_argument("--cov-on",     type=float, default=0.10,
                    help="Coverage >= this => label 'on' (default: 0.10)")
    ap.add_argument("--cov-off",    type=float, default=0.02,
                    help="Coverage <= this => label 'off' (default: 0.02)")
    ap.add_argument("--patch-size", type=int,   default=64,
                    help="Crop size in pixels (default: 64)")
    args = ap.parse_args()

    cfg       = Config.load(str(ROOT / "config" / "default.yaml"))
    inspector = Inspector(cfg)

    # Resolve target ROIs
    roi_map = {r.name: r for r in inspector.rois}
    if args.rois:
        bad = [n for n in args.rois if n not in roi_map]
        if bad:
            print(f"[ERROR] Unknown ROI names: {bad}")
            print(f"        Available: {sorted(roi_map.keys())}")
            sys.exit(1)
        target_rois = [roi_map[n] for n in args.rois]
    else:
        target_rois = [roi_map[n] for n in DEFAULT_DIGIT_ROIS if n in roi_map]

    print(f"\nTarget ROIs ({len(target_rois)}): {[r.name for r in target_rois]}")
    print(f"Thresholds: on >= {args.cov_on:.2f}  |  off <= {args.cov_off:.2f}  |  else: uncertain")
    print(f"Patch size: {args.patch_size}x{args.patch_size}px   Every: {args.every} frame(s)")

    # Collect video files
    inp = Path(args.input)
    if inp.is_file():
        videos = [inp]
    elif inp.is_dir():
        videos = sorted(p for p in inp.rglob("*")
                        if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
        if not videos:
            print(f"[ERROR] No video files found in {inp}")
            sys.exit(1)
        print(f"Found {len(videos)} video(s)")
    else:
        print(f"[ERROR] Not a file or directory: {args.input}")
        sys.exit(1)

    out_root = ROOT / args.out
    total_counts = {r.name: {"on": 0, "off": 0, "uncertain": 0} for r in target_rois}

    for i, vp in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}]", end=" ")
        counts = extract_from_video(
            video_path  = vp,
            inspector   = inspector,
            cfg         = cfg,
            target_rois = target_rois,
            out_root    = out_root,
            every       = args.every,
            cov_on      = args.cov_on,
            cov_off     = args.cov_off,
            patch_size  = args.patch_size,
        )
        for rn, c in counts.items():
            for lbl, n in c.items():
                total_counts[rn][lbl] += n

    # Summary table
    print("\n" + "=" * 68)
    print(f"  DONE  --  output: {out_root}")
    print("=" * 68)
    print(f"  {'ROI':<26} {'ON':>6} {'OFF':>6} {'UNCERTAIN':>10}  NOTE")
    print("  " + "-" * 60)
    for rn, c in total_counts.items():
        note = ""
        if c["on"] < 100:
            note = "<-- need more ON samples!"
        elif c["off"] < 100:
            note = "<-- need more OFF samples!"
        print(f"  {rn:<26} {c['on']:>6} {c['off']:>6} {c['uncertain']:>10}  {note}")
    print("=" * 68)
    print()
    print("NEXT STEPS:")
    print(f"  1. Open  {out_root / '<roi_name>' / 'uncertain'}  and manually")
    print(f"     move each crop to 'on' or 'off' as appropriate.")
    print(f"  2. When labels look good, run:")
    print(f"       python scripts/train_digit_detector.py --data {out_root}")


if __name__ == "__main__":
    main()
