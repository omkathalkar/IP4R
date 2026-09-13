"""Compute per-element thresholds from GOOD training videos.

For each element e across all successfully registered GOOD training videos:
  threshold(e) = mean(normalized_nc(e)) − std_mult × std(normalized_nc(e))
  p01(e)       = 1st percentile of normalized_nc(e)   (alternative for skewed distributions)

Thresholds are derived ONLY from the GOOD training splits (Jun-27 + Aug-28 good/).
Do NOT call this on the locked test sets (Jul-14, Sep-09).

Usage:
    python -m server_v6a.threshold \\
        --good-jun27 '/path/to/June-27-2026/good/' \\
        --good-aug28 '/path/to/August-28-2026/good/' \\
        --golden     data/reference/lcd_all_on.jpg \\
        --atlas      data/reference/rois.yaml \\
        --out        data/v6a_results/thresholds.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from .atlas import load_atlas, compute_normalized_nc
from .register import register_best_candidate
from .frame_extract import extract_splash_candidates

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _collect_good_nc(
    good_dirs: list[str | Path],
    golden_bgr: np.ndarray,
    atlas_path: str | Path,
    n_candidates: int = 3,
) -> dict[str, list[float]]:
    """Process every GOOD video in good_dirs; return {element_name: [nc_values]}."""
    atlas = load_atlas(atlas_path)
    element_ncs: dict[str, list[float]] = {roi.name: [] for roi in atlas}

    for d in good_dirs:
        d = Path(d)
        videos = sorted(p for p in d.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
        if not videos:
            log.warning("No videos in %s", d)
            continue
        log.info("Processing %d videos from %s …", len(videos), d)

        for vid in videos:
            candidates, cand_infos, session_info = extract_splash_candidates(
                vid, n_candidates=n_candidates
            )
            if not candidates:
                log.warning("  SKIP %s — no splash candidates (%s)",
                            vid.stem, session_info.get("error", "unknown"))
                continue

            warped, rho, path, best_idx = register_best_candidate(candidates, golden_bgr)
            if warped is None:
                # Bright-LCD sessions can't be ECC-aligned to the dark golden;
                # use the best raw crop (sorted by coverage) as fallback.
                best_idx = 0
                warped = candidates[0]
                path = "unregistered"
                log.info("  FALLBACK %s — using raw crop (best_rho=%.3f)", vid.stem, rho)

            fi = cand_infos[best_idx] if best_idx >= 0 else {}
            ncs = compute_normalized_nc(warped, atlas)
            if not ncs:
                log.warning("  SKIP %s — blank frame after registration", vid.stem)
                continue

            for name, nc in ncs.items():
                element_ncs[name].append(nc)

            log.debug("  OK %s  rho=%.3f  t=%.1fs  path=%s",
                      vid.stem, rho, fi.get("t_sec", -1), path)

    return element_ncs


def compute_thresholds(
    good_dirs: list[str | Path],
    golden_path: str | Path,
    atlas_path: str | Path,
    n_candidates: int = 3,
    std_mult: float = 3.0,
) -> dict:
    """Compute per-element thresholds from GOOD training videos.

    Returns the full thresholds dict (also suitable for JSON serialisation).
    """
    golden_bgr = cv2.imread(str(golden_path))
    if golden_bgr is None:
        raise FileNotFoundError(f"Cannot load golden image: {golden_path}")

    log.info("Golden: %s  (%dx%d)", golden_path, golden_bgr.shape[1], golden_bgr.shape[0])
    log.info("Atlas:  %s", atlas_path)

    element_ncs = _collect_good_nc(good_dirs, golden_bgr, atlas_path, n_candidates)

    elements: dict[str, dict] = {}
    for name, values in element_ncs.items():
        if not values:
            log.warning("Element '%s' has no data — skipping", name)
            continue
        arr   = np.array(values, dtype=np.float64)
        mean  = float(arr.mean())
        std   = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        p01   = float(np.percentile(arr, 1))
        # Use p01 (1st percentile of GOOD training) as threshold:
        # any test observation below the 1st-percentile of GOOD is anomalously low.
        # This is more robust than mean−k·σ for right-skewed / high-variance elements.
        thr   = p01
        elements[name] = {
            "mean":      round(mean, 5),
            "std":       round(std, 5),
            "p01":       round(p01, 5),
            "threshold": round(thr, 5),
            "n":         len(values),
        }
        log.info("  %-28s  mean=%.3f  std=%.3f  thr=%.3f  p01=%.3f  n=%d",
                 name, mean, std, thr, p01, len(values))

    result = {
        "golden_path": str(golden_path),
        "atlas_path":  str(atlas_path),
        "std_mult":    std_mult,
        "n_candidates": n_candidates,
        "good_dirs":   [str(d) for d in good_dirs],
        "elements":    elements,
    }
    return result


def load_thresholds(path: str | Path) -> dict[str, float]:
    """Load thresholds JSON and return {element_name: threshold_value}."""
    with open(path) as f:
        data = json.load(f)
    return {name: info["threshold"] for name, info in data["elements"].items()}


def main() -> None:
    p = argparse.ArgumentParser(description="IP4R v6b — compute per-element thresholds")
    p.add_argument("--good-jun27",   default=None,
                   help="Directory of GOOD 12-FPS Jun-27-2026 videos")
    p.add_argument("--good-aug28",   default=None,
                   help="Directory of GOOD 12-FPS Aug-28-2026 videos")
    p.add_argument("--golden",       default="data/reference/lcd_all_on.jpg",
                   help="Golden reference LCD crop (480×640 JPG)")
    p.add_argument("--atlas",        default="data/reference/rois.yaml",
                   help="ROI atlas YAML file")
    p.add_argument("--out",          default="data/v6a_results/thresholds.json",
                   help="Output JSON path")
    p.add_argument("--n-candidates", type=int, default=3,
                   help="Number of splash candidates per video")
    p.add_argument("--std-mult",     type=float, default=3.0,
                   help="Number of standard deviations below mean for threshold")
    args = p.parse_args()

    good_dirs = [d for d in [args.good_jun27, args.good_aug28] if d]
    if not good_dirs:
        p.error("At least one of --good-jun27 or --good-aug28 must be specified.")

    result = compute_thresholds(
        good_dirs=good_dirs,
        golden_path=args.golden,
        atlas_path=args.atlas,
        n_candidates=args.n_candidates,
        std_mult=args.std_mult,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Thresholds saved to %s (%d elements)", out, len(result["elements"]))


if __name__ == "__main__":
    main()
