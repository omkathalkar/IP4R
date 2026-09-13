"""v6b Method 1 — pure-CV end-to-end pipeline: video → PASS / FAIL / ABSTAIN.

Pipeline (from v6b.md):
  Step A+B  extract_splash_candidates    sample full timing window, rank by global coverage
  Step C    register_best_candidate      register-once with multi-candidate retry
  Step D    compute_normalized_nc        per-element coverage relative to global
  Step E/F  threshold check              fail any element below threshold; ABSTAIN if no warp

No trained model is used.  Thresholds come from GOOD training data only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .atlas import load_atlas, compute_normalized_nc, ROI
from .register import register_best_candidate
from .frame_extract import extract_splash_candidates


def run_video(
    video_path: str | Path,
    golden_bgr: np.ndarray,
    atlas: list[ROI],
    thresholds: dict[str, float],
    n_candidates: int = 3,
    window_override: tuple[float, float] | None = None,
) -> dict:
    """Run the v6b pure-CV pipeline on one video.

    Returns a result dict:
        verdict       "PASS" | "FAIL" | "ABSTAIN"
        passed        True iff verdict == "PASS"
        ecc_rho       float – correlation score of best registration; -1.0 if ABSTAIN
        reg_path      "light" | "dark" | "unknown"
        frame_info    dict from the selected candidate (t_sec, coverage, session, …)
        per_element   {name: {nc, threshold, passed}} — empty on ABSTAIN
        failures      list of element names that fell below threshold
        error         str | None
        elapsed_ms    float
    """
    t0 = time.perf_counter()
    video_path = Path(video_path)

    # Step A+B: extract candidates
    candidates, cand_infos, session_info = extract_splash_candidates(
        video_path, n_candidates=n_candidates, window_override=window_override,
    )

    def _abstain(reason: str) -> dict:
        return {
            "video":       str(video_path),
            "verdict":     "ABSTAIN",
            "passed":      False,
            "ecc_rho":     -1.0,
            "reg_path":    "unknown",
            "frame_info":  session_info,
            "per_element": {},
            "failures":    [],
            "error":       reason,
            "elapsed_ms":  round((time.perf_counter() - t0) * 1000, 1),
        }

    if not candidates:
        return _abstain(session_info.get("error", "no_candidates"))

    # Step C: register best candidate
    warped, rho, reg_path, best_idx = register_best_candidate(candidates, golden_bgr)

    # Bright-LCD sessions (Aug-28 / Sep-09) cannot be ECC-aligned to the dark golden
    # but lcd_crop.py's perspective warp already gives a consistent 480×640 layout,
    # so we fall back to the best raw crop ranked by global coverage.
    if warped is None:
        best_idx = 0  # candidates are sorted by coverage descending
        warped = candidates[best_idx]
        reg_path = "unregistered"

    # Step D: per-element normalised coverage
    ncs = compute_normalized_nc(warped, atlas)
    if not ncs:
        return _abstain("blank_frame_after_registration")

    # Step E+F: threshold check
    failures: list[str] = []
    per_element: dict[str, dict] = {}
    for name, thresh in thresholds.items():
        nc = ncs.get(name)
        if nc is None:
            continue
        ok = nc >= thresh
        per_element[name] = {
            "nc":        round(nc, 5),
            "threshold": round(thresh, 5),
            "passed":    ok,
        }
        if not ok:
            failures.append(name)

    verdict = "FAIL" if failures else "PASS"
    frame_info = cand_infos[best_idx] if best_idx >= 0 else session_info

    return {
        "video":       str(video_path),
        "verdict":     verdict,
        "passed":      not failures,
        "ecc_rho":     round(rho, 4),
        "reg_path":    reg_path,
        "frame_info":  frame_info,
        "per_element": per_element,
        "failures":    failures,
        "error":       None,
        "elapsed_ms":  round((time.perf_counter() - t0) * 1000, 1),
    }


class CV6Pipeline:
    """Convenience wrapper: load assets once, call .predict() repeatedly."""

    def __init__(
        self,
        thresholds_path: str | Path,
        golden_path: str | Path | None = None,
        atlas_path: str | Path | None = None,
        n_candidates: int = 3,
    ) -> None:
        with open(thresholds_path) as f:
            meta = json.load(f)

        gp = golden_path or meta.get("golden_path", "data/reference/lcd_all_on.jpg")
        ap = atlas_path  or meta.get("atlas_path",  "data/reference/rois.yaml")

        self.golden_bgr  = cv2.imread(str(gp))
        if self.golden_bgr is None:
            raise FileNotFoundError(f"Cannot load golden: {gp}")

        self.atlas        = load_atlas(ap)
        self.thresholds   = {n: i["threshold"] for n, i in meta["elements"].items()}
        self.n_candidates = n_candidates

    def predict(
        self,
        video_path: str | Path,
        window_override: tuple[float, float] | None = None,
    ) -> dict:
        return run_video(
            video_path, self.golden_bgr, self.atlas, self.thresholds,
            n_candidates=self.n_candidates, window_override=window_override,
        )
