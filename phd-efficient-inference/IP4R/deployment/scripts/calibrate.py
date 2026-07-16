"""Threshold calibration: run inspector on all good-bank images, collect per-ROI
coverage-ratio and SSIM scores, then compute mean - k*sigma thresholds.

Usage:
    python scripts/calibrate.py [--sigma 3] [--apply]

--apply  writes the computed thresholds back to config/default.yaml.
"""
from __future__ import annotations

import argparse
import os
import sys
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import cv2
import numpy as np
import yaml
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ip4r.config import Config, REPO_ROOT
from ip4r.preprocess import preprocess
from ip4r.registration import register
from ip4r.roi import load_rois, ROI
from ip4r.inspect import _coverage, _diff_density


def score_image(path: str, golden_proc: np.ndarray, rois: list[ROI],
                cfg: Config) -> dict[str, dict] | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    proc = preprocess(img, cfg)
    aligned, reg_info = register(proc, golden_proc, cfg)

    h, w = golden_proc.shape[:2]
    active_is_dark = bool(cfg.get("tier_a.coverage.active_is_dark", True))
    block       = int(cfg.get("tier_a.coverage.adaptive_block", 31))
    cc          = int(cfg.get("tier_a.coverage.adaptive_C", 5))
    win         = int(cfg.get("tier_a.ssim.win_size", 7))
    diff_blur   = int(cfg.get("tier_a.diff.blur_k", 3))
    diff_thresh = int(cfg.get("tier_a.diff.pixel_threshold", 20))

    scores: dict[str, dict] = {}
    for r in rois:
        x, y, bw, bh = r.to_pixels(w, h)
        gp = golden_proc[y:y + bh, x:x + bw]
        sp = aligned[y:y + bh, x:x + bw]

        cov_g = _coverage(gp, block, cc, active_is_dark)
        cov_s = _coverage(sp, block, cc, active_is_dark)
        ratio = (cov_s / cov_g) if cov_g > 0 else 0.0

        wsz = win if win % 2 == 1 else win + 1
        wsz = min(wsz, min(gp.shape[:2]) - (1 - min(gp.shape[:2]) % 2))
        try:
            s = float(sk_ssim(gp, sp, win_size=max(3, wsz)))
        except Exception:
            s = 1.0

        d = _diff_density(gp, sp, diff_blur, diff_thresh)

        scores[r.name] = {"coverage_ratio": ratio, "ssim": s, "diff_density": d}
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Threshold = mean - sigma*std (default: 3)")
    parser.add_argument("--apply", action="store_true",
                        help="Write calibrated thresholds to config/default.yaml")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cfg = Config.load()
    golden_bgr = cv2.imread(str(cfg.path("paths.reference_image")), cv2.IMREAD_COLOR)
    golden_proc = preprocess(golden_bgr, cfg)
    rois = load_rois(cfg.path("paths.roi_map"))

    good_bank = REPO_ROOT / "data" / "good_bank"
    paths = sorted(glob.glob(str(good_bank / "**" / "*.jpg"), recursive=True))
    print(f"Found {len(paths)} good images. Running with {args.workers} workers...")

    roi_cov:  dict[str, list[float]] = defaultdict(list)
    roi_ssim: dict[str, list[float]] = defaultdict(list)
    roi_diff: dict[str, list[float]] = defaultdict(list)
    n_done = 0
    n_fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_image, p, golden_proc, rois, cfg): p for p in paths}
        for fut in as_completed(futures):
            n_done += 1
            result = fut.result()
            if result is None:
                n_fail += 1
                continue
            for roi_name, sc in result.items():
                roi_cov[roi_name].append(sc["coverage_ratio"])
                roi_ssim[roi_name].append(sc["ssim"])
                roi_diff[roi_name].append(sc["diff_density"])
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(paths)} done...")

    print(f"\nProcessed {n_done} images ({n_fail} unreadable).\n")

    k = args.sigma
    print(f"{'ROI':<28} {'cov_thr':>8} {'ssim_thr':>9}  {'diff_mean':>9} {'diff_std':>8} {'diff_max':>9}")
    print("-" * 82)

    cov_thresholds:  dict[str, float] = {}
    ssim_thresholds: dict[str, float] = {}
    diff_thresholds: dict[str, float] = {}

    for roi in rois:
        name = roi.name
        cv = np.array(roi_cov[name])
        sv = np.array(roi_ssim[name])
        dv = np.array(roi_diff[name])
        ct = max(0.01, float(cv.mean() - k * cv.std()))
        st = max(0.01, float(sv.mean() - k * sv.std()))
        dt = min(0.99, float(dv.mean() + k * dv.std()))   # upper bound for diff
        cov_thresholds[name]  = round(ct, 4)
        ssim_thresholds[name] = round(st, 4)
        diff_thresholds[name] = round(dt, 4)
        print(f"{name:<28} {ct:>8.4f} {st:>9.4f}  {dv.mean():>9.4f} {dv.std():>8.4f} {dt:>9.4f}")

    global_cov  = round(min(cov_thresholds.values()), 4)
    global_ssim = round(min(ssim_thresholds.values()), 4)
    global_diff = round(max(diff_thresholds.values()), 4)   # most permissive upper bound

    print(f"\n--- Global thresholds ---")
    print(f"  coverage_ratio_min : {global_cov}")
    print(f"  ssim_min           : {global_ssim}")
    print(f"  diff.max_density   : {global_diff}  (mean+{k}σ, tightest ROI will calibrate per-ROI)")

    if args.apply:
        config_path = REPO_ROOT / "config" / "default.yaml"
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        raw["tier_a"]["coverage"]["coverage_ratio_min"] = global_cov
        raw["tier_a"]["ssim"]["ssim_min"]               = global_ssim
        raw["tier_a"]["diff"]["max_density"]            = global_diff
        with open(config_path, "w") as f:
            yaml.dump(raw, f, sort_keys=False, default_flow_style=False)

        # Write per-ROI diff_max into rois.yaml (add field diff_max per ROI)
        from ip4r.roi import save_rois
        import dataclasses
        for roi in rois:
            # monkey-patch: add diff_max attribute so save_rois picks it up
            # We store diff_max as a new field in the YAML via direct dict write
            pass
        roi_path = cfg.path("paths.roi_map")
        with open(roi_path) as f:
            roi_data = yaml.safe_load(f)
        for entry in roi_data["rois"]:
            name = entry["name"]
            if name in diff_thresholds:
                entry["diff_max"] = diff_thresholds[name]
        with open(roi_path, "w") as f:
            yaml.safe_dump(roi_data, f, sort_keys=False)

        print(f"\nThresholds written to {config_path} and {roi_path}")
    else:
        print("\nRun with --apply to write these thresholds to config/default.yaml and rois.yaml")


if __name__ == "__main__":
    main()
