"""Video processing worker for FQCT server.

Adapted from scripts/inspect_video.py:
  - Strips annotated-MP4 writing from the inference path (saves ~2s)
  - Saves trigger frame as JPEG for the 30-day archive
  - Returns structured result dict
  - Writes annotated MP4 in a second pass (archive only, non-blocking on result)
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ip4r.config import Config
from ip4r.inspect import _coverage
from ip4r.pipeline import Inspector
from ip4r.preprocess import preprocess
from ip4r.registration import register
from ip4r.report import write_report


def _fast_splash_check(
    warped: np.ndarray,
    golden: np.ndarray,
    digit_rois,
    cov_ratio_min: float,
    block: int,
    cc: int,
    active_is_dark: bool,
) -> bool:
    gh, gw = golden.shape[:2]
    for r in digit_rois:
        x, y, bw, bh = r.to_pixels(gw, gh)
        gp = golden[y : y + bh, x : x + bw]
        sp = warped[y : y + bh, x : x + bw]
        if gp.size == 0 or sp.size == 0:
            continue
        ratio = r.cov_ratio_min if r.cov_ratio_min is not None else cov_ratio_min
        cov_g = _coverage(gp, block, cc, active_is_dark)
        cov_s = _coverage(sp, block, cc, active_is_dark)
        if cov_s < cov_g * ratio:
            return False
    return True


def process_video(
    job_id: str,
    video_path: str,
    job_dir: Path,
    cfg: Config,
    inspector: Inspector,
    start_skip_s: float = 5.0,
    stable_needed: int = 3,
    cal_t_s: float = 2.0,
) -> dict:
    """
    Run splash-detection + full IP4R inspection on a DTS video.
    Returns the InspectionResult serialised as a dict.
    Raises RuntimeError if splash is not found.
    """
    t0 = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 12.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    golden = inspector.golden_proc
    gh, gw = golden.shape[:2]

    # ── Phase 1: register one calibration frame for fast warp ────────────────
    H_cal: np.ndarray | None = None
    cal_fi = min(int(cal_t_s * fps), total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cal_fi)
    ret, cal_frame = cap.read()
    if ret:
        cal_proc = preprocess(cal_frame, cfg)
        _, cal_reg = register(cal_proc, golden, cfg)
        if cal_reg.get("homography"):
            H_cal = np.array(cal_reg["homography"], dtype=np.float64)

    # ── Phase 2: fast scan for splash (skip first start_skip_s) ─────────────
    cov_ratio_min  = float(cfg.get("tier_a.coverage.coverage_ratio_min", 0.55))
    active_is_dark = bool(cfg.get("tier_a.coverage.active_is_dark", True))
    block = int(cfg.get("tier_a.coverage.adaptive_block", 31))
    cc    = int(cfg.get("tier_a.coverage.adaptive_C", 5))
    digit_rois = [r for r in inspector.rois if r.kind == "digit"]

    start_fi   = int(start_skip_s * fps)
    stable_cnt = 0
    trigger_fi: int | None = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_fi)
    for fi in range(start_fi, total):
        ret, frame = cap.read()
        if not ret:
            break
        proc   = preprocess(frame, cfg)
        warped = cv2.warpPerspective(proc, H_cal, (gw, gh)) if H_cal is not None else proc
        if _fast_splash_check(warped, golden, digit_rois, cov_ratio_min, block, cc, active_is_dark):
            stable_cnt += 1
            if stable_cnt == 1:
                trigger_fi = fi
        else:
            stable_cnt = 0
            trigger_fi = None
        if stable_cnt >= stable_needed:
            break

    if trigger_fi is None:
        cap.release()
        raise RuntimeError("splash_not_detected")

    # ── Phase 3: full 28-ROI inspection on the trigger frame ─────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, trigger_fi)
    _, trig_frame = cap.read()
    cap.release()

    result = inspector.inspect_array(trig_frame, video_path)

    t_inference_ms = (time.perf_counter() - t0) * 1000

    # ── Save artefacts ────────────────────────────────────────────────────────
    job_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(job_dir / "triggered_frame.jpg"), trig_frame)
    write_report(result, trig_frame, cfg, job_dir)

    result_dict = result.as_dict()
    result_dict["job_id"]          = job_id
    result_dict["inference_ms"]    = round(t_inference_ms, 1)
    result_dict["splash_frame"]    = trigger_fi
    result_dict["splash_time_s"]   = round(trigger_fi / fps, 2)

    return result_dict
