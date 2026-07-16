"""Live camera inspection for IP4R (splash-triggered).

Connects to a USB/built-in camera, waits for the AC-remote LCD splash screen
(all-segments-on state), runs a full IP4R inspection automatically, and
displays the PASS/FAIL verdict with an annotated overlay in real time.

Controls:
  q     — quit
  s     — manually trigger inspection on the current frame
  r     — reset and scan again after a verdict has been shown
  SPACE — save the current raw frame to data/samples/

Usage:
    python scripts/inspect_camera.py
    python scripts/inspect_camera.py --camera 1          # USB camera index 1
    python scripts/inspect_camera.py --stable-frames 5   # stricter splash confirmation
    python scripts/inspect_camera.py --no-window          # headless: save results only
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ip4r.config import Config
from ip4r.preprocess import preprocess
from ip4r.registration import register
from ip4r.pipeline import Inspector
from ip4r.report import write_report

# ── visual constants ─────────────────────────────────────────────────────────
PASS_COLOR = (60, 210, 60)
FAIL_COLOR = (40, 40, 230)
WARN_COLOR = (30, 170, 255)
INFO_COLOR = (200, 200, 200)
BAR_H      = 56
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ── coverage helper (mirrors inspect.py) ──────────────────────────────────────
def _coverage(patch: np.ndarray, block: int, c: int, active_is_dark: bool) -> float:
    if patch.size == 0:
        return 0.0
    block = block if block % 2 == 1 else block + 1
    block = max(3, min(block, (min(patch.shape[:2]) // 2) * 2 + 1))
    mode = cv2.THRESH_BINARY_INV if active_is_dark else cv2.THRESH_BINARY
    binar = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, block, c)
    return float((binar > 0).mean())


def _fast_splash_check(warped_gray: np.ndarray, golden_gray: np.ndarray, rois,
                        cov_ratio_min: float, block: int, cc: int,
                        active_is_dark: bool) -> tuple[bool, dict]:
    gh, gw = golden_gray.shape[:2]
    covs: dict = {}
    all_pass = True
    for r in rois:
        if r.kind != "digit":
            continue
        x, y, bw, bh = r.to_pixels(gw, gh)
        gp = golden_gray[y:y + bh, x:x + bw]
        sp = warped_gray[y:y + bh, x:x + bw]
        if gp.size == 0 or sp.size == 0:
            continue
        roi_ratio = r.cov_ratio_min if r.cov_ratio_min is not None else cov_ratio_min
        cov_s = _coverage(sp, block, cc, active_is_dark)
        covs[r.name] = round(cov_s, 3)
        cov_g = _coverage(gp, block, cc, active_is_dark)
        if cov_s < cov_g * roi_ratio:
            all_pass = False
    return all_pass, covs


# ── drawing helpers ───────────────────────────────────────────────────────────
def _draw_bar(frame: np.ndarray, text: str, color: tuple, sub: str = "") -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    cv2.rectangle(out, (0, H - BAR_H), (W, H), (20, 20, 20), -1)
    cv2.putText(out, text, (16, H - BAR_H + 36), FONT, 1.0, color, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (W - 500, H - BAR_H + 36), FONT, 0.60, (160, 160, 160), 1, cv2.LINE_AA)
    return out


def _draw_controls(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    hints = "[q] quit   [s] force inspect   [r] reset   [space] save frame"
    cv2.putText(out, hints, (12, 22), FONT, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
    return out


def _draw_roi_overlay(frame: np.ndarray, result, H_inv: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for r in result.roi_results:
        color = PASS_COLOR if r.passed else FAIL_COLOR
        x, y, w, h = r.bbox
        corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2).astype(int)
        cv2.polylines(out, [quad], True, color, 2 if r.passed else 3)
        top = tuple(quad[np.argmin(quad[:, 1])])
        cv2.putText(out, r.name.replace("_", " "), (top[0], max(20, top[1] - 6)),
                    FONT, 0.38, color, 1, cv2.LINE_AA)
    return out


def _draw_verdict(frame: np.ndarray, result) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    passed = result.passed
    text = "PASS" if passed else f"FAIL  ({len(result.failed_rois)} ROI)"
    color = PASS_COLOR if passed else FAIL_COLOR
    bg = (0, 60, 0) if passed else (0, 0, 80)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (W, 60), bg, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.putText(out, f"IP4R  |  {text}", (20, 44), FONT, 1.3, color, 3, cv2.LINE_AA)
    if not passed:
        fail_names = ", ".join(r.name for r in result.failed_rois)
        cv2.putText(out, fail_names, (20, 70), FONT, 0.55, FAIL_COLOR, 1, cv2.LINE_AA)
    return out


# ── main ──────────────────────────────────────────────────────────────────────
def run_camera(
    camera_index: int = 0,
    stable_needed: int = 3,
    show_window: bool = True,
    re_register_interval: int = 30,
) -> None:
    cfg = Config.load(REPO_ROOT / "config" / "default.yaml")
    inspector = Inspector(cfg)
    out_dir = REPO_ROOT / cfg.get("paths.results_dir", "data/results")
    samples_dir = REPO_ROOT / cfg.get("paths.samples_dir", "data/samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    cov_ratio_min  = float(cfg.get("tier_a.coverage.coverage_ratio_min", 0.55))
    active_is_dark = bool(cfg.get("tier_a.coverage.active_is_dark", True))
    block          = int(cfg.get("tier_a.coverage.adaptive_block", 31))
    cc             = int(cfg.get("tier_a.coverage.adaptive_C", 5))
    digit_rois     = [r for r in inspector.rois if r.kind == "digit"]

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "Try --camera 1 or --camera 2 for a USB camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Camera {camera_index} opened.")
    print("Point the camera at the AC remote in splash (all-segments-on) state.")
    print("Controls: [q] quit  [s] force-inspect  [r] reset  [space] save frame\n")

    H_warp: np.ndarray | None = None
    H_inv:  np.ndarray | None = None
    stable_cnt = 0
    frames_since_reg = 0
    state = "SCANNING"      # SCANNING | STABILIZING | RESULT
    result = None
    frozen_frame: np.ndarray | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed — retrying …")
            continue

        key = cv2.waitKey(1) & 0xFF if show_window else 0xFF

        # ── key handling ─────────────────────────────────────────────────────
        if key == ord('q'):
            break

        if key == ord(' '):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = samples_dir / f"capture_{ts}.jpg"
            cv2.imwrite(str(save_path), frame)
            print(f"Saved: {save_path}")

        if key == ord('r'):
            state = "SCANNING"
            stable_cnt = 0
            H_warp = None
            H_inv = None
            result = None
            frozen_frame = None
            print("Reset — scanning again …")

        if key == ord('s') and state != "RESULT":
            state = "FORCED"

        # ── show frozen result until reset ───────────────────────────────────
        if state == "RESULT":
            display = _draw_bar(
                frozen_frame, "RESULT — press [r] to scan again",
                PASS_COLOR if result.passed else FAIL_COLOR,
            )
            display = _draw_controls(display)
            if show_window:
                cv2.imshow("IP4R Camera", display)
            continue

        # ── re-register homography periodically ──────────────────────────────
        frames_since_reg += 1
        if H_warp is None or frames_since_reg >= re_register_interval:
            proc = preprocess(frame, cfg)
            _, reg_info = register(proc, inspector.golden_proc, cfg)
            H_raw = reg_info.get("homography")
            if H_raw is not None:
                H_warp = np.array(H_raw, dtype=np.float64)
                H_inv  = np.linalg.inv(H_warp)
            else:
                H_warp = None
                H_inv  = None
            frames_since_reg = 0

        # ── fast splash coverage check ────────────────────────────────────────
        gray = preprocess(frame, cfg)
        if H_warp is not None:
            gh, gw = inspector.golden_proc.shape[:2]
            warped = cv2.warpPerspective(gray, H_warp, (gw, gh))
        else:
            warped = gray

        splash_ok, covs = _fast_splash_check(
            warped, inspector.golden_proc, digit_rois,
            cov_ratio_min, block, cc, active_is_dark,
        )

        cov_str = "  ".join(f"{k.split('_')[-1]}={v:.2f}" for k, v in covs.items())

        if state == "FORCED" or (splash_ok and state in ("SCANNING", "STABILIZING")):
            if state != "FORCED":
                stable_cnt += 1
                state = "STABILIZING"
            status_text = (
                f"STABILIZING…  ({stable_cnt}/{stable_needed})"
                if state != "FORCED"
                else "FORCED INSPECT…"
            )
            display = _draw_bar(frame, status_text, WARN_COLOR, cov_str)
        else:
            stable_cnt = 0
            state = "SCANNING"
            display = _draw_bar(frame, "SCANNING…", INFO_COLOR, cov_str)

        trigger = (stable_cnt >= stable_needed) or (key == ord('s'))

        if trigger:
            print("Splash confirmed — running full inspection …")
            result = inspector.inspect_array(frame.copy(), "camera_live")

            ann = frame.copy()
            if H_inv is not None:
                ann = _draw_roi_overlay(ann, result, H_inv)
            ann = _draw_verdict(ann, result)
            frozen_frame = ann

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            overlay_path = out_dir / f"camera_{ts}_overlay.jpg"
            cv2.imwrite(str(overlay_path), ann)
            write_report(result, inspector.golden_bgr, cfg, out_dir)

            verdict = "PASS" if result.passed else "FAIL"
            print(f"\n{'='*50}")
            print(f"  VERDICT: {verdict}")
            if not result.passed:
                for r in result.failed_rois:
                    print(f"  FAIL  {r.name}: {r.reason}")
            else:
                print("  All ROIs passed.")
            print(f"  Overlay saved: {overlay_path}\n")

            state = "RESULT"
            stable_cnt = 0

        display = _draw_controls(display)
        if show_window:
            cv2.imshow("IP4R Camera", display)

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(
        description="IP4R live camera inspection (splash-triggered)"
    )
    ap.add_argument("--camera", type=int, default=0,
                    help="Camera device index (default 0 = built-in, 1/2 = USB)")
    ap.add_argument("--stable-frames", type=int, default=3,
                    help="Consecutive passing frames to confirm splash (default 3)")
    ap.add_argument("--no-window", action="store_true",
                    help="Headless mode: no display window, save results only")
    ap.add_argument("--re-register", type=int, default=30,
                    help="Re-run homography registration every N frames (default 30)")
    args = ap.parse_args()

    run_camera(
        camera_index=args.camera,
        stable_needed=args.stable_frames,
        show_window=not args.no_window,
        re_register_interval=args.re_register,
    )


if __name__ == "__main__":
    main()
