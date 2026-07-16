"""Splash-triggered video inspection for IP4R.

Detection strategy:
  1. Register one early calibration frame against the golden (ORB+ECC) to get H.
  2. For every subsequent frame, apply H via warpPerspective (fast) and check
     coverage in all digit ROIs — same threshold logic as the batch inspector.
  3. When `--stable-frames` consecutive frames all pass coverage, the splash
     screen is confirmed.  Run the full inspection on the first passing frame.
  4. Write an annotated MP4 showing SCANNING → STABILIZING → PASS/FAIL verdict.

Usage:
    python scripts/inspect_video.py <video>
    python scripts/inspect_video.py <video> --out data/results/vid --start-skip 5 --stable-frames 3
"""
from __future__ import annotations

import argparse
import subprocess
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


# ── visual constants ─────────────────────────────────────────────────────────
PASS_COLOR = (60, 210, 60)
FAIL_COLOR = (40, 40, 230)
WARN_COLOR = (30, 170, 255)
INFO_COLOR = (200, 200, 200)
BAR_H = 56
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── coverage helper (mirrors inspect.py, avoids private import) ───────────────
def _coverage(patch: np.ndarray, block: int, c: int, active_is_dark: bool = True) -> float:
    if patch.size == 0:
        return 0.0
    block = block if block % 2 == 1 else block + 1
    block = max(3, min(block, (min(patch.shape[:2]) // 2) * 2 + 1))
    mode = cv2.THRESH_BINARY_INV if active_is_dark else cv2.THRESH_BINARY
    binar = cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, block, c)
    return float((binar > 0).mean())


def _fast_splash_check(warped: np.ndarray, golden: np.ndarray, rois,
                        cov_ratio_min: float, block: int, cc: int,
                        active_is_dark: bool) -> tuple[bool, dict]:
    """Return (all_pass, {roi_name: cov_sample}) using pre-warped frame."""
    gh, gw = golden.shape[:2]
    covs: dict = {}
    all_pass = True
    for r in rois:
        if r.kind != "digit":
            continue
        x, y, bw, bh = r.to_pixels(gw, gh)
        gp = golden[y:y + bh, x:x + bw]
        sp = warped[y:y + bh, x:x + bw]
        if gp.size == 0 or sp.size == 0:
            continue
        roi_ratio = r.cov_ratio_min if r.cov_ratio_min is not None else cov_ratio_min
        cov_g = _coverage(gp, block, cc, active_is_dark)
        cov_s = _coverage(sp, block, cc, active_is_dark)
        covs[r.name] = round(cov_s, 3)
        if cov_s < cov_g * roi_ratio:
            all_pass = False
    return all_pass, covs


# ── drawing helpers ───────────────────────────────────────────────────────────
def _draw_status_bar(frame: np.ndarray, text: str, color: tuple, sub: str = "") -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    cv2.rectangle(out, (0, H - BAR_H), (W, H), (20, 20, 20), -1)
    cv2.putText(out, text, (16, H - BAR_H + 36), FONT, 1.1, color, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (W - 480, H - BAR_H + 36), FONT, 0.65, (160, 160, 160), 1, cv2.LINE_AA)
    return out


def _map_roi_quad(bbox: tuple, H_inv: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    mapped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_inv)
    return mapped.reshape(-1, 2).astype(int)


def _draw_roi_overlay(frame: np.ndarray, result, H_inv: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for r in result.roi_results:
        color = PASS_COLOR if r.passed else FAIL_COLOR
        quad = _map_roi_quad(r.bbox, H_inv)
        cv2.polylines(out, [quad], isClosed=True, color=color, thickness=2 if r.passed else 3)
        top_pt = tuple(quad[np.argmin(quad[:, 1])])
        label = r.name.replace("_", " ")
        cv2.putText(out, label, (top_pt[0], max(20, top_pt[1] - 8)),
                    FONT, 0.40, color, 1, cv2.LINE_AA)
        if r.coverage is not None:
            cv2.putText(out, f"c={r.coverage:.2f}", (top_pt[0], max(36, top_pt[1] + 10)),
                        FONT, 0.33, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def _draw_verdict_banner(frame: np.ndarray, result, t_trigger: float) -> np.ndarray:
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
    cv2.putText(out, f"splash @ t={t_trigger:.1f}s", (W - 340, 40), FONT, 0.75, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def _ffmpeg_writer(out_path: str, w: int, h: int, fps: float):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
        "-r", str(fps), "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast",
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── main inspection function ──────────────────────────────────────────────────
def inspect_video(
    video_path: str,
    out_dir: Path,
    start_skip_s: float = 5.0,
    stable_needed: int = 3,
    cal_t_s: float = 2.0,
) -> None:
    cfg = Config.load(REPO_ROOT / "config" / "default.yaml")
    inspector = Inspector(cfg)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 12.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gh, gw = inspector.golden_proc.shape[:2]

    print(f"Video: {W}x{H_vid}  {fps:.1f}fps  {total} frames  ({total/fps:.1f}s)")
    print(f"Start-skip: {start_skip_s}s  |  stable frames needed: {stable_needed}")

    # ── coverage params (mirrors inspect.py) ─────────────────────────────────
    cov_ratio_min  = float(cfg.get("tier_a.coverage.coverage_ratio_min", 0.55))
    active_is_dark = bool(cfg.get("tier_a.coverage.active_is_dark", True))
    block          = int(cfg.get("tier_a.coverage.adaptive_block", 31))
    cc             = int(cfg.get("tier_a.coverage.adaptive_C", 5))
    digit_rois     = [r for r in inspector.rois if r.kind == "digit"]

    # ── Phase 1: one-time registration for fast warp ─────────────────────────
    print(f"\nRegistering calibration frame at t={cal_t_s}s …")
    cal_fi = int(cal_t_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cal_fi)
    ret, cal_frame = cap.read()
    H_cal = None
    H_cal_inv = None
    if ret:
        cal_proc = preprocess(cal_frame, cfg)
        _, cal_reg = register(cal_proc, inspector.golden_proc, cfg)
        H_raw = cal_reg.get("homography")
        if H_raw is not None:
            H_cal = np.array(H_raw, dtype=np.float64)
            H_cal_inv = np.linalg.inv(H_cal)
            print(f"  Registration OK (method={cal_reg.get('method')})")
        else:
            print("  WARNING: registration failed, coverage scan may be inaccurate")

    # ── Phase 2: fast frame scan for splash ───────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = _ffmpeg_writer(str(out_dir / "annotated.mp4"), W, H_vid, fps)

    start_fi   = int(start_skip_s * fps)
    stable_cnt = 0
    triggered  = False
    result     = None
    H_result_inv = None
    t_trigger  = 0.0
    frozen_overlay = None
    trigger_fi = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"\nScanning for splash screen (skipping first {start_skip_s}s) …")
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = fi / fps

        if fi < start_fi or triggered:
            # ── before skip window or after trigger: just draw frozen state ──
            if triggered and frozen_overlay is not None:
                out_frame = frozen_overlay.copy()
                out_frame = _draw_status_bar(
                    out_frame,
                    f"RESULT  t={t_trigger:.1f}s",
                    PASS_COLOR if result.passed else FAIL_COLOR,
                    f"frame {fi+1}/{total}",
                )
            else:
                out_frame = _draw_status_bar(frame, "SCANNING…", INFO_COLOR,
                                              f"t={t_sec:.1f}s  frame {fi+1}/{total}")
            writer.stdin.write(out_frame.tobytes())
            continue

        # ── fast coverage check using pre-computed homography ─────────────────
        proc = preprocess(frame, cfg)
        if H_cal is not None:
            warped = cv2.warpPerspective(proc, H_cal, (gw, gh))
        else:
            warped = proc

        splash_ok, covs = _fast_splash_check(
            warped, inspector.golden_proc, digit_rois,
            cov_ratio_min, block, cc, active_is_dark
        )

        if splash_ok:
            stable_cnt += 1
            if stable_cnt == 1:
                trigger_fi = fi
            status_text = f"STABILIZING…  ({stable_cnt}/{stable_needed})"
            cov_str = "  ".join(f"{k.split('_')[-1]}={v:.2f}" for k, v in covs.items())
            out_frame = _draw_status_bar(frame, status_text, WARN_COLOR,
                                          f"t={t_sec:.1f}s  {cov_str}")
        else:
            stable_cnt = 0
            trigger_fi = None
            cov_str = "  ".join(f"{k.split('_')[-1]}={v:.2f}" for k, v in covs.items())
            out_frame = _draw_status_bar(frame, "SCANNING…", INFO_COLOR,
                                          f"t={t_sec:.1f}s  {cov_str}")

        if stable_cnt >= stable_needed:
            # ── Phase 3: full inspection on the first frame of stable window ──
            t_trigger = trigger_fi / fps
            print(f"\nSplash detected at t={t_trigger:.1f}s  (frame {trigger_fi}) — running full inspection …")
            cap.set(cv2.CAP_PROP_POS_FRAMES, trigger_fi)
            _, trig_frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi + 1)  # resume from current position

            result = inspector.inspect_array(trig_frame, video_path)
            cv2.imwrite(str(out_dir / "triggered_frame.jpg"), trig_frame)

            # Get per-frame homography for accurate overlay quads
            trig_proc = preprocess(trig_frame, cfg)
            _, trig_reg = register(trig_proc, inspector.golden_proc, cfg)
            H_trig = trig_reg.get("homography")
            H_result_inv = np.linalg.inv(np.array(H_trig)) if H_trig is not None else H_cal_inv

            ann = trig_frame.copy()
            if H_result_inv is not None:
                ann = _draw_roi_overlay(ann, result, H_result_inv)
            ann = _draw_verdict_banner(ann, result, t_trigger)
            frozen_overlay = ann

            triggered = True
            out_frame = _draw_status_bar(
                ann, f"t={t_trigger:.1f}s  TRIGGERED",
                PASS_COLOR if result.passed else FAIL_COLOR,
                f"frame {trigger_fi}/{total}",
            )

        writer.stdin.write(out_frame.tobytes())

    cap.release()
    writer.stdin.close()
    writer.wait()

    if not triggered:
        print("\nWARNING: no splash screen detected — no inspection triggered.")
        print("  Try reducing --start-skip or increasing --stable-frames.")
        return

    verdict = "PASS" if result.passed else "FAIL"
    print(f"\n{'='*56}")
    print(f"  VERDICT:  {verdict}   (splash at t={t_trigger:.1f}s)")
    print(f"{'='*56}")
    if not result.passed:
        for r in result.failed_rois:
            print(f"  FAIL  {r.name}: {r.reason}")
    else:
        print("  All ROIs passed.")
    print(f"\nSaved:")
    print(f"  {out_dir / 'annotated.mp4'}")
    print(f"  {out_dir / 'triggered_frame.jpg'}")


def main():
    ap = argparse.ArgumentParser(description="IP4R video inspection (splash-triggered)")
    ap.add_argument("video", help="Path to the video file")
    ap.add_argument("--out", default="data/results/video_triggered",
                    help="Output directory (default: data/results/video_triggered)")
    ap.add_argument("--start-skip", type=float, default=5.0,
                    help="Seconds to skip before looking for splash (default: 5.0)")
    ap.add_argument("--stable-frames", type=int, default=3,
                    help="Consecutive passing frames to confirm splash (default: 3)")
    ap.add_argument("--cal-t", type=float, default=2.0,
                    help="Timestamp (s) for calibration registration frame (default: 2.0)")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    inspect_video(
        args.video,
        out_dir,
        start_skip_s=args.start_skip,
        stable_needed=args.stable_frames,
        cal_t_s=args.cal_t,
    )


if __name__ == "__main__":
    main()
