"""Batch image inspection with correct ROI overlay (inverse homography projection).

Usage:
    python scripts/inspect_batch.py <images_dir> [--out <dir>] [--pattern *.jpg]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ip4r.config import Config
from ip4r.preprocess import preprocess
from ip4r.registration import register
from ip4r.pipeline import Inspector

PASS_COLOR = (60, 210, 60)
FAIL_COLOR = (40, 40, 230)
FONT       = cv2.FONT_HERSHEY_SIMPLEX
BAR_H      = 52


def _map_quad(bbox: tuple, H_inv: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox
    corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2).astype(int)


def annotate_image(frame: np.ndarray, result, H_inv: np.ndarray | None) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    passed = result.passed
    color  = PASS_COLOR if passed else FAIL_COLOR

    # ROI boxes
    for r in result.roi_results:
        rc = PASS_COLOR if r.passed else FAIL_COLOR
        thick = 2 if r.passed else 3
        if H_inv is not None:
            quad = _map_quad(r.bbox, H_inv)
            cv2.polylines(out, [quad], True, rc, thick)
            top = tuple(quad[np.argmin(quad[:, 1])])
        else:
            x, y, w, h = r.bbox
            cv2.rectangle(out, (x, y), (x+w, y+h), rc, thick)
            top = (x, y)

        label = r.name.replace("_", " ")
        cv2.putText(out, label, (top[0], max(18, top[1] - 8)),
                    FONT, 0.42, rc, 1, cv2.LINE_AA)
        if r.coverage is not None:
            cov_str = f"cov={r.coverage:.2f}"
            cv2.putText(out, cov_str, (top[0], max(34, top[1] + 12)),
                        FONT, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        if not r.passed and r.reason:
            cv2.putText(out, r.reason[:40], (top[0], max(50, top[1] + 26)),
                        FONT, 0.32, FAIL_COLOR, 1, cv2.LINE_AA)

    # Top banner
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (W, 54), (0, 60, 0) if passed else (0, 0, 70), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)
    verdict = "PASS" if passed else f"FAIL  ({len(result.failed_rois)} ROI)"
    cv2.putText(out, f"IP4R  |  {verdict}", (16, 38), FONT, 1.2, color, 2, cv2.LINE_AA)

    # Bottom status bar
    cv2.rectangle(out, (0, H - BAR_H), (W, H), (20, 20, 20), -1)
    failed_names = ", ".join(r.name for r in result.failed_rois) or "all ROIs passed"
    cv2.putText(out, failed_names, (12, H - BAR_H + 33),
                FONT, 0.65, color, 1, cv2.LINE_AA)

    return out


def inspect_one(path: Path, inspector: Inspector, cfg, out_dir: Path) -> dict:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return {"path": str(path), "error": "unreadable"}

    sample_proc = preprocess(frame.copy(), cfg)
    aligned, reg_info = register(sample_proc, inspector.golden_proc, cfg)

    H_mat = reg_info.get("homography")
    H_inv = np.linalg.inv(np.array(H_mat)) if H_mat is not None else None

    result = inspector.inspect_array(frame.copy(), str(path))

    ann = annotate_image(frame, result, H_inv)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{path.stem}_overlay.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 92])

    report = result.as_dict()
    with open(out_dir / f"{path.stem}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return {
        "file": path.name,
        "passed": result.passed,
        "failed_rois": [r.name for r in result.failed_rois],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir")
    ap.add_argument("--out", default="data/results/fault")
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg      = Config.load(REPO_ROOT / "config" / "default.yaml")
    inspector = Inspector(cfg)
    out_dir  = REPO_ROOT / args.out
    images   = sorted(Path(args.images_dir).glob(args.pattern))

    if not images:
        print(f"No images found in {args.images_dir} matching {args.pattern}")
        return

    print(f"Inspecting {len(images)} images → {out_dir}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(inspect_one, p, inspector, cfg, out_dir): p for p in images}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            status = "PASS" if r.get("passed") else f"FAIL {r.get('failed_rois', [])}"
            print(f"  [{i:3d}/{len(images)}]  {r['file']}  →  {status}")

    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed
    print(f"\n{'='*55}")
    print(f"  Total: {len(results)}   PASS: {passed}   FAIL: {failed}   ({100*failed/len(results):.1f}% detection)")
    print(f"  Overlays saved to: {out_dir}")

    # Summary JSON
    summary = {"total": len(results), "passed": passed, "failed": failed, "images": results}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
