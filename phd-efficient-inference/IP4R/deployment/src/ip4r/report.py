"""Reporting: annotated overlay + JSON. Don't print verdicts ad hoc — go through here."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .inspect import InspectionResult


def annotate(golden_bgr: np.ndarray, result: InspectionResult, cfg: Config) -> np.ndarray:
    pass_color = tuple(cfg.get("report.overlay_pass_color", [0, 180, 0]))
    fail_color = tuple(cfg.get("report.overlay_fail_color", [0, 0, 230]))
    canvas = golden_bgr.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for r in result.roi_results:
        x, y, w, h = r.bbox
        color = pass_color if r.passed else fail_color
        thick = 1 if r.passed else 2
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thick)
        if not r.passed:
            label = r.name
            cv2.putText(canvas, label, (x, max(10, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, fail_color, 1, cv2.LINE_AA)

    banner = "PASS" if result.passed else f"FAIL  ({len(result.failed_rois)} element(s))"
    bcolor = pass_color if result.passed else fail_color
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (30, 30, 30), -1)
    cv2.putText(canvas, f"IP4R: {banner}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bcolor, 1, cv2.LINE_AA)
    return canvas


def write_report(result: InspectionResult, golden_bgr: np.ndarray, cfg: Config,
                 results_dir: Path) -> dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_path).stem
    out: dict[str, str] = {}

    if cfg.get("report.save_overlay", True):
        overlay = annotate(golden_bgr, result, cfg)
        op = results_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(op), overlay)
        out["overlay"] = str(op)

    if cfg.get("report.save_json", True):
        jp = results_dir / f"{stem}_report.json"
        with open(jp, "w") as f:
            json.dump(result.as_dict(), f, indent=2)
        out["json"] = str(jp)

    return out
