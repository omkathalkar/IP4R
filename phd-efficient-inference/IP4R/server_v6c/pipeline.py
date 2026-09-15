"""v6c end-to-end pipeline: YOLO Phase 1 → Masked EfficientNet Phase 2 → EfficientNet Phase 3.

Usage:
    from server_v6c.pipeline import V6cPipeline

    pipe = V6cPipeline(
        yolo_model_path   = "data/macro_dataset/runs/macro_test/weights/best.pt",
        phase2_model_path = "models/dts_p2v2_best.pth",
        phase3_model_path = "models/v6a/best.pth",
    )
    result = pipe.predict(video_path)
    print(result["verdict"])   # 'PASS' | 'FAIL' | 'ABSTAIN'
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path

log = logging.getLogger(__name__)


class V6cPipeline:
    """
    Three-phase LCD QC pipeline.

    Phase 1 — YOLO checklist  : confirms 5 segment blocks, records T*
    Phase 2 — Masked EfficientNet : checks background for ghost/stray pixels  (optional)
    Phase 3 — Full EfficientNet   : verifies all 21 icons on YOLO-anchored crop
    """

    def __init__(
        self,
        yolo_model_path:   str | Path,
        phase3_model_path: str | Path,
        phase2_model_path: str | Path | None = None,   # None → skip Phase 2
        yolo_conf:    float = 0.50,
        phase2_thr:   float = 0.50,
        phase3_thr:   float = 0.50,
        device:       str | None = None,
        fps:          float = 12.0,
        n_phase3_frames: int = 3,
    ):
        self.yolo_model_path   = Path(yolo_model_path)
        self.phase2_model_path = Path(phase2_model_path) if phase2_model_path else None
        self.phase3_model_path = Path(phase3_model_path)
        self.yolo_conf         = yolo_conf
        self.phase2_thr        = phase2_thr
        self.phase3_thr        = phase3_thr
        self.device            = device
        self.fps               = fps
        self.n_phase3_frames   = n_phase3_frames

    def predict(self, video_path: str | Path) -> dict:
        """
        Run the full pipeline on one video.

        Returns a dict with keys:
            verdict      : 'PASS' | 'FAIL' | 'ABSTAIN'
            passed       : bool
            failed_at    : 'phase1' | 'phase2' | 'phase3' | None
            phase1       : Phase1Result as dict
            phase2       : Phase2Result as dict (or None if skipped)
            phase3       : Phase3Result as dict
            inference_ms : total wall-clock time in ms
        """
        from .yolo_phase1  import run_yolo_phase1
        from .phase2_masked import run_phase2
        from .phase3_cnn   import run_phase3

        t0 = time.perf_counter()

        # ── Phase 1 ────────────────────────────────────────────────────────────
        p1 = run_yolo_phase1(
            video_path   = video_path,
            model_path   = self.yolo_model_path,
            conf_threshold = self.yolo_conf,
        )

        if not p1.complete:
            ms = (time.perf_counter() - t0) * 1000
            return {
                "verdict":      "FAIL",
                "passed":       False,
                "failed_at":    "phase1",
                "phase1":       asdict(p1),
                "phase2":       None,
                "phase3":       None,
                "inference_ms": round(ms, 1),
            }

        # ── Phase 2 (optional) ─────────────────────────────────────────────────
        p2_dict = None
        if self.phase2_model_path is not None:
            p2 = run_phase2(
                video_path  = video_path,
                T_star      = p1.T_star,
                bboxes      = p1.bboxes,
                model_path  = self.phase2_model_path,
                threshold   = self.phase2_thr,
                device      = self.device,
            )
            p2_dict = asdict(p2)

            if not p2.passed:
                ms = (time.perf_counter() - t0) * 1000
                return {
                    "verdict":      "FAIL",
                    "passed":       False,
                    "failed_at":    "phase2",
                    "phase1":       asdict(p1),
                    "phase2":       p2_dict,
                    "phase3":       None,
                    "inference_ms": round(ms, 1),
                }

        # ── Phase 3 ────────────────────────────────────────────────────────────
        p3 = run_phase3(
            video_path  = video_path,
            T_star      = p1.T_star,
            model_path  = self.phase3_model_path,
            threshold   = self.phase3_thr,
            n_frames    = self.n_phase3_frames,
            fps         = self.fps,
            device      = self.device,
        )

        ms      = (time.perf_counter() - t0) * 1000
        passed  = p3.passed
        verdict = p3.verdict

        return {
            "verdict":      verdict,
            "passed":       passed,
            "failed_at":    None if passed else "phase3",
            "phase1":       asdict(p1),
            "phase2":       p2_dict,
            "phase3":       asdict(p3),
            "inference_ms": round(ms, 1),
        }
