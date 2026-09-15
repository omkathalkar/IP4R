"""v6c end-to-end pipeline: YOLO Phase 1 → Phase 2 → EfficientNet Phase 3.

Usage:
    from server_v6c.pipeline import V6cPipeline

    pipe = V6cPipeline(
        yolo_model_path      = "data/macro_dataset/runs/macro_test/weights/best.pt",
        phase3_model_path    = "models/v6a/best.pth",
        phase2_template_path = "models/golden_template.npz",  # Fix 2
    )
    result = pipe.predict(video_path)
    print(result["verdict"])   # 'PASS' | 'FAIL' | 'ABSTAIN'

Phase 2 options (mutually exclusive; template takes highest priority):
    phase2_template_path=<path> — Fix 2: IoU-based golden-template content check
    use_phase2_roi=True          — legacy: 8-region coverage + ghost check
    phase2_model_path=<path>     — legacy: masked EfficientNet ghost check

Fix 3 (2026-09-15): Phase 3 runs ONLY on Phase 2 AMBIGUOUS results.
  Phase 2 PASS → final PASS (no Phase 3).
  Phase 2 FAIL → final FAIL (no Phase 3).
  Phase 2 AMBIGUOUS → Phase 3 makes the call.
  (unchanged from original design — Phase 2 ROI PASS was already skipping Phase 3)
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

    Phase 1 — YOLO checklist     : confirms 5 segment blocks, records T*
    Phase 2 — ROI coverage check : 8-region coverage + ghost-pixel (new default)
               or Masked EfficientNet ghost-pixel check (legacy, via phase2_model_path)
    Phase 3 — Full EfficientNet  : verifies 21 icons on YOLO-anchored crop;
               also handles AMBIGUOUS results from Phase 2 ROI
    """

    def __init__(
        self,
        yolo_model_path:      str | Path,
        phase3_model_path:    str | Path,
        phase2_template_path: str | Path | None = None,  # Fix 2: template check
        phase2_model_path:    str | Path | None = None,  # legacy Phase 2
        use_phase2_roi:       bool = False,               # legacy ROI coverage check
        yolo_conf:    float = 0.50,
        phase2_thr:   float = 0.50,
        phase3_thr:   float = 0.50,
        phase2_roi_ghost_thr:           float = 0.40,
        phase2_template_iou_thr:        float = 0.20,
        phase2_template_ghost_thr:      float = 0.40,
        phase2_template_per_roi_iou_thr: dict | None = None,
        device:       str | None = None,
        fps:          float = 12.0,
        n_phase3_frames: int = 3,
    ):
        self.yolo_model_path           = Path(yolo_model_path)
        self.phase2_template_path      = Path(phase2_template_path) if phase2_template_path else None
        self.phase2_model_path         = Path(phase2_model_path) if phase2_model_path else None
        self.use_phase2_roi            = use_phase2_roi
        self.phase3_model_path         = Path(phase3_model_path)
        self.yolo_conf                 = yolo_conf
        self.phase2_thr                = phase2_thr
        self.phase3_thr                = phase3_thr
        self.phase2_roi_ghost_thr             = phase2_roi_ghost_thr
        self.phase2_template_iou_thr          = phase2_template_iou_thr
        self.phase2_template_ghost_thr        = phase2_template_ghost_thr
        self.phase2_template_per_roi_iou_thr  = phase2_template_per_roi_iou_thr
        self.device                           = device
        self.fps                       = fps
        self.n_phase3_frames           = n_phase3_frames

    def predict(self, video_path: str | Path) -> dict:
        """
        Run the full pipeline on one video.

        Returns a dict with keys:
            verdict         : 'PASS' | 'FAIL' | 'ABSTAIN'
            passed          : bool
            failed_at       : 'phase1' | 'phase2_tmpl' | 'phase2_roi' | 'phase2' | 'phase3' | None
            phase1          : Phase1Result as dict
            phase2_template : Phase2TemplateResult as dict (or None if skipped)
            phase2_roi      : Phase2ROIResult as dict (or None if skipped)
            phase2          : Phase2Result as dict (legacy, or None if skipped)
            phase3          : Phase3Result as dict (or None if Phase 2 was PASS/FAIL)
            inference_ms    : total wall-clock time in ms
        """
        from .yolo_phase1      import run_yolo_phase1
        from .phase2_masked    import run_phase2
        from .phase2_roi       import run_phase2_roi
        from .phase2_template  import run_phase2_template
        from .phase3_cnn       import run_phase3

        t0 = time.perf_counter()

        # ── Phase 1 ────────────────────────────────────────────────────────────
        p1 = run_yolo_phase1(
            video_path     = video_path,
            model_path     = self.yolo_model_path,
            conf_threshold = self.yolo_conf,
        )

        if not p1.complete:
            ms = (time.perf_counter() - t0) * 1000
            return {
                "verdict":         "FAIL",
                "passed":          False,
                "failed_at":       "phase1",
                "phase1":          asdict(p1),
                "phase2_template": None,
                "phase2_roi":      None,
                "phase2":          None,
                "phase3":          None,
                "inference_ms":    round(ms, 1),
            }

        p2_tmpl_dict = None
        p2_roi_dict  = None
        p2_dict      = None

        # ── Phase 2 Template check (Fix 2 — highest priority) ─────────────────
        if self.phase2_template_path is not None:
            p2_tmpl = run_phase2_template(
                video_path       = video_path,
                T_star           = p1.T_star,
                template_path    = self.phase2_template_path,
                fps              = self.fps,
                iou_thr          = self.phase2_template_iou_thr,
                ghost_thr        = self.phase2_template_ghost_thr,
                per_roi_iou_thr  = self.phase2_template_per_roi_iou_thr,
            )
            p2_tmpl_dict = asdict(p2_tmpl)

            if p2_tmpl.verdict == "FAIL":
                ms = (time.perf_counter() - t0) * 1000
                return {
                    "verdict":         "FAIL",
                    "passed":          False,
                    "failed_at":       "phase2_tmpl",
                    "phase1":          asdict(p1),
                    "phase2_template": p2_tmpl_dict,
                    "phase2_roi":      None,
                    "phase2":          None,
                    "phase3":          None,
                    "inference_ms":    round(ms, 1),
                }
            elif p2_tmpl.verdict == "PASS":
                ms = (time.perf_counter() - t0) * 1000
                return {
                    "verdict":         "PASS",
                    "passed":          True,
                    "failed_at":       None,
                    "phase1":          asdict(p1),
                    "phase2_template": p2_tmpl_dict,
                    "phase2_roi":      None,
                    "phase2":          None,
                    "phase3":          None,
                    "inference_ms":    round(ms, 1),
                }
            # AMBIGUOUS → fall through to Phase 3

        # ── Phase 2 ROI coverage check (legacy) ───────────────────────────────
        elif self.use_phase2_roi:
            p2_roi = run_phase2_roi(
                video_path  = video_path,
                T_star      = p1.T_star,
                fps         = self.fps,
                ghost_thr   = self.phase2_roi_ghost_thr,
            )
            p2_roi_dict = asdict(p2_roi)

            if p2_roi.verdict == "FAIL":
                ms = (time.perf_counter() - t0) * 1000
                return {
                    "verdict":         "FAIL",
                    "passed":          False,
                    "failed_at":       "phase2_roi",
                    "phase1":          asdict(p1),
                    "phase2_template": None,
                    "phase2_roi":      p2_roi_dict,
                    "phase2":          None,
                    "phase3":          None,
                    "inference_ms":    round(ms, 1),
                }
            elif p2_roi.verdict == "PASS":
                ms = (time.perf_counter() - t0) * 1000
                return {
                    "verdict":         "PASS",
                    "passed":          True,
                    "failed_at":       None,
                    "phase1":          asdict(p1),
                    "phase2_template": None,
                    "phase2_roi":      p2_roi_dict,
                    "phase2":          None,
                    "phase3":          None,
                    "inference_ms":    round(ms, 1),
                }
            # AMBIGUOUS → fall through to Phase 3

        # ── Phase 2 legacy (masked EfficientNet) ──────────────────────────────
        elif self.phase2_model_path is not None:
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
                    "verdict":         "FAIL",
                    "passed":          False,
                    "failed_at":       "phase2",
                    "phase1":          asdict(p1),
                    "phase2_template": None,
                    "phase2_roi":      None,
                    "phase2":          p2_dict,
                    "phase3":          None,
                    "inference_ms":    round(ms, 1),
                }

        # ── Phase 3 — runs only when Phase 2 is AMBIGUOUS or skipped (Fix 3) ──
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
            "verdict":         verdict,
            "passed":          passed,
            "failed_at":       None if passed else "phase3",
            "phase1":          asdict(p1),
            "phase2_template": p2_tmpl_dict,
            "phase2_roi":      p2_roi_dict,
            "phase2":          p2_dict,
            "phase3":          asdict(p3),
            "inference_ms":    round(ms, 1),
        }
