"""v6c end-to-end pipeline: YOLO Phase 1 → Phase 2 → EfficientNet Phase 3.

Usage:
    from server_v6c.pipeline import V6cPipeline

    pipe = V6cPipeline(
        yolo_model_path         = "data/macro_dataset/runs/macro_test/weights/best.pt",
        phase3_model_path       = "models/v6a/best.pth",
        use_phase2_elements     = True,
        elements_threshold_path = "models/element_thresholds.json",
        use_phase2_dino         = True,
        dino_bank_path          = "models/dino_reference_bank.npz",
        dino_thr_path           = "models/dino_threshold.json",
    )
    result = pipe.predict(video_path)
    print(result["verdict"])   # 'PASS' | 'FAIL' | 'ABSTAIN'

Phase 2 priority order (first enabled wins):
    use_phase2_elements / use_phase2_dino  — per-element (2A) + AnomalyDINO (2B) [new primary]
    use_phase2_ssim=True                   — SSIM on grayscale ROIs
    phase2_template_path=<path>            — IoU on binary Otsu masks (legacy Fix 2)
    use_phase2_roi=True                    — 8-region pixel-coverage check
    phase2_model_path=<path>               — masked EfficientNet ghost check (oldest)

Fix 3: Phase 3 runs ONLY for old-path AMBIGUOUS results — not for elements+dino path.
       Elements+dino returns PASS/FAIL directly (no AMBIGUOUS tier).
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

    Phase 1 — YOLO checklist   : confirms 5 segment blocks, records T*
    Phase 2 — Elements + DINO  : 28-element presence (2A) + AnomalyDINO (2B) [new default]
               OR legacy check : SSIM / IoU template / ROI / masked EfficientNet
    Phase 3 — Full EfficientNet: AMBIGUOUS fallback for legacy Phase 2 paths only
    """

    def __init__(
        self,
        yolo_model_path:      str | Path,
        phase3_model_path:    str | Path,
        # ── New primary Phase 2: per-element presence (2A) + AnomalyDINO (2B) ──
        use_phase2_elements:      bool            = False,
        elements_threshold_path:  str | Path | None = None,
        use_phase2_dino:          bool            = False,
        dino_bank_path:           str | Path | None = None,
        dino_thr_path:            str | Path | None = None,
        dino_model_id:            str             = "facebook/dinov2-small",
        dino_confirm_hits:        int             = 4,
        elements_weak_k:          int             = 3,
        elements_contrast_thr:    int             = 15,
        # ── Legacy Phase 2 paths (kept for backward compat + comparison) ───────
        phase2_template_path:    str | Path | None = None,
        phase2_model_path:       str | Path | None = None,
        use_phase2_ssim:         bool = False,
        use_phase2_roi:          bool = False,
        yolo_conf:    float = 0.50,
        phase2_thr:   float = 0.50,
        phase3_thr:   float = 0.50,
        phase2_roi_ghost_thr:            float = 0.40,
        phase2_ssim_ghost_thr:           float = 0.40,
        phase2_ssim_thr_margin:          float = 0.97,
        phase2_template_iou_thr:         float = 0.20,
        phase2_template_ghost_thr:       float = 0.40,
        phase2_template_per_roi_iou_thr: dict | None = None,
        device:          str | None = None,
        fps:             float = 12.0,
        n_phase3_frames: int   = 3,
    ):
        self.yolo_model_path             = Path(yolo_model_path)
        self.phase3_model_path           = Path(phase3_model_path)
        self.use_phase2_elements         = use_phase2_elements
        self.elements_threshold_path     = Path(elements_threshold_path) if elements_threshold_path else None
        self.use_phase2_dino             = use_phase2_dino
        self.dino_bank_path              = Path(dino_bank_path) if dino_bank_path else None
        self.dino_thr_path               = Path(dino_thr_path) if dino_thr_path else None
        self.dino_model_id               = dino_model_id
        self.dino_confirm_hits           = dino_confirm_hits
        self.elements_weak_k             = elements_weak_k
        self.elements_contrast_thr       = elements_contrast_thr
        self.phase2_template_path        = Path(phase2_template_path) if phase2_template_path else None
        self.phase2_model_path           = Path(phase2_model_path) if phase2_model_path else None
        self.use_phase2_ssim             = use_phase2_ssim
        self.use_phase2_roi              = use_phase2_roi
        self.yolo_conf                   = yolo_conf
        self.phase2_thr                  = phase2_thr
        self.phase3_thr                  = phase3_thr
        self.phase2_ssim_ghost_thr       = phase2_ssim_ghost_thr
        self.phase2_ssim_thr_margin      = phase2_ssim_thr_margin
        self.phase2_roi_ghost_thr        = phase2_roi_ghost_thr
        self.phase2_template_iou_thr     = phase2_template_iou_thr
        self.phase2_template_ghost_thr   = phase2_template_ghost_thr
        self.phase2_template_per_roi_iou_thr = phase2_template_per_roi_iou_thr
        self.device                      = device
        self.fps                         = fps
        self.n_phase3_frames             = n_phase3_frames

    def predict(self, video_path: str | Path) -> dict:
        """
        Run the full pipeline on one video.

        Returns a dict with keys:
            verdict          : 'PASS' | 'FAIL' | 'ABSTAIN'
            passed           : bool
            failed_at        : 'phase1' | 'phase2_elements' | 'phase2_dino' |
                               'phase2_ssim' | 'phase2_tmpl' | 'phase2_roi' |
                               'phase2' | 'phase3' | None
            phase1           : Phase1Result as dict
            phase2_elements  : Phase2ElementsResult as dict (or None)
            phase2_dino      : Phase2DinoResult as dict (or None)
            phase2_ssim      : Phase2SSIMResult as dict (or None)
            phase2_template  : Phase2TemplateResult as dict (or None)
            phase2_roi       : Phase2ROIResult as dict (or None)
            phase2           : Phase2Result as dict — legacy (or None)
            phase3           : Phase3Result as dict (or None)
            inference_ms     : total wall-clock time in ms
        """
        from .yolo_phase1     import run_yolo_phase1
        from .phase2_elements import run_phase2_elements
        from .phase2_dino     import run_phase2_dino
        from .phase2_masked   import run_phase2
        from .phase2_roi      import run_phase2_roi
        from .phase2_template import run_phase2_template
        from .phase2_ssim     import run_phase2_ssim
        from .phase3_cnn      import run_phase3

        t0 = time.perf_counter()

        # ── Phase 1 ─────────────────────────────────────────────────────────
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
                "phase2_elements": None,
                "phase2_dino":     None,
                "phase2_ssim":     None,
                "phase2_template": None,
                "phase2_roi":      None,
                "phase2":          None,
                "phase3":          None,
                "inference_ms":    round(ms, 1),
            }

        # ── Phase 2: Elements (2A) + AnomalyDINO (2B) — new primary path ────
        # Both checks run independently; OR-gate determines failure.
        # No AMBIGUOUS tier — returns PASS/FAIL directly, skips Phase 3.
        if self.use_phase2_elements or self.use_phase2_dino:
            p2_elem_dict = None
            p2_dino_dict = None
            failed_at    = None

            if self.use_phase2_elements and self.elements_threshold_path is not None:
                p2_elem = run_phase2_elements(
                    video_path      = video_path,
                    T_star          = p1.T_star,
                    thresholds_path = self.elements_threshold_path,
                    fps             = self.fps,
                    contrast_thr    = self.elements_contrast_thr,
                    weak_k          = self.elements_weak_k,
                )
                p2_elem_dict = asdict(p2_elem)
                if not p2_elem.passed:
                    failed_at = "phase2_elements"

            if self.use_phase2_dino and self.dino_bank_path is not None and self.dino_thr_path is not None:
                p2_dino = run_phase2_dino(
                    video_path   = video_path,
                    T_star       = p1.T_star,
                    bank_path    = self.dino_bank_path,
                    thr_path     = self.dino_thr_path,
                    model_id     = self.dino_model_id,
                    confirm_hits = self.dino_confirm_hits,
                    fps          = self.fps,
                    device       = self.device or "cpu",
                )
                p2_dino_dict = asdict(p2_dino)
                if not p2_dino.passed and failed_at is None:
                    failed_at = "phase2_dino"

            ms = (time.perf_counter() - t0) * 1000
            return {
                "verdict":         "FAIL" if failed_at else "PASS",
                "passed":          failed_at is None,
                "failed_at":       failed_at,
                "phase1":          asdict(p1),
                "phase2_elements": p2_elem_dict,
                "phase2_dino":     p2_dino_dict,
                "phase2_ssim":     None,
                "phase2_template": None,
                "phase2_roi":      None,
                "phase2":          None,
                "phase3":          None,
                "inference_ms":    round(ms, 1),
            }

        # ── Legacy Phase 2 paths ─────────────────────────────────────────────
        p2_ssim_dict = None
        p2_tmpl_dict = None
        p2_roi_dict  = None
        p2_dict      = None

        # ── Phase 2 SSIM check (highest-priority legacy) ─────────────────────
        if self.use_phase2_ssim and self.phase2_template_path is not None:
            p2_ssim = run_phase2_ssim(
                video_path      = video_path,
                T_star          = p1.T_star,
                template_path   = self.phase2_template_path,
                fps             = self.fps,
                ghost_thr       = self.phase2_ssim_ghost_thr,
                ssim_thr_margin = self.phase2_ssim_thr_margin,
            )
            p2_ssim_dict = asdict(p2_ssim)

            if p2_ssim.verdict == "FAIL":
                ms = (time.perf_counter() - t0) * 1000
                return {
                    "verdict":         "FAIL",
                    "passed":          False,
                    "failed_at":       "phase2_ssim",
                    "phase1":          asdict(p1),
                    "phase2_elements": None,
                    "phase2_dino":     None,
                    "phase2_ssim":     p2_ssim_dict,
                    "phase2_template": None,
                    "phase2_roi":      None,
                    "phase2":          None,
                    "phase3":          None,
                    "inference_ms":    round(ms, 1),
                }
            elif p2_ssim.verdict == "PASS":
                ms = (time.perf_counter() - t0) * 1000
                return {
                    "verdict":         "PASS",
                    "passed":          True,
                    "failed_at":       None,
                    "phase1":          asdict(p1),
                    "phase2_elements": None,
                    "phase2_dino":     None,
                    "phase2_ssim":     p2_ssim_dict,
                    "phase2_template": None,
                    "phase2_roi":      None,
                    "phase2":          None,
                    "phase3":          None,
                    "inference_ms":    round(ms, 1),
                }
            # AMBIGUOUS → fall through to Phase 3

        # ── Phase 2 IoU template check ────────────────────────────────────────
        elif self.phase2_template_path is not None:
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
                    "phase2_elements": None,
                    "phase2_dino":     None,
                    "phase2_ssim":     None,
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
                    "phase2_elements": None,
                    "phase2_dino":     None,
                    "phase2_ssim":     None,
                    "phase2_template": p2_tmpl_dict,
                    "phase2_roi":      None,
                    "phase2":          None,
                    "phase3":          None,
                    "inference_ms":    round(ms, 1),
                }
            # AMBIGUOUS → fall through to Phase 3

        # ── Phase 2 ROI coverage check ────────────────────────────────────────
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
                    "phase2_elements": None,
                    "phase2_dino":     None,
                    "phase2_ssim":     None,
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
                    "phase2_elements": None,
                    "phase2_dino":     None,
                    "phase2_ssim":     None,
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
                    "phase2_elements": None,
                    "phase2_dino":     None,
                    "phase2_ssim":     None,
                    "phase2_template": None,
                    "phase2_roi":      None,
                    "phase2":          p2_dict,
                    "phase3":          None,
                    "inference_ms":    round(ms, 1),
                }

        # ── Phase 3 — AMBIGUOUS fallback for legacy Phase 2 paths only ────────
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
            "phase2_elements": None,
            "phase2_dino":     None,
            "phase2_ssim":     p2_ssim_dict,
            "phase2_template": p2_tmpl_dict,
            "phase2_roi":      p2_roi_dict,
            "phase2":          p2_dict,
            "phase3":          asdict(p3),
            "inference_ms":    round(ms, 1),
        }
