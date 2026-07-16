"""End-to-end inspection of a single image against the golden reference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .preprocess import preprocess
from .registration import register
from .roi import ROI, load_rois
from .inspect import inspect_rois, InspectionResult


class Inspector:
    """Holds the golden reference + ROI map; inspects samples against them."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        ref_path = cfg.path("paths.reference_image")
        self.golden_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if self.golden_bgr is None:
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        self.golden_proc = preprocess(self.golden_bgr, cfg)
        self.rois: list[ROI] = load_rois(cfg.path("paths.roi_map"))

        # Optional CNN scorer — model_type selects which scorer to load
        self._cnn_scorer = None
        model_type = cfg.get("tier_a.cnn.model_type", "digit_cnn")

        if model_type == "mnist_hf":
            try:
                from .mnist_scorer import MnistDigitScorer
                active_is_dark = bool(cfg.get("tier_a.coverage.active_is_dark", True))
                self._cnn_scorer = MnistDigitScorer(invert=active_is_dark)
            except Exception as e:
                print(f"[Inspector] MNIST scorer load failed ({e}), continuing without CNN")
        else:
            model_path = Path(cfg.get("tier_a.cnn.model_path", "models/digit_cnn.pt"))
            if not model_path.is_absolute():
                model_path = Path(__file__).parent.parent.parent / model_path
            if model_path.exists():
                try:
                    from .digit_cnn import DigitCNNScorer
                    self._cnn_scorer = DigitCNNScorer(model_path)
                    print(f"[Inspector] Loaded DigitCNN from {model_path}")
                except Exception as e:
                    print(f"[Inspector] CNN load failed ({e}), continuing without it")

    def inspect_array(self, sample_bgr: np.ndarray, image_path: str) -> InspectionResult:
        if not self.rois:
            raise RuntimeError(
                "No ROIs defined. Author them first:  ip4r roi-edit"
            )
        sample_proc = preprocess(sample_bgr, self.cfg)
        aligned, reg_info = register(sample_proc, self.golden_proc, self.cfg)

        roi_kinds = self.cfg.get("tier_a.roi_kinds", [])
        active_rois = [r for r in self.rois if not roi_kinds or r.kind in roi_kinds]
        roi_results = inspect_rois(aligned, self.golden_proc, active_rois, self.cfg,
                                   cnn_scorer=self._cnn_scorer)

        fail_any = self.cfg.get("decision.fail_if_any_roi_fails", True)
        passed = not (fail_any and any(not r.passed for r in roi_results))

        return InspectionResult(
            image_path=image_path,
            passed=passed,
            roi_results=roi_results,
            registration={"method": reg_info["method"], "fallback": reg_info["fallback"]},
        )

    def inspect_path(self, image_path: str | Path) -> InspectionResult:
        image_path = str(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Sample image not found: {image_path}")
        return self.inspect_array(img, image_path)
