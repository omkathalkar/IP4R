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

    def _get_remote_bbox(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 5000:
            return None
        return cv2.boundingRect(c)

    def inspect_array(self, sample_bgr: np.ndarray, image_path: str) -> InspectionResult:
        if not self.rois:
            raise RuntimeError(
                "No ROIs defined. Author them first:  ip4r roi-edit"
            )
        sample_proc = preprocess(sample_bgr, self.cfg)
        
        # Detect remote to crop away the black jig, which confuses ORB
        bbox = self._get_remote_bbox(sample_bgr)
        if bbox:
            x, y, w, h = bbox
            sample_proc_crop = sample_proc[y:y+h, x:x+w]
            aligned_crop, reg_info = register(sample_proc_crop, self.golden_proc, self.cfg)
            H_crop = reg_info.get("homography")
            if H_crop is not None:
                T_inv = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=np.float64)
                H_full = H_crop @ T_inv
                reg_info["homography"] = H_full
                # warp full image for roi inspection
                aligned = cv2.warpPerspective(sample_proc, H_full, 
                                              (self.golden_proc.shape[1], self.golden_proc.shape[0]))
            else:
                aligned = sample_proc
        else:
            aligned, reg_info = register(sample_proc, self.golden_proc, self.cfg)

        roi_kinds = self.cfg.get("tier_a.roi_kinds", [])
        active_rois = [r for r in self.rois if not roi_kinds or r.kind in roi_kinds]
        roi_results = inspect_rois(aligned, self.golden_proc, active_rois, self.cfg,
                                   cnn_scorer=self._cnn_scorer)

        fail_any = self.cfg.get("decision.fail_if_any_roi_fails", True)
        min_fail = int(self.cfg.get("decision.min_fail_rois", 1))
        n_failed = sum(1 for r in roi_results if not r.passed)
        passed = not (fail_any and n_failed >= min_fail)

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
