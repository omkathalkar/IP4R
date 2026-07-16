"""Tier A: deterministic per-ROI checks (v02).
   Three orthogonal scores per element:
   - coverage   : 'is this element lit / present?'        (catches missing/partial)
   - diff_map   : 'does it look structurally different?'  (catches single missing stroke, smear)
   - SSIM       : 'is it shaped right / clean?'           (per-ROI calibrated floor)
   Plus per-segment occupancy for single-digit ROIs.
All computed AFTER registration (SSIM and diff are not shift-invariant).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .config import Config
from .roi import ROI


@dataclass
class ROIResult:
    name: str
    kind: str
    bbox: tuple[int, int, int, int]
    coverage: float | None = None
    coverage_golden: float | None = None
    coverage_pass: bool | None = None
    ssim: float | None = None
    ssim_pass: bool | None = None
    diff_density: float | None = None
    diff_pass: bool | None = None
    segment_fails: list = field(default_factory=list)
    segment_pass: bool | None = None
    cnn_defect_prob: float | None = None
    cnn_pass: bool | None = None
    passed: bool = True
    reason: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


@dataclass
class InspectionResult:
    image_path: str
    passed: bool
    roi_results: list[ROIResult] = field(default_factory=list)
    registration: dict = field(default_factory=dict)
    tier_b: dict | None = None

    @property
    def failed_rois(self) -> list[ROIResult]:
        return [r for r in self.roi_results if not r.passed]

    def as_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "passed": self.passed,
            "n_rois": len(self.roi_results),
            "n_failed": len(self.failed_rois),
            "registration": self.registration,
            "tier_b": self.tier_b,
            "rois": [r.as_dict() for r in self.roi_results],
        }


def _diff_density(gp: np.ndarray, sp: np.ndarray, blur_k: int = 3,
                  px_thresh: int = 20) -> float:
    """Fraction of pixels with significant structural difference.

    absdiff cancels global brightness shifts; local segment-scale differences
    (missing stroke, smear, corruption) survive and exceed px_thresh.
    """
    if gp.size == 0:
        return 0.0
    diff = cv2.absdiff(gp, sp)
    if blur_k > 1 and min(diff.shape[:2]) >= blur_k:
        diff = cv2.GaussianBlur(diff, (blur_k, blur_k), 1)
    _, mask = cv2.threshold(diff, px_thresh, 255, cv2.THRESH_BINARY)
    return float((mask > 0).mean())


def _coverage(patch: np.ndarray, block: int, c: int, active_is_dark: bool = True) -> float:
    """Fraction of 'lit' pixels via local adaptive threshold (robust to lighting).

    active_is_dark=True (reflective LCD): active segments are darker than local bg.
    active_is_dark=False (emissive/LED): active segments are brighter.
    """
    if patch.size == 0:
        return 0.0
    block = block if block % 2 == 1 else block + 1
    block = max(3, min(block, (min(patch.shape[:2]) // 2) * 2 + 1))
    mode = cv2.THRESH_BINARY_INV if active_is_dark else cv2.THRESH_BINARY
    binar = cv2.adaptiveThreshold(
        patch, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, block, c
    )
    return float((binar > 0).mean())


def inspect_rois(aligned: np.ndarray, golden: np.ndarray, rois: list[ROI],
                 cfg: Config, cnn_scorer=None) -> list[ROIResult]:
    from .segments import missing_segments  # local import avoids circular

    h, w = golden.shape[:2]

    # ── feature flags ────────────────────────────────────────────────────────
    cov_on  = cfg.get("tier_a.coverage.enabled", True)
    ssim_on = cfg.get("tier_a.ssim.enabled", True)
    diff_on = cfg.get("tier_a.diff.enabled", False)
    seg_on  = cfg.get("tier_a.segments.enabled", False)
    cnn_on  = cnn_scorer is not None
    cnn_thresh = float(cfg.get("tier_a.cnn.defect_threshold", 0.5))

    # ── coverage params ───────────────────────────────────────────────────────
    cov_ratio_min  = float(cfg.get("tier_a.coverage.coverage_ratio_min", 0.55))
    active_is_dark = bool(cfg.get("tier_a.coverage.active_is_dark", True))
    block          = int(cfg.get("tier_a.coverage.adaptive_block", 31))
    cc             = int(cfg.get("tier_a.coverage.adaptive_C", 5))

    # ── ssim params ───────────────────────────────────────────────────────────
    ssim_min = float(cfg.get("tier_a.ssim.ssim_min", 0.62))
    win      = int(cfg.get("tier_a.ssim.win_size", 7))

    # ── diff params ───────────────────────────────────────────────────────────
    diff_max    = float(cfg.get("tier_a.diff.max_density", 0.15))
    diff_blur   = int(cfg.get("tier_a.diff.blur_k", 3))
    diff_thresh = int(cfg.get("tier_a.diff.pixel_threshold", 20))

    # ── segment params ────────────────────────────────────────────────────────
    seg_max_aspect = float(cfg.get("tier_a.segments.max_aspect_ratio", 1.5))

    results: list[ROIResult] = []
    for r in rois:
        x, y, bw, bh = r.to_pixels(w, h)
        gp = golden[y:y + bh, x:x + bw]
        sp = aligned[y:y + bh, x:x + bw]
        res = ROIResult(name=r.name, kind=r.kind, bbox=(x, y, bw, bh))
        reasons: list[str] = []

        # per-ROI coverage ratio (used by both full-ROI coverage check and sub-digit check)
        roi_cov_ratio = r.cov_ratio_min if r.cov_ratio_min is not None else cov_ratio_min

        # ── 1. Coverage check ─────────────────────────────────────────────────
        if cov_on:
            cov_g = _coverage(gp, block, cc, active_is_dark)
            cov_s = _coverage(sp, block, cc, active_is_dark)
            res.coverage        = round(cov_s, 4)
            res.coverage_golden = round(cov_g, 4)
            thresh = cov_g * roi_cov_ratio
            res.coverage_pass = cov_s >= thresh
            if not res.coverage_pass:
                reasons.append(f"coverage {cov_s:.2f} < {thresh:.2f} (golden {cov_g:.2f})")

        # ── 2. Diff-map check ─────────────────────────────────────────────────
        if diff_on:
            roi_diff_max = r.diff_max if r.diff_max is not None else diff_max
            d = _diff_density(gp, sp, diff_blur, diff_thresh)
            res.diff_density = round(d, 4)
            res.diff_pass    = d <= roi_diff_max
            if not res.diff_pass:
                reasons.append(f"diff {d:.3f} > {roi_diff_max:.3f}")

        # ── 3. SSIM check ─────────────────────────────────────────────────────
        if ssim_on and min(gp.shape[:2]) >= win:
            wsz = win if win % 2 == 1 else win + 1
            wsz = min(wsz, min(gp.shape[:2]) - (1 - min(gp.shape[:2]) % 2))
            try:
                s = float(ssim(gp, sp, win_size=max(3, wsz)))
            except ValueError:
                s = 1.0
            roi_ssim_min = r.ssim_min if r.ssim_min is not None else ssim_min
            res.ssim      = round(s, 4)
            res.ssim_pass = s >= roi_ssim_min
            if not res.ssim_pass:
                reasons.append(f"ssim {s:.2f} < {roi_ssim_min:.2f}")

        # ── 4. Per-segment check (single-digit ROIs only) ─────────────────────
        if seg_on and r.kind == "digit" and not r.n_digits and (bw / max(bh, 1)) <= seg_max_aspect:
            fails = missing_segments(sp, cfg)
            res.segment_fails = fails
            res.segment_pass  = len(fails) == 0
            if fails:
                reasons.append(f"segments off: {','.join(fails)}")

        # ── 4b. Per-sub-digit coverage check (multi-digit strip ROIs) ─────────
        elif seg_on and r.kind == "digit" and r.n_digits and r.n_digits > 1:
            sub_w = sp.shape[1] // r.n_digits
            digit_fails: list[str] = []
            for i in range(r.n_digits):
                gp_sub = gp[:, i * sub_w:(i + 1) * sub_w]
                sp_sub = sp[:, i * sub_w:(i + 1) * sub_w]
                if gp_sub.size == 0 or sp_sub.size == 0:
                    continue
                cov_g_sub = _coverage(gp_sub, block, cc, active_is_dark)
                cov_s_sub = _coverage(sp_sub, block, cc, active_is_dark)
                thresh_sub = cov_g_sub * roi_cov_ratio
                if cov_s_sub < thresh_sub:
                    digit_fails.append(f"d{i + 1}(cov={cov_s_sub:.2f}<{thresh_sub:.2f})")
            res.segment_fails = digit_fails
            res.segment_pass  = len(digit_fails) == 0
            if digit_fails:
                reasons.append(f"sub-digit fail: {';'.join(digit_fails)}")

        # ── 5. CNN defect check (digit ROIs only) ─────────────────────────────
        if cnn_on and r.kind == "digit" and sp.size > 0:
            use_golden = hasattr(cnn_scorer, "score_vs_golden")
            if r.n_digits and r.n_digits > 1:
                # Wide strip: split into sub-digits, take worst-case score
                sub_w = sp.shape[1] // r.n_digits
                sub_scores = []
                for i in range(r.n_digits):
                    sp_sub = sp[:, i * sub_w:(i + 1) * sub_w]
                    if sp_sub.size == 0:
                        continue
                    if use_golden:
                        gp_sub = gp[:, i * sub_w:(i + 1) * sub_w]
                        sub_scores.append(cnn_scorer.score_vs_golden(sp_sub, gp_sub))
                    else:
                        sub_scores.append(cnn_scorer.score(sp_sub))
                prob = max(sub_scores) if sub_scores else 0.0
            else:
                prob = cnn_scorer.score_vs_golden(sp, gp) if use_golden else cnn_scorer.score(sp)
            res.cnn_defect_prob = round(prob, 4)
            res.cnn_pass        = prob < cnn_thresh
            if not res.cnn_pass:
                reasons.append(f"cnn_defect_prob {prob:.3f} >= {cnn_thresh:.2f}")

        res.passed = (
            (res.coverage_pass  is not False) and
            (res.diff_pass      is not False) and
            (res.ssim_pass      is not False) and
            (res.segment_pass   is not False) and
            (res.cnn_pass       is not False)
        )
        res.reason = "; ".join(reasons)
        results.append(res)

    return results
