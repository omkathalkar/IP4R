"""Smoke tests for the IP4R pipeline. Run with: pytest"""
from __future__ import annotations

import cv2

from ip4r.config import Config
from ip4r.pipeline import Inspector
from ip4r.roi import load_rois
from ip4r.synth import make_defect


def _inspector() -> Inspector:
    return Inspector(Config.load())


def test_selfcheck_passes():
    """Golden compared to itself must PASS on every ROI."""
    insp = _inspector()
    res = insp.inspect_array(insp.golden_bgr.copy(), "golden")
    assert res.passed
    assert len(res.failed_rois) == 0
    assert len(res.roi_results) > 0


def test_erased_element_fails_and_localizes():
    """Erasing one element must FAIL exactly that element, nothing else."""
    cfg = Config.load()
    insp = Inspector(cfg)
    rois = load_rois(cfg.path("paths.roi_map"))
    defect, desc = make_defect(insp.golden_bgr.copy(), rois, mode="erase", seed=7)
    target = desc.split(":")[1]
    res = insp.inspect_array(defect, "defect")
    assert not res.passed
    failed = {r.name for r in res.failed_rois}
    assert target in failed
    assert failed == {target}, f"unexpected extra failures: {failed - {target}}"


def test_pose_shift_recovers():
    """A small pose shift must be recovered by registration and still PASS."""
    cfg = Config.load()
    insp = Inspector(cfg)
    defect, _ = make_defect(insp.golden_bgr.copy(), [], mode="shift", seed=5)
    res = insp.inspect_array(defect, "shift")
    assert res.passed, f"registration failed to recover; {len(res.failed_rois)} ROIs failed"
