"""Tests for Phase 8: eval suite metric helpers."""
import sys, math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval_suite import (
    mean_std, fmt_ms, action_to_displacement, heading_error_deg,
    ade_fde, gt_first_offset, agg_column,
)


# ── mean_std ──────────────────────────────────────────────────────────────────

def test_mean_std_empty():
    assert mean_std([]) == (None, None)


def test_mean_std_single():
    mu, sd = mean_std([3.0])
    assert mu == pytest.approx(3.0)
    assert sd == pytest.approx(0.0)


def test_mean_std_known():
    mu, sd = mean_std([1.0, 3.0])
    assert mu == pytest.approx(2.0)
    assert sd == pytest.approx(math.sqrt(2.0))


def test_mean_std_filters_none():
    mu, sd = mean_std([1.0, None, 3.0])
    assert mu == pytest.approx(2.0)


# ── fmt_ms ────────────────────────────────────────────────────────────────────

def test_fmt_ms_none():
    assert fmt_ms(None, None) == "—"


def test_fmt_ms_with_unit():
    s = fmt_ms(1.234, 0.056, 2, "°")
    assert "1.23" in s and "0.06" in s and "°" in s


# ── action_to_displacement ────────────────────────────────────────────────────

def test_disp_straight():
    fwd, lat = action_to_displacement(0.3, 0.0, dt=0.2)
    assert fwd == pytest.approx(0.06)
    assert lat == pytest.approx(0.0, abs=1e-9)


def test_disp_turning():
    fwd, lat = action_to_displacement(0.3, 1.0, dt=0.2)
    assert fwd == pytest.approx(0.06)
    assert lat == pytest.approx(0.3 * math.sin(0.2))


def test_disp_zero_lin():
    fwd, lat = action_to_displacement(0.0, 0.5, dt=0.2)
    assert fwd == pytest.approx(0.0)
    assert lat == pytest.approx(0.0)


# ── heading_error_deg ─────────────────────────────────────────────────────────

def test_heading_err_zero():
    err = heading_error_deg(1.0, 0.0, 1.0, 0.0)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_heading_err_90():
    # pred going east, GT going north
    err = heading_error_deg(1.0, 0.0, 0.0, 1.0)
    assert err == pytest.approx(90.0, abs=1.0)


def test_heading_err_180():
    # pred going east, GT going west
    err = heading_error_deg(1.0, 0.0, -1.0, 0.0)
    assert err == pytest.approx(180.0, abs=1.0)


def test_heading_err_symmetric():
    # error should be unsigned
    e1 = heading_error_deg(1.0, 0.0, 0.0, 1.0)
    e2 = heading_error_deg(0.0, 1.0, 1.0, 0.0)
    assert e1 == pytest.approx(e2, abs=0.1)


# ── ade_fde ───────────────────────────────────────────────────────────────────

def _gt_future(steps, dx_per_step, dy_per_step):
    return [{"offset": [dx_per_step * (i + 1), dy_per_step * (i + 1), 0.0]}
            for i in range(steps)]


def test_ade_fde_perfect():
    # pred exactly matches GT
    gt = _gt_future(5, 0.06, 0.0)
    a, f = ade_fde(0.06, 0.0, gt, horizon=5)
    assert a == pytest.approx(0.0, abs=1e-6)
    assert f == pytest.approx(0.0, abs=1e-6)


def test_ade_fde_constant_offset():
    # pred displaced by 0.1m in x each step, GT goes in same direction
    gt = _gt_future(5, 0.06, 0.0)
    a, f = ade_fde(0.16, 0.0, gt, horizon=5)
    # each step: cum pred = 0.16*(i+1), cum gt = 0.06*(i+1), dist = 0.10*(i+1)
    expected_ade = 0.10 * (1 + 2 + 3 + 4 + 5) / 5
    assert a == pytest.approx(expected_ade, rel=1e-4)
    assert f == pytest.approx(0.10 * 5, rel=1e-4)


def test_ade_fde_empty_gt():
    a, f = ade_fde(0.06, 0.0, [], horizon=5)
    assert a is None
    assert f is None


def test_ade_fde_horizon_clamp():
    gt = _gt_future(3, 0.06, 0.0)
    a, f = ade_fde(0.06, 0.0, gt, horizon=10)  # horizon > len(gt)
    # Should use min(10, 3)=3 steps
    assert a is not None
    assert f is not None


# ── gt_first_offset ───────────────────────────────────────────────────────────

def test_gt_first_offset_ok():
    gt = [{"offset": [0.05, 0.02, 0.0]}]
    r  = gt_first_offset(gt)
    assert r == pytest.approx((0.05, 0.02))


def test_gt_first_offset_zero():
    gt = [{"offset": [0.0, 0.0, 0.0]}]
    assert gt_first_offset(gt) is None


def test_gt_first_offset_empty():
    assert gt_first_offset([]) is None


# ── agg_column ────────────────────────────────────────────────────────────────

def _make_metrics(n, h=10.0, ade=0.5, fde=1.0, lat=0.3, used_vla=True):
    return [
        {"used_vla": used_vla,
         "hybrid_h": h, "hybrid_ade": ade, "hybrid_fde": fde, "hybrid_lat": lat,
         "vla_h":    h, "vla_ade":   ade,  "vla_fde":   fde,  "vla_lat":   lat,
         "pp_h":     h, "pp_ade":    ade,  "pp_fde":    fde}
        for _ in range(n)
    ]


def test_agg_column_counts():
    m = _make_metrics(5)
    col = agg_column(m, "hybrid_h", "hybrid_ade", "hybrid_fde", "hybrid_lat")
    assert col["n"] == 5


def test_agg_column_mean():
    m = _make_metrics(4, h=20.0, ade=1.0, fde=2.0, lat=0.4)
    col = agg_column(m, "hybrid_h", "hybrid_ade", "hybrid_fde", "hybrid_lat")
    assert col["heading_mu"] == pytest.approx(20.0)
    assert col["ade_mu"]     == pytest.approx(1.0)
    assert col["fde_mu"]     == pytest.approx(2.0)
    assert col["lat_mu"]     == pytest.approx(0.4)


def test_agg_column_std_zero():
    m = _make_metrics(3, h=15.0)
    col = agg_column(m, "hybrid_h", "hybrid_ade", "hybrid_fde")
    assert col["heading_sd"] == pytest.approx(0.0)


def test_agg_column_no_lat():
    m = _make_metrics(3)
    col = agg_column(m, "pp_h", "pp_ade", "pp_fde")  # no lat_key
    assert col["lat_mu"] is None


def test_agg_column_handles_none_values():
    m = [
        {"hybrid_h": None, "hybrid_ade": 0.5, "hybrid_fde": 1.0, "hybrid_lat": 0.3},
        {"hybrid_h": 10.0, "hybrid_ade": None, "hybrid_fde": 1.0, "hybrid_lat": 0.3},
    ]
    col = agg_column(m, "hybrid_h", "hybrid_ade", "hybrid_fde", "hybrid_lat")
    assert col["n"] == 1                              # n counts non-None heading entries
    assert col["heading_mu"] == pytest.approx(10.0)  # single non-None heading
    assert col["ade_mu"]     == pytest.approx(0.5)   # single non-None ADE
    assert col["fde_mu"]     == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
