"""Tests for Phase 5: pure-pursuit controller and confidence gate."""
import sys, math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Waypoint
from control.pure_pursuit import PurePursuit
from control.confidence_gate import ConfidenceGate


# ── Pure Pursuit ──────────────────────────────────────────────────────────────

def _wps(*pts) -> list[Waypoint]:
    wps = []
    for i, (x, y) in enumerate(pts):
        nx, ny = pts[i + 1] if i + 1 < len(pts) else (x, y)
        theta = math.atan2(ny - y, nx - x)
        wps.append(Waypoint(x, y, theta))
    return wps


def test_pp_straight_ahead():
    # Robot at origin facing east (+X), path goes east
    pp  = PurePursuit(lookahead_dist=1.0, max_lin=0.3)
    wps = _wps((0, 0), (5, 0), (10, 0))
    lin, ang = pp.compute(0, 0, 0.0, wps, 0)
    assert lin > 0, "should move forward"
    assert abs(ang) < 0.05, f"should track straight, got ang={ang:.3f}"


def test_pp_turn_left():
    # Robot at (0,0) facing east, but path goes north
    pp  = PurePursuit(lookahead_dist=1.0, max_lin=0.3)
    wps = _wps((0, 0), (0, 5), (0, 10))
    lin, ang = pp.compute(0, 0, 0.0, wps, 0)
    assert lin > 0
    assert ang > 0, f"should steer left (positive ang), got {ang:.3f}"


def test_pp_turn_right():
    # Robot at (0,0) facing east, path goes south
    pp  = PurePursuit(lookahead_dist=1.0, max_lin=0.3)
    wps = _wps((0, 0), (0, -5), (0, -10))
    lin, ang = pp.compute(0, 0, 0.0, wps, 0)
    assert lin > 0
    assert ang < 0, f"should steer right (negative ang), got {ang:.3f}"


def test_pp_respects_max_ang():
    pp  = PurePursuit(lookahead_dist=1.0, max_lin=0.3, max_ang=0.8)
    wps = _wps((0, 0), (0, 5))
    lin, ang = pp.compute(0, 0, 0.0, wps, 0)
    assert abs(ang) <= 0.8 + 1e-9


def test_pp_respects_max_lin():
    pp  = PurePursuit(lookahead_dist=1.0, max_lin=0.25)
    wps = _wps((0, 0), (10, 0))
    lin, ang = pp.compute(0, 0, 0.0, wps, 0)
    assert lin <= 0.25 + 1e-9


def test_pp_min_lin_enforced():
    pp  = PurePursuit(lookahead_dist=1.0, min_lin=0.1, max_lin=0.3)
    wps = _wps((0, 0), (0, 5))   # large heading error → heading_scale reduced
    lin, ang = pp.compute(0, 0, 0.0, wps, 0)
    assert lin >= 0.1 - 1e-9, f"lin below min_lin: {lin}"


def test_pp_short_path_uses_last_wp():
    # Path much shorter than lookahead — should aim for last point
    pp  = PurePursuit(lookahead_dist=10.0, max_lin=0.3)
    wps = _wps((1, 0), (2, 0))
    lin, ang = pp.compute(0, 0, 0.0, wps, 0)
    assert lin > 0
    assert abs(ang) < 0.3   # mostly forward since goal is east


def test_pp_advances_wp_idx():
    # Robot far ahead — should use later waypoints
    pp  = PurePursuit(lookahead_dist=1.0, max_lin=0.3)
    # wp_idx=1 → start scanning from WP1
    wps = _wps((0, 0), (5, 0), (10, 0))
    lin, ang = pp.compute(4.5, 0, 0.0, wps, 1)
    assert lin > 0
    assert abs(ang) < 0.1


def test_pp_facing_away_from_path():
    # Robot facing west but path is east → large alpha → lots of turning
    pp  = PurePursuit(lookahead_dist=1.2, max_lin=0.3)
    wps = _wps((0, 0), (5, 0), (10, 0))
    lin, ang = pp.compute(0, 0, math.pi, wps, 0)  # facing west
    # Should still return valid velocities
    assert abs(lin) <= 0.3 + 1e-9
    assert abs(ang) <= 1.0 + 1e-9


# ── Confidence Gate ───────────────────────────────────────────────────────────

def _gate_step(gate, vla_lin, vla_ang, robot_x, robot_y,
               wp_x=10.0, wp_y=0.0, wp_idx=0):
    wps = [Waypoint(wp_x, wp_y, 0.0)]
    return gate.step(vla_lin, vla_ang, robot_x, robot_y, 0.0,
                     wp_x, wp_y, wp_idx, wps)


def test_gate_trusts_strong_vla():
    gate = ConfidenceGate(mag_thresh=0.15, progress_window=5, progress_min_m=0.05)
    # Robot moving steadily toward WP at x=10 — start at x=0, advance each step
    robot_x = 0.0
    for i in range(7):
        robot_x += 0.5   # closing in on WP at x=10
        lin, ang, used_vla, _ = _gate_step(gate, 0.33, 0.04, robot_x, 0.0)
    assert used_vla, "strong VLA with progress should be trusted"
    assert gate.vla_rate > 0.5


def test_gate_fallback_on_low_magnitude():
    gate = ConfidenceGate(mag_thresh=0.15, progress_window=3, progress_min_m=0.05)
    # VLA produces near-zero output (non-functional)
    lin, ang, used_vla, entry = _gate_step(gate, 0.001, 0.0, 5.0, 0.0)
    assert not used_vla
    assert not entry.mag_ok
    # Fallback should produce non-trivial forward motion
    assert lin >= 0.1


def test_gate_fallback_on_no_progress():
    gate = ConfidenceGate(mag_thresh=0.15, progress_window=5, progress_min_m=0.5)
    # VLA magnitude OK but robot not approaching WP (spinning in place)
    for _ in range(6):
        lin, ang, used_vla, entry = _gate_step(gate, 0.3, 0.8, 5.0, 0.0)
    assert not used_vla
    assert not entry.progress_ok


def test_gate_progress_window_insufficient_data():
    # Before progress_window fills, progress signal is True (benefit of doubt)
    gate = ConfidenceGate(mag_thresh=0.15, progress_window=20, progress_min_m=0.1)
    lin, ang, used_vla, entry = _gate_step(gate, 0.3, 0.0, 5.0, 0.0)
    assert entry.progress_ok   # window not full yet


def test_gate_both_signals_must_pass():
    gate = ConfidenceGate(mag_thresh=0.15, progress_window=5, progress_min_m=0.05)
    # Low magnitude + making progress → still fallback (mag fails)
    prev_x = 8.0
    for i in range(6):
        rx = prev_x - 0.3 * (i + 1)
        lin, ang, used_vla, _ = _gate_step(gate, 0.001, 0.0, rx, 0.0)
        prev_x = rx
    assert not used_vla   # magnitude killed it


def test_gate_resets_progress_window():
    gate = ConfidenceGate(mag_thresh=0.15, progress_window=5, progress_min_m=0.1)
    for _ in range(5):
        _gate_step(gate, 0.3, 0.0, 5.0, 0.0)   # fill window with stale data
    gate.reset_progress_window()
    # After reset window is empty → progress_ok is True again
    lin, ang, used_vla, entry = _gate_step(gate, 0.3, 0.0, 5.0, 0.0)
    assert entry.progress_ok


def test_gate_summary_counts():
    gate = ConfidenceGate(mag_thresh=0.15, progress_window=3, progress_min_m=0.05)
    for _ in range(4):
        _gate_step(gate, 0.33, 0.0, 5.0, 0.0)   # strong VLA
    _gate_step(gate, 0.001, 0.0, 5.0, 0.0)       # weak VLA → fallback
    s = gate.summary()
    assert s["steps_total"]   == 5
    assert s["steps_vla"]     >= 1
    assert s["steps_fallback"]>= 1
    assert abs(s["vla_rate"] + s["fallback_rate"] - 1.0) < 1e-9


def test_gate_log_length():
    gate = ConfidenceGate()
    wps  = [Waypoint(5.0, 0.0, 0.0)]
    for i in range(10):
        gate.step(0.3, 0.0, float(i) * 0.1, 0.0, 0.0, 5.0, 0.0, 0, wps)
    assert len(gate.log) == 10


def test_gate_log_as_dicts_schema():
    gate = ConfidenceGate()
    wps  = [Waypoint(5.0, 0.0, 0.0)]
    gate.step(0.33, 0.04, 2.0, 0.0, 0.0, 5.0, 0.0, 0, wps)
    d = gate.log_as_dicts()[0]
    for key in ("step", "robot_x", "robot_y", "wp_index", "vla_lin", "vla_ang",
                "vla_magnitude", "mag_ok", "progress_ok", "used_vla",
                "final_lin", "final_ang"):
        assert key in d, f"missing key: {key}"


def test_gate_straight_corridor_vla_rate():
    """
    Acceptance criterion: on a straight corridor with a functional VLA
    (lin≈0.33, ang≈0.04 — matches Phase 7 numbers), VLA mode > 90% of steps.
    """
    gate = ConfidenceGate(mag_thresh=0.15, progress_window=10, progress_min_m=0.05)
    wps  = [Waypoint(20.0, 0.0, 0.0)]
    robot_x = 0.0
    for step in range(60):
        robot_x += 0.33 * 0.2    # 5 Hz, 0.2s per step → 0.066m per step
        gate.step(0.33, 0.04, robot_x, 0.0, 0.0,
                  20.0, 0.0, 0, wps)
    assert gate.vla_rate > 0.90, f"VLA rate on corridor: {gate.vla_rate:.2%}"


def test_gate_ood_low_magnitude_trips_fallback():
    """
    Acceptance criterion: OOD scene gives low WP magnitude → gate trips to
    fallback rather than silently failing (mirrors Phase 8: 10% success,
    no STOP prediction from bare VLA).
    """
    gate = ConfidenceGate(mag_thresh=0.15)
    wps  = [Waypoint(5.0, 0.0, 0.0)]
    fallback_count = 0
    for _ in range(20):
        # Simulate OOD VLA: near-zero output like the random ActionExpert
        _, _, used_vla, _ = gate.step(0.0001, 0.0001, 1.0, 0.0, 0.0,
                                       5.0, 0.0, 0, wps)
        if not used_vla:
            fallback_count += 1
    assert fallback_count == 20, f"gate should always fallback on OOD magnitude, got {fallback_count}/20"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
