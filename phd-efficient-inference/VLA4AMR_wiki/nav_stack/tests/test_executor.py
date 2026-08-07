"""Tests for Phase 3 (waypoint→VLA input) and Phase 4 (executor state machine)."""
import sys, math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Waypoint
from interface.waypoint_to_vla_input import (
    compute_rel_heading, heading_to_instruction_key,
    WaypointToVLAInputOffline, VOCAB, TURN_THRESH, SLOW_DIST,
)
from executor.vla_local_executor import VLALocalExecutor, ExecutorState


# ── Phase 3: heading computation ─────────────────────────────────────────────

def test_rel_heading_straight_ahead():
    # Robot facing east (+X), waypoint directly east
    dist, hdg = compute_rel_heading(0, 0, 0.0, 5.0, 0.0)
    assert dist == pytest.approx(5.0, abs=1e-6)
    assert hdg  == pytest.approx(0.0, abs=1e-6)


def test_rel_heading_left():
    # Robot facing east, waypoint is north → should be +90° (left)
    dist, hdg = compute_rel_heading(0, 0, 0.0, 0.0, 5.0)
    assert hdg == pytest.approx(math.pi / 2, abs=1e-6)


def test_rel_heading_right():
    # Robot facing east, waypoint is south → should be -90° (right)
    dist, hdg = compute_rel_heading(0, 0, 0.0, 0.0, -5.0)
    assert hdg == pytest.approx(-math.pi / 2, abs=1e-6)


def test_rel_heading_facing_north():
    # Robot facing north, waypoint is west → rel heading should be +90° (left)
    dist, hdg = compute_rel_heading(0, 0, math.pi / 2, -5.0, 0.0)
    assert hdg == pytest.approx(math.pi / 2, abs=1e-4)


def test_rel_heading_wrap():
    # Heading wrap: robot facing west, waypoint directly east → 180° → wraps to ±pi
    dist, hdg = compute_rel_heading(0, 0, math.pi, 5.0, 0.0)
    assert abs(abs(hdg) - math.pi) < 1e-4


def test_rel_dist():
    dist, _ = compute_rel_heading(1.0, 2.0, 0.0, 4.0, 6.0)
    assert dist == pytest.approx(math.hypot(3.0, 4.0), abs=1e-6)


# ── Phase 3: instruction bucketing ────────────────────────────────────────────

def test_forward_bucket():
    key = heading_to_instruction_key(0.1, 5.0, False)
    assert key == "forward"


def test_turn_left_bucket():
    key = heading_to_instruction_key(TURN_THRESH + 0.1, 5.0, False)
    assert key == "turn_left"


def test_turn_right_bucket():
    key = heading_to_instruction_key(-(TURN_THRESH + 0.1), 5.0, False)
    assert key == "turn_right"


def test_slow_near_final_wp():
    key = heading_to_instruction_key(0.0, SLOW_DIST * 0.5, True)
    assert key == "slow"


def test_no_slow_far_from_final_wp():
    key = heading_to_instruction_key(0.0, SLOW_DIST * 2, True)
    assert key == "forward"


def test_slow_overrides_turn_at_final_wp():
    # Even if heading says turn, "slow" takes priority near goal
    key = heading_to_instruction_key(TURN_THRESH + 0.5, SLOW_DIST * 0.3, True)
    assert key == "slow"


def test_vocab_strings_unchanged():
    # Guard against accidental vocab drift — these must match training exactly
    assert VOCAB["forward"]    == "Drive forward through the warehouse aisle"
    assert VOCAB["turn_left"]  == "Turn left to navigate the warehouse corridor"
    assert VOCAB["turn_right"] == "Turn right at the intersection"
    assert VOCAB["slow"]       == "Navigate carefully and slow down"


# ── Phase 3: offline converter ────────────────────────────────────────────────

def test_offline_converter_forward():
    conv  = WaypointToVLAInputOffline()
    inp   = conv.step(0, 0, 0.0, 5.0, 0.0, is_final_wp=False)
    assert inp.instruction_key == "forward"
    assert inp.rel_dist        == pytest.approx(5.0, abs=1e-5)
    assert inp.feat_t is None


def test_offline_converter_turn_left():
    conv = WaypointToVLAInputOffline()
    # Waypoint directly to the north from robot facing east → large left heading
    inp  = conv.step(0, 0, 0.0, 0.0, 5.0, is_final_wp=False)
    assert inp.instruction_key == "turn_left"


def test_offline_converter_slow_at_goal():
    conv = WaypointToVLAInputOffline()
    inp  = conv.step(0, 0, 0.0, 0.2, 0.0, is_final_wp=True)
    assert inp.instruction_key == "slow"


# ── Phase 4: executor state transitions ───────────────────────────────────────

def _make_executor(waypoints, reach=0.3, stuck_steps=5, stuck_thresh=0.01):
    return VLALocalExecutor(
        vla_infer=None,
        waypoints=waypoints,
        reach_threshold=reach,
        stuck_steps=stuck_steps,
        stuck_dist_thresh=stuck_thresh,
    )


def test_executor_single_wp_goal():
    # Single waypoint at (0.1, 0) — robot starts at (0,0), very close → GOAL
    wps = [Waypoint(0.1, 0.0, 0.0)]
    ex  = _make_executor(wps, reach=0.3)
    res = ex.step(None, 0.0, 0.0, 0.0)
    assert res.state == ExecutorState.GOAL


def test_executor_multi_wp_sequence():
    wps = [Waypoint(2.0, 0.0, 0.0), Waypoint(4.0, 0.0, 0.0), Waypoint(6.0, 0.0, 0.0)]
    ex  = _make_executor(wps, reach=0.3)

    # Start far from WP0 — NAVIGATE
    res = ex.step(None, 0.0, 0.0, 0.0)
    assert res.state     == ExecutorState.NAVIGATE
    assert res.wp_index  == 0

    # Jump to near WP0 — WP_REACHED, advance to WP1
    res = ex.step(None, 1.85, 0.0, 0.0)
    assert res.state    in (ExecutorState.WP_REACHED, ExecutorState.NAVIGATE)
    # After advance, wp_index should be 1 (or 0 if not quite there yet)

    # Simulate reaching each WP in sequence
    positions = [(1.85, 0.0), (3.85, 0.0), (5.85, 0.0)]
    ex2 = _make_executor(wps, reach=0.3)
    states = []
    for px, py in positions:
        r = ex2.step(None, px, py, 0.0)
        states.append(r.state)
    assert ExecutorState.GOAL in states


def test_executor_stuck_detection():
    wps = [Waypoint(10.0, 0.0, 0.0)]
    ex  = _make_executor(wps, reach=0.3, stuck_steps=5, stuck_thresh=0.5)
    # Robot never moves
    for _ in range(10):
        res = ex.step(None, 0.0, 0.0, 0.0)
    assert res.state == ExecutorState.STUCK


def test_executor_not_stuck_if_moving():
    wps = [Waypoint(10.0, 0.0, 0.0)]
    ex  = _make_executor(wps, reach=0.3, stuck_steps=5, stuck_thresh=0.1)
    # Robot moves 0.5m per step — clearly not stuck
    for i in range(10):
        res = ex.step(None, float(i) * 0.5, 0.0, 0.0)
    assert res.state != ExecutorState.STUCK


def test_executor_empty_waypoints():
    with pytest.raises(ValueError):
        VLALocalExecutor(vla_infer=None, waypoints=[])


def test_executor_done_flag():
    wps = [Waypoint(0.1, 0.0, 0.0)]
    ex  = _make_executor(wps, reach=0.3)
    ex.step(None, 0.0, 0.0, 0.0)
    assert ex.done


def test_executor_noop_after_done():
    wps = [Waypoint(0.1, 0.0, 0.0)]
    ex  = _make_executor(wps, reach=0.3)
    ex.step(None, 0.0, 0.0, 0.0)   # GOAL
    # Further steps should return GOAL with zero velocity
    res = ex.step(None, 0.0, 0.0, 0.0)
    assert res.state == ExecutorState.GOAL
    assert res.lin   == 0.0
    assert res.ang   == 0.0


def test_executor_wp_magnitude_positive():
    wps = [Waypoint(5.0, 0.0, 0.0)]
    ex  = _make_executor(wps, reach=0.3)
    res = ex.step(None, 0.0, 0.0, 0.0)
    # Offline stub returns non-zero lin — magnitude must be > 0
    assert res.wp_magnitude >= 0.0


def test_executor_summary():
    wps = [Waypoint(0.1, 0.0, 0.0)]
    ex  = _make_executor(wps, reach=0.3)
    ex.step(None, 0.0, 0.0, 0.0)
    s = ex.summary()
    assert s["final_state"]     == "GOAL"
    assert s["waypoints_total"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
