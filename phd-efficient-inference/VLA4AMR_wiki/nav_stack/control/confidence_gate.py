"""
Phase 5 — Confidence gate (C3 contribution).

Decides per-step whether to trust the VLA's (lin, ang) or fall back to
pure-pursuit. Two complementary signals:

  1. WP-magnitude threshold — `sqrt(lin² + ang²)` below mag_thresh signals
     the VLA is producing near-zero, non-committed output (matches the
     "near-zero when non-functional" finding from Phase 8/9 eval where
     random ActionExpert gave magnitude ~0.0001 vs 0.46 for trained weights).

  2. Progress check — if the robot has not made net forward progress toward
     the current waypoint over the last `progress_window` steps, the VLA is
     stuck or spinning in place.  This catches the case where WP magnitude
     is plausibly large but misdirected.

Either signal tripping → gate selects pure-pursuit for that step.

Per-step log entry (list of dicts) is accumulated for the paper contribution
analysis: how often / where does the gate trigger, does it correlate with OOD
regions?
"""

import math
from collections import deque
from dataclasses import dataclass, field

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Waypoint
from control.pure_pursuit import PurePursuit


@dataclass
class GateLogEntry:
    step: int
    robot_x: float
    robot_y: float
    wp_index: int
    vla_lin: float
    vla_ang: float
    vla_magnitude: float
    progress_m: float           # net approach distance over window (m)
    mag_ok: bool                # True = magnitude above threshold
    progress_ok: bool           # True = making progress
    used_vla: bool
    final_lin: float
    final_ang: float


class ConfidenceGate:
    """
    Parameters
    ----------
    mag_thresh       : WP-magnitude floor. Calibrated from Phase 7/8/9:
                       trained VLA → ~0.46, non-functional → ~0.0001.
                       Default 0.15 gives comfortable margin.
    progress_window  : steps to look back for progress check.
    progress_min_m   : net approach distance over the window to count as progress.
    pure_pursuit     : PurePursuit instance (created with defaults if None).
    """

    def __init__(
        self,
        mag_thresh: float = 0.15,
        progress_window: int = 20,
        progress_min_m: float = 0.1,
        pure_pursuit: PurePursuit | None = None,
    ):
        self._mag_thresh   = mag_thresh
        self._prog_window  = progress_window
        self._prog_min     = progress_min_m
        self._pp           = pure_pursuit or PurePursuit()

        # Rolling window of (dist_to_wp) for the progress check
        self._dist_history: deque[float] = deque(maxlen=progress_window)
        self._step = 0
        self.log: list[GateLogEntry] = []

        # Running counts for quick summary
        self._n_vla       = 0
        self._n_fallback  = 0

    def step(
        self,
        vla_lin: float,
        vla_ang: float,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        wp_x: float,
        wp_y: float,
        wp_index: int,
        waypoints: list[Waypoint],
    ) -> tuple[float, float, bool, GateLogEntry]:
        """
        Decide whether to use VLA or pure-pursuit for this sim step.

        Returns
        -------
        (fin_lin, fin_ang, used_vla, log_entry)
        """
        self._step += 1
        dist_to_wp = math.hypot(wp_x - robot_x, wp_y - robot_y)
        self._dist_history.append(dist_to_wp)

        # ── Signal 1: WP magnitude ────────────────────────────────────────
        magnitude = math.hypot(vla_lin, vla_ang)
        mag_ok    = magnitude >= self._mag_thresh

        # ── Signal 2: Progress ────────────────────────────────────────────
        if len(self._dist_history) < self._prog_window:
            progress_ok = True   # not enough history yet — give benefit of doubt
            progress_m  = float("nan")
        else:
            oldest_dist = self._dist_history[0]
            progress_m  = oldest_dist - dist_to_wp   # positive = getting closer
            progress_ok = progress_m >= self._prog_min

        # ── Gate decision ─────────────────────────────────────────────────
        used_vla = mag_ok and progress_ok

        if used_vla:
            fin_lin, fin_ang = vla_lin, vla_ang
            self._n_vla += 1
        else:
            fin_lin, fin_ang = self._pp.compute(robot_x, robot_y, robot_theta, waypoints, wp_index)
            self._n_fallback += 1

        entry = GateLogEntry(
            step=self._step,
            robot_x=robot_x, robot_y=robot_y,
            wp_index=wp_index,
            vla_lin=vla_lin, vla_ang=vla_ang,
            vla_magnitude=magnitude,
            progress_m=progress_m,
            mag_ok=mag_ok,
            progress_ok=progress_ok,
            used_vla=used_vla,
            final_lin=fin_lin, final_ang=fin_ang,
        )
        self.log.append(entry)
        return fin_lin, fin_ang, used_vla, entry

    def reset_progress_window(self):
        """Call when advancing to a new waypoint — clears stale distance history."""
        self._dist_history.clear()

    @property
    def vla_rate(self) -> float:
        """Fraction of steps where VLA was trusted (0–1)."""
        total = self._n_vla + self._n_fallback
        return self._n_vla / total if total > 0 else float("nan")

    @property
    def fallback_rate(self) -> float:
        total = self._n_vla + self._n_fallback
        return self._n_fallback / total if total > 0 else float("nan")

    def summary(self) -> dict:
        return {
            "steps_total":    self._step,
            "steps_vla":      self._n_vla,
            "steps_fallback": self._n_fallback,
            "vla_rate":       round(self.vla_rate, 4),
            "fallback_rate":  round(self.fallback_rate, 4),
        }

    def log_as_dicts(self) -> list[dict]:
        return [
            {
                "step":          e.step,
                "robot_x":       round(e.robot_x, 3),
                "robot_y":       round(e.robot_y, 3),
                "wp_index":      e.wp_index,
                "vla_lin":       round(e.vla_lin, 4),
                "vla_ang":       round(e.vla_ang, 4),
                "vla_magnitude": round(e.vla_magnitude, 4),
                "progress_m":    round(e.progress_m, 4) if not math.isnan(e.progress_m) else None,
                "mag_ok":        e.mag_ok,
                "progress_ok":   e.progress_ok,
                "used_vla":      e.used_vla,
                "final_lin":     round(e.final_lin, 4),
                "final_ang":     round(e.final_ang, 4),
            }
            for e in self.log
        ]
