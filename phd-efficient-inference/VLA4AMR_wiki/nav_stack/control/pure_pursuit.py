"""
Phase 5 — Pure-pursuit controller fallback.

Classic pure-pursuit: given the robot's current pose and the A* waypoint path,
find the lookahead point on the path and compute (lin, ang) to reach it.

Reference: Coulter 1992, "Implementation of the Pure Pursuit Path Tracking Algorithm".
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Waypoint


class PurePursuit:
    """
    Parameters
    ----------
    lookahead_dist : metres ahead on the path to target (tune: 1–3× robot length).
                     NovaCarter length ≈ 0.8m → start at 1.2m.
    max_lin        : m/s cap on linear velocity.
    max_ang        : rad/s cap on angular velocity.
    min_lin        : m/s — slow but always move forward (avoids zero-velocity stall).
    """

    def __init__(
        self,
        lookahead_dist: float = 1.2,
        max_lin: float = 0.35,
        max_ang: float = 1.0,
        min_lin: float = 0.1,
    ):
        self.ld      = lookahead_dist
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.min_lin = min_lin

    def _find_lookahead(
        self,
        robot_x: float, robot_y: float,
        waypoints: list[Waypoint],
        wp_idx: int,
    ) -> tuple[float, float]:
        """
        Walk the path from wp_idx forward until we find a point that is
        ≥ lookahead_dist from the robot.  If the path ends before that,
        return the final waypoint.

        Returns world (lx, ly) of the lookahead point.
        """
        # Start from current target waypoint and scan forward
        for i in range(wp_idx, len(waypoints)):
            wx, wy = waypoints[i].x, waypoints[i].y
            if math.hypot(wx - robot_x, wy - robot_y) >= self.ld:
                return wx, wy
        # Path shorter than lookahead — aim for last waypoint
        return waypoints[-1].x, waypoints[-1].y

    def compute(
        self,
        robot_x: float, robot_y: float, robot_theta: float,
        waypoints: list[Waypoint],
        wp_idx: int,
    ) -> tuple[float, float]:
        """
        Compute (lin, ang) to track the path via pure pursuit.

        Pure-pursuit geometry (robot frame):
          α  = angle from robot heading to lookahead point
          κ  = 2 sin(α) / ld        (signed curvature)
          ang = lin * κ

        lin is reduced when |α| is large (we need to turn more than drive).
        """
        lx, ly = self._find_lookahead(robot_x, robot_y, waypoints, wp_idx)

        # Transform lookahead point into robot frame
        dx = lx - robot_x
        dy = ly - robot_y
        # Rotate by -robot_theta
        local_x =  dx * math.cos(robot_theta) + dy * math.sin(robot_theta)
        local_y = -dx * math.sin(robot_theta) + dy * math.cos(robot_theta)

        # Signed angle from robot heading to lookahead direction
        alpha = math.atan2(local_y, max(local_x, 1e-3))

        # Pure-pursuit curvature
        dist_to_la = math.hypot(local_x, local_y)
        if dist_to_la < 1e-6:
            return self.min_lin, 0.0

        kappa = 2.0 * math.sin(alpha) / max(dist_to_la, 1e-3)

        # Scale lin down when heading error is large (prioritise turning)
        heading_scale = max(0.3, 1.0 - abs(alpha) / math.pi)
        lin = float(max(self.min_lin, self.max_lin * heading_scale))
        ang = float(lin * kappa)

        lin = float(max(self.min_lin, min(self.max_lin, lin)))
        ang = float(max(-self.max_ang, min(self.max_ang, ang)))
        return lin, ang
