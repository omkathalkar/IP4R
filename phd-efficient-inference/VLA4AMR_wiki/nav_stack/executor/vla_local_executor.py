"""
Phase 4 — VLA local executor state machine.

Wraps FlowVLA-BW v3 (or any compatible VLA) in a state machine that
consumes a waypoint queue from the A* planner. Per-step, it:
  1. Computes the waypoint-grounded instruction via WaypointToVLAInputOffline
     (or the live version when a backbone is available).
  2. Runs FlowVLA-BW v3 inference to get (lin, ang).
  3. Checks waypoint-reached / stuck conditions.
  4. Returns (lin, ang, state) for the sim step.

The confidence signal (WP magnitude) used by Phase 5 (ConfidenceGate) is also
computed here and returned per step.

Designed to run standalone against scripted/replayed trajectories — no Isaac
Sim dependency.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import Waypoint
from interface.waypoint_to_vla_input import (
    WaypointToVLAInputOffline, WaypointToVLAInput, VLAInput,
)


class ExecutorState(Enum):
    NAVIGATE   = auto()   # actively pursuing current waypoint
    WP_REACHED = auto()   # just advanced to next waypoint (transient, 1 step)
    GOAL       = auto()   # final waypoint reached — episode complete
    STUCK      = auto()   # no progress for N steps — abort


@dataclass
class StepResult:
    lin: float
    ang: float
    state: ExecutorState
    wp_index: int                    # current target waypoint index
    wp_dist: float                   # metres to current waypoint
    wp_magnitude: float              # L2 norm of predicted (lin, ang) — confidence proxy
    instruction_key: str             # which vocab key was used
    vla_used: bool                   # True if VLA ran this step (False if upstream blocked)


class VLALocalExecutor:
    """
    State machine that drives the robot through a waypoint list.

    Parameters
    ----------
    vla_infer:       callable(feat_v, feat_t) → (lin, ang)
                     Pass None for offline / scripted testing (uses a stub).
    waypoints:       list[Waypoint] from A* planner.
    reach_threshold: metres — advance waypoint when within this distance.
    stuck_steps:     N steps of <stuck_dist_thresh progress → STUCK.
    stuck_dist_thresh: net displacement per step below this = "no progress".
    """

    def __init__(
        self,
        vla_infer,
        waypoints: list[Waypoint],
        wp_to_vla=None,
        reach_threshold: float = 0.3,
        stuck_steps: int = 30,
        stuck_dist_thresh: float = 0.05,
    ):
        if not waypoints:
            raise ValueError("waypoints list is empty")

        self._infer           = vla_infer
        self._waypoints       = waypoints
        self._wp_idx          = 0
        self._reach_thresh    = reach_threshold
        self._stuck_steps     = stuck_steps
        self._stuck_thresh    = stuck_dist_thresh
        self._state           = ExecutorState.NAVIGATE
        self._wp_to_vla       = wp_to_vla or WaypointToVLAInputOffline()

        # Stuck detection: rolling window of last N positions
        self._pos_history: deque[tuple[float, float]] = deque(maxlen=stuck_steps)
        self._step_count  = 0

    @property
    def state(self) -> ExecutorState:
        return self._state

    @property
    def done(self) -> bool:
        return self._state in (ExecutorState.GOAL, ExecutorState.STUCK)

    def _current_wp(self) -> Waypoint:
        return self._waypoints[self._wp_idx]

    def _is_final_wp(self) -> bool:
        return self._wp_idx == len(self._waypoints) - 1

    def _check_stuck(self, robot_x: float, robot_y: float) -> bool:
        self._pos_history.append((robot_x, robot_y))
        if len(self._pos_history) < self._stuck_steps:
            return False
        oldest_x, oldest_y = self._pos_history[0]
        net_disp = math.hypot(robot_x - oldest_x, robot_y - oldest_y)
        return net_disp < self._stuck_thresh * self._stuck_steps

    def step(
        self,
        camera_frame,            # np.ndarray BGR (H,W,3) or None for offline
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        feat_v: torch.Tensor | None = None,  # pre-extracted vision feature if available
    ) -> StepResult:
        """
        Execute one sim step. Returns StepResult with (lin, ang, state).

        camera_frame: not used directly here — caller should extract feat_v
                      from it using backbone.vlm.extract_feature() and pass as feat_v.
        feat_v:       (896,) tensor on model device. If None, uses zero vector (offline mode).
        """
        if self._state in (ExecutorState.GOAL, ExecutorState.STUCK):
            return StepResult(0.0, 0.0, self._state, self._wp_idx, 0.0, 0.0, "forward", False)

        self._step_count += 1
        wp = self._current_wp()

        # ── Waypoint-grounded instruction ─────────────────────────────────
        vla_input: VLAInput = self._wp_to_vla.step(
            robot_x, robot_y, robot_theta,
            wp.x, wp.y,
            is_final_wp=self._is_final_wp(),
        )

        # ── VLA inference ─────────────────────────────────────────────────
        if self._infer is not None and feat_v is not None:
            lin, ang = self._infer(feat_v, vla_input.feat_t)
            vla_used = True
        else:
            # Offline stub: proportional heading controller
            lin = float(np.clip(0.3, 0.0, 0.4))
            ang = float(np.clip(vla_input.rel_heading * 1.5, -1.0, 1.0))
            vla_used = False

        wp_magnitude = math.hypot(lin, ang)

        # ── Waypoint reached? ─────────────────────────────────────────────
        wp_dist = vla_input.rel_dist
        if wp_dist < self._reach_thresh:
            if self._is_final_wp():
                self._state = ExecutorState.GOAL
            else:
                self._wp_idx += 1
                self._state  = ExecutorState.WP_REACHED
        else:
            self._state = ExecutorState.NAVIGATE

        # ── Stuck detection ───────────────────────────────────────────────
        if self._check_stuck(robot_x, robot_y):
            self._state = ExecutorState.STUCK

        return StepResult(
            lin=lin,
            ang=ang,
            state=self._state,
            wp_index=self._wp_idx,
            wp_dist=wp_dist,
            wp_magnitude=wp_magnitude,
            instruction_key=vla_input.instruction_key,
            vla_used=vla_used,
        )

    def summary(self) -> dict:
        return {
            "steps": self._step_count,
            "waypoints_total": len(self._waypoints),
            "waypoints_reached": self._wp_idx,
            "final_state": self._state.name,
        }


# ── Convenience: load FlowVLA-BW v3 checkpoint and return an infer callable ──

def load_flowvla_v3(ckpt_path: str, device: str = "cuda:0"):
    """
    Load flowvla_v3_best.pt and return a callable:
        infer(feat_v: Tensor(896,), feat_t: Tensor(896,)) → (lin, ang)
    """
    import torch.nn as nn

    T_EMB_DIM = 64
    N_STEPS   = 20

    def sinusoidal_embed(t, dim=T_EMB_DIM):
        half  = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    class FlowActionHead2D(nn.Module):
        def __init__(self, feat_dim, hidden, t_emb_dim=T_EMB_DIM, n_layers=3):
            super().__init__()
            self.cond_enc = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden * 2), nn.SiLU(),
                nn.Linear(hidden * 2, hidden),
            )
            self.t_enc = nn.Sequential(
                nn.Linear(t_emb_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.action_in = nn.Linear(2, hidden)
            layers, in_dim = [], hidden * 3
            for i in range(n_layers):
                out_dim = hidden if i < n_layers - 1 else 2
                layers.append(nn.Linear(in_dim, out_dim))
                if i < n_layers - 1:
                    layers.append(nn.SiLU())
                in_dim = hidden
            self.denoiser = nn.Sequential(*layers)

        def forward(self, x_t, t, cond):
            h = torch.cat([self.action_in(x_t), self.cond_enc(cond), self.t_enc(sinusoidal_embed(t))], dim=-1)
            return self.denoiser(h)

        @torch.no_grad()
        def sample(self, cond, n_steps=N_STEPS):
            B = cond.shape[0]
            x = torch.randn(B, 2, device=cond.device)
            dt = 1.0 / n_steps
            for i in range(n_steps):
                t = torch.full((B,), 1.0 - i * dt, device=cond.device)
                x = x - self.forward(x, t, cond) * dt
            return x

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = FlowActionHead2D(cfg["feat_dim"], cfg["hidden"],
                             cfg["t_emb_dim"], cfg["n_layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feat_mean = torch.tensor(ckpt["feat_mean"], dtype=torch.float32).to(device)
    feat_std  = torch.tensor(ckpt["feat_std"],  dtype=torch.float32).to(device)
    act_mean  = torch.tensor(ckpt["act_mean"],  dtype=torch.float32).to(device)
    act_std   = torch.tensor(ckpt["act_std"],   dtype=torch.float32).to(device)

    def infer(feat_v: torch.Tensor, feat_t: torch.Tensor) -> tuple[float, float]:
        feat   = torch.cat([feat_v, feat_t]).unsqueeze(0)
        feat_n = (feat - feat_mean) / feat_std
        with torch.no_grad():
            pred_n = model.sample(feat_n)
        pred = (pred_n * act_std + act_mean).squeeze(0).cpu().tolist()
        lin = float(np.clip(pred[0], -0.5, 0.5))
        ang = float(np.clip(pred[1], -1.0,  1.0))
        return lin, ang

    return infer
