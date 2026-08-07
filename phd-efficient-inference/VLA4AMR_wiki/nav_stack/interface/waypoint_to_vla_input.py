"""
Phase 3 — Waypoint → VLA input interface.

FlowVLA-BW v3 conditioning:
  cond = cat(feat_v, feat_t)  where feat_t = embed_tokens(instruction).mean(0)

The model was trained on exactly 4 instruction strings (auto-assigned from
action patterns). Any unseen string lands OOD. This module maps the relative
heading from current pose → next waypoint onto the correct training vocabulary
string, using the same bucketing as the training label_frame() function.

feat_t is updated once per waypoint advance (cheap: tokenize + embed → mean),
not once per sim step.
"""

import math
import torch
from dataclasses import dataclass


# Matches the INSTRUCTIONS dict in flowvla_v3_train.py exactly.
VOCAB = {
    "forward":    "Drive forward through the warehouse aisle",
    "turn_left":  "Turn left to navigate the warehouse corridor",
    "turn_right": "Turn right at the intersection",
    "slow":       "Navigate carefully and slow down",
}

# Heading thresholds (radians). |Δθ| >= TURN_THRESH → turn instruction.
# Mirrors the ang_vel >= 0.5 threshold used during training auto-labeling,
# but applied to the geometric heading error rather than the commanded ang_vel.
TURN_THRESH = 0.35   # ~20°
SLOW_DIST   = 0.8    # metres — switch to "slow" when within this of goal


@dataclass
class VLAInput:
    instruction_key: str    # "forward" | "turn_left" | "turn_right" | "slow"
    instruction_str: str    # full sentence passed to tokenizer
    rel_dist: float         # metres to next waypoint
    rel_heading: float      # radians, signed (+= left, -= right)
    feat_t: torch.Tensor    # (896,) text embedding on the model's device


def heading_to_instruction_key(rel_heading: float, rel_dist: float, is_final_wp: bool) -> str:
    """
    Map relative heading + distance to a vocabulary key.

    rel_heading: signed angle from robot's current heading to the direction
                 of the next waypoint. Positive = CCW = left, Negative = CW = right.
    rel_dist:    distance to the next waypoint in metres.
    is_final_wp: True when this is the last waypoint (goal) → switch to slow
                 when close.
    """
    if is_final_wp and rel_dist < SLOW_DIST:
        return "slow"
    if rel_heading > TURN_THRESH:
        return "turn_left"
    if rel_heading < -TURN_THRESH:
        return "turn_right"
    return "forward"


def compute_rel_heading(robot_x: float, robot_y: float, robot_theta: float,
                        wp_x: float, wp_y: float) -> tuple[float, float]:
    """
    Compute (rel_dist, rel_heading) from the robot's current pose to a waypoint.

    rel_heading is in (-pi, pi]: positive = left (CCW), negative = right (CW).
    """
    dx = wp_x - robot_x
    dy = wp_y - robot_y
    rel_dist = math.hypot(dx, dy)
    abs_heading = math.atan2(dy, dx)
    rel_heading = abs_heading - robot_theta
    # Wrap to (-pi, pi]
    rel_heading = (rel_heading + math.pi) % (2 * math.pi) - math.pi
    return rel_dist, rel_heading


class WaypointToVLAInput:
    """
    Stateful converter: holds the InternVL3-1B tokenizer + embed_tokens layer
    and caches feat_t for the current instruction. feat_t is recomputed only
    when the instruction key changes (i.e. at a waypoint transition).
    """

    def __init__(self, backbone, tokenizer, device: str):
        """
        backbone: loaded TICVLA model (needs backbone.vlm.language_model.model.embed_tokens)
        tokenizer: AutoTokenizer loaded from internvl3-1b
        device: "cuda:0" etc.
        """
        self._embed = backbone.vlm.language_model.model.embed_tokens
        self._tok   = tokenizer
        self._dev   = device
        self._current_key: str | None = None
        self._current_feat_t: torch.Tensor | None = None

    def _compute_feat_t(self, instruction_str: str) -> torch.Tensor:
        ids = self._tok(instruction_str, return_tensors="pt",
                        add_special_tokens=True).input_ids.to(self._dev)
        with torch.no_grad():
            feat_t = self._embed(ids)[0].float().mean(0)   # (896,)
        return feat_t

    def step(
        self,
        robot_x: float, robot_y: float, robot_theta: float,
        wp_x: float, wp_y: float,
        is_final_wp: bool = False,
    ) -> VLAInput:
        """
        Called each sim step. Recomputes feat_t only when the instruction
        key changes (waypoint advance or heading bucket crosses threshold).
        """
        rel_dist, rel_heading = compute_rel_heading(robot_x, robot_y, robot_theta, wp_x, wp_y)
        key = heading_to_instruction_key(rel_heading, rel_dist, is_final_wp)

        if key != self._current_key:
            self._current_key    = key
            self._current_feat_t = self._compute_feat_t(VOCAB[key])

        return VLAInput(
            instruction_key=key,
            instruction_str=VOCAB[key],
            rel_dist=rel_dist,
            rel_heading=rel_heading,
            feat_t=self._current_feat_t,
        )


class WaypointToVLAInputOffline:
    """
    Offline version — no backbone/tokenizer dependency.
    Returns VLAInput with feat_t=None. For testing Phase 3 logic
    without loading InternVL3-1B.
    """

    def step(
        self,
        robot_x: float, robot_y: float, robot_theta: float,
        wp_x: float, wp_y: float,
        is_final_wp: bool = False,
    ) -> VLAInput:
        rel_dist, rel_heading = compute_rel_heading(robot_x, robot_y, robot_theta, wp_x, wp_y)
        key = heading_to_instruction_key(rel_heading, rel_dist, is_final_wp)
        return VLAInput(
            instruction_key=key,
            instruction_str=VOCAB[key],
            rel_dist=rel_dist,
            rel_heading=rel_heading,
            feat_t=None,
        )
