# C3 — Confidence-Gated Nav2 Handoff

**Type:** contribution
**Status:** active
**Last updated:** 2026-05-23
**Related:** [[C1-AdaCoT]], [[C2-MidLevelActionHead]], [[VLingNav]], [[Language-as-Cost]]

## Summary
Switch monitor node subscribes to /amcl_pose covariance, dynamic obstacle detector, and goal type. Publishes /control_mode. Arbitrator node forwards cmd_vel from Nav2 or VLA with velocity smoothing at transitions.

## Novelty claim
Novel architecture for selective VLA override of classical planners. Switch criterion design (threshold derivation + stability analysis) is itself a publishable component.

## Trigger conditions
1. AMCL covariance > σ_thresh (localisation uncertain)
2. Dynamic obstacle detected within safety radius
3. Goal expressed in natural language (not coordinates)

## ROS 2 nodes
- `switch_monitor` — subscribes: /amcl_pose, /obstacle_detection, /goal_type → publishes: /control_mode
- `arbitrator` — subscribes: /nav2/cmd_vel, /vla/cmd_vel, /control_mode → publishes: /cmd_vel (with velocity smoothing)

## Gap vs Language-as-Cost
[[Language-as-Cost]] uses VLM only as additional semantic cost in Nav2 costmap — Nav2 always in control.
Our C3 goes further: full VLA handoff in high-uncertainty zones. Switch criterion design is novel.

## Implementation plan (BW06)
