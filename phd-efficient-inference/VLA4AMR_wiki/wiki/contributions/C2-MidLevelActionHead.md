# C2 — Mid-Level Language Action Head

**Type:** contribution
**Status:** active
**Last updated:** 2026-05-23
**Related:** [[OpenVLA-OFT]], [[NaVILA]], [[C3-ConfidenceGatedHandoff]], [[C4-FusionAblation]]

## Summary
Replace discrete token prediction head with a structured language action decoder outputting (direction, distance, context) tuples. Passed to Nav2 as semantic waypoints via a custom ROS 2 message type. Based on the OFT recipe from [[OpenVLA-OFT]].

## Novelty claim
Extends NaVILA's mid-level action concept from legged robots to wheeled AMR with Nav2 integration. Auditable action trace for Bosch deployment requirements.

## Technical design
```
Camera frame → VLA backbone → L1 regression head
→ output: (direction: str, distance: float, context: str)
→ custom ROS 2 msg: VLAWaypoint.msg
→ Nav2 global planner as semantic waypoint
```

## OFT recipe applied
- Parallel decoding → all action dimensions in one forward pass
- K=8 chunking → 8 future waypoints per inference step
- L1 regression head → continuous (direction, distance, context) values
- LoRA → parameter-efficient, ~500 demos sufficient

## Advantage over raw Twist output
Raw cmd_vel (linear_x, angular_z) is uninterpretable. Mid-level tuples give:
- Auditable action trace ("turn left 1.2m, avoid forklift")
- Direct Nav2 integration via waypoints
- Bosch deployment requirement: reasoning transparency

## Implementation plan (BW05)
1. Replace OFT L1 head with structured decoder
2. Define VLAWaypoint.msg (direction: string, distance: float32, context: string)
3. Write Nav2 waypoint publisher node
4. Test end-to-end: camera → VLA → Nav2 waypoint → robot motion
