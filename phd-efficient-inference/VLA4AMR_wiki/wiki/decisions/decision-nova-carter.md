# Decision: Nova Carter over BCR Bot

**Type:** decision
**Status:** active
**Last updated:** 2026-05-23
**Related:** [[NovaCarter]], [[IsaacSim]]

## Decision
Migrate from BCR Bot (Gazebo) to Nova Carter (Isaac Sim) for VLA4AMR simulation.

## Rationale
BCR Bot on Gazebo: Nav2 + BCR Bot not reliably moving via Nav2 goals (TF chain issue). Nova Carter is NVIDIA's validated AMR with full Isaac integration, correct sensor suite (stereo RGB + LiDAR), and pre-validated Nav2 pipeline. Enables publishable simulation environment matching Bosch-class AMR deployment.
