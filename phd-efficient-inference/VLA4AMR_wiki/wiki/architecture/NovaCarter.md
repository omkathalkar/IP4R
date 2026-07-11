# Nova Carter

**Type:** architecture
**Status:** active
**Last updated:** 2026-06-15
**Related:** [[IsaacSim]], [[simulator-machine]], [[decision-nova-carter]]

## Summary
NVIDIA reference AMR. Hawk stereo + 3D LiDAR. Full Isaac Sim integration. Replaces BCR Bot.
In carter_warehouse_navigation.usd: appears at `/World/Nova_Carter_ROS`, 5,267 total scene prims.

## Why Nova Carter
- Isaac-native, plug-and-play, sensor suite matches VLA input needs
- Bosch ACTIVE Shuttle class — same category as deployment target
- Full Nav2 + RViz2 pre-validated by NVIDIA

## Camera prim paths (confirmed 2026-06-15)

Primary camera for VLA input: **front Hawk left** — `/World/Nova_Carter_ROS/chassis_link/sensors/front_hawk/left/camera_left`

| Rig | Left | Right |
|-----|------|-------|
| front_hawk | `.../front_hawk/left/camera_left` | `.../front_hawk/right/camera_right` |
| left_hawk | `.../left_hawk/left/camera_left` | `.../left_hawk/right/camera_right` |
| right_hawk | `.../right_hawk/left/camera_left` | `.../right_hawk/right/camera_right` |
| back_hawk | `.../back_hawk/left/camera_left` | `.../back_hawk/right/camera_right` |

Base path: `/World/Nova_Carter_ROS/chassis_link/sensors/`

17 camera prims total in scene (includes warehouse overview camera at `/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime/Camera`).
