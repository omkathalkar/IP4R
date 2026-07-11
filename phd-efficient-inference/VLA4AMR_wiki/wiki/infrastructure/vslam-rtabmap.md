# VSLAM — RTAB-Map Stereo SLAM on Isaac Sim

**Type:** infrastructure
**Status:** active
**Last updated:** 2026-06-24
**Related:** [[simulator-machine]], [[C3-ConfidenceGatedHandoff]], [[C6-EvaluationProtocol]], [[NovaCarter]], [[IsaacSim]]

## Summary

Stereo visual SLAM using RTAB-Map v0.22.1 (ROS2 Jazzy) receiving front hawk stereo camera frames from Nova Carter in Isaac Sim 6.0.0.1 headless. Publishes `/rtabmap/odom` (camera-frame odometry) and saves a 3D map database. Used to generate VSLAM pose estimates for the C3 confidence gate and C6 evaluation protocol.

## Setup

### Isaac Sim side (`bw09_isaac_sim_ros2.py`)
- Scene: `{ASSETS}/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd`
- Robot: `{ASSETS}/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd` (ROS2 OmniGraph variant)
- ROS2 bridge enabled via `omni.kit.app.get_extension_manager().set_extension_enabled_immediate("isaacsim.ros2.bridge", True)`
- `/clock` published via OmniGraph: `ROS2PublishClock` + `IsaacReadSimulationTime` — **must be created AFTER `open_stage()`**
- Image writers: `LdrColorSDROS2PublishImage` on `rp_left`/`rp_right` (640×480 RGB)
- Camera info: **from Carter's built-in OmniGraph only** — do NOT attach `ROS2PublishCameraInfo` replicator writers (publishes empty messages)
- Carter moved via USD session layer translate (`xformOp:translate`) at 0.4 m/s

### Topics published by Isaac Sim
| Topic | Type | Notes |
|-------|------|-------|
| `/front_stereo_camera/left/image_rect_color` | `sensor_msgs/Image` | rgb8, 640×480 |
| `/front_stereo_camera/right/image_rect_color` | `sensor_msgs/Image` | rgb8, 640×480 |
| `/front_stereo_camera/left/camera_info` | `sensor_msgs/CameraInfo` | from Carter OmniGraph |
| `/front_stereo_camera/right/camera_info` | `sensor_msgs/CameraInfo` | from Carter OmniGraph |
| `/clock` | `rosgraph_msgs/Clock` | sim time, enables `use_sim_time=true` |

### RTAB-Map side (`bw09_launch_rtabmap.py`)
- `use_sim_time=true` (requires `/clock`)
- `frame_id='front_stereo_camera_left_optical'` — no `base_link→camera` TF exists since Carter physics is inactive
- Static TF: `front_stereo_camera_left_optical → front_stereo_camera_right_optical` at (0.12, 0, 0) — Nova Carter hawk baseline = 12 cm
- stereo_odometry remaps output: `('odom', '/rtabmap/odom')` so rtabmap can sync odom + images
- Parameters: `Vis/MinInliers=5`, `Odom/ResetCountdown=5`, `Odom/Strategy=0` (Frame-to-Map)
- Map DB: `/tmp/vla4amr_warehouse_map.db`

## Run command

```bash
source /opt/ros/jazzy/setup.bash && CUDA_VISIBLE_DEVICES=0 DISPLAY=:1 \
    /home/cvit-car-simulator/miniconda3/envs/isaac6/bin/python -u \
    ~/VLA4AMR/code/bw09_isaac_sim_ros2.py > ~/Desktop/bw09_ros2.log 2>&1
```

RTAB-Map is launched automatically as a subprocess inside `bw09_isaac_sim_ros2.py`.

## Known Issues

- Stereo left/right frame timestamps can differ by ~33ms (2 frames) — use `approx_sync=true`
- Carter kinematic teleport (session layer translate) causes occasional feature tracking resets — odometry quality oscillates 5–25 inliers
- The replicator `ROS2PublishCameraInfo` writer publishes `width=0, height=0` (empty) — only Carter's OmniGraph camera_info works

## Key Bug History

1. `No module named 'omni.isaac.core'` — deprecated in Isaac Sim 6.x
2. `Carter children (0): []` — wrong Carter USD path (correct: `Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd`)
3. `/clock Publisher count: 0` — OmniGraph created before `open_stage()` was wiped by stage load
4. `use_sim_time=true` freezing — no `/clock` publisher (fix: OmniGraph ROS2PublishClock after stage load)
5. `stereo baseline (0.000000)` — `ROS2PublishCameraInfo` writer has empty CameraInfo; removed it
6. TF extrapolation error — timestamp mismatch (sim time vs wall clock) fixed by publishing `/clock`
7. `odom` topic mismatch — stereo_odometry publishes to `/odom`, rtabmap needs `/rtabmap/odom`

## Open Questions

- Can Carter be moved via physics (ArticulationController) instead of session layer to get smoother motion and TF tree?
- Can the RTAB-Map map DB be converted to a Nav2 costmap for planning?
- Does the 12 cm baseline assumption match actual Nova Carter USD camera positions?

## Sources

- [[simulator-machine]] — hardware specs
- Isaac Sim 6.0.0.1 ROS2 bridge documentation
- RTAB-Map ROS2 wiki (rtabmap-ros v0.22.1)
