# Isaac Sim

**Type:** architecture
**Status:** active
**Last updated:** 2026-06-15
**Related:** [[NovaCarter]], [[Isaac-Synthetic]], [[simulator-machine]]

## Summary

NVIDIA Isaac Sim 6.0.0.1 (conda env `isaac6`, Python 3.12). Primary simulation environment for VLA4AMR: Nova Carter in warehouse world, synthetic data generation, and closed-loop Nav2 evaluation. Isaac Sim 4.5.0 (`conda env: isaac`, Python 3.10) is deprecated — do not use.

## Working launch config (2026-06-15)

See [[simulator-machine]] for full command. Short summary:
- `DISPLAY=:1`, `no_window: True`, never set `CUDA_VISIBLE_DEVICES`
- Use direct Python path: `/home/cvit-car-simulator/miniconda3/envs/isaac6/bin/python -u`
- Driver `nvidia-driver-570-open` (570.211.01) — 595.x crashes Blackwell CC 12.0
- Startup time: ~7.8s (vs ~230s in 4.5.0)

## carter_warehouse_navigation.usd (BW03 scene)

**S3 path (6.0):** `{assets_root}/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd`
where `assets_root` = `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0`
(Note: `get_assets_root_path()` returns this automatically in isaac6 env.)

**Load result (2026-06-15):** 5,267 prims, 17 cameras, full scene verified ✅

**Camera paths — Nova Carter:**
| Camera | Prim path |
|--------|-----------|
| Front Hawk left | `/World/Nova_Carter_ROS/chassis_link/sensors/front_hawk/left/camera_left` |
| Front Hawk right | `/World/Nova_Carter_ROS/chassis_link/sensors/front_hawk/right/camera_right` |
| Left Hawk left/right | same pattern, `left_hawk/` |
| Right Hawk left/right | same pattern, `right_hawk/` |
| Back Hawk left/right | same pattern, `back_hawk/` |
| Warehouse overview | `/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime/Camera` |

**Correct async load polling — `get_stage_loading_status()` is broken in 6.0.0.1 (always 0/0).**
Use prim-count probe instead:
```python
omni.usd.get_context().open_stage(nav_usd)
for i in range(1200):
    app.update()
    stage = omni.usd.get_context().get_stage()
    if stage and stage.GetPrimAtPath('/World').IsValid():
        count = sum(1 for _ in stage.Traverse())
        if count > 100:
            print(f"Stage ready at iter {i}: {count} prims", flush=True)
            break
    if i % 60 == 0:
        print(f"[iter {i}] waiting...", flush=True)
```

## ROS2 bridge (confirmed working 2026-06-15)

`isaacsim.ros2.bridge` is installed in isaac6 `exts/` as part of `isaacsim-ros2==6.0.0.1`.
It is **NOT loaded by default** — must be explicitly enabled before opening any stage.

**Working pattern:**
```python
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "no_window": True, "renderer": "RayTracedLighting"})

import omni.kit.app
mgr = omni.kit.app.get_app().get_extension_manager()
mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
app.update()
# NOW open stage — OmniGraph ROS2 nodes will resolve correctly
```

**Load sequence (confirmed):**
- `isaacsim.ros2.core-1.9.1` → "Attempting to load system rclpy" → "rclpy loaded" ✅
- `isaacsim.ros2.nodes-1.18.11` → `isaacsim.ros2.bridge-5.1.1` ✅
- Bridge version: 5.1.1 (independent of isaac sim 6.0.0.1 package version)

**Uses system Jazzy rclpy** — NOT the pip-installed standalone ros2. Keep ROS_DISTRO, PYTHONPATH,
AMENT_PREFIX_PATH set (i.e., have sourced `/opt/ros/jazzy/setup.bash`). Python 3.12 match makes it seamless.

**What was wrong before:** `isaacsim-ros2-bridge` pip package doesn't exist — wrong name. The correct
package is `isaacsim-ros2==6.0.0.1` (already installed via `isaacsim[ros2]` or `isaacsim[all]`).

## Nova Carter status

- Asset: `{assets_root}/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd` ✅
- carter_warehouse_navigation.usd load: 5,267 prims ✅ (vs 2,141 for robot-only USD)
- ROS2 action graph: dormant — bridge not loaded (see above)
- See [[NovaCarter]] for full sensor details and camera prim paths

## Blocker history (resolved)

**4.5.0 → ROS2 bridge + Python version mismatch (resolved 2026-06-13):**
Isaac Sim 4.5.0 used Python 3.10. Jazzy ROS2 was compiled for Python 3.12.
The bridge's rclpy C extension (`_rclpy_pybind11.cpython-310`) could not load Jazzy's py3.12 packages.
**Fix:** Upgraded to Isaac Sim 6.0.0.1 (Python 3.12 = Jazzy Python). Bridge works natively once installed.

**4.5.0 → Blackwell CC 12.0 unsupported (resolved 2026-06-13):**
Kit SDK 106.5 predates Blackwell. Driver 595.x also incompatible.
**Fix:** Isaac Sim 6.0.0.1 has native CC 12.0 support. Driver 570.211.01 required (not 595.x).

## Sources
- [[simulator-machine]] — machine setup, driver config, install commands
- [[NovaCarter]] — sensor suite, camera paths
