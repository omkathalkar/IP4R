#!/usr/bin/env python3
"""
wms_step3_vla_drive.py — Step 3: VLA drives Nova Carter on warehouse_20x20.usd.

300 action steps (60s sim), VLA output applied via og.Controller.set().
Two cameras:
  - Carter front hawk camera → fed to VLA for inference
  - Static world-space camera (fixed LookAt, no per-frame update) → compiled into demo.mp4
"""

import sys, os, time, signal, math, json, datetime
import numpy as np
from pathlib import Path

SCENE_PATH   = os.path.expanduser("~/TIC-VLA/DynaNav/assets/warehouse_20x20/warehouse_20x20.usd")
CARTER_URL   = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/6.0/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
)
CARTER_PRIM_PATH = "/World/Nova_Carter_ROS"
# Camera fed to VLA (Carter's sensor)
VLA_CAMERA_PATH = f"{CARTER_PRIM_PATH}/chassis_link/sensors/front_hawk/left/camera_left"
# Static world-space camera for demo video (positioned via LookAt, not attached to Carter)
DEMO_CAM_PATH   = "/World/DemoCam"

IPC_DIR      = "/tmp/navstack_ipc"
FRAME_IN     = os.path.join(IPC_DIR, "frame.jpg")
FRAME_RD     = os.path.join(IPC_DIR, "frame.ready")
INSTR_FILE   = os.path.join(IPC_DIR, "current_instruction.txt")
ROBOT_STATE  = os.path.join(IPC_DIR, "robot_state.json")
ACT_OUT      = os.path.join(IPC_DIR, "action.json")
ACT_RD       = os.path.join(IPC_DIR, "action.ready")
QUIT_F       = os.path.join(IPC_DIR, "quit")

INSTRUCTION  = "Drive forward to aisle six"
GOAL_X, GOAL_Y = 2.0, 17.0   # aisle_06 goal position (must match TARGETS in large_collect.py)
SIM_HZ       = 60
SEND_HZ      = 5
SEND_STEP    = SIM_HZ // SEND_HZ
ACTION_STEPS = 300
TOTAL_SIM    = ACTION_STEPS * SEND_STEP

ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.expanduser(f"~/Desktop/wms_step3_aisle6/{ts}")
FRM_DIR = os.path.join(OUT_DIR, "frames")       # chase-cam frames for video
VLA_DIR = os.path.join(OUT_DIR, "vla_frames")   # front-cam frames fed to VLA
VID_OUT = os.path.join(OUT_DIR, "demo.mp4")
LOG_OUT = os.path.join(OUT_DIR, "step3_log.jsonl")
os.makedirs(FRM_DIR, exist_ok=True)
os.makedirs(VLA_DIR, exist_ok=True)
os.makedirs(IPC_DIR, exist_ok=True)

print("=" * 64)
print("Step 3 — VLA-driven Nova Carter on warehouse_20x20.usd")
print(f"  Instruction  : {INSTRUCTION!r}")
print(f"  Action steps : {ACTION_STEPS}  ({ACTION_STEPS/SEND_HZ:.0f}s sim time)")
print(f"  Output       : {OUT_DIR}")
print("=" * 64)

from isaacsim import SimulationApp
app = SimulationApp({
    "headless":   True,
    "no_window":  True,
    "multi_gpu":  False,
    "active_gpu": 1,
})

import omni.usd, omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, Gf

print("Loading warehouse_20x20.usd ...")
omni.usd.get_context().open_stage(SCENE_PATH)
stage = None
for i in range(1000):
    app.update()
    s = omni.usd.get_context().get_stage()
    if s and sum(1 for _ in s.Traverse()) > 30:
        stage = s; break
    if i % 200 == 0:
        print(f"  [{i}] waiting...", flush=True)
    time.sleep(0.05)
if stage is None:
    print("ERROR: stage failed to load"); app.close(); sys.exit(1)
print(f"  Loaded ({sum(1 for _ in stage.Traverse())} prims)")

# ── Spawn Carter ───────────────────────────────────────────────────────────────
from isaacsim.core.utils.stage import add_reference_to_stage
SPAWN_X, SPAWN_Y, SPAWN_Z = 5.0, -8.0, 0.15
add_reference_to_stage(usd_path=CARTER_URL, prim_path=CARTER_PRIM_PATH)
carter_prim = stage.GetPrimAtPath(CARTER_PRIM_PATH)
xf = UsdGeom.Xformable(carter_prim)
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set(Gf.Vec3d(SPAWN_X, SPAWN_Y, SPAWN_Z)); break
print("  Waiting 200 ticks for sublayers ...")
for _ in range(200): app.update()

# ── Static world demo camera ──────────────────────────────────────────────────
# Carter spawns at (5,-8,0.15) facing +X, drives ~20m along +X over 60s.
# Camera: side-rear elevated position covering the aisle. Never moves.
demo_cam_usd = UsdGeom.Camera.Define(stage, DEMO_CAM_PATH)
demo_cam_xf  = UsdGeom.Xformable(demo_cam_usd.GetPrim())
demo_cam_usd.GetHorizontalApertureAttr().Set(24.0)
demo_cam_usd.GetFocalLengthAttr().Set(11.0)   # ~85° FOV

eye_pt    = Gf.Vec3d(-1.0, -14.0, 6.0)
center_pt = Gf.Vec3d(8.0,  -3.0, 0.4)
up_vec    = Gf.Vec3d(0.0,   0.0, 1.0)
_m = Gf.Matrix4d()
_m.SetLookAt(eye_pt, center_pt, up_vec)
demo_cam_xf.MakeMatrixXform().Set(_m.GetInverse())
print(f"  Static demo camera at {DEMO_CAM_PATH}: eye=(-1,-14,6) → (8,-3,0.4)")

# ── Extensions ────────────────────────────────────────────────────────────────
mgr = omni.kit.app.get_app().get_extension_manager()
mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
for _ in range(20): app.update()
mgr.set_extension_enabled_immediate("isaacsim.robot.wheeled_robots.nodes", True)
for _ in range(5): app.update()

import omni.timeline
omni.timeline.get_timeline_interface().play()
for _ in range(10): app.update()

# ── OmniGraph ─────────────────────────────────────────────────────────────────
import omni.graph.core as og
keys = og.Controller.Keys
og.Controller.edit(
    {"graph_path": "/ActionGraph_wms3", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("on_tick",    "omni.graph.action.OnTick"),
            ("diff_drive", "isaacsim.robot.wheeled_robots.DifferentialController"),
            ("art_ctrl",   "isaacsim.core.nodes.IsaacArticulationController"),
        ],
        keys.CONNECT: [
            ("on_tick.outputs:tick",               "diff_drive.inputs:execIn"),
            ("on_tick.outputs:tick",               "art_ctrl.inputs:execIn"),
            ("diff_drive.outputs:velocityCommand", "art_ctrl.inputs:velocityCommand"),
        ],
        keys.SET_VALUES: [
            ("diff_drive.inputs:wheelDistance", 0.413),
            ("diff_drive.inputs:wheelRadius",   0.100),
        ],
    }
)
for _ in range(30): app.update()
og.Controller.set(og.Controller.attribute("/ActionGraph_wms3/art_ctrl.inputs:robotPath"),  CARTER_PRIM_PATH)
og.Controller.set(og.Controller.attribute("/ActionGraph_wms3/art_ctrl.inputs:jointNames"), ["joint_wheel_left", "joint_wheel_right"])
for _ in range(30): app.update()
diff_lin_attr = og.Controller.attribute("/ActionGraph_wms3/diff_drive.inputs:linearVelocity")
diff_ang_attr = og.Controller.attribute("/ActionGraph_wms3/diff_drive.inputs:angularVelocity")
print("OmniGraph ready ✓")

print("Warming up (120 ticks) ...")
for _ in range(120): app.update()

# ── Cameras ───────────────────────────────────────────────────────────────────
# Front hawk camera for VLA inference
vla_cam_prim = stage.GetPrimAtPath(VLA_CAMERA_PATH)
vla_cam_use  = VLA_CAMERA_PATH if vla_cam_prim.IsValid() else DEMO_CAM_PATH
print(f"VLA camera : {vla_cam_use}")
print(f"Demo camera: {DEMO_CAM_PATH}")

rp_demo = rep.create.render_product(DEMO_CAM_PATH, (1280, 720))
rp_vla  = rep.create.render_product(vla_cam_use,   (640, 360))
demo_ann = rep.AnnotatorRegistry.get_annotator("rgb")
vla_ann  = rep.AnnotatorRegistry.get_annotator("rgb")
demo_ann.attach([rp_demo])
vla_ann.attach([rp_vla])
for _ in range(60): app.update()

# ── IPC setup ─────────────────────────────────────────────────────────────────
for f in [FRAME_RD, ACT_RD, QUIT_F]:
    if os.path.exists(f): os.remove(f)
with open(INSTR_FILE, "w") as fi:
    fi.write(INSTRUCTION)

log_f   = open(LOG_OUT, "w")
cur_lin = cur_ang = 0.0
waiting = False
vla_lin = vla_ang = 0.0

def stop(sig, _):
    open(QUIT_F, "w").close()
    og.Controller.set(diff_lin_attr, 0.0)
    og.Controller.set(diff_ang_attr, 0.0)
    log_f.close(); app.close(); sys.exit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

print(f"\nRunning {ACTION_STEPS} VLA-driven action steps ...")
print(f"  {'Step':>5}  {'Lin':>7}  {'Ang':>7}  {'Mag':>6}  {'Src'}")
print("  " + "─" * 40)

step = n_frames = 0
robot_x = float(SPAWN_X)
robot_y = float(SPAWN_Y)
robot_theta = 0.0
dt_send = 1.0 / SEND_HZ

from PIL import Image as PILImage

while step < TOTAL_SIM:
    app.update()
    step += 1
    og.Controller.set(diff_lin_attr, float(cur_lin))
    og.Controller.set(diff_ang_attr, float(cur_ang))

    if step % SEND_STEP != 0:
        continue

    rep.orchestrator.step()

    # Demo-cam frame → saved for video
    demo_rgb = demo_ann.get_data()
    if demo_rgb is not None and len(demo_rgb) > 0:
        frame = demo_rgb[:, :, :3]
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        PILImage.fromarray(frame).save(
            os.path.join(FRM_DIR, f"frame_{n_frames:06d}.jpg"), quality=90)

    # Write robot state for GoalCond worker (always, regardless of model type)
    with open(ROBOT_STATE, "w") as _rs:
        json.dump({"robot_x": robot_x, "robot_y": robot_y, "robot_theta": robot_theta,
                   "goal_x": GOAL_X, "goal_y": GOAL_Y}, _rs)

    # VLA front-cam frame → sent to worker
    vla_rgb = vla_ann.get_data()
    if vla_rgb is not None and len(vla_rgb) > 0:
        vla_frame = vla_rgb[:, :, :3]
        if vla_frame.dtype != np.uint8:
            vla_frame = (vla_frame * 255).clip(0, 255).astype(np.uint8)
        if not waiting:
            PILImage.fromarray(vla_frame).save(FRAME_IN, quality=90)
            open(FRAME_RD, "w").close()
            waiting = True

    src = "hold"
    if os.path.exists(ACT_RD):
        try:
            act = json.load(open(ACT_OUT))
            vla_lin = float(act.get("lin_vel", 0.0))
            vla_ang = float(act.get("ang_vel", 0.0))
            src = "VLA"
        except Exception:
            pass
        os.remove(ACT_RD)
        waiting = False
        cur_lin, cur_ang = vla_lin, vla_ang

    robot_theta += cur_ang * dt_send
    robot_x     += cur_lin * math.cos(robot_theta) * dt_send
    robot_y     += cur_lin * math.sin(robot_theta) * dt_send

    mag = math.hypot(cur_lin, cur_ang)
    if n_frames % 20 == 0:
        print(f"  {step:>5}  {cur_lin:>+7.3f}  {cur_ang:>+7.3f}  {mag:>6.3f}  {src}", flush=True)

    log_f.write(json.dumps({
        "step": step, "frame": n_frames,
        "vla_lin": round(vla_lin, 4), "vla_ang": round(vla_ang, 4),
        "fin_lin": round(cur_lin, 4), "fin_ang": round(cur_ang, 4),
        "robot_x": round(robot_x, 3), "robot_y": round(robot_y, 3),
        "src": src,
    }) + "\n")
    n_frames += 1

open(QUIT_F, "w").close()
og.Controller.set(diff_lin_attr, 0.0)
og.Controller.set(diff_ang_attr, 0.0)
log_f.close()

dist = math.hypot(robot_x, robot_y)
print("\n" + "=" * 64)
print("Step 3 — Episode complete")
print(f"  Frames      : {n_frames}")
print(f"  Dead-reckoned displacement: x={robot_x:.3f}m  y={robot_y:.3f}m  ({dist:.2f}m)")
print("=" * 64)

def compile_video(frame_dir, out_path, fps=5):
    import subprocess
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frames:
        print(f"  No frames in {frame_dir}"); return
    r = subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i",
         os.path.join(frame_dir, "frame_%06d.jpg"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path],
        capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  Video: {out_path}  ({os.path.getsize(out_path)//1024} KB)")
    else:
        print(f"  ffmpeg error: {r.stderr[-300:]}")

compile_video(FRM_DIR, VID_OUT)
app.close()
print("Step 3 complete.")
