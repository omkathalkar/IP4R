#!/usr/bin/env python3
"""
large_collect.py — Large diverse dataset collection for VLA training.

100 episodes × 10 DynaNav targets, each with:
  - Random start position (2–10m from goal) sampled from floor-plan free space
  - Random initial heading (uniform 0–2π)
  - Proportional waypoint controller to goal
  - Front-hawk frame + (lin_vel, ang_vel) saved at 5 Hz

Target: ~10,000 frames across varied visual contexts and headings.

Requires:
  ~/Desktop/vslam_data/<ts>/map.pgm  ← occupancy grid (5 cm/px, from build_grid_from_gt.py)

Usage (via large_collect.sh):
  conda activate isaac6
  python3 large_collect.py [--map <map.pgm>] [--episodes 100] [--seed 42]
"""

import sys, os, time, math, json, argparse, datetime, signal
import numpy as np
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--map", default=None,
    help="Path to map.pgm (default: newest ~/Desktop/vslam_data/*/map.pgm)")
ap.add_argument("--episodes", type=int, default=100,
    help="Total episodes to collect (default 100, 10 per target)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max_steps", type=int, default=400,  # 80s @ 5Hz
    help="Max controller steps per episode")
args = ap.parse_args()

# ── Locate map ─────────────────────────────────────────────────────────────────
if args.map is None:
    candidates = sorted(Path.home().glob("Desktop/vslam_data/*/map.pgm"))
    if not candidates:
        print("ERROR: no map.pgm found. Run build_grid_from_gt.py first."); sys.exit(1)
    MAP_PATH = candidates[-1]
else:
    MAP_PATH = Path(args.map)
print(f"Floor plan: {MAP_PATH}")

# ── Floor plan free-space ──────────────────────────────────────────────────────
from PIL import Image as PILImage
grid_img = np.array(PILImage.open(MAP_PATH))
# PGM is saved flipped (row0=top=maxY); flip back so row0=south
grid = np.flipud(grid_img)
FREE_MASK = (grid == 254)
RES      = 0.05      # m/pixel
ORIGIN_X = -29.5
ORIGIN_Y = -11.0
MAP_H, MAP_W = grid.shape

def world_to_px(wx, wy):
    return int((wx - ORIGIN_X) / RES), int((wy - ORIGIN_Y) / RES)

def is_free(wx, wy, margin_m=0.5):
    """True if world position + margin is in free space."""
    r_px = int(margin_m / RES)
    col, row = world_to_px(wx, wy)
    for dr in range(-r_px, r_px+1):
        for dc in range(-r_px, r_px+1):
            rr, cc = row+dr, col+dc
            if not (0 <= rr < MAP_H and 0 <= cc < MAP_W):
                return False
            if not FREE_MASK[rr, cc]:
                return False
    return True

# ── Episode targets ────────────────────────────────────────────────────────────
TARGETS = [
    {"id": 0, "label": "forklift",  "goal": ( 2.5, 10.6), "instruction": "Go to the forklift area"},
    {"id": 1, "label": "aisle_06",  "goal": ( 2.0, 17.0), "instruction": "Drive forward to aisle six"},
    {"id": 2, "label": "danger",    "goal": (-3.0, 12.5), "instruction": "Avoid the danger zone"},
    {"id": 3, "label": "aisle_05",  "goal": (-3.0, 19.0), "instruction": "Navigate left to aisle five"},
    {"id": 4, "label": "aisle_04",  "goal": (-8.2, 17.0), "instruction": "Navigate to aisle four"},
    {"id": 5, "label": "aisle_03",  "goal": (-13.0,17.0), "instruction": "Go to aisle three"},
    {"id": 6, "label": "aisle_02",  "goal": (-18.0,17.0), "instruction": "Navigate to aisle two"},
    {"id": 7, "label": "aisle_01",  "goal": (-23.0,17.0), "instruction": "Navigate to aisle one"},
    {"id": 8, "label": "first_aid", "goal": (-26.0, 1.1), "instruction": "Find the first aid kit station"},
    {"id": 9, "label": "safety_sw", "goal": ( 0.5, 30.0), "instruction": "Navigate to the safety switch"},
]

# ── Generate episode specs ─────────────────────────────────────────────────────
np.random.seed(args.seed)
N_EP_PER_TARGET = args.episodes // len(TARGETS)
episodes = []

for tgt in TARGETS:
    gx, gy = tgt["goal"]
    count = 0
    attempts = 0
    while count < N_EP_PER_TARGET and attempts < 2000:
        attempts += 1
        dist    = np.random.uniform(2.5, 9.0)
        angle   = np.random.uniform(0, 2 * math.pi)
        sx      = gx + dist * math.cos(angle)
        sy      = gy + dist * math.sin(angle)
        heading = np.random.uniform(0, 2 * math.pi)   # random initial orientation
        if is_free(sx, sy, margin_m=0.45):
            episodes.append({
                "ep_id":       len(episodes),
                "target_id":   tgt["id"],
                "label":       tgt["label"],
                "instruction": tgt["instruction"],
                "start":       [round(sx, 3), round(sy, 3)],
                "heading":     round(heading, 4),   # radians
                "goal":        [gx, gy],
                "dist_m":      round(math.hypot(gx-sx, gy-sy), 2),
            })
            count += 1

# Fill remaining with spawn-based starts if needed
while len(episodes) < args.episodes:
    tgt = np.random.choice(TARGETS)
    gx, gy = tgt["goal"]
    sx = np.random.uniform(-25, 5)
    sy = np.random.uniform(-8, 28)
    if is_free(sx, sy):
        episodes.append({
            "ep_id": len(episodes), "target_id": tgt["id"],
            "label": tgt["label"], "instruction": tgt["instruction"],
            "start": [round(sx,3), round(sy,3)],
            "heading": round(np.random.uniform(0, 2*math.pi), 4),
            "goal": [gx, gy],
            "dist_m": round(math.hypot(gx-sx, gy-sy), 2),
        })

np.random.shuffle(episodes)
for i, ep in enumerate(episodes): ep["ep_id"] = i

print(f"Generated {len(episodes)} episodes across {len(TARGETS)} targets")
for tgt in TARGETS:
    n = sum(1 for e in episodes if e["target_id"] == tgt["id"])
    print(f"  {tgt['label']:12s}: {n} episodes")

# ── Output ────────────────────────────────────────────────────────────────────
ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path.home() / f"Desktop/large_dataset/{ts}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_F = OUT_DIR / "dataset.jsonl"
SUMMARY_F = OUT_DIR / "summary.json"
EPISODE_SPEC_F = OUT_DIR / "episode_specs.json"

with open(EPISODE_SPEC_F, "w") as f:
    json.dump({"n": len(episodes), "episodes": episodes}, f, indent=2)

# ── Isaac Sim ─────────────────────────────────────────────────────────────────
SCENE_PATH       = os.path.expanduser("~/TIC-VLA/DynaNav/assets/warehouse_20x20/warehouse_20x20.usd")
CARTER_URL       = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/6.0/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
)
CARTER_PRIM_PATH = "/World/Nova_Carter_ROS"
FRONT_CAM_PATH   = f"{CARTER_PRIM_PATH}/chassis_link/sensors/front_hawk/left/camera_left"
SIM_HZ   = 60
SEND_HZ  = 5
SEND_STEP = SIM_HZ // SEND_HZ

K_ANG   = 1.5; K_LIN = 0.40; ANG_MAX = 0.80; LIN_MAX = 0.40; LIN_MIN = 0.08
GOAL_RADIUS = 1.0

def normalize_angle(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def controller(rx, ry, rtheta, gx, gy):
    dx, dy    = gx - rx, gy - ry
    dist      = math.hypot(dx, dy)
    err       = normalize_angle(math.atan2(dy, dx) - rtheta)
    ang_vel   = float(np.clip(K_ANG * err, -ANG_MAX, ANG_MAX))
    lin_frac  = max(0.0, 1.0 - abs(err) / (math.pi / 2))
    lin_vel   = float(np.clip(K_LIN * lin_frac, LIN_MIN, LIN_MAX))
    return lin_vel, ang_vel, dist

print("\nStarting Isaac Sim...")
from isaacsim import SimulationApp
app = SimulationApp({
    "headless": True, "no_window": True, "multi_gpu": False, "active_gpu": 1,
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
    if i % 200 == 0: print(f"  [{i}] waiting...", flush=True)
    import time; time.sleep(0.05)
if stage is None:
    print("ERROR: stage failed to load"); app.close(); sys.exit(1)
print(f"  Loaded ({sum(1 for _ in stage.Traverse())} prims)")

from isaacsim.core.utils.stage import add_reference_to_stage
SPAWN = (5.0, -8.0, 0.15)
add_reference_to_stage(usd_path=CARTER_URL, prim_path=CARTER_PRIM_PATH)
carter_prim = stage.GetPrimAtPath(CARTER_PRIM_PATH)
xf = UsdGeom.Xformable(carter_prim)
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set(Gf.Vec3d(*SPAWN)); break
for _ in range(200): app.update()

mgr = omni.kit.app.get_app().get_extension_manager()
mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
for _ in range(20): app.update()
mgr.set_extension_enabled_immediate("isaacsim.robot.wheeled_robots.nodes", True)
for _ in range(5): app.update()

import omni.timeline
omni.timeline.get_timeline_interface().play()
for _ in range(10): app.update()

import omni.graph.core as og
keys = og.Controller.Keys
og.Controller.edit(
    {"graph_path": "/ActionGraph_lc", "evaluator_name": "execution"},
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
og.Controller.set(og.Controller.attribute("/ActionGraph_lc/art_ctrl.inputs:robotPath"),  CARTER_PRIM_PATH)
og.Controller.set(og.Controller.attribute("/ActionGraph_lc/art_ctrl.inputs:jointNames"), ["joint_wheel_left", "joint_wheel_right"])
for _ in range(30): app.update()
diff_lin = og.Controller.attribute("/ActionGraph_lc/diff_drive.inputs:linearVelocity")
diff_ang = og.Controller.attribute("/ActionGraph_lc/diff_drive.inputs:angularVelocity")
print("OmniGraph ready ✓")
for _ in range(120): app.update()

# Camera
front_prim = stage.GetPrimAtPath(FRONT_CAM_PATH)
cam_use    = FRONT_CAM_PATH if front_prim.IsValid() else "/World/OverviewCam"
rp_front   = rep.create.render_product(cam_use, (640, 360))
ann_front  = rep.AnnotatorRegistry.get_annotator("rgb")
ann_front.attach([rp_front])
for _ in range(60): app.update()
print(f"Camera: {cam_use}")

_teleport_diag_done = False

def teleport_carter(sx, sy, heading_rad):
    """Move Carter to (sx, sy, 0.20) facing heading_rad and settle."""
    global _teleport_diag_done
    og.Controller.set(diff_lin, 0.0); og.Controller.set(diff_ang, 0.0)
    for _ in range(10): app.update()

    h2 = heading_rad / 2.0
    rot_applied = False
    for op in xf.GetOrderedXformOps():
        ot = op.GetOpType()
        if not _teleport_diag_done:
            print(f"  [diag] XformOp: {ot}  precision={op.GetPrecision()}", flush=True)
        if ot == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(sx, sy, 0.20))
        elif ot == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(heading_rad)))
            rot_applied = True
        elif ot == UsdGeom.XformOp.TypeOrient:
            # quaternion: rotation around Z by heading_rad (double precision required by NovaCarter)
            op.Set(Gf.Quatd(math.cos(h2), 0.0, 0.0, math.sin(h2)))
            rot_applied = True
        elif ot == UsdGeom.XformOp.TypeRotateZ:
            op.Set(float(math.degrees(heading_rad)))
            rot_applied = True

    if not _teleport_diag_done:
        print(f"  [diag] rot_applied={rot_applied}", flush=True)
        _teleport_diag_done = True

    # 80 ticks (~1.3 s at 60 Hz) for physics to settle after teleport
    for _ in range(80): app.update()

def stop(sig, _):
    og.Controller.set(diff_lin, 0.0); og.Controller.set(diff_ang, 0.0)
    app.close(); sys.exit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

# ── Collection loop ────────────────────────────────────────────────────────────
dt         = 1.0 / SEND_HZ
dataset_f  = open(DATASET_F, "w")
ep_summaries = []
total_frames = 0

print(f"\nCollecting {len(episodes)} episodes ...")
print(f"{'EP':>4}  {'Label':12}  {'Dist':>6}  {'Steps':>6}  {'Frames':>7}  Status")
print("─" * 56)

for ep in episodes:
    ep_id   = ep["ep_id"]
    gx, gy  = ep["goal"]
    sx, sy  = ep["start"]
    heading = ep["heading"]
    instr   = ep["instruction"]
    label   = ep["label"]
    ep_dist = ep["dist_m"]

    ep_dir = OUT_DIR / "frames" / f"ep_{ep_id:04d}_{label}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    teleport_carter(sx, sy, heading)

    robot_x, robot_y, robot_theta = sx, sy, heading
    cur_lin = cur_ang = 0.0
    ep_step = ep_frame = 0
    done = False
    sim_tick = 0

    while ep_step < args.max_steps and not done:
        app.update()
        sim_tick += 1
        og.Controller.set(diff_lin, float(cur_lin))
        og.Controller.set(diff_ang, float(cur_ang))

        if sim_tick % SEND_STEP != 0:
            continue

        lin_vel, ang_vel, dist = controller(robot_x, robot_y, robot_theta, gx, gy)
        if dist < GOAL_RADIUS:
            done = True; lin_vel = ang_vel = 0.0
        cur_lin, cur_ang = lin_vel, ang_vel

        rep.orchestrator.step()
        rgb = ann_front.get_data()
        frame_path = None
        if rgb is not None and len(rgb) > 0:
            frame = rgb[:, :, :3]
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame_path = ep_dir / f"frame_{ep_frame:06d}.jpg"
            PILImage.fromarray(frame).save(frame_path, quality=90)

        dataset_f.write(json.dumps({
            "ep_id": ep_id, "ep_label": label, "ep_step": ep_step,
            "global_frame": total_frames,
            "frame_path": str(frame_path.relative_to(OUT_DIR)) if frame_path else None,
            "instruction": instr,
            "lin_vel": round(lin_vel, 4), "ang_vel": round(ang_vel, 4),
            "robot_x": round(robot_x, 3), "robot_y": round(robot_y, 3),
            "robot_theta": round(robot_theta, 4),
            "dist_to_goal": round(dist, 3),
            "goal_x": gx, "goal_y": gy,
            "start_x": sx, "start_y": sy, "start_heading": heading,
        }) + "\n")

        robot_theta = normalize_angle(robot_theta + cur_ang * dt)
        robot_x    += cur_lin * math.cos(robot_theta) * dt
        robot_y    += cur_lin * math.sin(robot_theta) * dt

        ep_step += 1; ep_frame += 1; total_frames += 1

    og.Controller.set(diff_lin, 0.0); og.Controller.set(diff_ang, 0.0)
    status = "REACHED" if done else "TIMEOUT"
    dist_f = math.hypot(gx - robot_x, gy - robot_y)
    ep_summaries.append({
        "ep_id": ep_id, "label": label, "instruction": instr,
        "start": [sx, sy], "heading": heading, "goal": [gx, gy],
        "steps": ep_step, "frames": ep_frame, "status": status,
        "dist_final": round(dist_f, 3), "end_pos": [round(robot_x,3), round(robot_y,3)],
    })
    print(f"{ep_id:>4}  {label:12}  {ep_dist:>5.1f}m  {ep_step:>6}  {ep_frame:>7}  {status}",
          flush=True)
    for _ in range(20): app.update()

dataset_f.close()

reached = sum(1 for e in ep_summaries if e["status"] == "REACHED")
with open(SUMMARY_F, "w") as f:
    json.dump({
        "ts": ts, "total_frames": total_frames,
        "n_episodes": len(episodes), "reached": reached,
        "episodes": ep_summaries,
    }, f, indent=2)

app.close()
print("\n" + "=" * 56)
print(f"Large dataset collection complete")
print(f"  Episodes   : {reached}/{len(episodes)} REACHED")
print(f"  Total frames: {total_frames:,}")
print(f"  Dataset dir : {OUT_DIR}")
print(f"  JSONL       : {DATASET_F}")
print("=" * 56)
