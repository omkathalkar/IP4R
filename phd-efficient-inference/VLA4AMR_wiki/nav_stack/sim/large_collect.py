#!/usr/bin/env python3
"""
large_collect.py — Logical dataset collection with A* + Pure Pursuit navigation.

Architecture (mirrors VLA4AMR target architecture):
  Goal position → A* planner → waypoints → Pure Pursuit → cmd_vel → Isaac Sim

This replaces the P-controller:
  - A* plans collision-free paths through warehouse aisles (no rack crashes)
  - Pure Pursuit provides smooth curved path-following
  - Grid is generated once from the known warehouse layout

Usage (isaac6 env, GPU 1):
  conda activate isaac6
  python3 large_collect.py [--episodes 500] [--seed 42]
"""

import sys, os, time, math, json, argparse, datetime, signal
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation

# ── Nav stack imports ──────────────────────────────────────────────────────────
NAV_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NAV_STACK))
from common import GridMeta, Waypoint
from planning.astar_planner import plan_path
from control.pure_pursuit import PurePursuit

# ── Args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--episodes",  type=int, default=500)
ap.add_argument("--seed",      type=int, default=42)
ap.add_argument("--max_steps", type=int, default=600,   # 120s @ 5Hz
                help="Max Pure Pursuit steps per episode (A* paths are longer than straight lines)")
args = ap.parse_args()

# ── Warehouse occupancy grid ───────────────────────────────────────────────────
# Layout derived from spawn zones + goal positions of carter_warehouse_navigation.
# Six north-south aisles at x ∈ {-23,-18,-13,-8.2,-3,2}, each 2.8m wide.
# Robots spawn on the open south floor (y < 0) and navigate north through aisles.
GRID_DIR   = NAV_STACK / "grid"
GRID_NPY   = GRID_DIR / "warehouse_occupancy_grid.npy"
GRID_YAML  = GRID_DIR / "warehouse_occupancy_meta.yaml"

_G_RES    = 0.1         # 10 cm/cell
_G_OX     = -30.0       # world X of cell (0,0)
_G_OY     = -10.0       # world Y of cell (0,0)
_G_W      = 42.0        # arena width  → x: -30 to +12
_G_H      = 45.0        # arena height → y: -10 to +35
_AISLE_CX = [-23.0, -18.0, -13.0, -8.2, -3.0, 2.0]
_AISLE_HW = 1.4         # half-width → 2.8 m corridors
_RACK_Y0  = 2.0         # racks start at y = 2 m
_RACK_Y1  = 25.0        # racks end   at y = 25 m
_R_RADIUS = 0.4         # NovaCarter inflation radius (m)


def _build_grid():
    import yaml
    cols = int(_G_W / _G_RES)
    rows = int(_G_H / _G_RES)
    g    = np.zeros((rows, cols), dtype=np.uint8)

    def wc(x): return int((x - _G_OX) / _G_RES)
    def wr(y): return int((y - _G_OY) / _G_RES)
    wall = int(0.5 / _G_RES)                     # 0.5 m outer wall

    # Outer walls
    g[:wall, :] = 255;  g[-wall:, :] = 255
    g[:, :wall] = 255;  g[:, -wall:] = 255

    # Mark rack zone as occupied, then carve aisle corridors
    r0, r1 = wr(_RACK_Y0), wr(_RACK_Y1)
    g[r0:r1, wall:cols - wall] = 255
    for ax in _AISLE_CX:
        c0 = max(wall, wc(ax - _AISLE_HW))
        c1 = min(cols - wall, wc(ax + _AISLE_HW))
        g[r0:r1, c0:c1] = 0

    # Inflate by robot radius so A* paths have clearance
    rad    = int(np.ceil(_R_RADIUS / _G_RES))
    struct = np.ones((2 * rad + 1, 2 * rad + 1), dtype=bool)
    inf_g  = binary_dilation(g > 127, structure=struct).astype(np.uint8) * 255

    meta = GridMeta(_G_RES, _G_OX, _G_OY, cols, rows)
    GRID_DIR.mkdir(exist_ok=True)
    np.save(GRID_NPY, inf_g)
    yaml.dump({"resolution": _G_RES, "origin_x": _G_OX, "origin_y": _G_OY,
               "width": cols, "height": rows},
              open(GRID_YAML, "w"))
    return inf_g, meta


def _load_grid():
    import yaml
    g    = np.load(GRID_NPY)
    d    = yaml.safe_load(GRID_YAML.read_text())
    return g, GridMeta(**d)


if GRID_NPY.exists() and GRID_YAML.exists():
    occ_grid, grid_meta = _load_grid()
    print(f"Grid loaded: {grid_meta.width}×{grid_meta.height} @ {_G_RES}m/cell")
else:
    print("Generating warehouse occupancy grid ...")
    occ_grid, grid_meta = _build_grid()
    free = (occ_grid == 0).sum()
    print(f"Grid generated: {grid_meta.width}×{grid_meta.height}  free={free:,} cells")

pp = PurePursuit(lookahead_dist=1.2, max_lin=0.35, max_ang=1.0, min_lin=0.10)
WP_ADVANCE = 0.6   # advance waypoint when within this distance (m)


def plan_episode(sx, sy, gx, gy):
    """Return A* Waypoint list or None on failure."""
    try:
        return plan_path(occ_grid, (sx, sy), (gx, gy), grid_meta)
    except Exception as e:
        print(f"  [A* skip] {e}", flush=True)
        return None


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

SPAWN_ZONES = {
    "forklift"  : (-6.0,  7.0, -5.0, -1.0),
    "aisle_06"  : (-5.0,  7.0, -5.0, -1.0),
    "danger"    : (-9.0,  3.0, -5.0, -1.0),
    "aisle_05"  : (-9.0,  3.0, -5.0, -1.0),
    "aisle_04"  : (-14.0, -2.0,-5.0, -1.0),
    "aisle_03"  : (-19.0, -7.0,-5.0, -1.0),
    "aisle_02"  : (-25.0,-12.0,-5.0, -1.0),
    "aisle_01"  : (-26.0,-18.0,-5.0, -1.0),
    "first_aid" : (-26.0,-20.0,-7.0, -1.0),
    "safety_sw" : ( -4.0,  5.0, -5.0, -1.0),
}

# ── Generate episode specs ─────────────────────────────────────────────────────
np.random.seed(args.seed)
N_EP_PER_TARGET = args.episodes // len(TARGETS)
episodes = []

for tgt in TARGETS:
    gx, gy = tgt["goal"]
    zone   = SPAWN_ZONES[tgt["label"]]
    x_lo, x_hi, y_lo, y_hi = zone
    count = attempts = 0
    while count < N_EP_PER_TARGET and attempts < 4000:
        attempts += 1
        sx = np.random.uniform(x_lo, x_hi)
        sy = np.random.uniform(y_lo, y_hi)
        d  = math.hypot(gx - sx, gy - sy)
        if d < 2.0 or d > 25.0:
            continue
        goal_angle = math.atan2(gy - sy, gx - sx)
        heading    = goal_angle + np.random.uniform(-math.pi / 4, math.pi / 4)
        episodes.append({
            "ep_id": len(episodes), "target_id": tgt["id"],
            "label": tgt["label"],  "instruction": tgt["instruction"],
            "start": [round(sx, 3), round(sy, 3)],
            "heading": round(heading % (2 * math.pi), 4),
            "goal":  [gx, gy], "dist_m": round(d, 2),
        })
        count += 1
    if count < N_EP_PER_TARGET:
        print(f"  WARNING: {tgt['label']} only got {count}/{N_EP_PER_TARGET} spawns")

while len(episodes) < args.episodes:
    tgt   = TARGETS[np.random.randint(len(TARGETS))]
    gx, gy = tgt["goal"]
    zone   = SPAWN_ZONES[tgt["label"]]
    x_lo, x_hi, y_lo, y_hi = zone
    sx = np.random.uniform(x_lo, x_hi)
    sy = np.random.uniform(y_lo, y_hi)
    d  = math.hypot(gx - sx, gy - sy)
    if d < 2.0 or d > 25.0:
        continue
    goal_angle = math.atan2(gy - sy, gx - sx)
    heading    = goal_angle + np.random.uniform(-math.pi / 4, math.pi / 4)
    episodes.append({
        "ep_id": len(episodes), "target_id": tgt["id"],
        "label": tgt["label"],  "instruction": tgt["instruction"],
        "start": [round(sx, 3), round(sy, 3)],
        "heading": round(heading % (2 * math.pi), 4),
        "goal": [gx, gy], "dist_m": round(d, 2),
    })

np.random.shuffle(episodes)
for i, ep in enumerate(episodes): ep["ep_id"] = i

print(f"Generated {len(episodes)} episodes across {len(TARGETS)} targets")
for tgt in TARGETS:
    n = sum(1 for e in episodes if e["target_id"] == tgt["id"])
    print(f"  {tgt['label']:12s}: {n} episodes")

# ── Output paths ──────────────────────────────────────────────────────────────
ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path.home() / f"Desktop/large_dataset_v3/{ts}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_F      = OUT_DIR / "dataset.jsonl"
SUMMARY_F      = OUT_DIR / "summary.json"
EPISODE_SPEC_F = OUT_DIR / "episode_specs.json"

with open(EPISODE_SPEC_F, "w") as f:
    json.dump({"n": len(episodes), "episodes": episodes}, f, indent=2)

# ── Isaac Sim setup ────────────────────────────────────────────────────────────
SCENE_PATH       = os.path.expanduser("~/TIC-VLA/DynaNav/assets/warehouse_20x20/warehouse_20x20.usd")
CARTER_URL       = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/6.0/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
)
CARTER_PRIM_PATH = "/World/Nova_Carter_ROS"
FRONT_CAM_PATH   = f"{CARTER_PRIM_PATH}/chassis_link/sensors/front_hawk/left/camera_left"
SIM_HZ    = 60
SEND_HZ   = 5
SEND_STEP = SIM_HZ // SEND_HZ
GOAL_RADIUS = 1.0

def normalize_angle(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

print("\nStarting Isaac Sim...")
from isaacsim import SimulationApp
app = SimulationApp({
    "headless": True, "no_window": True, "multi_gpu": False, "active_gpu": 1,
})

import omni.usd, omni.kit.app
import omni.replicator.core as rep
from pxr import UsdGeom, Gf
from PIL import Image as PILImage

print("Loading warehouse_20x20.usd ...")
omni.usd.get_context().open_stage(SCENE_PATH)
stage = None
for i in range(1000):
    app.update()
    s = omni.usd.get_context().get_stage()
    if s and sum(1 for _ in s.Traverse()) > 30:
        stage = s; break
    if i % 200 == 0: print(f"  [{i}] waiting...", flush=True)
    time.sleep(0.05)
if stage is None:
    print("ERROR: stage failed to load"); app.close(); sys.exit(1)
print(f"  Loaded ({sum(1 for _ in stage.Traverse())} prims)")

from isaacsim.core.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path=CARTER_URL, prim_path=CARTER_PRIM_PATH)
carter_prim = stage.GetPrimAtPath(CARTER_PRIM_PATH)
xf = UsdGeom.Xformable(carter_prim)
for op in xf.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
        op.Set(Gf.Vec3d(5.0, -8.0, 0.15)); break
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

front_prim = stage.GetPrimAtPath(FRONT_CAM_PATH)
cam_use    = FRONT_CAM_PATH if front_prim.IsValid() else "/World/OverviewCam"
rp_front   = rep.create.render_product(cam_use, (640, 360))
ann_front  = rep.AnnotatorRegistry.get_annotator("rgb")
ann_front.attach([rp_front])
for _ in range(60): app.update()
print(f"Camera: {cam_use}")

_teleport_diag_done = False

def teleport_carter(sx, sy, heading_rad):
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
            op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(heading_rad))); rot_applied = True
        elif ot == UsdGeom.XformOp.TypeOrient:
            op.Set(Gf.Quatd(math.cos(h2), 0.0, 0.0, math.sin(h2))); rot_applied = True
        elif ot == UsdGeom.XformOp.TypeRotateZ:
            op.Set(float(math.degrees(heading_rad))); rot_applied = True
    if not _teleport_diag_done:
        print(f"  [diag] rot_applied={rot_applied}", flush=True)
        _teleport_diag_done = True
    for _ in range(80): app.update()

def stop(sig, _):
    og.Controller.set(diff_lin, 0.0); og.Controller.set(diff_ang, 0.0)
    app.close(); sys.exit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

# ── Collection loop ────────────────────────────────────────────────────────────
dt           = 1.0 / SEND_HZ
dataset_f    = open(DATASET_F, "w")
ep_summaries = []
total_frames = 0
n_skipped    = 0

print(f"\nCollecting {len(episodes)} episodes (A* + Pure Pursuit) ...")
print(f"{'EP':>4}  {'Label':12}  {'Dist':>6}  {'WPs':>4}  {'Steps':>6}  {'Frames':>7}  Status")
print("─" * 63)

for ep in episodes:
    ep_id   = ep["ep_id"]
    gx, gy  = ep["goal"]
    sx, sy  = ep["start"]
    heading = ep["heading"]
    instr   = ep["instruction"]
    label   = ep["label"]
    ep_dist = ep["dist_m"]

    # Plan collision-free A* path before spawning
    waypoints = plan_episode(sx, sy, gx, gy)
    if waypoints is None:
        n_skipped += 1
        print(f"{ep_id:>4}  {label:12}  {ep_dist:>5.1f}m  {'--':>4}  {'--':>6}  {'--':>7}  SKIP (A* fail)",
              flush=True)
        continue

    ep_dir = OUT_DIR / "frames" / f"ep_{ep_id:04d}_{label}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    teleport_carter(sx, sy, heading)

    robot_x, robot_y, robot_theta = sx, sy, heading
    cur_lin = cur_ang = 0.0
    ep_step = ep_frame = 0
    wp_idx  = 0
    done    = False
    sim_tick = 0

    while ep_step < args.max_steps and not done:
        app.update()
        sim_tick += 1
        og.Controller.set(diff_lin, float(cur_lin))
        og.Controller.set(diff_ang, float(cur_ang))

        if sim_tick % SEND_STEP != 0:
            continue

        # Advance waypoint index when robot is close enough
        while wp_idx < len(waypoints) - 1:
            if math.hypot(robot_x - waypoints[wp_idx].x,
                          robot_y - waypoints[wp_idx].y) < WP_ADVANCE:
                wp_idx += 1
            else:
                break

        dist = math.hypot(gx - robot_x, gy - robot_y)
        if dist < GOAL_RADIUS:
            done = True
            lin_vel = ang_vel = 0.0
        else:
            lin_vel, ang_vel = pp.compute(robot_x, robot_y, robot_theta, waypoints, wp_idx)

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
            "wp_idx": wp_idx, "n_waypoints": len(waypoints),
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
        "n_waypoints": len(waypoints),
        "steps": ep_step, "frames": ep_frame, "status": status,
        "dist_final": round(dist_f, 3), "end_pos": [round(robot_x,3), round(robot_y,3)],
    })
    print(f"{ep_id:>4}  {label:12}  {ep_dist:>5.1f}m  {len(waypoints):>4}  "
          f"{ep_step:>6}  {ep_frame:>7}  {status}", flush=True)
    for _ in range(20): app.update()

dataset_f.close()

reached = sum(1 for e in ep_summaries if e["status"] == "REACHED")
with open(SUMMARY_F, "w") as f:
    json.dump({
        "ts": ts, "total_frames": total_frames, "controller": "astar_pure_pursuit",
        "n_episodes": len(ep_summaries), "reached": reached, "skipped": n_skipped,
        "episodes": ep_summaries,
    }, f, indent=2)

app.close()
print("\n" + "=" * 63)
print(f"Dataset collection complete (A* + Pure Pursuit)")
print(f"  Episodes   : {reached}/{len(ep_summaries)} REACHED  ({n_skipped} skipped A* fail)")
print(f"  Total frames: {total_frames:,}")
print(f"  Dataset dir : {OUT_DIR}")
print(f"  JSONL       : {DATASET_F}")
print("=" * 63)
