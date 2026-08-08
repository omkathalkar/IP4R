#!/usr/bin/env python3
"""
goalcond_eval_all_targets.py — Closed-loop evaluation of ActionRegressorGoalCond
on all 10 DynaNav targets, 3 random starts each (30 episodes total).

Self-contained: loads GoalCond checkpoint + runs Isaac Sim in a single process.
No separate VLA worker needed — GoalCond inference is pure MLP (~1 ms).

Usage (isaac6 env, GPU 1):
  python3 goalcond_eval_all_targets.py \
      --ckpt ~/Desktop/flowvla_goalcond/flowvla_goalcond_best.pt \
      --out  ~/Desktop/goalcond_eval/
"""

import argparse, json, math, os, sys, time, datetime, signal
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.expanduser(
    "~/Desktop/flowvla_goalcond/flowvla_goalcond_best.pt"))
ap.add_argument("--out",  default=os.path.expanduser("~/Desktop/goalcond_eval"))
ap.add_argument("--starts-per-target", type=int, default=3)
ap.add_argument("--max-steps",         type=int, default=500)
ap.add_argument("--seed",              type=int, default=0)
ap.add_argument("--map",               default=None,
    help="map.pgm for free-space sampling (default: newest ~/Desktop/vslam_data/*/map.pgm)")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

TICVLA_REPO = "/home/cvit-car-simulator/VLA4AMR/TIC-VLA-main"
BASE_MODEL  = "/home/cvit-car-simulator/VLA4AMR/checkpoints/internvl3-1b"
DEVICE      = "cuda:0"

# ── Load checkpoint ───────────────────────────────────────────────────────────
print(f"Loading checkpoint: {args.ckpt}")
ckpt     = torch.load(args.ckpt, map_location=DEVICE)
cfg      = ckpt["config"]
FEAT_DIM = cfg["feat_dim"]   # 899 = 896 text + 3 geo
HIDDEN   = cfg.get("hidden", 256)

class ActionRegressorGoalCond(nn.Module):
    def __init__(self, feat_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden*2), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(hidden*2, hidden), nn.SiLU(),
            nn.Dropout(0.1), nn.Linear(hidden, 2))
    def forward(self, x): return self.net(x)

model = ActionRegressorGoalCond(FEAT_DIM, HIDDEN).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

feat_mean = torch.tensor(ckpt["feat_mean"], dtype=torch.float32, device=DEVICE)
feat_std  = torch.tensor(ckpt["feat_std"],  dtype=torch.float32, device=DEVICE)
act_mean  = torch.tensor(ckpt["act_mean"],  dtype=torch.float32, device=DEVICE)
act_std   = torch.tensor(ckpt["act_std"],   dtype=torch.float32, device=DEVICE)
print(f"  ActionRegressorGoalCond loaded  feat_dim={FEAT_DIM}  hidden={HIDDEN}")

# ── Load text features for each instruction ───────────────────────────────────
print("Loading InternVL3-1B tokenizer for text features...")
sys.path.insert(0, TICVLA_REPO)
import logging; logging.basicConfig(level=logging.WARNING)
from ticvla.models.ticvla import TICVLA
from transformers import AutoTokenizer

backbone     = TICVLA(model_path=BASE_MODEL, action_horizon_steps=30,
                      action_num_layers=3, train_vlm=False).to(DEVICE).eval()
tokenizer    = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
embed_tokens = backbone.vlm.language_model.model.embed_tokens

# ── Targets ───────────────────────────────────────────────────────────────────
TARGETS = [
    {"id":0, "label":"forklift",  "goal":(2.5, 10.6), "instruction":"Go to the forklift area"},
    {"id":1, "label":"aisle_06",  "goal":(2.0, 17.0), "instruction":"Drive forward to aisle six"},
    {"id":2, "label":"danger",    "goal":(-3.0, 12.5),"instruction":"Avoid the danger zone"},
    {"id":3, "label":"aisle_05",  "goal":(-3.0, 19.0),"instruction":"Navigate left to aisle five"},
    {"id":4, "label":"aisle_04",  "goal":(-8.2, 17.0),"instruction":"Navigate to aisle four"},
    {"id":5, "label":"aisle_03",  "goal":(-13.0,17.0),"instruction":"Go to aisle three"},
    {"id":6, "label":"aisle_02",  "goal":(-18.0,17.0),"instruction":"Navigate to aisle two"},
    {"id":7, "label":"aisle_01",  "goal":(-23.0,17.0),"instruction":"Navigate to aisle one"},
    {"id":8, "label":"first_aid", "goal":(-26.0, 1.1),"instruction":"Find the first aid kit station"},
    {"id":9, "label":"safety_sw", "goal":(0.5,  30.0),"instruction":"Navigate to the safety switch"},
]

# Pre-extract text features
print("Pre-extracting text features...")
text_cache = {}
for tgt in TARGETS:
    instr = tgt["instruction"]
    if instr not in text_cache:
        ids = tokenizer(instr, return_tensors="pt",
                        add_special_tokens=True).input_ids.to(DEVICE)
        with torch.no_grad():
            text_cache[instr] = embed_tokens(ids)[0].float().mean(0)
print(f"  {len(text_cache)} instruction embeddings cached")

# ── Floor plan free-space ──────────────────────────────────────────────────────
from PIL import Image as PILImage

if args.map is None:
    candidates = sorted(Path.home().glob("Desktop/vslam_data/*/map.pgm"))
    MAP_PATH = candidates[-1] if candidates else None
else:
    MAP_PATH = Path(args.map)

if MAP_PATH and MAP_PATH.exists():
    grid_img  = np.array(PILImage.open(MAP_PATH))
    grid      = np.flipud(grid_img)
    FREE_MASK = (grid == 254)
    RES, ORIGIN_X, ORIGIN_Y = 0.05, -29.5, -11.0
    MAP_H, MAP_W = grid.shape

    def is_free(wx, wy, margin_m=0.5):
        r_px = int(margin_m / RES)
        col  = int((wx - ORIGIN_X) / RES)
        row  = int((wy - ORIGIN_Y) / RES)
        for dr in range(-r_px, r_px+1):
            for dc in range(-r_px, r_px+1):
                rr, cc = row+dr, col+dc
                if not (0 <= rr < MAP_H and 0 <= cc < MAP_W): return False
                if not FREE_MASK[rr, cc]: return False
        return True
    print(f"Floor plan: {MAP_PATH}")
else:
    def is_free(wx, wy, margin_m=0.5): return True
    print("No floor plan — sampling without free-space check")

# ── Sample starts ─────────────────────────────────────────────────────────────
np.random.seed(args.seed)
episodes = []
for tgt in TARGETS:
    gx, gy = tgt["goal"]
    count, attempts = 0, 0
    while count < args.starts_per_target and attempts < 2000:
        attempts += 1
        dist  = np.random.uniform(2.5, 8.0)
        angle = np.random.uniform(0, 2*math.pi)
        sx, sy = gx + dist*math.cos(angle), gy + dist*math.sin(angle)
        heading = np.random.uniform(0, 2*math.pi)
        if is_free(sx, sy, 0.45):
            episodes.append({**tgt, "start":(round(sx,3), round(sy,3)),
                              "heading": round(heading,4),
                              "ep_id": len(episodes),
                              "dist_m": round(math.hypot(gx-sx, gy-sy), 2)})
            count += 1
    if count < args.starts_per_target:
        print(f"  WARNING: only {count}/{args.starts_per_target} starts for {tgt['label']}")

print(f"\n{len(episodes)} episodes across {len(TARGETS)} targets")

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_angle(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

@torch.no_grad()
def infer(instr, robot_x, robot_y, robot_theta, goal_x, goal_y):
    dx, dy   = goal_x - robot_x, goal_y - robot_y
    dist     = math.hypot(dx, dy)
    h_err    = normalize_angle(math.atan2(dy, dx) - robot_theta)
    geo      = torch.tensor([dist, math.cos(h_err), math.sin(h_err)],
                            dtype=torch.float32, device=DEVICE)
    feat     = torch.cat([text_cache[instr], geo]).unsqueeze(0)
    feat_n   = (feat - feat_mean) / feat_std
    pred_n   = model(feat_n)
    pred     = (pred_n * act_std + act_mean).squeeze(0).cpu().tolist()
    lin = float(np.clip(pred[0], -0.5, 0.5))
    ang = float(np.clip(pred[1], -1.0,  1.0))
    return lin, ang, dist

# ── Isaac Sim ─────────────────────────────────────────────────────────────────
SCENE_PATH       = os.path.expanduser("~/TIC-VLA/DynaNav/assets/warehouse_20x20/warehouse_20x20.usd")
CARTER_URL       = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com"
                    "/Assets/Isaac/6.0/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd")
CARTER_PRIM_PATH = "/World/Nova_Carter_ROS"
GOAL_RADIUS      = 1.0
SIM_HZ, SEND_HZ  = 60, 5
SEND_STEP        = SIM_HZ // SEND_HZ
DT               = 1.0 / SEND_HZ

print("\nStarting Isaac Sim...")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "no_window": True, "multi_gpu": False, "active_gpu": 1})

import omni.usd, omni.kit.app
from pxr import UsdGeom, Gf

print("Loading warehouse_20x20.usd...")
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
        op.Set(Gf.Vec3d(5.0, -8.0, 0.20)); break
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
    {"graph_path": "/ActionGraph_eval", "evaluator_name": "execution"},
    {keys.CREATE_NODES: [
        ("on_tick",    "omni.graph.action.OnTick"),
        ("diff_drive", "isaacsim.robot.wheeled_robots.DifferentialController"),
        ("art_ctrl",   "isaacsim.core.nodes.IsaacArticulationController"),
    ], keys.CONNECT: [
        ("on_tick.outputs:tick",               "diff_drive.inputs:execIn"),
        ("on_tick.outputs:tick",               "art_ctrl.inputs:execIn"),
        ("diff_drive.outputs:velocityCommand", "art_ctrl.inputs:velocityCommand"),
    ], keys.SET_VALUES: [
        ("diff_drive.inputs:wheelDistance", 0.413),
        ("diff_drive.inputs:wheelRadius",   0.100),
    ]})
for _ in range(30): app.update()
og.Controller.set(og.Controller.attribute("/ActionGraph_eval/art_ctrl.inputs:robotPath"),  CARTER_PRIM_PATH)
og.Controller.set(og.Controller.attribute("/ActionGraph_eval/art_ctrl.inputs:jointNames"), ["joint_wheel_left","joint_wheel_right"])
for _ in range(120): app.update()
diff_lin = og.Controller.attribute("/ActionGraph_eval/diff_drive.inputs:linearVelocity")
diff_ang = og.Controller.attribute("/ActionGraph_eval/diff_drive.inputs:angularVelocity")
print("OmniGraph ready ✓")

def teleport(sx, sy, heading):
    og.Controller.set(diff_lin, 0.0); og.Controller.set(diff_ang, 0.0)
    for _ in range(10): app.update()
    h2 = heading / 2.0
    for op in xf.GetOrderedXformOps():
        ot = op.GetOpType()
        if ot == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(sx, sy, 0.20))
        elif ot == UsdGeom.XformOp.TypeOrient:
            op.Set(Gf.Quatd(math.cos(h2), 0.0, 0.0, math.sin(h2)))
        elif ot == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(heading)))
    for _ in range(80): app.update()

def stop_sig(sig, _):
    og.Controller.set(diff_lin, 0.0); og.Controller.set(diff_ang, 0.0)
    app.close(); sys.exit(0)
signal.signal(signal.SIGINT, stop_sig)
signal.signal(signal.SIGTERM, stop_sig)

# ── Evaluation loop ───────────────────────────────────────────────────────────
results = []
print(f"\n{'EP':>4}  {'Label':12}  {'Dist':>6}  {'Steps':>6}  Status  Final_dist")
print("─" * 58)

for ep in episodes:
    sx, sy   = ep["start"]
    gx, gy   = ep["goal"]
    heading  = ep["heading"]
    instr    = ep["instruction"]
    label    = ep["label"]
    ep_id    = ep["ep_id"]

    teleport(sx, sy, heading)

    rx, ry, rt = sx, sy, heading
    cur_lin = cur_ang = 0.0
    sim_tick = ep_step = 0
    done = False

    while ep_step < args.max_steps and not done:
        app.update()
        sim_tick += 1
        og.Controller.set(diff_lin, float(cur_lin))
        og.Controller.set(diff_ang, float(cur_ang))

        if sim_tick % SEND_STEP != 0:
            continue

        lin, ang, dist = infer(instr, rx, ry, rt, gx, gy)
        if dist < GOAL_RADIUS:
            done = True; lin = ang = 0.0
        cur_lin, cur_ang = lin, ang

        rt += cur_ang * DT
        rx += cur_lin * math.cos(rt) * DT
        ry += cur_lin * math.sin(rt) * DT
        ep_step += 1

    og.Controller.set(diff_lin, 0.0); og.Controller.set(diff_ang, 0.0)
    dist_final = math.hypot(gx - rx, gy - ry)
    status     = "REACHED" if done else "TIMEOUT"
    results.append({
        "ep_id": ep_id, "label": label, "instruction": instr,
        "start": [sx, sy], "heading": heading, "goal": [gx, gy],
        "dist_m": ep["dist_m"], "steps": ep_step,
        "status": status, "dist_final": round(dist_final, 3),
        "end_pos": [round(rx, 3), round(ry, 3)],
    })
    print(f"{ep_id:>4}  {label:12}  {ep['dist_m']:>5.1f}m  {ep_step:>6}  {status}  {dist_final:.2f}m",
          flush=True)
    for _ in range(20): app.update()

# ── Aggregate ─────────────────────────────────────────────────────────────────
reached    = sum(1 for r in results if r["status"] == "REACHED")
total      = len(results)
success_rt = reached / total * 100

print("\n" + "=" * 58)
print(f"Closed-loop Evaluation — GoalCond ActionRegressor")
print(f"  Checkpoint : {args.ckpt}")
print(f"  Episodes   : {total}  ({args.starts_per_target} starts × {len(TARGETS)} targets)")
print(f"  Success    : {reached}/{total}  ({success_rt:.1f}%)")
print("─" * 58)
print(f"  {'Label':12}  {'REACHED':>8}  {'TIMEOUT':>8}  {'SR%':>6}")
for tgt in TARGETS:
    tgt_results = [r for r in results if r["label"] == tgt["label"]]
    n_reached   = sum(1 for r in tgt_results if r["status"] == "REACHED")
    print(f"  {tgt['label']:12}  {n_reached:>8}  {len(tgt_results)-n_reached:>8}  {n_reached/len(tgt_results)*100:>5.0f}%")
print("=" * 58)

out_file = os.path.join(args.out, f"eval_{ts}.json")
with open(out_file, "w") as f:
    json.dump({
        "ts": ts, "checkpoint": args.ckpt,
        "n_episodes": total, "reached": reached,
        "success_rate_pct": round(success_rt, 1),
        "episodes": results,
    }, f, indent=2)
print(f"Results: {out_file}")

app.close()
