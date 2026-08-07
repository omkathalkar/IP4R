#!/usr/bin/env python3
"""
nav_stack/sim/run_episode_simulator.py — Hybrid nav stack, Isaac Sim side.

Isaac Sim: GPU 1 (Blackwell).  VLA worker: separate process, GPU 0 (4060 Ti).

Per-episode flow:
  1. Generate or load parametric occupancy grid → A* plan (waypoints).
  2. Load carter_warehouse_navigation.usd, enable ROS2 bridge after open_stage().
  3. Build cmd_vel OmniGraph (same wiring as flowvla_v3_run.py — do not modify).
  4. Sim loop at 60 Hz, camera + action at 5 Hz:
       a. Write current instruction to IPC (updated per waypoint).
       b. Send frame to VLA worker → get (vla_lin, vla_ang).
       c. ConfidenceGate: trust VLA or override with PurePursuit.
       d. Inject final (lin, ang) via og.Controller.set() (vectord[3]→double fix).
       e. Dead-reckon robot pose for waypoint tracking.
       f. Advance waypoint on reach; stop on GOAL or STUCK.
  5. Compile demo.mp4 BEFORE app.close().

Usage (via run_episode_simulator.sh — sets env vars):
  python3 run_episode_simulator.py --start-xy 3.0 1.0 --goal-xy 22.0 15.0
"""

import sys, os, json, time, signal, argparse, math, datetime
import numpy as np
from pathlib import Path

# ── nav_stack imports (works when nav_stack/ is on the Desktop) ───────────────
_NAV = Path(__file__).resolve().parents[2]   # two levels up from sim/ → repo root
sys.path.insert(0, str(_NAV))
from nav_stack.common import Waypoint
from nav_stack.grid.generate_occupancy_grid import (
    build_parametric_grid, inflate_grid, save_grid, load_grid,
)
from nav_stack.planning.astar_planner import plan_path, load_grid_and_meta
from nav_stack.interface.waypoint_to_vla_input import WaypointToVLAInputOffline
from nav_stack.executor.vla_local_executor import ExecutorState
from nav_stack.control.confidence_gate import ConfidenceGate
from nav_stack.control.pure_pursuit import PurePursuit

# ── IPC paths (must match vla_inference_worker.py) ───────────────────────────
IPC_DIR      = "/tmp/navstack_ipc"
FRAME_IN     = os.path.join(IPC_DIR, "frame.jpg")
FRAME_RD     = os.path.join(IPC_DIR, "frame.ready")
INSTR_FILE   = os.path.join(IPC_DIR, "current_instruction.txt")
ACT_OUT      = os.path.join(IPC_DIR, "action.json")
ACT_RD       = os.path.join(IPC_DIR, "action.ready")
QUIT_F       = os.path.join(IPC_DIR, "quit")

SCENE_URL   = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com"
               "/Assets/Isaac/6.0/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd")
CAMERA_PATH = "/World/Nova_Carter_ROS/chassis_link/sensors/front_hawk/left/camera_left"
FRAME_W, FRAME_H = 640, 360
SIM_HZ  = 60
SEND_HZ = 5
SEND_STEP = SIM_HZ // SEND_HZ     # inject + capture every 12 sim steps

# ── Grid cache (on Desktop, persists across episodes) ────────────────────────
GRID_DIR = os.path.expanduser("~/Desktop/nav_stack_grid")


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_grid(resolution: float = 0.05, robot_radius: float = 0.4):
    gd = Path(GRID_DIR)
    if (gd / "warehouse_occupancy_grid.npy").exists():
        print(f"Using cached grid at {GRID_DIR}")
        return load_grid_and_meta(GRID_DIR)
    print("Generating parametric occupancy grid...")
    grid, meta = build_parametric_grid(resolution=resolution)
    inflated   = inflate_grid(grid, robot_radius, resolution)
    save_grid(inflated, meta, gd)
    return inflated, meta


def make_vel_node():
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    rclpy.init()
    class VelNode(Node):
        def __init__(self):
            super().__init__("navstack_episode")
            self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        def send(self, lin, ang):
            msg = Twist()
            msg.linear.x  = float(lin)
            msg.angular.z = float(ang)
            self.pub.publish(msg)
    return rclpy, VelNode()


def compile_video(frame_dir, out_path, fps=5):
    import subprocess
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frames:
        print("No frames captured."); return
    pattern = os.path.join(frame_dir, "frame_%06d.jpg")
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  Video: {out_path}  ({os.path.getsize(out_path)/1e3:.0f} KB)")
    else:
        print(f"  ffmpeg error: {r.stderr[-300:]}")


def dead_reckon(x, y, theta, lin, ang, dt):
    """Euler-integrate cmd_vel to estimate robot pose."""
    theta_new = theta + ang * dt
    x_new     = x + lin * math.cos((theta + theta_new) / 2) * dt
    y_new     = y + lin * math.sin((theta + theta_new) / 2) * dt
    # Normalise theta to (-pi, pi]
    theta_new = (theta_new + math.pi) % (2 * math.pi) - math.pi
    return x_new, y_new, theta_new


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-xy",    nargs=2, type=float, default=[3.0, 1.0],
                    metavar=("X", "Y"), help="Start position in world coords (metres)")
    ap.add_argument("--goal-xy",     nargs=2, type=float, default=[22.0, 15.0],
                    metavar=("X", "Y"), help="Goal position in world coords (metres)")
    ap.add_argument("--steps",       type=int, default=600,
                    help="Max sim steps (60 Hz). 600 = 10s sim time, ~50 frames at 5 Hz")
    ap.add_argument("--reach-thresh",type=float, default=0.5,
                    help="Waypoint-reached threshold (metres)")
    ap.add_argument("--stuck-steps", type=int,   default=30,
                    help="Steps of no-progress before STUCK declared (at 5 Hz: 6s)")
    ap.add_argument("--mag-thresh",  type=float, default=0.15,
                    help="ConfidenceGate WP-magnitude floor")
    ap.add_argument("--skip-vla",    action="store_true",
                    help="Bypass VLA entirely — run pure-pursuit only (wiring sanity check)")
    ap.add_argument("--robot-start-theta", type=float, default=0.0,
                    help="Initial robot heading (radians, 0 = east / +X)")
    args = ap.parse_args()

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.expanduser(f"~/Desktop/navstack_episode/{ts}")
    frm_dir = os.path.join(out_dir, "frames")
    vid_out = os.path.join(out_dir, "episode.mp4")
    log_out = os.path.join(out_dir, "episode_log.jsonl")
    gate_log_out = os.path.join(out_dir, "gate_log.json")
    os.makedirs(frm_dir, exist_ok=True)

    start_world = tuple(args.start_xy)
    goal_world  = tuple(args.goal_xy)

    print("=" * 64)
    print("nav_stack — Hybrid A*/VLA Episode")
    print(f"  Start  : {start_world}")
    print(f"  Goal   : {goal_world}")
    print(f"  Steps  : {args.steps}  ({args.steps/SIM_HZ:.1f}s sim time)")
    print(f"  VLA    : {'DISABLED (pure-pursuit only)' if args.skip_vla else 'enabled'}")
    print(f"  Output : {out_dir}")
    print("=" * 64)

    # ── Phase 1+2: A* plan ────────────────────────────────────────────────
    grid, meta = ensure_grid()
    print(f"Grid: {meta.width}×{meta.height} @ {meta.resolution}m/cell")

    waypoints = plan_path(grid, start_world, goal_world, meta)
    print(f"A* plan: {len(waypoints)} waypoints")
    for i, wp in enumerate(waypoints):
        print(f"  WP{i:02d}  x={wp.x:.2f}  y={wp.y:.2f}  θ={math.degrees(wp.theta):+.1f}°")

    # ── Phase 3: instruction interface (offline — no backbone in Isaac process) ─
    wp_to_vla = WaypointToVLAInputOffline()

    # ── Phase 5: confidence gate + pure-pursuit ───────────────────────────
    gate = ConfidenceGate(
        mag_thresh      = args.mag_thresh,
        progress_window = args.stuck_steps,
        progress_min_m  = 0.05,
        pure_pursuit    = PurePursuit(lookahead_dist=1.2, max_lin=0.3, min_lin=0.1),
    )

    # ── IPC setup ─────────────────────────────────────────────────────────
    os.makedirs(IPC_DIR, exist_ok=True)
    for f in [FRAME_RD, ACT_RD, QUIT_F]:
        if os.path.exists(f): os.remove(f)

    # ── Isaac Sim ─────────────────────────────────────────────────────────
    from isaacsim import SimulationApp
    app = SimulationApp({
        "headless":   True,
        "no_window":  True,
        "multi_gpu":  False,
        "active_gpu": 1,     # Blackwell for rendering/physics
    })

    import omni.usd, omni.kit.app
    import omni.replicator.core as rep
    from PIL import Image as PILImage

    # Load scene WITHOUT ROS2 bridge — pre-wired OmniGraph crashes if bridge
    # is active during open_stage(). (Bug F4 — do not revert this ordering.)
    print("Loading warehouse scene (~30s)...")
    omni.usd.get_context().open_stage(SCENE_URL)
    stage = None
    for _ in range(400):
        app.update()
        s = omni.usd.get_context().get_stage()
        if s and s.GetPrimAtPath("/World").IsValid():
            if sum(1 for _ in s.Traverse()) > 500:
                stage = s; break
        time.sleep(0.1)

    if stage is None:
        print("ERROR: Scene failed to load."); app.close(); return

    print(f"Scene loaded ({sum(1 for _ in stage.Traverse())} prims).")

    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    for _ in range(20): app.update()
    mgr.set_extension_enabled_immediate("isaacsim.robot.wheeled_robots.nodes", True)
    for _ in range(5):  app.update()

    # Start physics BEFORE the sim loop — PhysX won't step without this.
    import omni.timeline
    omni.timeline.get_timeline_interface().play()
    for _ in range(5): app.update()

    # Build cmd_vel OmniGraph.
    # KEY: linearVelocity/angularVelocity NOT wired (vectord[3]→double mismatch
    # causes silent zeros). Scalars injected from Python each step instead.
    print("Building cmd_vel OmniGraph...")
    import omni.graph.core as og
    keys = og.Controller.Keys
    diff_lin_attr = diff_ang_attr = None
    try:
        og.Controller.edit(
            {"graph_path": "/ActionGraph_navstack", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("on_tick",    "omni.graph.action.OnTick"),
                    ("ros_sub",    "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                    ("diff_drive", "isaacsim.robot.wheeled_robots.DifferentialController"),
                    ("art_ctrl",   "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                keys.CONNECT: [
                    ("on_tick.outputs:tick",               "ros_sub.inputs:execIn"),
                    ("ros_sub.outputs:execOut",            "diff_drive.inputs:execIn"),
                    ("ros_sub.outputs:execOut",            "art_ctrl.inputs:execIn"),
                    ("diff_drive.outputs:velocityCommand", "art_ctrl.inputs:velocityCommand"),
                    # linearVelocity/angularVelocity intentionally omitted — type mismatch.
                ],
                keys.SET_VALUES: [
                    ("ros_sub.inputs:topicName",       "/cmd_vel"),
                    ("diff_drive.inputs:wheelDistance", 0.413),
                    ("diff_drive.inputs:wheelRadius",   0.100),
                    ("art_ctrl.inputs:robotPath",       "/World/Nova_Carter_ROS"),
                    ("art_ctrl.inputs:jointNames",      ["joint_wheel_left", "joint_wheel_right"]),
                ],
            }
        )
        for _ in range(30): app.update()
        diff_lin_attr = og.Controller.attribute("/ActionGraph_navstack/diff_drive.inputs:linearVelocity")
        diff_ang_attr = og.Controller.attribute("/ActionGraph_navstack/diff_drive.inputs:angularVelocity")
        print("  OmniGraph ready — scalar injection active.")
    except Exception as e:
        print(f"  WARNING: OmniGraph creation failed: {e}")

    rclpy, vel_node = make_vel_node()

    print("Warming up (120 steps)...")
    for _ in range(120): app.update()

    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    cam_path = CAMERA_PATH if cam_prim.IsValid() else \
               "/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime/Camera"
    rp      = rep.create.render_product(cam_path, (FRAME_W, FRAME_H))
    rgb_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_ann.attach([rp])
    for _ in range(60): app.update()

    # ── Episode state ─────────────────────────────────────────────────────
    robot_x     = float(start_world[0])
    robot_y     = float(start_world[1])
    robot_theta = args.robot_start_theta
    wp_idx      = 0
    episode_state = "NAVIGATE"

    # Stuck detection (at 5 Hz level)
    stuck_counter = 0

    step      = 0
    n_frames  = 0
    cur_lin   = 0.0
    cur_ang   = 0.0
    waiting   = False
    dt_send   = 1.0 / SEND_HZ    # 0.2 s per camera/action step

    print(f"\nRunning — {args.steps} steps, {args.steps/SIM_HZ:.1f}s sim time")
    print(f"  {'Step':>6}  {'Lin':>7}  {'Ang':>7}  {'Src':>4}  "
          f"{'WP':>3}  {'Dist':>6}  {'State'}")
    print("  " + "─" * 54)

    log_f = open(log_out, "w")

    def stop(sig, _):
        open(QUIT_F, "w").close()
        vel_node.send(0, 0)
        log_f.close()
        app.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while step < args.steps and episode_state not in ("GOAL", "STUCK"):
        app.update()
        rclpy.spin_once(vel_node, timeout_sec=0.0)
        step += 1

        # Only act at 5 Hz (every SEND_STEP sim steps)
        if step % SEND_STEP != 0:
            continue

        # ── Rep step + frame capture ──────────────────────────────────────
        rep.orchestrator.step()
        rgb = rgb_ann.get_data()
        if rgb is None or len(rgb) == 0:
            continue

        frame = rgb[:, :, :3]
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame_path = os.path.join(frm_dir, f"frame_{n_frames:06d}.jpg")
        PILImage.fromarray(frame).save(frame_path, quality=92)

        # ── Waypoint-grounded instruction (Phase 3) ───────────────────────
        wp = waypoints[wp_idx]
        vla_input = wp_to_vla.step(
            robot_x, robot_y, robot_theta,
            wp.x, wp.y,
            is_final_wp=(wp_idx == len(waypoints) - 1),
        )

        # Write instruction for VLA worker to read
        with open(INSTR_FILE, "w") as f:
            f.write(vla_input.instruction_str)

        # ── VLA inference via IPC ─────────────────────────────────────────
        vla_lin = vla_ang = 0.0
        if not args.skip_vla and not waiting:
            PILImage.fromarray(frame).save(FRAME_IN, quality=90)
            open(FRAME_RD, "w").close()
            waiting = True

        # Poll for VLA action (non-blocking — use previous action if not ready)
        if os.path.exists(ACT_RD):
            try:
                act = json.load(open(ACT_OUT))
                vla_lin = float(act.get("lin_vel", 0.0))
                vla_ang = float(act.get("ang_vel", 0.0))
            except Exception:
                pass
            os.remove(ACT_RD)
            waiting = False

        # ── Confidence gate (Phase 5) ─────────────────────────────────────
        if args.skip_vla:
            # Bypass gate entirely — pure-pursuit drives
            pp = PurePursuit(lookahead_dist=1.2, max_lin=0.3, min_lin=0.1)
            fin_lin, fin_ang = pp.compute(robot_x, robot_y, robot_theta, waypoints, wp_idx)
            used_vla = False
            gate_entry = None
        else:
            fin_lin, fin_ang, used_vla, gate_entry = gate.step(
                vla_lin, vla_ang,
                robot_x, robot_y, robot_theta,
                wp.x, wp.y, wp_idx, waypoints,
            )

        cur_lin, cur_ang = fin_lin, fin_ang

        # ── Inject into OmniGraph + ROS2 ─────────────────────────────────
        vel_node.send(cur_lin, cur_ang)
        if diff_lin_attr is not None:
            og.Controller.set(diff_lin_attr, float(cur_lin))
            og.Controller.set(diff_ang_attr, float(cur_ang))

        # ── Dead-reckon pose (Phase 4 state tracking) ────────────────────
        robot_x, robot_y, robot_theta = dead_reckon(
            robot_x, robot_y, robot_theta, cur_lin, cur_ang, dt_send
        )

        # ── Waypoint reached? ────────────────────────────────────────────
        dist_to_wp = math.hypot(wp.x - robot_x, wp.y - robot_y)
        if dist_to_wp < args.reach_thresh:
            if wp_idx < len(waypoints) - 1:
                wp_idx += 1
                gate.reset_progress_window()
                stuck_counter = 0
                episode_state = "WP_REACHED"
            else:
                episode_state = "GOAL"
        else:
            episode_state = "NAVIGATE"

        # ── Stuck check (progress window at 5 Hz) ────────────────────────
        # Gate's progress signal already covers this — if gate is always
        # falling back AND distance is not decreasing, declare STUCK.
        if (not args.skip_vla and gate_entry is not None
                and not gate_entry.progress_ok and not gate_entry.mag_ok):
            stuck_counter += 1
        else:
            stuck_counter = max(0, stuck_counter - 1)

        if stuck_counter >= args.stuck_steps:
            episode_state = "STUCK"

        # ── Log ───────────────────────────────────────────────────────────
        src = "VLA" if used_vla else "PP "
        print(f"  {step:>6}  {cur_lin:>+7.3f}  {cur_ang:>+7.3f}  {src}  "
              f"{wp_idx:>3}  {dist_to_wp:>6.2f}  {episode_state}", flush=True)

        log_entry = {
            "step": step, "frame": n_frames,
            "robot_x": round(robot_x, 3), "robot_y": round(robot_y, 3),
            "robot_theta": round(robot_theta, 4),
            "wp_index": wp_idx, "dist_to_wp": round(dist_to_wp, 3),
            "vla_lin": round(vla_lin, 4), "vla_ang": round(vla_ang, 4),
            "fin_lin": round(cur_lin, 4),  "fin_ang": round(cur_ang, 4),
            "used_vla": used_vla,
            "instruction": vla_input.instruction_key,
            "episode_state": episode_state,
        }
        log_f.write(json.dumps(log_entry) + "\n")
        n_frames += 1

    # ── Episode end ───────────────────────────────────────────────────────
    vel_node.send(0, 0)
    log_f.close()
    open(QUIT_F, "w").close()

    gate_summary = gate.summary()
    print("\n" + "=" * 64)
    print(f"Episode complete — {episode_state}")
    print(f"  Frames     : {n_frames}")
    print(f"  Waypoints  : {wp_idx}/{len(waypoints)} reached")
    print(f"  VLA rate   : {gate_summary['vla_rate']:.1%}")
    print(f"  Fallback   : {gate_summary['fallback_rate']:.1%}")
    dist_to_goal = math.hypot(goal_world[0] - robot_x, goal_world[1] - robot_y)
    print(f"  Dist to goal (dead-reckoned): {dist_to_goal:.2f}m")
    print("=" * 64)

    # Save gate log for paper analysis
    with open(gate_log_out, "w") as f:
        json.dump({
            "summary": gate_summary,
            "waypoints": [{"x": w.x, "y": w.y, "theta": w.theta} for w in waypoints],
            "start": start_world, "goal": goal_world,
            "episode_state": episode_state,
            "steps": step, "n_frames": n_frames,
            "dist_to_goal_m": round(dist_to_goal, 3),
            "gate_log": gate.log_as_dicts(),
        }, f, indent=2)
    print(f"  Gate log   : {gate_log_out}")

    # Compile video BEFORE app.close() — app.close() calls sys.exit() internally.
    compile_video(frm_dir, vid_out, fps=SEND_HZ)
    print(f"  Video      : {vid_out}")

    app.close()


if __name__ == "__main__":
    main()
