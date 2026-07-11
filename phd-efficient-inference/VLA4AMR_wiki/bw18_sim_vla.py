#!/usr/bin/env python3
"""
bw18_sim_vla.py — BW18 TIC-VLA live inference in Isaac Sim.
Pairs with bw11_sim_isaac.py (isaac6 env, unchanged IPC protocol).

Two loading modes:
  A) Separate VLM + action checkpoints (BW18 fine-tuned):
       --vlm-ckpt <path> --action-ckpt <path>   (action_num_layers=6)
  B) Full Lightning checkpoint (original TIC-VLA authors' pretrained):
       --full-ckpt <path>                        (action_num_layers=3, inferred from hyper_parameters)

TIC-VLA predicts cumulative FLU waypoints → proportional heading controller
converts to (lin, ang) velocity commands sent to Isaac Sim.
Records composite ego+iso MP4.

Run (tic-vla env, SECOND terminal after bw11_sim_isaac.py is waiting):
  # BW18 fine-tuned:
  source ~/VLA4AMR/code/.env.bw18
  CUDA_VISIBLE_DEVICES=0 python3 ~/VLA4AMR/code/bw18_sim_vla.py \\
      --instruction "Drive forward and turn right at the intersection" \\
      --output-video ~/Desktop/bw18_sim_demo.mp4 \\
      2>&1 | tee ~/Desktop/bw18_sim_vla.log

  # Original TIC-VLA pretrained:
  CUDA_VISIBLE_DEVICES=0 python3 ~/VLA4AMR/code/bw18_sim_vla.py \\
      --full-ckpt ~/Desktop/tic-vla-full-ckpt/TIC-VLA-model.ckpt \\
      --instruction "Drive forward and turn right at the intersection" \\
      --output-video ~/Desktop/bw18_sim_demo_fullckpt.mp4 \\
      2>&1 | tee ~/Desktop/bw18_sim_vla_fullckpt.log
"""

import os, json, argparse, time, shutil, tempfile
import numpy as np
from pathlib import Path
from collections import deque
from PIL import Image
import cv2
import torch
import logging

import sys
sys.path.insert(0, os.path.expanduser("~/VLA4AMR/TIC-VLA-main"))
from ticvla.models.ticvla import TICVLA

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("--full-ckpt",    default=None,
                help="Path to a full TIC-VLA Lightning checkpoint (VLM+action in one file). "
                     "When set, --vlm-ckpt and --action-ckpt are ignored.")
ap.add_argument("--vlm-ckpt", default=os.path.expanduser(
    "~/Desktop/bw18_ticvla_output/checkpoints/ticvla/vlm/"
    "ticvla-vlm-epoch=00-val_language_loss=0.2781.ckpt"))
ap.add_argument("--action-ckpt", default=os.path.expanduser(
    "~/Desktop/bw18_ticvla_output/checkpoints/ticvla/action/"
    "ticvla-action-epoch=14-val_total_loss=0.0041.ckpt"))
ap.add_argument("--base-model",   default=os.path.expanduser("~/VLA4AMR/checkpoints/internvl3-1b"))
ap.add_argument("--instruction",  default="Drive forward and turn right at the intersection")
ap.add_argument("--output-video", default=os.path.expanduser("~/Desktop/bw18_sim_demo.mp4"))
ap.add_argument("--fps",          type=int, default=5)
ap.add_argument("--ipc-dir",      default="/tmp/bw11_ipc")
ap.add_argument("--history-len",  type=int, default=5,
                help="Number of delayed frames fed to VLM (TIC-VLA context window)")
ap.add_argument("--lookahead",    type=int, default=4,
                help="Waypoint lookahead index for heading estimation (default=4 → 0.5s)")
ap.add_argument("--ang-gain",     type=float, default=2.5,
                help="Proportional gain for angular velocity from heading error")
ap.add_argument("--max-lin",      type=float, default=0.4, help="Max linear velocity (m/s)")
ap.add_argument("--device",       default="cuda:0")
args = ap.parse_args()

IPC    = Path(args.ipc_dir)
DEVICE = args.device
VID_W, VID_H = 1280, 720
CAM_W, CAM_H = 640, 720
MAP_W, MAP_H = 640, 720

TMP_FRAMES = Path(tempfile.mkdtemp(prefix="bw18_frames_"))
log.info(f"Temp frame dir: {TMP_FRAMES}")


# ── Model ──────────────────────────────────────────────────────────────────────

def _load_action_expert(model: TICVLA, sd: dict, label: str):
    ae_sd = {k[len("model.action_expert."):]: v
             for k, v in sd.items() if k.startswith("model.action_expert.")}
    if ae_sd:
        miss, unex = model.action_expert.load_state_dict(ae_sd, strict=False)
        log.info(f"Action expert ← {label}: {len(ae_sd)} keys | missing={len(miss)} unexpected={len(unex)}")
    else:
        log.warning(f"No model.action_expert.* keys found in {label}!")


def load_model() -> TICVLA:
    log.info("Loading TIC-VLA model ...")

    if args.full_ckpt:
        # Full Lightning checkpoint: read action_num_layers from hyper_parameters
        full_path = os.path.expanduser(args.full_ckpt)
        log.info(f"Full ckpt ← {full_path}")
        raw = torch.load(full_path, map_location="cpu")
        hp  = raw.get("hyper_parameters", {})
        action_layers = int(hp.get("action_num_layers", 3))
        action_steps  = int(hp.get("action_horizon_steps", 30))
        log.info(f"action_num_layers={action_layers}  action_horizon_steps={action_steps}  "
                 f"(from checkpoint hyper_parameters)")

        model = TICVLA(model_path=args.base_model,
                       action_horizon_steps=action_steps,
                       action_num_layers=action_layers).to(DEVICE)

        model.load_vlm_checkpoint(full_path)
        _load_action_expert(model, raw.get("state_dict", {}), full_path)
    else:
        model = TICVLA(model_path=args.base_model, action_horizon_steps=30,
                       action_num_layers=6).to(DEVICE)

        log.info(f"VLM ← {args.vlm_ckpt}")
        model.load_vlm_checkpoint(args.vlm_ckpt)

        ckpt = torch.load(args.action_ckpt, map_location="cpu")
        _load_action_expert(model, ckpt.get("state_dict", ckpt), args.action_ckpt)

    model.eval()
    log.info(f"TIC-VLA ready on {DEVICE}")
    return model


# ── Waypoints → velocity ───────────────────────────────────────────────────────

def waypoints_to_vel(wp: np.ndarray) -> tuple[float, float]:
    """
    Convert cumulative FLU waypoints (T, 2) to (lin_vel, ang_vel).
    wp[t] = cumulative (dx_forward, dy_left) at (t+1)*0.1 s.
    Speed from first-step magnitude; heading from lookahead waypoint.
    """
    if len(wp) == 0:
        return 0.2, 0.0

    dx0, dy0 = float(wp[0, 0]), float(wp[0, 1])
    li = min(args.lookahead, len(wp) - 1)
    dx_la, dy_la = float(wp[li, 0]), float(wp[li, 1])

    speed = np.sqrt(dx0**2 + dy0**2) / 0.1
    lin   = float(np.clip(speed, 0.0, args.max_lin))

    heading = np.arctan2(dy_la, max(dx_la, 0.01))
    ang     = float(np.clip(heading * args.ang_gain, -1.0, 1.0))

    return lin, ang


# ── Video panels ───────────────────────────────────────────────────────────────

def ego_panel(bgr, lin, ang, response, query_step, frame_step):
    panel  = cv2.resize(bgr, (CAM_W, CAM_H - 220))
    canvas = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    canvas[:CAM_H - 220] = panel
    canvas[CAM_H - 220:] = (20, 20, 30)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (230, 230, 230)
    CYAN  = (220, 200, 80)
    GREEN = (80, 220, 80)

    cv2.rectangle(canvas, (0, CAM_H - 220), (CAM_W, CAM_H - 195), (35, 35, 55), -1)
    cv2.putText(canvas, f"BW18 TIC-VLA  query={query_step:03d}  frame={frame_step:04d}",
                (10, CAM_H - 200), font, 0.50, WHITE, 1, cv2.LINE_AA)

    cv2.putText(canvas, "CMD VELOCITY",          (10, CAM_H - 168), font, 0.44, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"lin = {lin:+.3f} m/s", (10, CAM_H - 144), font, 0.60, GREEN, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"ang = {ang:+.3f} r/s", (10, CAM_H - 110), font, 0.60, GREEN, 2, cv2.LINE_AA)

    dir_str = "FWD" if abs(ang) < 0.2 else ("LEFT" if ang > 0 else "RIGHT")
    dir_col = ((80,220,80) if dir_str == "FWD" else
               (40,200,220) if dir_str == "LEFT" else (200,140,40))
    cv2.putText(canvas, dir_str, (CAM_W - 110, CAM_H - 120), font, 0.9, dir_col, 2, cv2.LINE_AA)

    cot_line = response.split("\n")[0][:90] if response else ""
    cv2.putText(canvas, f"CoT: {cot_line}", (10, CAM_H - 70), font, 0.34, CYAN, 1, cv2.LINE_AA)

    instr_disp = args.instruction[:85] + ("…" if len(args.instruction) > 85 else "")
    cv2.putText(canvas, instr_disp, (10, CAM_H - 30), font, 0.36, (180,180,255), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, CAM_H - 220), (CAM_W, CAM_H - 220), (60,60,80), 2)
    return canvas


def iso_panel(iso_bgr, query_step):
    panel = cv2.resize(iso_bgr, (MAP_W, MAP_H))
    font  = cv2.FONT_HERSHEY_SIMPLEX
    ov    = panel.copy()
    cv2.rectangle(ov, (4, 4), (460, 82), (15,15,25), -1)
    cv2.addWeighted(ov, 0.75, panel, 0.25, 0, panel)
    cv2.putText(panel, "WAREHOUSE — ISOMETRIC VIEW", (12, 30),
                font, 0.60, (230,230,230), 1, cv2.LINE_AA)
    cv2.putText(panel, f"BW18 TIC-VLA (InternVL3-1B + ActionExpert)  q={query_step:03d}",
                (12, 58), font, 0.46, (100,220,255), 1, cv2.LINE_AA)
    cv2.line(panel, (0,0), (0,MAP_H), (60,60,80), 2)
    return panel


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    model = load_model()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, args.fps, (VID_W, VID_H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {args.output_video}")

    IPC.mkdir(parents=True, exist_ok=True)
    (IPC / "vla_ready").write_text("1")
    log.info(f"Signalled vla_ready. Watching {IPC} ...")

    frame_history: deque[str] = deque(maxlen=args.history_len)
    ticvla_history: list      = []

    query_step    = 0
    frame_step    = 0
    last_seen     = -1
    last_response = ""
    ep_log        = []

    try:
        while True:
            if (IPC / "done").exists() and not (IPC / "frame_ready").exists():
                log.info("Isaac Sim signalled done.")
                break

            if not (IPC / "frame_ready").exists():
                time.sleep(0.02)
                continue

            frame_id = int((IPC / "frame_ready").read_text().strip())
            if frame_id == last_seen:
                time.sleep(0.02)
                continue
            last_seen = frame_id

            ego_path = IPC / "ego_frame.png"
            iso_path = IPC / "iso_frame.png"
            if not ego_path.exists():
                (IPC / "frame_ready").unlink(missing_ok=True)
                continue

            tmp_frame = str(TMP_FRAMES / f"frame_{frame_step:06d}.png")
            shutil.copy2(str(ego_path), tmp_frame)
            frame_history.append(tmp_frame)

            t0 = time.time()

            delayed = list(frame_history)[:-1] or [tmp_frame]
            current = tmp_frame
            current_ts = float(frame_step) * 0.1

            response, waypoints, _ = model.predict(
                delayed_image_paths=delayed,
                current_image_path=current,
                instruction=args.instruction,
                robot_state=None,
                history=ticvla_history if ticvla_history else None,
                current_timestamp=current_ts,
                time_delay=len(delayed) * 0.2,
            )
            lat = time.time() - t0
            query_step += 1

            wp = waypoints.detach().float().cpu().numpy()
            if wp.ndim == 3:
                wp = wp[0]   # (T, 2)

            lin, ang = waypoints_to_vel(wp)
            last_response = response

            ticvla_history.append({
                "timestamp": current_ts,
                "waypoints": wp[:3].tolist(),
            })
            if len(ticvla_history) > 10:
                ticvla_history = ticvla_history[-10:]

            log.info(f"q={query_step:3d}  lin={lin:+.3f}  ang={ang:+.3f}  "
                     f"lat={lat:.2f}s  wp0=({wp[0,0]:.3f},{wp[0,1]:.3f})")

            (IPC / "action.json").write_text(json.dumps({"lin": lin, "ang": ang}))
            (IPC / "action_ready").write_text("1")
            (IPC / "frame_ready").unlink(missing_ok=True)
            frame_step += 1

            # Build video frame
            pil_ego = Image.open(ego_path).convert("RGB")
            ego_bgr = cv2.cvtColor(np.array(pil_ego), cv2.COLOR_RGB2BGR)
            ep = ego_panel(ego_bgr, lin, ang, response, query_step, frame_step)

            if iso_path.exists():
                iso_bgr = cv2.imread(str(iso_path))
                if iso_bgr is None:
                    iso_bgr = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
            else:
                iso_bgr = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
                cv2.putText(iso_bgr, "ISO WARMING UP...", (MAP_W//2-130, MAP_H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,180,180), 2)

            ip = iso_panel(iso_bgr, query_step)
            writer.write(np.hstack([ep, ip]))

            ep_log.append({
                "frame": frame_step,
                "query": query_step,
                "lin":   round(lin, 4),
                "ang":   round(ang, 4),
                "lat_s": round(lat, 3),
                "wp0":   [round(float(wp[0,0]),4), round(float(wp[0,1]),4)],
            })

    finally:
        writer.release()
        shutil.rmtree(TMP_FRAMES, ignore_errors=True)

    log.info(f"\nVideo saved → {args.output_video}")
    log_path = Path(args.output_video).with_suffix(".json")
    log_path.write_text(json.dumps(ep_log, indent=2))
    log.info(f"Episode log → {log_path}")

    if ep_log:
        lins = [r["lin"] for r in ep_log]
        angs = [r["ang"] for r in ep_log]
        lats = [r["lat_s"] for r in ep_log]
        log.info(f"\n{frame_step} frames | {query_step} queries | "
                 f"avg_lin={np.mean(lins):.3f}  avg_ang={np.mean(angs):.3f}  "
                 f"avg_lat={np.mean(lats):.2f}s")


if __name__ == "__main__":
    main()
