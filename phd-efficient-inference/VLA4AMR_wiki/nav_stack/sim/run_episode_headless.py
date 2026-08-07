#!/usr/bin/env python3
"""
nav_stack/sim/run_episode_headless.py — Replay-based hybrid-stack eval on Ada HPC.

No Isaac Sim. Runs FlowVLA-BW v3 + ConfidenceGate on one BW17 DynaNav window
(pre-collected image data with GT future waypoints). Directly comparable to
Phase 9 (TIC-VLA baseline) — same ADE/FDE/heading-error metrics, plus the
new fallback-trigger-rate column.

Decision: Isaac Sim is NOT used on Ada (no GDM session, EGL untested on u22
nodes, and replay-based eval is more repeatable for paper numbers anyway).
The Phase 6 Isaac Sim integration on cvit-car-simulator provides the demo video;
Phase 7 here provides quantitative paper metrics.

Usage (via SLURM — see run_episode_ada.slurm):
  python3 run_episode_headless.py \
      --window-dir /ssd_scratch/om.kathalkar/bw17_dynav/test/DynaNav_json/<window> \
      --ckpt       /ssd_scratch/om.kathalkar/checkpoints/flowvla_v3_best.pt \
      --grid-dir   /ssd_scratch/om.kathalkar/nav_stack_grid \
      --out-dir    /home2/om.kathalkar/logs/phase7 \
      --data-root-old /home/cvit-car-simulator/Desktop/bw17_dynav \
      --data-root-new /ssd_scratch/om.kathalkar/bw17_dynav
"""

import sys, os, json, math, time, argparse, logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# ── nav_stack imports ─────────────────────────────────────────────────────────
_NAV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_NAV))
from nav_stack.common import Waypoint
from nav_stack.grid.generate_occupancy_grid import build_parametric_grid, inflate_grid, save_grid, load_grid
from nav_stack.planning.astar_planner import plan_path, load_grid_and_meta
from nav_stack.interface.waypoint_to_vla_input import (
    WaypointToVLAInputOffline, VOCAB, heading_to_instruction_key, compute_rel_heading,
)
from nav_stack.control.confidence_gate import ConfidenceGate
from nav_stack.control.pure_pursuit import PurePursuit

T_EMB_DIM = 64
N_ODE_STEPS = 20


# ── FlowActionHead2D (self-contained — no dep on training script) ─────────────

def _sinusoidal_embed(t, dim=T_EMB_DIM):
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FlowActionHead2D(nn.Module):
    def __init__(self, feat_dim, hidden, t_emb_dim=T_EMB_DIM, n_layers=3):
        super().__init__()
        self.cond_enc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.t_enc = nn.Sequential(
            nn.Linear(t_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.action_in = nn.Linear(2, hidden)
        layers, in_dim = [], hidden * 3
        for i in range(n_layers):
            out_dim = hidden if i < n_layers - 1 else 2
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = hidden
        self.denoiser = nn.Sequential(*layers)

    def forward(self, x_t, t, cond):
        h = torch.cat([self.action_in(x_t), self.cond_enc(cond), self.t_enc(_sinusoidal_embed(t))], dim=-1)
        return self.denoiser(h)

    @torch.no_grad()
    def sample(self, cond, n_steps=N_ODE_STEPS):
        B = cond.shape[0]
        x = torch.randn(B, 2, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), 1.0 - i * dt, device=cond.device)
            x = x - self.forward(x, t, cond) * dt
        return x


def load_flowvla(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = FlowActionHead2D(cfg["feat_dim"], cfg["hidden"],
                             cfg["t_emb_dim"], cfg["n_layers"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return (model,
            torch.tensor(ckpt["feat_mean"], dtype=torch.float32).to(device),
            torch.tensor(ckpt["feat_std"],  dtype=torch.float32).to(device),
            torch.tensor(ckpt["act_mean"],  dtype=torch.float32).to(device),
            torch.tensor(ckpt["act_std"],   dtype=torch.float32).to(device))


# ── BW17 DynaNav data loading ─────────────────────────────────────────────────

def remap_path(p: str, old_root: str, new_root: str) -> str:
    if old_root and new_root and p.startswith(old_root):
        return new_root + p[len(old_root):]
    return p


def load_window(window_dir: Path, old_root: str, new_root: str) -> dict | None:
    """Load one BW17 DynaNav window. Returns None if unusable."""
    instr_path = window_dir / "instruction.txt"
    if not instr_path.exists():
        return None
    instruction = instr_path.read_text().strip().strip('"')

    json_files = sorted(window_dir.glob("t_*.json"))
    if not json_files:
        return None
    data = json.loads(json_files[0].read_text())

    current   = data.get("current", {})
    gt_future = data.get("future",  [])
    history   = data.get("history", [])

    curr_img = remap_path(current.get("img", ""), old_root, new_root)
    if not curr_img or not os.path.exists(curr_img):
        return None

    hist_imgs = [
        remap_path(h["img"], old_root, new_root)
        for h in history if h.get("img")
    ]
    hist_imgs = [p for p in hist_imgs if os.path.exists(p)]

    return {
        "instruction": instruction,
        "curr_img":    curr_img,
        "hist_imgs":   hist_imgs,
        "gt_future":   gt_future,
        "window_name": window_dir.name,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def gt_first_offset(gt_future: list) -> tuple[float, float] | None:
    """Return (dx, dy) of first GT step in robot frame, or None if zero/missing."""
    if not gt_future:
        return None
    off = gt_future[0].get("offset", [0, 0, 0])
    dx, dy = float(off[0]), float(off[1])
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return None
    return dx, dy


def action_to_displacement(lin: float, ang: float, dt: float = 0.2) -> tuple[float, float]:
    """
    Approximate robot-frame displacement from (lin, ang) over one time step dt.
    Uses Euler integration: forward = lin*dt, lateral = (lin*sin(ang*dt)).
    """
    fwd  = lin * dt
    lat  = lin * math.sin(ang * dt) if abs(ang) > 1e-6 else 0.0
    return fwd, lat


def compute_heading_error(pred_dx, pred_dy, gt_dx, gt_dy) -> float:
    """Unsigned heading error in degrees between two 2D displacements."""
    pred_angle = math.atan2(pred_dy, max(abs(pred_dx), 1e-6) * (1 if pred_dx >= 0 else -1))
    gt_angle   = math.atan2(gt_dy,   max(abs(gt_dx),   1e-6) * (1 if gt_dx   >= 0 else -1))
    err = abs(pred_angle - gt_angle) % (2 * math.pi)
    if err > math.pi:
        err = 2 * math.pi - err
    return math.degrees(err)


def compute_ade_fde(pred_dx, pred_dy, gt_future, dt=0.2, horizon=10):
    """
    Roll out predicted (constant) action for `horizon` steps and compute
    ADE/FDE against GT cumulative offsets.
    """
    n = min(horizon, len(gt_future))
    if n == 0:
        return None, None

    dists = []
    cum_pred_x = cum_pred_y = 0.0
    theta = 0.0   # assume robot facing forward at step 0
    for i in range(n):
        # Step in robot frame → accumulate (simplified: constant heading)
        cum_pred_x += pred_dx
        cum_pred_y += pred_dy
        gt = gt_future[i].get("offset", [0, 0, 0])
        ex = cum_pred_x - float(gt[0])
        ey = cum_pred_y - float(gt[1])
        dists.append(math.hypot(ex, ey))

    return sum(dists) / len(dists), dists[-1]


# ── Occupancy grid (cached) ───────────────────────────────────────────────────

def ensure_grid_ada(grid_dir: str, resolution: float = 0.05, robot_radius: float = 0.4):
    gd = Path(grid_dir)
    if (gd / "warehouse_occupancy_grid.npy").exists():
        return load_grid_and_meta(grid_dir)
    print(f"Generating parametric grid → {grid_dir}", flush=True)
    grid, meta = build_parametric_grid(resolution=resolution)
    inflated   = inflate_grid(grid, robot_radius, resolution)
    save_grid(inflated, meta, gd)
    return inflated, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-dir",     required=True)
    ap.add_argument("--ckpt",           required=True)
    ap.add_argument("--grid-dir",       required=True)
    ap.add_argument("--out-dir",        required=True)
    ap.add_argument("--ticvla-repo",    default="/home2/om.kathalkar/VLA4AMR/TIC-VLA-main")
    ap.add_argument("--base-model",     default="/home2/om.kathalkar/VLA4AMR/checkpoints/internvl3-1b")
    ap.add_argument("--device",         default="cuda:0")
    ap.add_argument("--data-root-old",  default="",
                    help="Prefix in JSON image paths from the simulator machine")
    ap.add_argument("--data-root-new",  default="",
                    help="Replacement prefix for image paths on Ada")
    ap.add_argument("--start-xy",  nargs=2, type=float, default=[3.0, 1.0])
    ap.add_argument("--goal-xy",   nargs=2, type=float, default=[22.0, 15.0])
    ap.add_argument("--mag-thresh",type=float, default=0.15)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_dir = Path(args.window_dir)
    result_path = out_dir / f"{window_dir.name}_result.json"

    # Skip if already done (SLURM retry safety)
    if result_path.exists():
        print(f"Already done: {result_path}", flush=True)
        return

    print(f"=== Phase 7 eval: {window_dir.name} ===", flush=True)

    # ── Load window data ──────────────────────────────────────────────────
    window = load_window(window_dir, args.data_root_old, args.data_root_new)
    if window is None:
        print(f"  SKIP: unusable window ({window_dir.name})", flush=True)
        result_path.write_text(json.dumps({"window": window_dir.name, "skipped": True}))
        return

    # ── A* plan ───────────────────────────────────────────────────────────
    grid, meta = ensure_grid_ada(args.grid_dir)
    waypoints  = plan_path(grid, tuple(args.start_xy), tuple(args.goal_xy), meta)
    print(f"  A* plan: {len(waypoints)} waypoints", flush=True)

    # ── Load backbone + FlowVLA ───────────────────────────────────────────
    sys.path.insert(0, args.ticvla_repo)
    from ticvla.models.ticvla import TICVLA
    from ticvla.utils.vision import load_image
    from transformers import AutoTokenizer

    t0 = time.time()
    backbone = TICVLA(model_path=args.base_model, action_horizon_steps=30,
                      action_num_layers=3, train_vlm=False).to(args.device).eval()
    tokenizer    = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    embed_tokens = backbone.vlm.language_model.model.embed_tokens

    model, feat_mean, feat_std, act_mean, act_std = load_flowvla(args.ckpt, args.device)
    print(f"  Models loaded in {time.time()-t0:.1f}s", flush=True)

    # ── Confidence gate (magnitude-only: no progress history in replay mode) ─
    gate = ConfidenceGate(
        mag_thresh       = args.mag_thresh,
        progress_window  = 1,    # progress check requires sequential frames
        progress_min_m   = 0.0,  # disable: set min to 0 so progress always passes
        pure_pursuit     = PurePursuit(lookahead_dist=1.2, max_lin=0.3, min_lin=0.1),
    )

    # ── Per-frame evaluation (one window = one inference call) ────────────
    wp_to_vla   = WaypointToVLAInputOffline()
    # Use first waypoint as the nominal target for this window
    target_wp   = waypoints[0]
    vla_input   = wp_to_vla.step(
        args.start_xy[0], args.start_xy[1], 0.0,
        target_wp.x, target_wp.y, is_final_wp=False,
    )

    # Tokenise instruction
    instr_str = vla_input.instruction_str
    ids = tokenizer(instr_str, return_tensors="pt", add_special_tokens=True).input_ids.to(args.device)
    with torch.no_grad():
        feat_t = embed_tokens(ids)[0].float().mean(0)

    # Vision feature
    t_infer = time.time()
    try:
        img_tensor = load_image(window["curr_img"], input_size=448, max_num=1).to(torch.bfloat16).to(args.device)
        with torch.no_grad():
            img_emb = backbone.vlm.extract_feature(img_tensor)
            feat_v  = img_emb.reshape(-1, img_emb.shape[-1]).mean(0).float()
    except Exception as e:
        print(f"  ERROR loading image: {e}", flush=True)
        result_path.write_text(json.dumps({"window": window_dir.name, "error": str(e)}))
        return

    feat   = torch.cat([feat_v, feat_t]).unsqueeze(0)
    feat_n = (feat - feat_mean) / feat_std
    with torch.no_grad():
        pred_n = model.sample(feat_n)
    pred = (pred_n * act_std + act_mean).squeeze(0).cpu().tolist()

    vla_lin = float(np.clip(pred[0], -0.5, 0.5))
    vla_ang = float(np.clip(pred[1], -1.0,  1.0))
    latency = time.time() - t_infer

    # ── ConfidenceGate decision ───────────────────────────────────────────
    fin_lin, fin_ang, used_vla, gate_entry = gate.step(
        vla_lin, vla_ang,
        args.start_xy[0], args.start_xy[1], 0.0,
        target_wp.x, target_wp.y, 0, waypoints,
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    DT = 0.2   # 5 Hz frame rate
    pred_dx, pred_dy = action_to_displacement(fin_lin, fin_ang, DT)
    vla_dx,  vla_dy  = action_to_displacement(vla_lin, vla_ang, DT)

    gt_off = gt_first_offset(window["gt_future"])
    if gt_off is not None:
        gt_dx, gt_dy = gt_off
        heading_err_final = compute_heading_error(pred_dx, pred_dy, gt_dx, gt_dy)
        heading_err_vla   = compute_heading_error(vla_dx,  vla_dy,  gt_dx, gt_dy)
        ade_final, fde_final = compute_ade_fde(pred_dx, pred_dy, window["gt_future"])
        ade_vla,   fde_vla   = compute_ade_fde(vla_dx,  vla_dy,  window["gt_future"])
    else:
        heading_err_final = heading_err_vla = ade_final = fde_final = ade_vla = fde_vla = None

    vla_magnitude = math.hypot(vla_lin, vla_ang)

    result = {
        "window":           window_dir.name,
        "instruction_vla":  instr_str,
        "instruction_data": window["instruction"],
        # VLA raw output
        "vla_lin":          round(vla_lin, 4),
        "vla_ang":          round(vla_ang, 4),
        "vla_magnitude":    round(vla_magnitude, 4),
        # Gate decision
        "used_vla":         used_vla,
        "mag_ok":           gate_entry.mag_ok,
        "fin_lin":          round(fin_lin, 4),
        "fin_ang":          round(fin_ang, 4),
        # Metrics vs GT
        "heading_err_final_deg": round(heading_err_final, 2) if heading_err_final is not None else None,
        "heading_err_vla_deg":   round(heading_err_vla,   2) if heading_err_vla   is not None else None,
        "ade_final_m":           round(ade_final, 4)         if ade_final          is not None else None,
        "fde_final_m":           round(fde_final, 4)         if fde_final          is not None else None,
        "ade_vla_m":             round(ade_vla, 4)           if ade_vla            is not None else None,
        "fde_vla_m":             round(fde_vla, 4)           if fde_vla            is not None else None,
        # Meta
        "latency_s":        round(latency, 3),
        "n_waypoints":      len(waypoints),
        "start_xy":         args.start_xy,
        "goal_xy":          args.goal_xy,
    }

    result_path.write_text(json.dumps(result, indent=2))
    h = f"{heading_err_final:.1f}°" if heading_err_final is not None else "n/a"
    src = "VLA" if used_vla else "PP "
    print(f"  [{src}] mag={vla_magnitude:.3f}  h_err={h}  "
          f"ade={ade_final:.4f}m  lat={latency:.2f}s", flush=True)
    print(f"  Result → {result_path}", flush=True)


if __name__ == "__main__":
    main()
