#!/usr/bin/env python3
"""
flowvla_train_goalcond.py — Train goal-conditioned ActionRegressor.

Input features: text_feat(896) ‖ geo_feat(3)
  text_feat : instruction → InternVL3-1B embed_tokens mean (896-d)
  geo_feat  : [dist_to_goal, cos(heading_err), sin(heading_err)]
              where heading_err = normalize_angle(atan2(dy,dx) - robot_theta)

Labels: (lin_vel, ang_vel) from P-controller demonstrations.

This avoids the null-model collapse of the vision-only approach: the P-controller
labels are derived directly from geometry, so (text+geo) → action is a learnable
mapping with a consistent signal.

Usage (tic-vla env, GPU 0):
  CUDA_VISIBLE_DEVICES=0 python3 flowvla_train_goalcond.py \\
      --dataset ~/Desktop/large_dataset/<ts>/dataset.jsonl \\
      --out     ~/Desktop/flowvla_goalcond/
"""

import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--dataset",  required=True)
ap.add_argument("--out",      required=True)
ap.add_argument("--epochs",   type=int,   default=300)
ap.add_argument("--lr",       type=float, default=1e-3)
ap.add_argument("--batch",    type=int,   default=256)
ap.add_argument("--hidden",   type=int,   default=256)
ap.add_argument("--device",   default="cuda:0")
args = ap.parse_args()

DEVICE       = args.device
TICVLA_REPO  = "/home/cvit-car-simulator/VLA4AMR/TIC-VLA-main"
BASE_MODEL   = "/home/cvit-car-simulator/VLA4AMR/checkpoints/internvl3-1b"
os.makedirs(args.out, exist_ok=True)
CKPT_OUT     = os.path.join(args.out, "flowvla_goalcond_best.pt")
LOG_OUT      = os.path.join(args.out, "train_log.jsonl")

print("=" * 64)
print("FlowVLA Goal-Conditioned Training")
print(f"  Dataset  : {args.dataset}")
print(f"  Output   : {args.out}")
print(f"  Epochs   : {args.epochs}  LR={args.lr}  Batch={args.batch}")
print(f"  Device   : {DEVICE}")
print("=" * 64)

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_angle(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

# ── Load dataset.jsonl ────────────────────────────────────────────────────────
dataset_dir = str(Path(args.dataset).parent)
with open(args.dataset) as f:
    records = [json.loads(l) for l in f if l.strip()]

print(f"\nLoaded {len(records)} steps from {len(set(r['ep_id'] for r in records))} episodes")
instructions_used = sorted(set(r["instruction"] for r in records))
print(f"Unique instructions ({len(instructions_used)}):")
for instr in instructions_used:
    n = sum(1 for r in records if r["instruction"] == instr)
    print(f"  [{n:4d} steps]  {instr}")

# ── Load InternVL3-1B tokenizer for text features ────────────────────────────
print(f"\nLoading InternVL3-1B tokenizer on {DEVICE} ...")
sys.path.insert(0, TICVLA_REPO)
import logging; logging.basicConfig(level=logging.WARNING)
from ticvla.models.ticvla import TICVLA
from transformers import AutoTokenizer

t0 = time.time()
backbone = TICVLA(model_path=BASE_MODEL, action_horizon_steps=30,
                  action_num_layers=3, train_vlm=False).to(DEVICE).eval()
tokenizer    = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
embed_tokens = backbone.vlm.language_model.model.embed_tokens
print(f"  Loaded in {time.time()-t0:.1f}s  (tokenizer only — no vision encoder needed)")

# ── Pre-extract text features (one per unique instruction) ────────────────────
print("\nPre-extracting text features ...")
text_feat_cache = {}
for instr in instructions_used:
    ids = tokenizer(instr, return_tensors="pt",
                    add_special_tokens=True).input_ids.to(DEVICE)
    with torch.no_grad():
        feat_t = embed_tokens(ids)[0].float().mean(0)
    text_feat_cache[instr] = feat_t.cpu()
    print(f"  OK  {instr[:60]}")

TEXT_DIM = next(iter(text_feat_cache.values())).shape[0]  # 896
GEO_DIM  = 3   # [dist, cos_heading_err, sin_heading_err]
FEAT_DIM = TEXT_DIM + GEO_DIM

# ── Build feature matrix from dataset records ─────────────────────────────────
print(f"\nBuilding (text ‖ geo) features for {len(records)} steps ...")
all_feats   = []
all_actions = []
skip = 0

for i, rec in enumerate(records):
    gx, gy = rec["goal_x"], rec["goal_y"]
    rx, ry, rt = rec["robot_x"], rec["robot_y"], rec["robot_theta"]

    dx, dy = gx - rx, gy - ry
    dist   = math.hypot(dx, dy)
    h_to_g = math.atan2(dy, dx)
    h_err  = normalize_angle(h_to_g - rt)

    geo = torch.tensor([dist, math.cos(h_err), math.sin(h_err)], dtype=torch.float32)
    feat = torch.cat([text_feat_cache[rec["instruction"]], geo])  # (899,)
    action = torch.tensor([rec["lin_vel"], rec["ang_vel"]], dtype=torch.float32)

    all_feats.append(feat)
    all_actions.append(action)

    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(records)} done", flush=True)

print(f"  Done: {len(all_feats)} steps  ({skip} skipped)")

feats   = torch.stack(all_feats)    # (N, 899)
actions = torch.stack(all_actions)  # (N, 2)

# ── Normalise ─────────────────────────────────────────────────────────────────
feat_mean = feats.mean(0)
feat_std  = feats.std(0).clamp(min=1e-6)
act_mean  = actions.mean(0)
act_std   = actions.std(0).clamp(min=1e-6)

feats_n   = (feats   - feat_mean) / feat_std
actions_n = (actions - act_mean)  / act_std

print(f"\nAction stats (raw):")
print(f"  lin_vel  mean={act_mean[0]:+.4f}  std={act_std[0]:.4f}"
      f"  min={actions[:,0].min():+.4f}  max={actions[:,0].max():+.4f}")
print(f"  ang_vel  mean={act_mean[1]:+.4f}  std={act_std[1]:.4f}"
      f"  min={actions[:,1].min():+.4f}  max={actions[:,1].max():+.4f}")
print(f"\nGeo feature stats (dist, cos_h, sin_h):")
geo_raw = feats[:, TEXT_DIM:]
print(f"  dist     mean={geo_raw[:,0].mean():.3f}  std={geo_raw[:,0].std():.3f}"
      f"  min={geo_raw[:,0].min():.3f}  max={geo_raw[:,0].max():.3f}")
print(f"  cos_h    mean={geo_raw[:,1].mean():.3f}  std={geo_raw[:,1].std():.3f}")
print(f"  sin_h    mean={geo_raw[:,2].mean():.3f}  std={geo_raw[:,2].std():.3f}")

# ── Model ─────────────────────────────────────────────────────────────────────
class ActionRegressorGoalCond(nn.Module):
    def __init__(self, feat_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden * 2), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, feat):
        return self.net(feat)

    @torch.no_grad()
    def sample(self, feat, n_steps=None):
        return self.forward(feat)


model    = ActionRegressorGoalCond(FEAT_DIM, args.hidden).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nActionRegressorGoalCond: {n_params:,} parameters  "
      f"feat_dim={FEAT_DIM} (text={TEXT_DIM}+geo={GEO_DIM})  hidden={args.hidden}")

# ── Training ──────────────────────────────────────────────────────────────────
dataset_t = TensorDataset(feats_n, actions_n)
loader    = DataLoader(dataset_t, batch_size=args.batch, shuffle=True, drop_last=False)

optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)

print(f"\nTraining for {args.epochs} epochs ...")
print(f"  {'Epoch':>6}  {'Loss':>10}  {'lin_MAE':>8}  {'ang_MAE':>8}  {'LR':>10}")
print("  " + "─" * 52)

log_f     = open(LOG_OUT, "w")
best_loss = float("inf")
feats_n_d   = feats_n.to(DEVICE)
actions_n_d = actions_n.to(DEVICE)
actions_d   = actions.to(DEVICE)

for epoch in range(1, args.epochs + 1):
    model.train()
    ep_loss = 0.0
    for fb, ab in loader:
        fb, ab = fb.to(DEVICE), ab.to(DEVICE)
        optim.zero_grad()
        pred = model(fb)
        loss = nn.functional.mse_loss(pred, ab)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        ep_loss += loss.item() * fb.shape[0]
    ep_loss /= len(feats_n)
    sched.step()

    freq = 10 if args.epochs <= 300 else 25
    if epoch % freq == 0 or epoch == args.epochs:
        model.eval()
        with torch.no_grad():
            pred_n = model(feats_n_d)
            pred   = pred_n * act_std.to(DEVICE) + act_mean.to(DEVICE)
            mae_lin = (pred[:, 0] - actions_d[:, 0]).abs().mean().item()
            mae_ang = (pred[:, 1] - actions_d[:, 1]).abs().mean().item()
        lr_now = sched.get_last_lr()[0]
        print(f"  {epoch:>6}  {ep_loss:>10.6f}  {mae_lin:>8.4f}  {mae_ang:>8.4f}  {lr_now:>10.2e}",
              flush=True)
        log_f.write(json.dumps({
            "epoch": epoch, "loss": round(ep_loss, 6),
            "mae_lin": round(mae_lin, 5), "mae_ang": round(mae_ang, 5),
            "lr": lr_now,
        }) + "\n"); log_f.flush()

        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({
                "model_state":  model.state_dict(),
                "config":       {"feat_dim": FEAT_DIM, "hidden": args.hidden,
                                 "text_dim": TEXT_DIM, "geo_dim": GEO_DIM},
                "feat_mean":    feat_mean.tolist(),
                "feat_std":     feat_std.tolist(),
                "act_mean":     act_mean.tolist(),
                "act_std":      act_std.tolist(),
                "instructions": {r["ep_label"]: r["instruction"] for r in records},
                "val_mae":      {"lin": round(mae_lin, 5), "ang": round(mae_ang, 5)},
                "epoch":        epoch,
                "dataset":      args.dataset,
                "n_train":      len(feats),
                "model_type":   "ActionRegressorGoalCond",
            }, CKPT_OUT)

log_f.close()

print(f"\nBest loss  : {best_loss:.6f}")
print(f"Checkpoint : {CKPT_OUT}")
print(f"Train log  : {LOG_OUT}")
print("Done.")
