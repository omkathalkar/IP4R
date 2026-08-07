#!/usr/bin/env python3
"""
nav_stack/sim/vla_inference_worker.py — VLA inference process for hybrid nav stack.

Run in tic-vla conda env (GPU 0, 4060 Ti).  Mirrors flowvla_v3_gui_vla.py but:
  - Reads the current instruction from IPC_DIR/current_instruction.txt each step
    (written by the Isaac Sim process when it advances to a new waypoint).
  - Caches feat_t per instruction string — only re-tokenises when the string changes.
  - Writes IPC-format action.json with keys "lin_vel" and "ang_vel" (same as before).

Usage (started by run_episode_simulator.sh — do not invoke directly):
  CUDA_VISIBLE_DEVICES=0 python3 vla_inference_worker.py \
      --ckpt ~/Desktop/flowvla_v3_output/flowvla_v3_best.pt
"""

import sys, os, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn

IPC_DIR      = "/tmp/navstack_ipc"
FRAME_IN     = os.path.join(IPC_DIR, "frame.jpg")
FRAME_RD     = os.path.join(IPC_DIR, "frame.ready")
INSTR_FILE   = os.path.join(IPC_DIR, "current_instruction.txt")
ACT_OUT      = os.path.join(IPC_DIR, "action.json")
ACT_RD       = os.path.join(IPC_DIR, "action.ready")
QUIT_F       = os.path.join(IPC_DIR, "quit")

TICVLA_REPO  = "/home/cvit-car-simulator/VLA4AMR/TIC-VLA-main"
BASE_MODEL   = "/home/cvit-car-simulator/VLA4AMR/checkpoints/internvl3-1b"
DEVICE       = "cuda:0"
T_EMB_DIM    = 64
N_STEPS      = 20


def sinusoidal_embed(t, dim=T_EMB_DIM):
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
        h = torch.cat([self.action_in(x_t), self.cond_enc(cond), self.t_enc(sinusoidal_embed(t))], dim=-1)
        return self.denoiser(h)

    @torch.no_grad()
    def sample(self, cond, n_steps=N_STEPS):
        B = cond.shape[0]
        x = torch.randn(B, 2, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), 1.0 - i * dt, device=cond.device)
            x = x - self.forward(x, t, cond) * dt
        return x


def load_model(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    cfg   = ckpt["config"]
    model = FlowActionHead2D(cfg["feat_dim"], cfg["hidden"],
                             cfg["t_emb_dim"], cfg["n_layers"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    feat_mean = torch.tensor(ckpt["feat_mean"], dtype=torch.float32).to(DEVICE)
    feat_std  = torch.tensor(ckpt["feat_std"],  dtype=torch.float32).to(DEVICE)
    act_mean  = torch.tensor(ckpt["act_mean"],  dtype=torch.float32).to(DEVICE)
    act_std   = torch.tensor(ckpt["act_std"],   dtype=torch.float32).to(DEVICE)
    return model, feat_mean, feat_std, act_mean, act_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/Desktop/flowvla_v3_output/flowvla_v3_best.pt"))
    args = ap.parse_args()

    os.makedirs(IPC_DIR, exist_ok=True)
    for f in [FRAME_RD, ACT_RD, QUIT_F]:
        if os.path.exists(f): os.remove(f)

    print("=" * 58)
    print("VLA Inference Worker — nav_stack hybrid")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  IPC dir    : {IPC_DIR}")
    print("=" * 58)

    sys.path.insert(0, TICVLA_REPO)
    import logging; logging.basicConfig(level=logging.WARNING)
    from ticvla.models.ticvla import TICVLA
    from ticvla.utils.vision import load_image

    print("Loading InternVL3-1B backbone...")
    t0 = time.time()
    backbone = TICVLA(model_path=BASE_MODEL, action_horizon_steps=30,
                      action_num_layers=3, train_vlm=False).to(DEVICE).eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    from transformers import AutoTokenizer
    tokenizer    = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    embed_tokens = backbone.vlm.language_model.model.embed_tokens

    model, feat_mean, feat_std, act_mean, act_std = load_model(args.ckpt)
    print(f"  FlowActionHead2D loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # feat_t cache — recompute only when instruction string changes
    cached_instr: str | None = None
    cached_feat_t: torch.Tensor | None = None

    def get_feat_t(instr: str) -> torch.Tensor:
        nonlocal cached_instr, cached_feat_t
        if instr != cached_instr:
            ids = tokenizer(instr, return_tensors="pt",
                            add_special_tokens=True).input_ids.to(DEVICE)
            with torch.no_grad():
                cached_feat_t = embed_tokens(ids)[0].float().mean(0)
            cached_instr = instr
            print(f"  [vla] instruction updated → \"{instr[:60]}\"")
        return cached_feat_t

    # Default instruction (forward) until Isaac Sim writes its first instruction
    default_instr = "Drive forward through the warehouse aisle"
    _ = get_feat_t(default_instr)

    print("\nWaiting for frames... (Ctrl+C to stop)\n")

    import cv2
    WIN = "VLA Worker — Nova Carter POV"
    _gui = False
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 640, 360)
        cv2.imshow(WIN, np.zeros((360, 640, 3), dtype=np.uint8))
        cv2.waitKey(1)
        _gui = True
    except cv2.error:
        pass

    step = 0
    while True:
        if os.path.exists(QUIT_F):
            print("Quit signal received.")
            break

        if not os.path.exists(FRAME_RD):
            if _gui: cv2.waitKey(10)
            else:    time.sleep(0.01)
            continue

        t_infer = time.time()

        # Read current instruction (updated per-waypoint by Isaac process)
        if os.path.exists(INSTR_FILE):
            try:
                instr = open(INSTR_FILE).read().strip()
            except Exception:
                instr = default_instr
        else:
            instr = default_instr

        feat_t = get_feat_t(instr)

        try:
            img_tensor = load_image(FRAME_IN, input_size=448, max_num=1).to(torch.bfloat16).to(DEVICE)
            with torch.no_grad():
                img_emb = backbone.vlm.extract_feature(img_tensor)
                feat_v  = img_emb.reshape(-1, img_emb.shape[-1]).mean(0).float()
        except Exception as e:
            print(f"  [step {step}] Frame error: {e}")
            if os.path.exists(FRAME_RD): os.remove(FRAME_RD)
            continue

        feat   = torch.cat([feat_v, feat_t]).unsqueeze(0)
        feat_n = (feat - feat_mean) / feat_std
        with torch.no_grad():
            pred_n = model.sample(feat_n)
        pred = (pred_n * act_std + act_mean).squeeze(0).cpu().tolist()

        lin = float(np.clip(pred[0], -0.5, 0.5))
        ang = float(np.clip(pred[1], -1.0,  1.0))
        lat = time.time() - t_infer
        step += 1
        print(f"  step {step:4d} | lin={lin:+.3f}  ang={ang:+.3f} | {lat*1000:.0f}ms | \"{instr[:40]}\"")

        if _gui:
            raw = cv2.imread(FRAME_IN)
            if raw is not None:
                h, w = raw.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                ov = raw.copy()
                cv2.rectangle(ov, (0, h-50), (w, h), (0,0,0), -1)
                cv2.addWeighted(ov, 0.6, raw, 0.4, 0, raw)
                cv2.putText(raw, instr[:60], (8, 22), font, 0.45, (180,255,180), 1, cv2.LINE_AA)
                cv2.putText(raw, f"lin:{lin:+.3f}  ang:{ang:+.3f}  step:{step}",
                            (8, h-14), font, 0.46, (255,255,255), 1, cv2.LINE_AA)
                cv2.imshow(WIN, raw); cv2.waitKey(1)

        os.remove(FRAME_RD)
        with open(ACT_OUT, "w") as f:
            json.dump({"lin_vel": lin, "ang_vel": ang, "step": step,
                       "instruction": instr}, f)
        open(ACT_RD, "w").close()

    if _gui: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
