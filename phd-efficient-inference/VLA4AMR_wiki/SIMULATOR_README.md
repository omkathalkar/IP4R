# VLA4AMR — Simulator Machine Onboarding Guide

Welcome to the project. This document covers everything you need to get up and running on the simulator machine.

---

## 1. Machine Overview

| Item | Details |
|------|---------|
| **Host** | `cvit-car-simulator` |
| **IP** | `10.2.141.227` |
| **OS** | Ubuntu 24.04 LTS |
| **CPU** | AMD Ryzen 9 7950X (16-core, 32-thread) |
| **RAM** | 62 GB |
| **GPU 0** | NVIDIA GeForce RTX 4060 Ti — 16 GB (Isaac Sim / light tasks) |
| **GPU 1** | NVIDIA RTX PRO 5000 Blackwell — 48 GB (VLA training / inference) |
| **Driver** | 570.211.01 (CUDA 12.8) |

---

## 2. SSH Access

```bash
ssh cvit-car-simulator@10.2.141.227
# password: cvit@123!
```

For passwordless commands from your own machine, install `sshpass`:
```bash
sshpass -p 'cvit@123!' ssh cvit-car-simulator@10.2.141.227
```

To copy files to/from the machine:
```bash
# local → simulator
sshpass -p 'cvit@123!' scp myfile.py cvit-car-simulator@10.2.141.227:~/VLA4AMR/code/

# simulator → local
sshpass -p 'cvit@123!' scp cvit-car-simulator@10.2.141.227:~/Desktop/output.mp4 .
```

---

## 3. Directory Layout

```
~/ (home: /home/cvit-car-simulator)
├── miniconda3/                  ← all Python environments live here
├── VLA4AMR/
│   ├── code/                   ← ALL experiment scripts (bwXX_*.py)
│   ├── checkpoints/            ← model weights
│   │   ├── internvl3-1b/       ← base VLM for TIC-VLA (BW16)
│   │   ├── openvla-7b/         ← base model for OpenVLA experiments
│   │   ├── bw11_lora/ … bw14_lora/   ← LoRA adapters per experiment
│   │   └── c4/                 ← C4 fusion ablation checkpoints
│   ├── TIC-VLA-main/           ← TIC-VLA source code (BW16 model)
│   ├── CorrectNav-main/        ← CorrectNav source code (BW15 model)
│   ├── assets/                 ← USD assets for Isaac Sim scenes
│   └── datasets/               ← symlinks / raw collected data
└── Desktop/
    ├── bw16_ticvla_dataset/    ← BW16 training/val data (377+57 episodes)
    ├── bw16_ticvla_output/     ← BW16 checkpoints + logs
    │   └── checkpoints/ticvla/
    │       ├── vlm/            ← Stage-1 VLM checkpoints
    │       └── action/         ← Stage-2 action head checkpoints
    ├── bw16_eval/              ← BW16 offline eval outputs
    └── logs/                   ← misc experiment logs
```

---

## 4. Conda Environments

| Env | Python | Purpose |
|-----|--------|---------|
| `tic-vla` | 3.11 | **BW16** — TIC-VLA training & inference (torch 2.8+cu128) |
| `isaac6` | 3.12 | **Isaac Sim 6.0.0.1** simulation (torch 2.11) |
| `openvla` | 3.10 | BW11–BW15 OpenVLA / Qwen2.5-VL inference (torch 2.11+cu128) |
| `annotate` | — | CoT annotation (BW10) |
| `vla_ros` | — | ROS 2 + VLA bridge nodes |
| `carla` | — | CARLA sim (not active) |
| `nerfstudio` | — | NeRF scene reconstruction |

### Activating an environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tic-vla          # or isaac6, openvla, etc.
```

---

## 5. GPU Assignment Convention

```bash
# GPU 0 = RTX 4060 Ti 16 GB  → Isaac Sim (headless rendering)
# GPU 1 = RTX PRO 5000 48 GB → VLA model (training / inference)

# Always set this before running Isaac Sim:
export CUDA_VISIBLE_DEVICES=0

# Always set this before running VLA inference/training:
export CUDA_VISIBLE_DEVICES=1   # or use .env.bw16 which sets it to 0
                                 # (remapped because Isaac Sim is on the other terminal)
```

> **Note:** When both Isaac Sim and the VLA run simultaneously, Isaac Sim gets GPU 0 and the VLA gets GPU 1. The `.env.bw16` file sets `CUDA_VISIBLE_DEVICES=0` which, inside the tic-vla terminal (where Isaac Sim is NOT running), maps to the RTX 5000 Pro.

---

## 6. ROS 2

The machine runs **ROS 2 Jazzy** (not Humble).

```bash
source /opt/ros/jazzy/setup.bash
ros2 --version    # should print "jazzy"
```

Always source this before running any ROS 2 nodes (data collection, live inference).

---

## 7. Isaac Sim

Isaac Sim 6.0.0.1 is installed as a **Python package** inside the `isaac6` conda environment (not as a standalone app).

### Launch (headless)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac6
source /opt/ros/jazzy/setup.bash

DISPLAY=:1 CUDA_VISIBLE_DEVICES=0 \
~/miniconda3/envs/isaac6/bin/python -u \
    ~/VLA4AMR/code/bw11_sim_isaac.py \
    --task-type cross_turn_right \
    2>&1 | tee ~/Desktop/bw11_sim_isaac.log
```

### Available task types

| Task | Description |
|------|-------------|
| `cross_turn_right` | Drive to intersection, turn right, depart |
| `cross_turn_left` | Drive to intersection, turn left, depart |
| `aisle_fwd` | Drive straight down the aisle |

### Known Isaac Sim bugs (already fixed in scripts)

- **Multi-GPU crash:** Always set `CUDA_VISIBLE_DEVICES=0` before launching — Isaac Sim crashes on multi-GPU without this.
- **Robot spawn xform:** Physics breaks if you modify the robot's Xform after spawning. Spawn once, move via velocity commands only.
- **XformCache:** Does not return physics-updated positions — use `dc.get_rigid_body_pose()` instead.

---

## 8. BW16 — TIC-VLA (Current Active Experiment)

BW16 is the current model: **TIC-VLA** (InternVL3-1B + ActionExpert) fine-tuned on our warehouse dataset.

### Best checkpoints (already trained)

```
VLM (Stage 1):
  ~/Desktop/bw16_ticvla_output/checkpoints/ticvla/vlm/
  └── ticvla-vlm-epoch=01-val_language_loss=0.1272.ckpt  ← USE THIS

Action Head (Stage 2):
  ~/Desktop/bw16_ticvla_output/checkpoints/ticvla/action/
  └── ticvla-action-epoch=09-val_total_loss=0.0331.ckpt  ← USE THIS
```

### Environment file

```bash
source ~/VLA4AMR/code/.env.bw16
# Sets: BW16_DATA_ROOT, BW16_BASE_MODEL, BW16_OUTPUT_DIR, CUDA_VISIBLE_DEVICES=0
```

### Run offline evaluation (generates MP4 + ADE/FDE metrics)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tic-vla
source ~/VLA4AMR/code/.env.bw16

python3 ~/VLA4AMR/code/bw16_infer.py \
    --vlm-ckpt   ~/Desktop/bw16_ticvla_output/checkpoints/ticvla/vlm/ticvla-vlm-epoch=01-val_language_loss=0.1272.ckpt \
    --action-ckpt ~/Desktop/bw16_ticvla_output/checkpoints/ticvla/action/ticvla-action-epoch=09-val_total_loss=0.0331.ckpt \
    --data-dir   ~/Desktop/bw16_ticvla_dataset/val/DynaNav_json \
    --out-dir    ~/Desktop/bw16_eval \
    --n-samples  50 \
    2>&1 | tee ~/Desktop/bw16_infer.log
```

Output: `~/Desktop/bw16_eval/bw16_eval.mp4` + `summary.json` + per-sample plots.

**Eval results (BW16):** ADE = 0.475 ± 0.280 m, FDE = 0.848 ± 0.816 m, ~0.85 s/query.

---

## 9. Live Isaac Sim Demo with VLA (Two-Terminal Setup)

The live demo runs two processes simultaneously via IPC in `/tmp/bw11_ipc/`.

### Terminal 1 — Isaac Sim (isaac6 env)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac6
source /opt/ros/jazzy/setup.bash

DISPLAY=:1 CUDA_VISIBLE_DEVICES=0 \
~/miniconda3/envs/isaac6/bin/python -u \
    ~/VLA4AMR/code/bw11_sim_isaac.py \
    --task-type cross_turn_right \
    2>&1 | tee ~/Desktop/bw11_sim_isaac.log
```

Wait until you see `[SIM] Waiting for VLA ready...` before starting Terminal 2.

### Terminal 2 — TIC-VLA VLA process (tic-vla env)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tic-vla
source ~/VLA4AMR/code/.env.bw16

python3 ~/VLA4AMR/code/bw16_sim_vla.py \
    --instruction "Move ahead in the aisle. At the end, if a dead end approaches, decide to turn right or left." \
    --output-video ~/Desktop/bw16_sim_demo.mp4 \
    2>&1 | tee ~/Desktop/bw16_sim_vla.log
```

The video is written live to `--output-video` and is complete when both processes exit.

### IPC protocol (for reference)

```
/tmp/bw11_ipc/
  ego_frame.png   ← Isaac Sim writes current egocentric camera frame
  iso_frame.png   ← Isaac Sim writes isometric overhead view
  vla_ready       ← VLA writes "1" when model is loaded and ready
  frame_ready     ← Isaac Sim writes frame_id when new frame available
  action.json     ← VLA writes {"lin": float, "ang": float}
  action_ready    ← VLA writes "1" when action is ready
  done            ← Isaac Sim writes "1" when episode ends
```

---

## 10. tmux — Running Jobs in Background

All long-running jobs use tmux so they survive SSH disconnection.

```bash
# Start a new named session
tmux new-session -d -s my_session 'command here'

# Attach to a running session
tmux attach -t my_session

# List all sessions
tmux ls

# Detach from a session (stay inside tmux)
Ctrl+B, then D

# Kill a session
tmux kill-session -t my_session

# Capture last 50 lines of output without attaching
tmux capture-pane -t my_session -p -S -50
```

---

## 11. Checking GPU Usage

```bash
# Live GPU monitor
watch -n 1 nvidia-smi

# Quick snapshot
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader
```

---

## 12. Quick Troubleshooting

| Symptom | Fix |
|---------|-----|
| Isaac Sim crashes immediately | Add `CUDA_VISIBLE_DEVICES=0` before the launch command |
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python-headless` in the active env |
| Video plays as solid green on macOS | Re-encode: `ffmpeg -y -i in.mp4 -vcodec libx264 -pix_fmt yuv420p out.mp4` |
| `Permission denied` on SSH | Password is `cvit@123!` — use `sshpass` or type it when prompted |
| `conda: command not found` | Run `source ~/miniconda3/etc/profile.d/conda.sh` first |
| IPC stale flags causing hang | `rm -rf /tmp/bw11_ipc && mkdir /tmp/bw11_ipc` |
| Training OOM on GPU 1 | Reduce `batch_size` in the YAML or add `--gradient_checkpointing` |

---

## 13. Key Contacts

| Role | Person |
|------|--------|
| Project lead (Mr. K) | Om Kathalkar |
| PI | Prof. CV Jawahar (CVIT, IIIT Hyderabad) |
| Manager | Dr. Shankar |
| Target venue | ICRA 2027, Seoul — submission deadline **15 Sep 2026** |
