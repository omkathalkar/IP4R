#!/usr/bin/env bash
# nav_stack/sim/run_episode_simulator.sh — Hybrid A*/VLA episode launcher.
#
# Preserves the exact env setup validated on 2026-08-04/05/06:
#   - DISPLAY=:1 / XAUTHORITY (Xorg GDM session, not Xvfb)
#   - CUDA_DEVICE_ORDER=PCI_BUS_ID  (4060 Ti=0, Blackwell=1)
#   - VK_ICD_FILENAMES (force NVIDIA Vulkan ICD)
#   - No CUDA_VISIBLE_DEVICES for Isaac (carb.cudainterop crashes with it)
#   - CUDA_VISIBLE_DEVICES=0 for VLA (4060 Ti)
#
# Usage:
#   bash run_episode_simulator.sh [start_x start_y goal_x goal_y] [--steps N] [--skip-vla]
#
# Examples:
#   bash run_episode_simulator.sh                           # defaults (3,1)→(22,15)
#   bash run_episode_simulator.sh 3.0 1.0 22.0 15.0
#   bash run_episode_simulator.sh 3.0 1.0 22.0 15.0 --steps 900 --skip-vla

set -eo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
START_X="${1:-3.0}"
START_Y="${2:-1.0}"
GOAL_X="${3:-22.0}"
GOAL_Y="${4:-15.0}"
shift 4 2>/dev/null || true    # remaining args passed through to Python

CKPT="$HOME/Desktop/flowvla_v3_output/flowvla_v3_best.pt"
IPC_DIR="/tmp/navstack_ipc"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
TICVLA_PY="$HOME/miniconda3/envs/tic-vla/bin/python3"
ISAAC_PY="$HOME/miniconda3/envs/isaac6/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="$SCRIPT_DIR/vla_inference_worker.py"
ISAAC_SCRIPT="$SCRIPT_DIR/run_episode_simulator.py"
VLA_LOG="$HOME/Desktop/navstack_vla_worker.log"
ISAAC_LOG="$HOME/Desktop/navstack_isaac.log"

echo "================================================================"
echo "nav_stack — Hybrid A*/VLA Episode — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Start   : ($START_X, $START_Y)"
echo "  Goal    : ($GOAL_X,  $GOAL_Y)"
echo "  Isaac GPU: 1 (Blackwell)   VLA GPU: 0 (4060 Ti)"
echo "================================================================"

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
[ -f "$WORKER_SCRIPT" ] || { echo "ERROR: VLA worker not found: $WORKER_SCRIPT"; exit 1; }
[ -f "$ISAAC_SCRIPT"  ] || { echo "ERROR: Isaac script not found: $ISAAC_SCRIPT";  exit 1; }

source "$CONDA_SH"
mkdir -p "$IPC_DIR"
rm -f "$IPC_DIR"/{frame.jpg,frame.ready,current_instruction.txt,action.json,action.ready,quit}

# ── Environment (identical to flowvla_v3_run_launch.sh — do not change) ──────
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export XDG_RUNTIME_DIR=/run/user/1000
export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=ubuntu:GNOME
export GDK_BACKEND=x11
export QT_QPA_PLATFORM=xcb

echo "  DISPLAY  : $DISPLAY"
echo "================================================================"

# ── VLA worker on 4060 Ti (background) ───────────────────────────────────────
echo "[1/2] Starting VLA inference worker (GPU 0, 4060 Ti)..."
CUDA_VISIBLE_DEVICES=0 "$TICVLA_PY" -u "$WORKER_SCRIPT" \
    --ckpt "$CKPT" \
    > "$VLA_LOG" 2>&1 &
VLA_PID=$!
echo "      PID=$VLA_PID  log → $VLA_LOG"
echo "      Waiting 25s for InternVL3-1B to load..."
sleep 25

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    echo "Shutting down..."
    touch "$IPC_DIR/quit" 2>/dev/null || true
    kill "$VLA_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# ── Isaac Sim on Blackwell (foreground) ───────────────────────────────────────
echo "[2/2] Starting Isaac Sim (GPU 1, Blackwell)..."
source /opt/ros/jazzy/setup.bash
# Do NOT set CUDA_VISIBLE_DEVICES for Isaac — carb.cudainterop crashes if set.
# GPU is selected via active_gpu=1 in SimulationApp config.
"$ISAAC_PY" -u "$ISAAC_SCRIPT" \
    --start-xy "$START_X" "$START_Y" \
    --goal-xy  "$GOAL_X"  "$GOAL_Y"  \
    "$@" \
    2>&1 | tee "$ISAAC_LOG"

echo "================================================================"
echo "Episode complete. Output: ~/Desktop/navstack_episode/"
echo "================================================================"
