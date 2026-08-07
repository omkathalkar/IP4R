#!/bin/bash
# nav_stack/sim/run_episode_ada.slurm — Phase 7 SLURM array job.
#
# Runs 20 BW17 DynaNav windows through the hybrid stack (FlowVLA-BW v3 +
# ConfidenceGate) in replay mode.  Results written to:
#   /home2/om.kathalkar/logs/phase7/<window_name>_result.json
#
# PRE-SUBMIT CHECKLIST (run once on Ada before sbatch):
#   mkdir -p /home2/om.kathalkar/logs/phase7
#   mkdir -p /ssd_scratch/om.kathalkar/nav_stack_grid
#   conda activate tic-vla && \
#     python3 /home2/om.kathalkar/nav_stack/grid/generate_occupancy_grid.py \
#       --out-dir /ssd_scratch/om.kathalkar/nav_stack_grid --backend parametric
#
# Submit:
#   sbatch /home2/om.kathalkar/nav_stack/sim/run_episode_ada.slurm
#
# Aggregate results (after all tasks finish):
#   python3 /home2/om.kathalkar/nav_stack/eval/aggregate_phase7.py \
#       --results-dir /home2/om.kathalkar/logs/phase7

#SBATCH --job-name=vla4amr_phase7
#SBATCH --partition=u22
#SBATCH --account=om.kathalkar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --output=/home2/om.kathalkar/logs/phase7/slurm_%A_%a.out
#SBATCH --error=/home2/om.kathalkar/logs/phase7/slurm_%A_%a.err

set -eo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
ADA_HOME=/home2/om.kathalkar
ADA_SCRATCH=/ssd_scratch/om.kathalkar

DATA_ROOT="$ADA_SCRATCH/bw17_dynav"
CKPT="$ADA_SCRATCH/checkpoints/flowvla_v3_best.pt"
GRID_DIR="$ADA_SCRATCH/nav_stack_grid"
OUT_DIR="$ADA_HOME/logs/phase7"
NAV_STACK="$ADA_HOME/nav_stack"
TICVLA_REPO="$ADA_HOME/VLA4AMR/TIC-VLA-main"
BASE_MODEL="$ADA_HOME/VLA4AMR/checkpoints/internvl3-1b"

SIM_DATA_ROOT="/home/cvit-car-simulator/Desktop/bw17_dynav"

# ── Conda ──────────────────────────────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tic-vla

mkdir -p "$OUT_DIR"

# ── Window selection ───────────────────────────────────────────────────────────
DYNAV_DIR="$DATA_ROOT/test/DynaNav_json"
if [ ! -d "$DYNAV_DIR" ]; then
    echo "ERROR: DynaNav_json not found at $DYNAV_DIR"
    exit 1
fi

mapfile -t WINDOWS < <(find "$DYNAV_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
N_WINDOWS=${#WINDOWS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "$TASK_ID" -ge "$N_WINDOWS" ]; then
    echo "TASK $TASK_ID >= $N_WINDOWS available windows — nothing to do."
    exit 0
fi

WINDOW_DIR="${WINDOWS[$TASK_ID]}"
WINDOW_NAME="$(basename "$WINDOW_DIR")"

echo "================================================================"
echo "Phase 7 — SLURM Task $TASK_ID / window $WINDOW_NAME"
echo "  Job ID   : ${SLURM_JOB_ID}_${TASK_ID}"
echo "  Host     : $(hostname)"
echo "  GPU      : $CUDA_VISIBLE_DEVICES"
echo "  Window   : $WINDOW_DIR"
echo "================================================================"

# ── Run eval ──────────────────────────────────────────────────────────────────
python3 "$NAV_STACK/sim/run_episode_headless.py" \
    --window-dir    "$WINDOW_DIR"    \
    --ckpt          "$CKPT"          \
    --grid-dir      "$GRID_DIR"      \
    --out-dir       "$OUT_DIR"       \
    --ticvla-repo   "$TICVLA_REPO"   \
    --base-model    "$BASE_MODEL"    \
    --data-root-old "$SIM_DATA_ROOT" \
    --data-root-new "$DATA_ROOT"

echo "Task $TASK_ID complete."
