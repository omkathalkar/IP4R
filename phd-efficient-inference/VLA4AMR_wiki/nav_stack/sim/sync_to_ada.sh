#!/usr/bin/env bash
# nav_stack/sim/sync_to_ada.sh — Sync data, checkpoint, and code to Ada HPC.
#
# Run FROM cvit-car-simulator:
#   bash ~/Desktop/nav_stack/sim/sync_to_ada.sh
#
# After sync, submit the SLURM array job from Ada:
#   ssh om.kathalkar@ada.iiit.ac.in \
#     "sbatch /home2/om.kathalkar/nav_stack/sim/run_episode_ada.slurm"

set -eo pipefail

ADA_USER=om.kathalkar
ADA_HOST=ada.iiit.ac.in
ADA_HOME=/home2/om.kathalkar
ADA_SCRATCH=/ssd_scratch/om.kathalkar

LOCAL_DATA=~/Desktop/bw17_dynav
LOCAL_CKPT=~/Desktop/flowvla_v3_output/flowvla_v3_best.pt
LOCAL_NAV=~/Desktop/nav_stack

echo "================================================================"
echo "VLA4AMR Phase 7 — Sync to Ada HPC"
echo "  Target   : $ADA_USER@$ADA_HOST"
echo "  Scratch  : $ADA_SCRATCH"
echo "================================================================"

# ── Validate locals ────────────────────────────────────────────────────────────
[ -d "$LOCAL_DATA"  ] || { echo "ERROR: $LOCAL_DATA not found"; exit 1; }
[ -f "$LOCAL_CKPT"  ] || { echo "ERROR: $LOCAL_CKPT not found"; exit 1; }
[ -d "$LOCAL_NAV"   ] || { echo "ERROR: $LOCAL_NAV not found";  exit 1; }

# ── 1. BW17 DynaNav data ──────────────────────────────────────────────────────
echo ""
echo "[1/4] BW17 DynaNav data → $ADA_HOST:$ADA_SCRATCH/bw17_dynav/"
ssh "$ADA_USER@$ADA_HOST" "mkdir -p $ADA_SCRATCH/bw17_dynav"
rsync -avzP --exclude='*.pyc' --exclude='__pycache__' \
    "$LOCAL_DATA/" \
    "$ADA_USER@$ADA_HOST:$ADA_SCRATCH/bw17_dynav/"

# ── 2. FlowVLA-BW v3 checkpoint ───────────────────────────────────────────────
echo ""
echo "[2/4] Checkpoint → $ADA_HOST:$ADA_SCRATCH/checkpoints/flowvla_v3_best.pt"
ssh "$ADA_USER@$ADA_HOST" "mkdir -p $ADA_SCRATCH/checkpoints"
rsync -avzP \
    "$LOCAL_CKPT" \
    "$ADA_USER@$ADA_HOST:$ADA_SCRATCH/checkpoints/flowvla_v3_best.pt"

# ── 3. nav_stack code ─────────────────────────────────────────────────────────
echo ""
echo "[3/4] nav_stack code → $ADA_HOST:$ADA_HOME/nav_stack/"
rsync -avzP \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='*.npy' \
    --exclude='*.yaml' \
    "$LOCAL_NAV/" \
    "$ADA_USER@$ADA_HOST:$ADA_HOME/nav_stack/"

# ── 4. Pre-generate occupancy grid on Ada ─────────────────────────────────────
echo ""
echo "[4/4] Pre-generating warehouse grid on Ada (skips if already cached)..."
ssh "$ADA_USER@$ADA_HOST" bash <<'EOF'
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate tic-vla
    GRID_DIR=/ssd_scratch/om.kathalkar/nav_stack_grid
    GRID_NPY=$GRID_DIR/warehouse_occupancy_grid.npy
    mkdir -p "$GRID_DIR"
    if [ -f "$GRID_NPY" ]; then
        echo "  Grid already cached at $GRID_DIR — skipping."
    else
        echo "  Generating parametric grid → $GRID_DIR ..."
        python3 /home2/om.kathalkar/nav_stack/grid/generate_occupancy_grid.py \
            --out-dir "$GRID_DIR" \
            --backend parametric
        echo "  Grid generated OK."
    fi
EOF

echo ""
echo "================================================================"
echo "Sync complete."
echo ""
echo "Next steps:"
echo "  1. Ensure log dir exists on Ada:"
echo "       ssh $ADA_USER@$ADA_HOST 'mkdir -p $ADA_HOME/logs/phase7'"
echo ""
echo "  2. Submit the SLURM array job:"
echo "       ssh $ADA_USER@$ADA_HOST \\"
echo "         'sbatch $ADA_HOME/nav_stack/sim/run_episode_ada.slurm'"
echo ""
echo "  3. Monitor:"
echo "       ssh $ADA_USER@$ADA_HOST 'squeue -u $ADA_USER'"
echo ""
echo "  4. Aggregate results after all tasks finish:"
echo "       ssh $ADA_USER@$ADA_HOST \\"
echo "         'python3 $ADA_HOME/nav_stack/eval/aggregate_phase7.py \\"
echo "            --results-dir $ADA_HOME/logs/phase7'"
echo "================================================================"
