# Decision: BW18 — TIC-VLA training on BW17 warehouse dataset

**Type:** decision
**Status:** complete
**Last updated:** 2026-07-11
**Related:** [[decision-bw16-ticvla-warehouse]], [[bw17-warehouse-dataset]], [[C2-MidLevelActionHead]]

## Summary

BW18 trains TIC-VLA (InternVL3-1B + ActionExpert) on the BW17 warehouse dataset — a purpose-built
9-task synthetic dataset with richer task diversity than BW16's 6-task Nav-AMR-WH data.
Two-stage: Stage 1 VLM CoT fine-tuning → Stage 2 action head training.

## Motivation

BW16 used the existing Nav-AMR-WH dataset (converted from bw11_dataset). BW17 was built specifically
to address BW16's task coverage gaps:

| Aspect | BW16 | BW18 |
|--------|------|------|
| Tasks | 6 | 9 (adds slow variants + u-turn) |
| Episodes | 434 train | 392 train |
| Windows | ~50K frames | 3,590 windows |
| Format | per-frame JSONs | per-window DynaNav_json |
| Instructions | static | 4 augmented phrasings/window |
| CoT | phase-based | decision-frame only (Qwen2.5-VL) |
| Action supervision | linear+angular vel | 30-step FLU waypoint horizon |

BW17's 30-step waypoint horizon (3s) is better matched to TIC-VLA's ActionExpert architecture
than per-frame action labels.

## Key design choices

### 1. Every-5th-file filter bypass
TICVLADataset_VLM._find_samples() takes every 5th file per directory. Solution: one subdir per
(window × instruction-variant) so each dir has exactly 1 JSON → 100% sample retention.

### 2. Decision-frame CoT only
CoT annotation is expensive (~486 chars/sample via Qwen2.5-VL-7B). Non-decision windows get
`cot=""` which TIC-VLA's `_build_messages()` handles gracefully (omits CoT block from prompt).

### 3. FLU waypoints vs raw velocities
BW16 converted (lin_vel, ang_vel) to cumulative FLU via Euler integration. BW17 uses the same
FLU convention but with the cleaner `world_to_flu()` function:
```python
flu_x = dx*sin(h) + dy*cos(h)
flu_y = -dx*cos(h) + dy*sin(h)
```
30-waypoint horizon at 10Hz gives the action head 3s of future context.

### 4. GPU allocation
- Collection + instruct/annotate: `openvla` conda env, `cuda:0` = RTX 5000 Pro (47.3 GiB)
  (REVERSED vs nvidia-smi in openvla env)
- Training: `tic-vla` conda env, `CUDA_VISIBLE_DEVICES=0` = RTX 5000 Pro

## Training configuration

### Stage 1 — VLM CoT fine-tuning
```yaml
batch_size: 4
accumulate_grad_batches: 4   # effective batch = 16
learning_rate: 2.0e-5
max_epochs: 10
warmup_steps: 300
precision: bf16-mixed
```
Script: `bash ~/VLA4AMR/code/bw18_train_vlm.sh`
Log: `~/Desktop/bw18_vlm.log`

### Stage 2 — Action head training
```yaml
batch_size: 32
learning_rate: 1.0e-4
max_epochs: 15
action_horizon_steps: 30
warmup_steps: 300
```
Script: `bash ~/VLA4AMR/code/bw18_train_action.sh [vlm_ckpt]`
Log: `~/Desktop/bw18_action.log`

## Expected outcomes

- Stage 1: val CoT loss convergence (expect ~10 epochs, ~2h on RTX 5000 Pro)
- Stage 2: val_ade < 0.3m, val_fde < 0.5m (BW16 reference: ade=0.206, fde=0.338)
- Closed-loop: live Isaac Sim demo replacing BW16 model

## Results (2026-07-11)

| Metric | BW16 | BW18 | Δ |
|--------|------|------|---|
| val_ADE | 0.206 m | **0.090 m** | **−56%** |
| val_FDE | 0.338 m | **0.185 m** | **−45%** |

Per-task test ADE: aisle_straight=0.044m, turns=0.058–0.066m, u_turn=0.089m,
multi_turn=0.077–0.097m, goal_seek=0.217m, obstacle_avoid=0.349m.

**Bugs encountered and fixed:**
- Stage 2 `IndexError` in `policy_data.py:322`: `delayed_idx = max(0, current_idx - files_back)`
  (single-file dirs + non-zero timestamp → look-back index out of range for 1-element list)

## Status

- [x] BW17 dataset built and verified (14,360 DynaNav_json dirs)
- [x] Training scripts uploaded to simulator
- [x] Stage 1 VLM training — stopped at epoch 0 (best val=0.278, overfitting from epoch 1)
- [x] Stage 2 Action training — 15 epochs, best val_ADE=0.077m at epoch 14
- [x] Offline eval — ADE=0.090m, FDE=0.185m on 2,160 test samples
- [x] Closed-loop Isaac Sim demo — `bw18_sim_vla.py`, 40 queries, avg_lat=2.99s, right turn executed per instruction

## Sources

- `bw17_collect.py` through `bw17_convert.py`
- `bw18_train_vlm.{sh,yaml}`, `bw18_train_action.{sh,yaml}`, `.env.bw18`
- [[bw17-warehouse-dataset]] — full dataset documentation
