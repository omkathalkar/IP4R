# Decision: BW16 — TIC-VLA (InternVL3-1B + ActionExpert) on Warehouse Data

**Type:** decision
**Status:** active
**Last updated:** 2026-07-09
**Related:** [[decision-bw15-correctnav-fullstack]], [[Nav-AMR-WH]], [[IsaacSim]], [[C2-MidLevelActionHead]], [[C6-EvaluationProtocol]]

## Summary

BW16 introduces TIC-VLA (Think-in-Control VLA, ICML 2026) as an additional VLA
baseline alongside CorrectNav (BW15). TIC-VLA uses InternVL3-1B VLM + an
ActionExpert cross-attention head that reads KV cache states to predict T=10
waypoints at 10Hz. No pretrained warehouse checkpoint is publicly available, so
we train from scratch on bw11_dataset using a two-stage supervised pipeline.

## Context

- CorrectNav (BW15) uses LLaVA-Video-7B-Qwen2 + discrete {Forward, Left, Right, Stop}
  tokens → deterministic but no continuous trajectory.
- TIC-VLA outputs continuous waypoints (x,y at 10Hz for 3s) via a dedicated
  ActionExpert head, giving smoother Nav2 integration.
- TIC-VLA paper is ICML 2026 (arXiv 2602.02459); code + data at
  `handsomeYun/TIC-VLA` (HuggingFace dataset only, no checkpoint).
- BW16 enables direct comparison between discrete-token (BW15) and continuous-
  waypoint (BW16) approaches for C4-FusionAblation.

## Architecture

| Component | Detail |
|-----------|--------|
| VLM | InternVL3-1B (OpenGVLab/InternVL3-1B), ~1B params, bf16 |
| Vision encoder | InternViT-300M (SigLIP-style, integrated into InternVL3) |
| Action head | ActionExpert: cross-attention transformer (6 layers) on VLM KV cache |
| Action output | T=10 waypoints (x,y) at 0.1s intervals → 3s look-ahead |
| Training | Stage 1: VLM CoT fine-tune; Stage 2: ActionExpert frozen VLM |
| Data | bw16_ticvla_dataset (converted from bw11_dataset, 50,971 JSON files) |

## Data conversion (bw16_convert_ticvla.py)

bw11_dataset (HF format) → TIC-VLA JSON format:

**Output directory structure:**
```
~/Desktop/bw16_ticvla_dataset/
├── train/DynaNav_json/<episode_id>/img_0.json … img_N.json
├── train/DynaNav_json/<episode_id>/instruction.txt
├── train/DynaNav_json/<episode_id>/cot.txt  (if episode has CoT)
└── val/DynaNav_json/<episode_id>/...
```

**Key conversion logic:**
- `image_path` fix: `/Desktop/bwXX/` → `/Desktop/datasets/bwXX/` (all 50,971 images verified to exist)
- `future` offsets: cumulative world-frame (FLU) positions from current frame via Euler integration
  `x += lin_vel * cos(heading) * 0.1; y += lin_vel * sin(heading) * 0.1; heading += ang_vel * 0.1`
- `current.orientation = [0,0,0,1]` (identity); `future[0].orientation = yaw_to_quat(ang_vel * 0.1)`
  → `_compute_robot_state_from_future` recovers `yaw_rate = ang_vel` exactly
- `future[29].offset` = cumulative position at 3s → used as VLM 3-s waypoint
- 90 future frames stored per JSON (required for VLM 3s/6s/9s guidance waypoints)
- Frames near episode end (<90 future frames): `guidance_waypoint = -100` (masked in VLM loss)
- `DynaNav_json/` path component satisfies `_detect_dataset_info` → 'wheeled robot' robot type
- Per-episode `instruction.txt` + `cot.txt` referenced by absolute path in each JSON

**Stats:** 377 train episodes, 57 val episodes, 50,971 total JSON files.

## Training setup

**Conda env:** `tic-vla` (Python 3.11, torch 2.8.0+cu128, transformers>=4.40,<5.0)
**GPU:** RTX 5000 Pro 48GB (CUDA_VISIBLE_DEVICES=0, same as BW15)
**Base model:** `~/VLA4AMR/checkpoints/internvl3-1b` (downloaded from HF)

### Stage 1 — VLM CoT fine-tuning

```bash
bash ~/VLA4AMR/code/bw16_train_vlm.sh
# Config: ~/VLA4AMR/code/bw16_train_vlm.yaml
# batch=4, accum=4, lr=3e-5, epochs=10, bf16, warmup=500 steps
# Output: ~/Desktop/bw16_ticvla_output/checkpoints/ticvla/vlm/last.ckpt
```

### Stage 2 — ActionExpert training (VLM frozen)

```bash
bash ~/VLA4AMR/code/bw16_train_action.sh
# Config: ~/VLA4AMR/code/bw16_train_action.yaml
# batch=32, accum=1, lr=1e-4, epochs=10, bf16
# Requires Stage 1 checkpoint
```

### Environment variables (.env.bw16)

```bash
BW16_DATA_ROOT=~/Desktop/bw16_ticvla_dataset
BW16_BASE_MODEL=~/VLA4AMR/checkpoints/internvl3-1b
BW16_OUTPUT_DIR=~/Desktop/bw16_ticvla_output
CUDA_VISIBLE_DEVICES=0
TRANSFORMERS_OFFLINE=1
```

## Isaac Sim integration (planned)

TIC-VLA ships `DynaNav/behavior/nova_carter_test_ticvla.py` as a BehaviorScript.
This requires adaptation for Isaac Sim 6.0.0.1 (TIC-VLA targets 5.0.0):
- Import paths for `isaacsim.robot.wheeled_robots` may differ
- `TICVLA_CHECKPOINT_PATH` env var points to Stage 2 checkpoint
- BehaviorScript runs inference in a background thread (non-blocking)
- Outputs T=10 waypoints → converted to (lin_vel, ang_vel) via diff-drive kinematics

## Open questions

- Stage 1 converges: InternVL3-1B is pretrained on natural images; warehouse
  domain gap may require more epochs or a lower initial LR (try 1e-5 if val_loss
  plateaus early).
- Action head with 30-step horizon (3s): episodes average ~80 frames; frames
  within 30 of episode end get zero-padded waypoints (masked by zeros, not -100,
  in the action head loss). Confirm this doesn't bias the model toward stopping.
- BehaviorScript API compatibility: Isaac Sim 6.0.0.1 vs 5.0.0 — test imports
  before full integration.
- CorrectNav (BW15) vs TIC-VLA (BW16) navigation quality: need closed-loop
  comparison metric (success rate, collision rate, path efficiency).

## Sources

- [[decision-bw15-correctnav-fullstack]] — prior CorrectNav pivot decision
- TIC-VLA paper: arXiv 2602.02459, ICML 2026
- Code: `~/VLA4AMR/TIC-VLA-main/` on cvit-car-simulator
- Dataset: `~/Desktop/bw16_ticvla_dataset/` on cvit-car-simulator
- Training scripts: `~/VLA4AMR/code/bw16_*.{sh,yaml}` on cvit-car-simulator
