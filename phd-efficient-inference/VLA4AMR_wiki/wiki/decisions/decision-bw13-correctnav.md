# Decision: BW13 CorrectNav-Inspired Training — Four Fixes for Closed-Loop Failure

**Type:** decision
**Status:** active
**Last updated:** 2026-07-01
**Related:** [[C1-AdaCoT]], [[decision-phase-free-bw12]], [[Nav-AMR-WH]], [[overview]]

## Decision

Adapt the CorrectNav Self-Correction Flywheel paradigm (Yu et al., AAAI 2026) to fix the BW12 closed-loop failure (F3). BW13 implements four static training-side fixes before dynamic online correction (BW14).

**BW13 output format:**
```
<think>optional reasoning</think>
<summary>scene description — ALWAYS present</summary>
ACTION_0: lin=x ang=y
ACTION_1: lin=x ang=y
ACTION_2: lin=x ang=y
ACTION_3: lin=x ang=y
```

**BW13 key parameters:** batch=1, grad_accum=32, max_len=1024, chunk_size=4, history_len=3, turn_weight=3.0, epochs=3, lr=2e-4, LoRA rank=32 α=64.

## Root Cause Analysis (F3)

BW12 offline val: ang MAE=0.0003 on turning frames. BW12 Isaac Sim closed-loop: ang=0.0000 for ALL 40 frames.
The per-frame imitation metric completely masks four failure modes:

| # | Failure Mode | Evidence |
|---|-------------|----------|
| F3.1 | **Imitation ≠ decision** | Offline turning frames are already mid-turn; live inference must initiate turns proactively from approach frames where GT is still ang≈0 |
| F3.2 | **Mode collapse** | ~80% straight frames → ang=0 learned as overwhelming default prior |
| F3.3 | **Memory self-locking** | CoT fires once → summary frozen → context reinforces "straight ahead" → CoT never fires again |
| F3.4 | **No covariate recovery data** | Pure expert trajectories → model never trained on approach-phase states where turn initiation is needed |

## The Four Fixes

### Fix 1 — Multi-Frame History (H=3)
Pass the last 3 frames as a sequence rather than a single image.

**What it solves:** F3.1, F3.4 — temporal context "robot has been moving straight for 3 steps → intersection NOW visible" provides the proactive cue. Single-frame inference sees only the current state; multi-frame exposes trajectory.

**Implementation:** Qwen2.5-VL supports multi-image input natively. Images concatenated as `[frame_{i-2}, frame_{i-1}, frame_i]` in the user message content list.

### Fix 2 — Action Chunking (N=4)
Predict 4 consecutive actions per query instead of 1.

**What it solves:** F3.1 (primary fix for turn initiation). Approach frames 1–3 steps before the turn boundary now have turning actions in chunk positions 1–3. This is free implicit look-ahead supervision from the existing dataset — no new data collection needed.

**Example:** Frame i=97 (approaching turn, GT ang≈0, turn starts at i=100):
- ACTION_0: lin=0.800 ang=0.000  ← current frame (still straight)
- ACTION_1: lin=0.600 ang=-0.300 ← 1 step ahead (starting to slow+turn)
- ACTION_2: lin=0.400 ang=-0.800 ← 2 steps ahead (mid-turn)
- ACTION_3: lin=0.300 ang=-1.200 ← 3 steps ahead (full turn)

The model learns "I see intersection approaching → I should turn in ~1-2 steps."

**Inference:** Execute all 4 actions sequentially in sim before next VLA query (1 query → 4 sim steps → 1 query ...).

### Fix 3 — Turn Loss Reweighting (3×)
Apply 3× loss weight to any sample where any action in the chunk has `|ang| > 0.05 rad/s`.

**What it solves:** F3.2 (mode collapse). Without reweighting, 80% straight samples dominate gradient signal. 3× weight pushes the effective ratio from 83%/17% to 67%/33% — turning frames now represent 1/3 of the effective training signal even though they are only 16.9% of samples.

**Implementation:** Custom `weighted_loss()` using per-sample CE before reduction, then `where(has_turn, TURN_WEIGHT, 1.0)` mask applied per sample before mean.

### Fix 4 — Always-On Summary (decoupled from CoT)
Every frame outputs `<summary>` regardless of whether `<think>` is triggered.

**What it solves:** F3.3 (memory self-locking). In BW12, CoT fires at 13.2% of frames. If CoT doesn't fire → no summary → memory stays stale → next frame also unlikely to trigger CoT. Fix: annotate 100% of frames with synthetic summaries. CoT (`<think>`) remains optional and entropy-gated. Summary generation is deterministic from the action chunk.

**Summary synthesis rules:**
- Any `|ang| > 0.05` in chunk → "approaching junction, preparing to turn {right|left}"
- `lin_0 < 0.55` → "clear warehouse aisle ahead, moving at reduced speed"
- else → "clear warehouse aisle ahead, moving forward"

**Inference:** Extract `<summary>` from every frame's output (guaranteed present), pass forward as `Memory:` for next frame.

## CorrectNav Alignment

| CorrectNav component | BW13 implementation |
|---------------------|---------------------|
| Keyframe perception (MLLM sees key states) | Multi-frame input (H=3) |
| Action chunking (N=4 future actions) | Chunk size N=4, ACTION_0..3 format |
| Self-correction data (deviation→correction) | BW14 (dynamic Isaac Sim flywheel) |
| Continual learning (retrain on correction data) | BW14 |

BW13 = CorrectNav Fixes 1+2 (static). BW14 = CorrectNav Fixes 3+4 (dynamic online).

## Training Run

**Launched:** 2026-07-01, tmux session `bw13_train` on cvit-car-simulator
- Dataset: 46,035 train / 4,936 val (same bw11_dataset, chunked)
- Turning chunks: 16.9% train, 12.6% val (up from 13.2% due to look-ahead)
- Optimizer steps: 4,314 total (3 epochs, effective batch=32)
- Output: `~/VLA4AMR/checkpoints/bw13_lora/`

## How to Apply

All BW13+ inference scripts must:
1. Pass 3 consecutive frames as multi-image input
2. Parse ACTION_0..3 from output; execute all 4 sequentially before next query
3. Extract `<summary>` from every frame (guaranteed) for rolling Memory:
4. Keep entropy-gated `<think>` for CoT (unchanged from BW12)
