# Decision: BW14 — DiscreteNav + CorrectNav Flywheel Architecture

**Type:** decision
**Status:** active
**Last updated:** 2026-07-03
**Related:** [[C1-AdaCoT]], [[decision-bw13-correctnav]], [[Nav-AMR-WH]], [[IsaacSim]], [[NovaCarter]]

## Summary

BW13 closed-loop demo (2026-07-01) still outputs ang=0.0000 for all 40 frames — identical failure to BW12 despite 4 CorrectNav-inspired fixes. Root cause is structural: continuous lin/ang regression has a mode-collapse attractor at (0,0). BW14 replaces continuous action prediction with discrete navigation tokens, motivated by VLN-CE and CorrectNav, and adds multi-frame video history to break memory self-locking.

## Problem with BW13

| Failure | BW13 fix | Why it still fails |
|---------|----------|-------------------|
| Mode collapse (80% straight) | 3× turn loss weight | Weight helps training but doesn't change the output space — model still outputs continuous (0.0, 0.0) when uncertain |
| Memory self-locking | Always-on summary | Still writes "clear aisle ahead" from own output, not from visual change |
| Imitation ≠ decision | Action chunking N=4 | Chunk positions 1-3 showed turn signals in offline eval but don't translate to closed-loop initiation |
| No recovery data | Not addressed | Deferred to Phase 2 |

The BW13 offline eval showed an alternating pattern: non-zero ang at chunk positions 1 and 3 during approach (mean |ang|≈0.1-0.3 at pos 1,3). This proves the model "knows" a turn is coming but cannot commit to it at position 0 (current frame). The continuous output space allows infinitely small values; the model hedges.

## Core change: discrete action tokens

Replace `ACTION_i: lin=x ang=y` with `ACTIONS: FORWARD,TURN_RIGHT,...` (N=6).

**Why this eliminates mode collapse:**
- "TURN_RIGHT" is a single vocabulary token. There is no gradient path from "I'm uncertain" to "output a small TURN_RIGHT". Either the token is predicted or it isn't.
- Discrete output is also zero-shot interpretable: we can inspect the exact sequence and understand model reasoning without decoding continuous values.
- Loss: cross-entropy over the action token sequence. Correct token = 0 loss regardless of magnitude.

**Token vocabulary and velocity mapping (execution):**

| Token | lin (m/s) | ang (rad/s) | Condition from GT |
|-------|-----------|-------------|-------------------|
| FORWARD | 0.35 | 0.0 | lin>0.05, \|ang\|≤0.05 |
| TURN_LEFT | 0.25 | +0.7 | ang > +0.05 |
| TURN_RIGHT | 0.25 | -0.7 | ang < -0.05 |
| STOP | 0.0 | 0.0 | lin≤0.05, \|ang\|≤0.05 |

The ANG_THR=0.05 matches bw13_infer.py — same as the threshold used to define "turning" in all prior evals.

**Token distribution in bw11_dataset (estimated):**
- FORWARD: ~80% (same as continuous ang≈0)
- TURN_LEFT: ~8%
- TURN_RIGHT: ~8%
- STOP: ~4%

Turn reweighting (3×) still applied for sequences containing TURN_LEFT/TURN_RIGHT.

## Multi-frame history: 6 frames at 64 tokens/image

BW13 attempted H=3 multi-frame but hit OOM (3×576=1728 tokens >> 1024 max_len). 

**BW14 fix:** `max_pixels=64×28×28=50,176` per image → each 336×336 frame downsampled to 224×224 effective → 64 visual tokens per image.

With H=6 frames: 6×64=384 vision tokens + ~256 text = ~640 total → well within 1024 max_len.

**Why H=6 breaks memory self-locking:**
The model now sees visual change across 6 frames (~6 seconds). At 3-4 frames before an intersection, the aisle end and opening become visible in the image sequence. The model can detect "aisle narrowing → opening ahead" purely from visual context, without relying on the memory string. The `<summary>` still updates but is now secondary to visual evidence.

## CorrectNav flywheel plan (two phases)

**Phase 1 (BW14 base): supervised learning on GT discrete tokens**
- Convert bw11_dataset (lin_vel, ang_vel) → discrete tokens on-the-fly
- Train H=6, N=6 with turn reweighting 3×
- Expected outcome: ~100% parse rate, turning token accuracy >90%

**Phase 2 (BW14-fly): deviation detection + correction data**
Adapt CorrectNav `find_point()` to Isaac Sim:
1. Run BW14 base in Isaac Sim closed-loop; log robot (x,y) pose per frame via `/chassis/odom`
2. Deviation detection: if distance from reference path > 0.3m for 3+ consecutive frames → trigger correction
3. Correction data collection: snap pose to nearest reference path point, re-execute with scripted planner from that point, record frames as correction episode
4. Label correction frames: FORWARD/TURN_LEFT/TURN_RIGHT from ShortestPath recovery actions
5. Add correction episodes to training set, retrain
6. Repeat flywheel 2-3 iterations

Phase 2 directly addresses the "no recovery data" failure mode from BW12/BW13. The correction data teaches the model what to do when already-deviated — the critical missing supervision signal.

## Why VLN-CE was not directly adopted

VLN-CE uses discrete actions {STOP, FORWARD, TURN_LEFT, TURN_RIGHT} on Habitat-sim + Matterport3D. We adopt the same token vocabulary but NOT the dataset or simulator:
- Matterport3D is indoor photorealistic but NOT warehouse industrial environments
- Habitat-sim ≠ Isaac Sim (different physics, sensor model, ROS2 bridge)
- VLN-CE's waypoint predictor outputs (r, θ) → converted to discrete tokens at each step

The CorrectNav flywheel is the direct inspiration for Phase 2. VLN-CE motivates the discrete token vocabulary (proven to work for navigation without regression collapse).

## Key metrics to track (BW14 eval)

**Offline (bw14_infer.py):**
- Parse rate (target: 100%)
- Token accuracy at position 0 (FORWARD/TURN_* accuracy, overall and per task)
- Turn-initiation check: TURN token rate at approach frames, positions 2-5 (should be high for cross_turn tasks)

**Closed-loop (bw14_sim_vla.py):**
- Token distribution over episode: must include TURN_LEFT or TURN_RIGHT tokens
- ang nonzero fraction (target: >30% for cross_turn episodes)
- Success: robot completes turn and arrives at destination aisle

## Files

| File | Role |
|------|------|
| `bw14_train.py` | Phase 1 training — discrete token format, H=6, N=6 |
| `bw14_infer.py` | Offline eval — token accuracy, turn-initiation check |
| `bw14_sim_vla.py` | Live demo — pairs with `bw11_sim_isaac.py`, token→vel mapper |

## Sources

- [[decision-bw13-correctnav]] — prior CorrectNav adaptation and F3 root cause
- CorrectNav (Yu et al., AAAI 2026) — `find_point()` deviation detection, correction flywheel
- VLN-CE (Krantz et al., ECCV 2020) — discrete navigation token vocabulary
