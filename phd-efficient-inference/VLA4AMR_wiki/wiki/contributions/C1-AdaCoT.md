# C1 — AdaCoT Trigger Mechanism

**Type:** contribution
**Status:** active
**Last updated:** 2026-07-01 (BW13 launch)
**Related:** [[VLingNav]], [[C3-ConfidenceGatedHandoff]], [[overview]], [[decision-entropy-trigger]], [[Nav-AMR-WH]], [[decision-bw13-correctnav]]

## Summary

Entropy-gated adaptive chain-of-thought reasoning. When VLA token probability entropy exceeds threshold θ, chain-of-thought is invoked before action prediction. Threshold tuned per environment via ablation.

## Novelty claim

First entropy-gated adaptive reasoning applied to wheeled AMR navigation. No existing paper combines AdaCoT with 2D-SLAM in industrial warehouse settings.

## Technical design

```
Input: VLA token probability distribution at each step
Compute: H = -Σ p(t) log p(t)  [Shannon entropy]
If H > θ:  invoke CoT → explicit reasoning → action
If H ≤ θ:  direct action (fast path)
θ: tuned per environment via ablation on Isaac Sim scenes
```

Key distinction from [[VLingNav]]: VLingNav *learns* when to think from 2.9M annotated samples. Our entropy trigger *computes* it directly from model uncertainty — no additional training data needed.

## Why entropy gating

- VLingNav AdaCoT fires at 2.1% of steps. Dense CoT at every step actually hurts performance (ObjectNav: 25.3% vs 36.2% no-CoT).
- Fixed interval (k=5) gets 42.5% vs our adaptive design target of 50.1%.
- Entropy directly measures model uncertainty — high entropy = model unsure = need to reason.
- No annotation cost. VLingNav needs Nav-AdaCoT-2.9M; we need just θ calibration.

## Trigger conditions (expected)

- Intersections / multiple valid paths
- Partial occlusions blocking goal
- Goal-proximity ambiguity
- Dynamic obstacle crossing path
- Low AMCL localisation confidence (synergy with [[C3-ConfidenceGatedHandoff]])

## Ablation results (BW04, 2026-06-20, 450 robot-front warehouse frames)

**θ decision: θ = 2.795 nats**

Full 11-condition sweep across two rounds:

| Mode | θ | CoT rate | Mean latency |
|---|---|---|---|
| no_cot | — | 0.0% | 223ms |
| dense_cot | — | 100.0% | 419ms |
| fixed_k5 | — | 20.0% | 266ms |
| fixed_k20 | — | 5.1% | 238ms |
| entropy_gated | 2.830 | 2.0% | 231ms |
| **entropy_gated** | **2.795** | **3.1%** | **233ms** |
| entropy_gated | 2.719 | 5.1% | 237ms |
| entropy_gated | 2.664 | 10.0% | 245ms |

θ=2.795 chosen: 3.1% activation (nearest to VLingNav's 2.1%), +10ms overhead only.
CoT fires on the top 3% highest-entropy steps — intersections, occlusions, ambiguous goals.
When CoT fires, mean entropy drops 2.401→2.095, confirming CoT provides genuine disambiguation.

See [[decision-c1-theta]].

## Live integration status (BW04, 2026-06-20)

`bw04_c1_live.py` deployed on simulator and integrated into the full Isaac Sim loop via `bw04_openvla_node.py`.

Live entropy values on clear warehouse corridor: H = 0.5–1.5 nats (well below θ=2.795 → CoT off).
This is correct: straight-aisle frames are low-uncertainty. CoT will fire at intersections/occlusions.

**Key bug found and fixed:** Entropy probe used bare TASK_INSTR as prompt. With 32K-token vocabulary,
model probability is near-uniform → H ≈ ln(32000) = 10.37 nats always. Fix: use
`"In: What action should the robot take to {TASK_INSTR}? Out:"` — the trailing "Out:" primes model
to predict from the 256-bin action vocabulary, producing calibrated entropy.

Live performance:
- Inference: 0.34s/step (no CoT), 0.68s/step (with CoT)
- ROS2 pipeline: /front_stereo_camera → C1 server → /cmd_vel, entropy, cot_active, cot_text
- Action quality: base model outputs near-zero actions (expected — will improve with C2/C4 fine-tuning)

## Open questions

- What is the right entropy estimator? Token-level H, action-dimension H, or sequence-level H?
- Does θ need to be per-scene-type or globally calibrated?
- Interaction with VLingMem-style memory: should past CoT summaries inform current entropy estimate?
- CoT text quality: will improve significantly once model is fine-tuned on warehouse data (C2/C4).

## BW11 End-to-End Training (2026-06-30)

The full AdaCoT pipeline is now trained end-to-end on [[Nav-AMR-WH]]:

**Dataset:** bw11_dataset — 50,971 frames, 13.2% CoT annotated with `<think>...</think><summary>...</summary>` format  
**Annotation model:** Qwen2.5-VL-7B-Instruct (local, RTX PRO 5000 Blackwell, bfloat16)  
**Training:** Qwen2.5-VL-7B + LoRA rank=32 α=64, 3 epochs, best val_loss=**0.0551** (epoch 1)  
**Checkpoint:** `~/VLA4AMR/checkpoints/bw11_lora/best_lora/`  

**Training objective (AdaCoT, VLingNav-style):**
- CoT frames (13.2%): predict `<think>...</think><summary>...</summary>\nACTION: lin=x ang=y`
- Plain frames (86.8%): predict `ACTION: lin=x ang=y`
- Loss: cross-entropy on assistant tokens only (system+user masked with -100)

**Val set results (4,936 samples):**
- 100% parse rate — model always outputs valid `ACTION: lin=x ang=y`
- Overall lin MAE = 0.0000, ang MAE = 0.0105
- Turn phases: lin MAE=0.0000, ang MAE=0.0013 (near-perfect)
- obj_goal: ang MAE=0.0904 (most variable task, fewest samples)

## BW12 Phase-Free Training (2026-07-01)

**Key advance:** Eliminated the `Phase:` field dependency identified as load-bearing in BW11. See [[decision-phase-free-bw12]].

**Changes from BW11:**
- User prompt: `Task: {instruction}\nMemory: {summary}\nOutput action:` (no Phase:)
- Rolling memory: last `<summary>` from any prior CoT frame in same episode propagated forward
- 115 instruction variants (15-20 phrasings per task type) sampled randomly during training
- Same LoRA config (rank=32, α=64), same dataset (bw11_dataset)

**BW12 val results (4,936 samples):**

| Task | lin MAE | ang MAE | Note |
|------|---------|---------|------|
| aisle_fwd | 0.0000 | 0.0000 | Perfect |
| aisle_fwd_slow | 0.0000 | 0.0000 | Perfect |
| cross_turn_left | 0.0070 | **0.0028** | Near-perfect turn without Phase: |
| cross_turn_right | 0.0051 | **0.0003** | Near-perfect turn without Phase: |
| obj_goal | 0.1015 | 0.0951 | Higher error (variable trajectory) |
| obstacle_slalom | 0.0207 | 0.0110 | Good |

Best val_loss = **0.0542** (step 2400, epoch 2) vs BW11 0.0551.
100% parse rate, 0 failures.
Model learns to turn from vision + memory alone — Phase: dependency resolved.

**Key implementation finding:** The `Phase:` field in the user prompt is load-bearing. The model learned to condition turning behaviour strongly on phase name. In live ROS2 mode, phase must be provided by the nav stack or a heuristic detector.

**Live closed-loop demo:** `~/Desktop/bw11_sim_demo.mp4` — dual-camera Isaac Sim recording (egocentric + overhead isometric view), two-process IPC (isaac6 + openvla envs). See `bw11_sim_isaac.py` + `bw11_sim_vla.py`.

## BW13 CorrectNav Training (2026-07-01)

**Core finding from BW12 live demo (F3):** Model predicts ang=0.0000 for all 40 closed-loop frames despite offline ang MAE=0.0003 on turns. Root cause: imitation≠decision, mode collapse, memory self-locking, no recovery data.

**BW13 fixes applied to AdaCoT pipeline:**

| Fix | Change | Addresses |
|-----|--------|-----------|
| Multi-frame input (H=3) | 3 consecutive frames as multi-image input | F3.1 — temporal proactive context |
| Action chunking (N=4) | Predict 4 next actions; approach frames get turning actions in future positions | F3.1, F3.4 — implicit look-ahead |
| Turn reweighting (3×) | Loss ×3 for chunks with `|ang|>0.05` | F3.2 — mode collapse |
| Always-on summary | Every frame outputs `<summary>` (deterministic synthetic); CoT still optional | F3.3 — memory self-locking |

**New output format (BW13):**
```
<think>optional reasoning</think>
<summary>scene description — ALWAYS present</summary>
ACTION_0: lin=x ang=y
ACTION_1: lin=x ang=y
ACTION_2: lin=x ang=y
ACTION_3: lin=x ang=y
```

**Turning chunk ratio:** 16.9% train / 12.6% val (up from 13.2% BW11/BW12 — chunking adds look-ahead supervision for free from existing dataset).

Training launched 2026-07-01 on cvit-car-simulator (tmux `bw13_train`, openvla env). 4,314 optimizer steps, ~15h ETA.
See [[decision-bw13-correctnav]] for full root cause analysis and fix rationale.

## Sources

- [[VLingNav]] — Table 6: AdaCoT ablation (dense CoT hurts, adaptive best at 2.1% activation)
- [[decision-entropy-trigger]] — rationale for entropy over learned trigger
- [[Nav-AMR-WH]] — training dataset with AdaCoT annotations
- [[decision-bw13-correctnav]] — CorrectNav-inspired 4-fix training for closed-loop failure (BW13)
