# VLA4AMR — Project Overview

**Type:** overview
**Status:** active
**Last updated:** 2026-07-03 (BW14 launch)
**Related:** [[C1-AdaCoT]], [[C2-MidLevelActionHead]], [[C3-ConfidenceGatedHandoff]], [[IsaacSim]], [[NovaCarter]], [[simulator-machine]], [[Nav-AMR-WH]]

## Summary

VLA4AMR applies Vision-Language-Action models to Autonomous Mobile Robot navigation in industrial warehouse settings. The core thesis: current VLA models are black-box reactive policies with no reasoning or memory — unacceptable for industrial AMR deployment. We address this with six interlocking contributions targeting ICRA 2027.

## Current status (BW13 — Jul 1, 2026)

| Item | Status |
|------|--------|
| ICRA 2027 plan submitted to Prof. Jawahar | ✅ Done |
| Dr. Shankar call prep (slides + script) | ✅ Done |
| OpenVLA-OFT & VLingNav papers reviewed | ✅ Done |
| Wiki initialised | ✅ Done |
| GPU driver install on simulator machine | ✅ Done (570.211.01, downgraded 2026-06-13 — 595.x crashes Blackwell in Isaac Sim) |
| Isaac Sim + Nova Carter setup | ✅ Done (2026-06-13 — Nova_Carter_ROS.usd, 31 children, 2141 prims, all sensors verified) |
| Isaac Sim 6.0.0.1 + carter_warehouse_navigation.usd | ✅ Done (2026-06-15 — 5267 prims, 17 cameras, S3 load verified) |
| ROS2 pipeline end-to-end verified (BW02 gate) | ✅ Done (2026-06-16 — 5/5 topics PASS: /tf, /chassis/odom, front camera, /front_3d_lidar/lidar_points) |
| OpenVLA-OFT integration (BW03 gate) | ✅ Done (2026-06-17 — 0.20s/inf, 5 Hz sustained, /cmd_vel verified, inf#615+) |
| C4 FusionAblation — all 3 strategies trained | ✅ Done (2026-06-19 — token_concat best val_loss=0.1669, FiLM=0.3152, cross_attn=0.4066) |
| C4 FusionAblation — open-loop RMSE eval | ✅ Done (2026-06-20 — token_concat best: total=0.1697, nav=0.3174) |
| C4 strategy decision | ✅ Done (2026-06-20 — token_concat → C2 and C5, see [[decision-c4-strategy]]) |
| C1 AdaCoT script written | ✅ Done (2026-06-20 — bw04_c1_adacot.py, 5 ablation modes) |
| Dataset curation script written | ✅ Done (2026-06-20 — bw04_dataset_curator.py, BridgeData-V2 + HM3D) |
| C1 AdaCoT θ sweep (11 conditions) | ✅ Done (2026-06-20 — θ=2.795, 3.1% activation, +10ms, see [[decision-c1-theta]]) |
| C1 live integration into inference server | ✅ Done (2026-06-20 — bw04_c1_live.py smoke test PASS: H=10.286, action=[7], dt=0.61s, JSON valid) |
| BW04 Isaac-Synthetic dataset v1 (straight) | ✅ Done (2026-06-23 — 200 eps, 16,802 frames, ~/Desktop/bw04_dataset/) |
| BW04 Isaac-Synthetic dataset v2 (straight + zigzag) | ✅ Done (2026-06-23 — 200 eps, 16,802 frames, ~/Desktop/bw04_dataset_v2/, 6 task types) |
| BW05 HuggingFace conversion (v1) | ✅ Done (2026-06-23 — 16,798 samples, train=15,118 / val=1,680, 238.6 MB) |
| BW05 HuggingFace conversion (v2, combined) | ✅ Done (2026-06-23 — 33,597 samples, train=30,237 / val=3,360, 477.5 MB) |
| BW06 LoRA fine-tuning (OpenVLA-7B + bw05_dataset_v2) | ✅ Done (2026-06-23 — rank=32, converged val_loss=0.0001 at step 1000/5670) |
| BW06 Inference evaluation (300 val samples) | ✅ Done (2026-06-23 — see results below) |
| BridgeData-V2 / HM3D download + curation | ⏳ BW04 — parallel track |
| BW09 VSLAM pipeline (RTAB-Map stereo) | ✅ Done (2026-06-24 — stereo_odometry + rtabmap on Nova Carter hawk cameras, /rtabmap/odom verified, map saved) |
| BW10 Nav-AMR-WH dataset collection | ✅ Done (2026-06-29 — 640 episodes, warehouse_multiple_shelves.usd + Nova Carter, 6 task types) |
| BW10 CoT annotation (Qwen2.5-VL-7B local) | ✅ Done (2026-06-29 — 50,971 frames annotated in parallel, 13.2% CoT ratio, VLingNav AdaCoT format) |
| BW11 dataset merge (bw11_dataset) | ✅ Done (2026-06-29 — 46,035 train / 4,936 val, HF DatasetDict, norm_stats.json) |
| BW11 LoRA fine-tune (Qwen2.5-VL-7B) | ✅ Done (2026-06-30 — rank=32, best val_loss=0.0551 epoch 1, 95M/8.4B trainable) |
| BW11 offline eval on val set | ✅ Done (2026-06-30 — 100% parse rate, lin MAE=0.0000, ang MAE=0.0105 overall) |
| BW11 live inference + Isaac Sim dual-camera recording | ✅ Done (2026-06-30 — two-process IPC: isaac6 + openvla, 1280×720 MP4 with egocentric + isometric overhead view) |
| BW12 phase-free LoRA fine-tune (Qwen2.5-VL-7B) | ✅ Done (2026-07-01 — no Phase:, rolling Memory:, 115 instruction variants, best val_loss=0.0542, 12.5h) |
| BW12 offline eval on val set | ✅ Done (2026-07-01 — 100% parse rate, lin MAE=0.0165, ang MAE=0.0124; turning ang MAE=0.0003–0.0028 **without Phase: hint**) |
| BW12 live demo (phase-free, Isaac Sim) | ✅ Done (2026-07-01 — bw12_sim_vla.py, natural language instructions, rolling memory, dual-camera MP4) |
| BW13 F3 root-cause analysis | ✅ Done (2026-07-01 — 4 failure modes identified: imitation≠decision, mode collapse, memory self-locking, no recovery data) |
| BW13 CorrectNav-inspired training (bw13_train.py) | ✅ Done (2026-07-01 — best val_loss=0.0449, better than BW12; closed-loop STILL ang=0.0000, root cause = mode collapse structural) |
| BW14 DiscreteNav architecture decision | ✅ Done (2026-07-03 — discrete tokens + H=6 + N=6, see [[decision-bw14-discrete-nav]]) |
| BW14 training scripts written | ✅ Done (2026-07-03 — bw14_train.py / bw14_infer.py / bw14_sim_vla.py) |
| BW14 training on simulator | ⏳ Pending (deploy to cvit-car-simulator, tmux bw14_train) |

## BW12 Evaluation Results (2026-07-01)

Fine-tuned Qwen2.5-VL-7B (LoRA rank=32, best val_loss=0.0542) — **phase-free**, memory-chained, 115 instruction variants.
Evaluated on 4,936 val samples from bw11_dataset (same val split as BW11).

**Key innovation:** NO `Phase:` field in prompt. Model learns to turn from vision + rolling `<summary>` memory.

**Overall (n=4,936):**

| Metric | lin_vel (m/s) | ang_vel (rad/s) |
|--------|---------------|-----------------|
| Parse rate | **100%** | — |
| MAE | 0.0165 | **0.0124** |

**Per task type:**

| Task | n | lin MAE | ang MAE | CoT% |
|------|---|---------|---------|------|
| aisle_fwd | 960 | **0.0000** | **0.0000** | 14.8% |
| aisle_fwd_slow | 320 | **0.0000** | **0.0000** | 15.0% |
| cross_turn_left | 1280 | 0.0070 | **0.0028** | 12.7% |
| cross_turn_right | 1280 | 0.0051 | **0.0003** | 11.8% |
| obj_goal | 536 | 0.1015 | 0.0951 | 12.5% |
| obstacle_slalom | 560 | 0.0207 | 0.0110 | 13.4% |

**vs BW11 (Phase-dependent):**

| Metric | BW11 (Phase:) | BW12 (no Phase:) |
|--------|--------------|-----------------|
| Val loss | 0.0551 | **0.0542** |
| Overall ang MAE | 0.0105 | 0.0124 |
| Turning ang MAE | 0.0013 | 0.0003–0.0028 |
| Phase dependency | Required | **Eliminated** |

**Artifacts:**
- Checkpoint: `~/VLA4AMR/checkpoints/bw12_lora/best_lora/`
- Eval results: `~/Desktop/bw12_eval_results.json`
- Demo video: `~/Desktop/bw12_sim_demo.mp4`

## BW11 Evaluation Results (2026-06-30)

Fine-tuned Qwen2.5-VL-7B (LoRA rank=32, best_lora, val_loss=0.0551) evaluated on 4,936 val samples from bw11_dataset.

**Overall (n=4,936):**

| Metric | lin_vel (m/s) | ang_vel (rad/s) |
|--------|---------------|-----------------|
| Parse rate | 100% | — |
| MAE | **0.0000** | **0.0105** |

**Per task type:**

| Task | n | lin MAE | ang MAE |
|------|---|---------|---------|
| aisle_fwd | 960 | 0.0000 | 0.0000 |
| aisle_fwd_slow | 320 | 0.0000 | 0.0000 |
| cross_turn_left | 1280 | 0.0000 | 0.0013 |
| cross_turn_right | 1280 | 0.0000 | 0.0013 |
| **obj_goal** | 536 | 0.0000 | **0.0904** |
| obstacle_slalom | 560 | 0.0000 | 0.0000 |

**Key finding:** `Phase:` field in user prompt is load-bearing — model conditions strongly on it to predict turning behaviour. In live inference, phase must be supplied (from nav stack or heuristic) for correct turn commands. obj_goal shows higher ang MAE due to variable angular behaviour and fewer training samples.

**Artifacts:**
- Checkpoint: `~/VLA4AMR/checkpoints/bw11_lora/best_lora/`
- Eval results: `~/VLA4AMR/checkpoints/bw11_lora/eval_results.json`
- Demo videos: `~/Desktop/bw11_sim_demo.mp4` (Isaac Sim, dual-camera), `~/Desktop/bw11_demo_iso.mp4` (offline, isometric viz)

## BW06 Evaluation Results

Fine-tuned OpenVLA-7B (LoRA rank=32) evaluated on 300 val samples from bw05_dataset_v2.
6 task types: aisle_{a,b} × {fwd, slow, zigzag}.

**Overall (n=300):**

| Metric | lin_vel | ang_vel |
|--------|---------|---------|
| MAE | 0.1660 m/s | 0.0382 rad/s |
| Accuracy (±0.05 tol) | 17.0% | **85.0%** |

**Per task breakdown:**

| Task | n | MAE lin | MAE ang | Acc lin | Acc ang |
|------|---|---------|---------|---------|---------|
| aisle_a_fwd | 91 | 0.1451 | 0.0699 | 27% | 74% |
| aisle_a_slow | 90 | 0.2000 | 0.0000 | 0% | **100%** |
| aisle_a_zigzag | 10 | **0.0000** | 0.2237 | **100%** | 20% |
| aisle_b_fwd | 54 | 0.2000 | 0.0000 | 0% | **100%** |
| aisle_b_slow | 39 | 0.2000 | 0.0000 | 0% | **100%** |
| aisle_b_zigzag | 16 | **0.0000** | 0.1780 | **100%** | 19% |

**Diagnosis:**
- **ang_vel (straight/slow):** Near-perfect — model correctly suppresses rotation for non-zigzag tasks.
- **lin_vel (zigzag):** Perfect — "weaving/adjusting" keyword maps cleanly to full-speed (1.0 m/s).
- **lin_vel (slow):** Consistently off by +0.2 m/s (predicts ≈0.8 instead of GT=0.6). Keyword "slowly" learned but output mis-calibrated — likely needs separate action-space normalization per speed tier.
- **ang_vel (zigzag):** Only 19–20% accuracy — model cannot track the sinusoidal variation without temporal context. Root cause: 84% straight training frames, so visual grounding for angular velocity is weak. Confirmed fast-convergence limitation.

**Artifacts:** `~/Desktop/bw06_eval_results.json`, `~/Desktop/bw06_annotated_episode.mp4` (ep 229, zigzag, 84 frames, GT/PRED overlay)

## Bi-weekly plan

| BW | Dates | Goal |
|----|-------|------|
| 01 | May 15–28 | Isaac Sim + Nova Carter + literature |
| 02 | Jun 1–13 | Nav2 baseline benchmarks (open + closed loop) |
| 03 | Jun 16–28 | OpenVLA-OFT integration + fusion ablation start |
| 04 | Jul 1–13 | AdaCoT trigger (C1) + dataset curation |
| 05 | Jul 14–27 | Mid-level action head (C2) |
| 06 | Jul 28–Aug 10 | LoRA SFT + C3 handoff + PPO start |
| 07 | Aug 11–24 | Complete experiments + demo video |
| 08 | Aug 25–Sep 7 | Full paper draft v1 → v2 |
| 09 | Sep 8–15 | Final submit via PaperPlaza |

## Submission target

**ICRA 2027** · Seoul, May 24–28, 2027
**Deadline:** Sep 15, 2026

## Six contributions

1. [[C1-AdaCoT]] — Entropy-gated adaptive reasoning trigger
2. [[C2-MidLevelActionHead]] — Language action decoder for Nav2 waypoints
3. [[C3-ConfidenceGatedHandoff]] — Selective VLA override of Nav2
4. [[C4-FusionAblation]] — Fusion strategy comparison (token concat / FiLM / cross-attn)
5. [[C5-RLFinetuning]] — PPO online alignment for navigation robustness
6. [[C6-EvaluationProtocol]] — Dual open-loop + closed-loop evaluation

## Training data plan

- [[BridgeData-V2]] — ~15K episodes (primary SFT)
- [[HM3D]] — ~8K trajectories (language-described goal nav)
- [[Isaac-Synthetic]] — ~5K episodes (our warehouse dataset)
- [[RL-Rollouts]] — ~50K PPO steps online

## Open questions

- ~~Will nvidia-driver-595-open support RTX 5000 Pro Blackwell?~~ **Resolved 2026-05-26:** Yes (but 595.x crashes Blackwell in Isaac Sim — driver 570 required)
- ~~Isaac Sim GPU assignment?~~ **Resolved 2026-06-13:** Never set `CUDA_VISIBLE_DEVICES` for Isaac Sim (carb.cudainterop crash). Blackwell = Omniverse GPU 0 (fastest-first order). For openvla env: `CUDA_VISIBLE_DEVICES=0` = RTX PRO 5000 Blackwell.
- ~~ROS 2 Jazzy installed?~~ **Resolved 2026-05-26:** Installed (ros-jazzy-desktop + ros-dev-tools)
- **Isaac Sim 6.0.0 upgrade (BW03 gate)**: 4.5.0 has no Blackwell CC 12.0 support (Kit SDK 106.5 predates it) and uses Python 3.10 which is incompatible with Jazzy's Python 3.12 rclpy. Upgrading to 6.0.0.1 (Python 3.12) fixes both.
- ~~**carter_warehouse_navigation.usd S3 path**~~ **Resolved 2026-06-15:** `{assets_root}/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd` — 5,267 prims, 17 cameras, stage load verified. `get_stage_loading_status()` broken in 6.0.0.1 — use prim-count probe. See [[IsaacSim]].
- ~~**`isaacsim.ros2.bridge` — BW03 blocker**~~ **Resolved 2026-06-15:** Extension IS installed (as `isaacsim-ros2==6.0.0.1`), just not loaded by default. Fix: `mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)` before opening stage. Uses system Jazzy rclpy (Python 3.12 match = seamless). See [[IsaacSim]].
- ~~**BW02/BW03 gate: ROS2 pipeline verification**~~ **Resolved 2026-06-16:** 5/5 topics PASS. /odom is /chassis/odom; LiDAR is /front_3d_lidar/lidar_points (PointCloud2, no 2D /scan). app.update() must be called during topic subscription checks or sim doesn't tick.
- ~~**BW03: OpenVLA-OFT inference node**~~ **Resolved 2026-06-17:** Two-process architecture (ROS2 node in system Python 3.12 + inference server in openvla env Python 3.10). 0.20s/inference on Blackwell. Bug: `predict_action()` returns ndarray not tensor — fixed with `hasattr(action, 'cpu')` guard. See [[bw03_inference_server]], [[bw03_openvla_node]].
- ~~**Current BW03 task:** C4 FusionAblation~~ **Resolved 2026-06-20:** token_concat wins (val_loss=0.1669 vs FiLM=0.3152 vs cross_attn=0.4066). See [[decision-c4-strategy]].
- ~~**BW04/BW05/BW06: Isaac-Synthetic dataset + LoRA fine-tune**~~ **Resolved 2026-06-23:** 200-ep v1 (straight) + 200-ep v2 (straight+zigzag) collected; bw05_dataset_v2 (33,597 samples); OpenVLA-7B LoRA rank=32 converged val_loss=0.0001. Eval: ang_vel acc=85% overall, lin_vel zigzag acc=100%, lin_vel slow miscalibrated (+0.2 m/s). See BW06 results table above.
- ~~**BW07 priority:** Re-balance Isaac-Synthetic~~ **Resolved BW10/BW11:** New Nav-AMR-WH dataset (bw11_dataset) with 640 episodes, 6 task types including turns and slalom; CoT annotated at 13.2% ratio. Supersedes bw04/bw05/bw06 datasets. Model now uses Qwen2.5-VL-7B instead of OpenVLA-7B.
- ~~**BW11/BW12: Phase dependency**~~ **Resolved BW12:** BW12 eliminates Phase: token. Model infers turning from vision + rolling memory. Turning ang MAE=0.0003–0.0028 without Phase:.
- ~~**BW12 closed-loop failure (F3)**~~ **Root cause resolved 2026-07-01:** ang=0.0000 throughout live demo despite offline ang MAE=0.0003. Four causes: imitation≠decision, mode collapse (~80% straight frames), memory self-locking loop, no recovery data. Fix: BW13 (action chunking N=4, multi-frame H=3, turn reweight 3×, always-summary). See [[decision-bw13-correctnav]].
- **BW13/BW14 priorities:** (1) BW13 training complete → offline eval → confirm turn-initiation MAE at chunk positions 0-3. (2) BW14 CorrectNav dynamic Flywheel: run BW13 in Isaac Sim, detect deviation frames, script correction trajectories, add to dataset, retrain. (3) obj_goal ang MAE=0.0951 — needs more variable episodes. (4) Closed-loop success-rate metric (C6). (5) C3 confidence-gated handoff. (6) RL fine-tuning (C5).
