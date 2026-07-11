# VLA4AMR Wiki — Master Index

**Last updated:** 2026-07-11 (BW18 launch)
**Total pages:** 35
**Wiki maintainer:** Claude (Chetak)

---

## Project overview

| Page | Type | Status | Summary |
|------|------|--------|---------|
| [[overview]] | overview | active | Project status, bi-weekly progress, immediate next steps |

---

## Contributions (C1–C6)

| Page | Type | Status | Summary |
|------|------|--------|---------|
| [[C1-AdaCoT]] | contribution | active | Entropy-gated adaptive chain-of-thought trigger for VLA navigation |
| [[C2-MidLevelActionHead]] | contribution | active | Mid-level language action head outputting (direction, distance, context) tuples |
| [[C3-ConfidenceGatedHandoff]] | contribution | active | Switch monitor + arbitrator for selective VLA override of Nav2 |
| [[C4-FusionAblation]] | contribution | active | Token concat vs FiLM vs cross-attention; scaffold done BW03, training pending |
| [[C5-RLFinetuning]] | contribution | pending | PPO post-SFT with navigation reward function |
| [[C6-EvaluationProtocol]] | contribution | pending | Dual open-loop + closed-loop evaluation protocol |

---

## Related papers

| Page | Type | Status | Summary |
|------|------|--------|---------|
| [[OpenVLA-OFT]] | paper | active | Stanford fine-tuning recipe — parallel decoding, L1 head, LoRA (C2/C4 base) |
| [[VLingNav]] | paper | active | ByteDance adaptive CoT + linguistic memory navigation (C1/C3/C5 inspiration) |
| [[NaVILA]] | paper | active | USC legged robot VLA with mid-level language actions (C2 structural inspiration) |
| [[Language-as-Cost]] | paper | active | VLM as semantic cost layer into Nav2 costmap (C3 baseline) |
| [[VLFM]] | paper | active | Vision-language frontier maps, zero-shot object nav (ICRA 2024 Best Paper) |
| [[OpenVLA-OFT-Stanford]] | paper | active | Same as OpenVLA-OFT — use [[OpenVLA-OFT]] |

---

## Datasets

| Page | Type | Status | Summary |
|------|------|--------|---------|
| [[BridgeData-V2]] | dataset | active | Primary SFT corpus, ~15K nav episodes, filter: forward >1m, CLIP >0.3 |
| [[HM3D]] | dataset | active | PointNav + ObjectNav, ~8K trajectories, filter: Nav2 failure episodes |
| [[Isaac-Synthetic]] | dataset | active | Early warehouse dataset (bw04–bw06), OpenVLA 7D format — superseded by Nav-AMR-WH |
| [[Nav-AMR-WH]] | dataset | active | **Primary dataset (BW11+)** — 640 eps, 50,971 frames, AdaCoT annotations, bw11_dataset |
| [[bw17-warehouse-dataset]] | dataset | complete | BW18 TIC-VLA dataset — 560 eps, 9 tasks, 14,360 DynaNav_json dirs, 4× instruction aug + CoT |
| [[RL-Rollouts]] | dataset | pending | PPO online rollouts, ~50K steps, curriculum static → dynamic |

---

## Architecture & system components

| Page | Type | Status | Summary |
|------|------|--------|---------|
| [[NovaCarter]] | architecture | active | NVIDIA reference AMR — Hawk stereo + LiDAR, Isaac-native, our simulation platform |
| [[IsaacSim]] | architecture | active | NVIDIA Isaac Sim, ROS 2 Jazzy bridge, warehouse_with_forklifts world |
| [[ROS2-Pipeline]] | architecture | active | VLA inference node, file-based IPC bridge, async 10Hz pipeline |

---

## Infrastructure

| Page | Type | Status | Summary |
|------|------|--------|---------|
| [[simulator-machine]] | infrastructure | active | cvit-car-simulator: AMD Ryzen 9 7950X, RTX 4060 Ti 16GB + RTX 5000 Pro 48GB |
| [[openvla-env]] | infrastructure | active | Conda env: openvla, Python 3.10, torch+cu12x, OpenVLA-7B working stack |
| [[vslam-rtabmap]] | infrastructure | active | RTAB-Map stereo VSLAM on Isaac Sim — Nova Carter hawk cameras, /rtabmap/odom |

---

## Decisions log

| Page | Type | Status | Summary |
|------|------|--------|---------|
| [[decision-entropy-trigger]] | decision | active | Why entropy-gating over learned AdaCoT for C1 |
| [[decision-nova-carter]] | decision | active | Why Nova Carter over BCR Bot for simulation |
| [[decision-oft-recipe]] | decision | active | Why OFT recipe (L1 + parallel + chunking) over diffusion for C2 |
| [[decision-c4-strategy]] | decision | active | Token concat chosen for C2/C5 — data-scarce regime favours zero-param baseline |
| [[decision-c1-theta]] | decision | active | θ=2.795 nats for AdaCoT entropy gate — 3.1% activation, +10ms overhead, p97 of warehouse entropy |
| [[decision-qwen-bw11]] | decision | active | Switch to Qwen2.5-VL-7B for BW11+ — native CoT text output, same family as annotation model |
| [[decision-ipc-two-process]] | decision | active | File-based IPC for Isaac Sim + VLA — necessary due to isaac6/openvla CUDA driver incompatibility |
| [[decision-phase-free-bw12]] | decision | active | Remove Phase: field in BW12 — model infers turning from vision + rolling memory |
| [[decision-bw13-correctnav]] | decision | active | CorrectNav-inspired 4-fix training for F3 closed-loop failure (multi-frame, chunking, reweight, always-summary) |
| [[decision-bw14-discrete-nav]] | decision | active | DiscreteNav: replace continuous regression with {FORWARD,TURN_LEFT,TURN_RIGHT,STOP} tokens + H=6 video history + CorrectNav flywheel |
| [[decision-bw15-correctnav-fullstack]] | decision | active | BW15: Full CorrectNav stack (LLaVA-Video-7B-Qwen2 + SigLIP + MP4 video), 1-GPU LoRA adaptation, Isaac Sim warehouse |
| [[decision-bw16-ticvla-warehouse]] | decision | active | BW16: TIC-VLA (InternVL3-1B + ActionExpert) trained on bw11_dataset — continuous waypoint baseline |
| [[decision-bw18-ticvla-bw17]] | decision | active | BW18: TIC-VLA re-trained on BW17 9-task dataset — richer tasks, 30-step FLU horizon, 4× instruction aug |
