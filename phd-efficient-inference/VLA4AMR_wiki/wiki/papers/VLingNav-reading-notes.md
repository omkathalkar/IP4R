# Reading Notes — VLingNav
**Paper:** VLingNav: Embodied Navigation with Adaptive Reasoning and Visual-Assisted Linguistic Memory
**Authors:** Shaoan Wang*, Yuanfei Luo* et al. (ByteDance Seed & Peking University)
**arXiv:** 2601.08665 · Jan 13, 2026
**Read:** 2026-05-26 · Om Kathalkar
**Relevance:** Direct inspiration for C1 (AdaCoT trigger), C3 (handoff logic), C5 (RL fine-tuning)

---

## 1. The Problem They Are Solving

Current navigation VLAs are **reactive systems** — they map observations directly to actions with a fixed inference budget per step. Three structural failures:

1. **No reasoning:** Cannot increase deliberation when the scene is ambiguous (intersection, occlusion, never-seen room). Fixed compute per step means complex situations get the same budget as trivial ones.
2. **No persistent memory:** Models rely on limited context windows. Without long-horizon semantic memory, agents loop, revisit dead ends, cannot adapt to dynamic changes. Prior implicit memory (visual feature compression) loses semantic detail.
3. **Imitation ceiling:** SFT from expert demos suffers covariate shift — the agent encounters states never seen in demos and has no recovery mechanism.

The cognitive science framing is deliberate: they invoke Kahneman's **dual-process theory** (System 1 = fast/reactive, System 2 = slow/deliberative) and argue VLAs need both modes.

---

## 2. Core Contributions

1. **AdaCoT** — Adaptive Chain-of-Thought: dynamic binary trigger per step (`<think_on>` / `<think_off>`)
2. **VLingMem** — Visual-Assisted Linguistic Memory: persistent cross-modal semantic memory built from CoT summaries
3. **Nav-AdaCoT-2.9M** — largest navigation dataset with reasoning annotations (2.9M steps, 472K CoT labels)
4. **Expert-guided online RL** — hybrid rollout with a shortest-path planner as expert corrector

---

## 3. Architecture (Detailed)

### 3.1 Base Model
**LLaVA-Video-7B** + MLP action head.
- Backbone: video-based VLM with SigLIP-400M vision encoder
- N = 729 patches per frame, C = 1152 embedding dimension
- Action model: MLP A_θ(h_t^pred) → motion trajectory τ̃_t
- Action space: τ = {a₁, a₂, ..., aₙ} where each a ∈ ℝ³ = (x, y, θ) — position + orientation waypoints over horizon n

### 3.2 Observation Encoding — Dynamic FPS Sampling

**Problem:** Video VLMs accumulate frames over time → growing compute. Uniform subsampling loses recent detail. Merging visual tokens distorts semantics.

**Solution:** Inspired by the Ebbinghaus forgetting curve:

$$f_s(i) = f_s^{max} \cdot e^{-\Delta T / s}$$

where f_s = sampling rate, ΔT = t − i (time interval from frame i to current t), s = memory stability parameter.

**Effect:** Recent frames sampled at high rate (preserved in detail), older frames at low rate (implicitly forgotten). Controls total token count while preserving short-term temporal resolution.

For historical frames, also apply **grid pooling** to further reduce spatial resolution:
$$g(i) = \lfloor e^{\Delta T / g} \rfloor, \quad \mathbf{V}'_{t_i} = \mathcal{G}(\mathbf{V}_{t_i}, g(i))$$

**Temporal-aware tokens:** To resolve temporal ambiguity after variable-rate sampling, inject a temporal indicator before each frame:
$$E^T(\Delta T) = E^T_{base} + \text{RoPE}(\Delta T)$$

This lets the model perceive absolute time intervals between historical and current frames.

**Cross-modal projector:** 2-layer MLP P(·) maps visual features V → projected tokens E^V in LLM latent space.

### 3.3 AdaCoT — Adaptive Chain-of-Thought

The model receives concatenated tokens: [E^I (instruction), E^T (temporal), E^V (visual), E^M (memory)].

Step 1: Predict CoT indicator token — `<think_on>` or `<think_off>`
- If `<think_off>`: proceed directly to action trajectory prediction
- If `<think_on>`: generate full CoT content autoregressively

CoT content has two structured components:
- **Reasoning:** `<think>...</think>` — perception, task decomposition, location assessment, action determination
- **Memory summary:** `<summary>...</summary>` — environmental summary of current observation, stored as linguistic memory

The summary is incorporated into subsequent steps as VLingMem input. This creates a **write-then-read** loop: CoT generates memory, memory conditions future CoT.

**Empirical activation rate: 2.1% of steps** — the model learns that reasoning is only worth it at decision-critical moments (intersections, occlusions, ambiguous scenes).

### 3.4 Action Model — Probabilistic Continuous Head

The MLP action head parameterises a **multivariate Gaussian** distribution:
$$\pi_\theta(a_t | s_t) = \mathcal{N}\left(\mu_\theta(h_t), \text{diag}(\sigma_\theta(h_t)^2)\right)$$

where h_t = hidden state of the last token predicted by the VLM.

- **During RL rollout:** Sample stochastically from π_θ for exploration
- **During inference:** Use deterministic mean a_t = μ_θ(h_t)

This is the right design: L1 heads predict means, diffusion heads are too slow. Adding uncertainty (σ) is cheap and enables proper RL.

> **Comparison with OFT:** VLingNav uses continuous actions with a probabilistic head (Gaussian). OpenVLA-OFT uses L1 (mean prediction only). VLingNav's explicit σ enables proper PPO-style exploration without perturbation tricks.

---

## 4. Training — Three Stages

### Stage 1: Model Pre-training
- Data: open-world adaptive CoT video dataset (1.6M samples from LLaVA-Video-178K, Video-R1, ScanQA)
- Objective: standard CE loss at token level
- Duration: 1 epoch
- Purpose: teach the backbone *when* to reason (adaptive CoT), not just *what to do*
- Note: only this stage teaches the adaptive trigger; it's essentially "CoT curriculum" before navigation

### Stage 2: Supervised Fine-Tuning (SFT)
$$\min_\theta \mathcal{L}_{SFT}(\theta) = \alpha \mathcal{L}_{MSE}(\tilde{\tau}_t, \tau_t^{gt}) + (1-\alpha) \mathcal{L}_{CE}(E_t^{pred}, E_t^{gt})$$

- α = 0.5 (equal weight between trajectory MSE and textual CE)
- Data: Nav-AdaCoT-2.9M (navigation) + 1.6M open-world video = 4.5M total
- Co-training: interleave navigation + video data (prevents forgetting of general visual reasoning)
- 20K steps, batch size 512, 128 × NVIDIA A100 GPUs
- Visual encoder frozen; all other parameters updated

### Stage 3: Online Expert-Guided Post-Training
**Objective:** Overcome covariate shift from offline SFT. The agent sees states not in the training distribution and must learn recovery.

$$\min_\theta \mathcal{L}_{post}(\theta) = \lambda \mathcal{L}_{RL}(\theta) + (1-\lambda) \mathcal{L}_{SFT}(\theta)$$

where λ = 0.01 (small RL weight to prevent catastrophic forgetting).

$$\mathcal{L}_{RL}(\theta) = -\mathbb{E}_t\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

This is PPO-style REINFORCE++.

**Hybrid rollout** (Fig. 5):
- *Naive rollout:* Policy interacts independently. Store successful trajectories only (on-policy data; positive examples only).
- *Expert-guided rollout:* When agent oscillates or gets stuck for k=15 steps, an expert policy (shortest-path planner in simulator) intervenes, demonstrates recovery, then returns control. These corrective demonstrations go into the hybrid buffer.

**Why hybrid?** Naive RL alone: stuck agents generate no reward signal → sparse gradients. Expert takeover provides dense corrective supervision at failure states. Critically, the expert provides *how to escape*, not *how to succeed from the start* — so the policy learns generalised recovery behaviours.

**Implementation:** 10 rollout iterations, 128 on-policy episodes per iteration. HM3D OVON + HM3D Instance ImageNav + EVT-Bench for online environments.

---

## 5. Dataset — Nav-AdaCoT-2.9M

### Scale comparison:
| Dataset | Scenes | Steps | CoT labels | Action type |
|---------|--------|-------|------------|-------------|
| HM3D ObjNav | 80 | — | — | Description |
| Nav-CoT-110K | 342 | 110K | 110K | Description |
| **Nav-AdaCoT-2.9M (ours)** | **718** | **2.9M** | **472K** | **Trajectory** |

Key innovations over prior datasets:
- **Trajectory-based actions** (not description-based): finer-grained supervision
- **Adaptive CoT annotations** (not dense): 16.4% of steps have CoT (472K / 2.9M)
- **Multi-task:** ObjNav + EVT + ImageNav in one dataset
- **Generated by Qwen2.5-VL-72B** with a 5-component prompt:
  1. Navigation instruction
  2. Last 10 egocentric frames
  3. Prior memory context
  4. Expert action trajectories
  5. Explicit format requirements (JSON with `chain_of_thought` and `summary` keys)

**Two-stage filtering:**
1. Rule-based: discard incomplete / logically inconsistent responses
2. Quality verification: cross-validate decisions against expert navigation trajectories

---

## 6. Experimental Results

### 6.1 Object Goal Navigation

**HM3D ObjNav (Table 2):**

| Method | HM3Dv1 SR | HM3Dv1 SPL | HM3Dv2 SR | MP3D SR |
|--------|-----------|------------|-----------|---------|
| VLFM | 52.5 | 30.4 | 63.6 | 36.4 |
| Uni-NaVid | 73.7 | 37.1 | — | — |
| FiLM-Nav | 61.7 | 37.3 | 77.0 | — |
| CogNav | 72.5 | 26.2 | — | 46.6 |
| VLingNav (SFT only) | 70.6 | 38.2 | 76.4 | 47.4 |
| **VLingNav (full)** | **79.1 ★** | **42.9 ★** | **83.0 ★** | **58.9 ★** |

+5.4 SR (+7.3%) over Uni-NaVid on HM3Dv1. +26.4% SR improvement over CogNav on MP3D.

**Key finding on MP3D:** Long-range exploration is where VLingNav most outperforms — memory is critical for multi-room search without revisiting.

### 6.2 Embodied Visual Tracking (EVT-Bench)

| Method | Single-Target SR | TR | Distracted SR | TR |
|--------|------------------|----|---------------|----|
| TrackVLA++ | 86.0 | 81.0 | 66.5 | 68.8 |
| NavFoM* | 88.4 | 80.7 | 62.0 | 67.9 |
| VLingNav (SFT) | 87.2 | 78.9 | 66.1 | 69.7 |
| **VLingNav (full)** | **88.4 ★** | **81.2 ★** | **67.6 ★** | **73.5 ★** |

Distracted tracking is most improved (+4.7% TR vs TrackVLA++): adaptive reasoning helps re-identify targets after occlusion.

### 6.3 Image Goal Navigation

| Method | HM3D ImageNav SR | SPL |
|--------|------------------|-----|
| UniGoal | 60.2 | 23.7 |
| **VLingNav** | **60.8 ★** | **37.4 ★** |

Modest SR gain but +13.7 SPL (+57.8%) — dramatically more efficient paths. This is the linguistic memory benefit: don't revisit, navigate efficiently.

### 6.4 Real-World Transfer (Zero-Shot)
Robot: Unitree Go2 quadruped + Intel RealSense D457 + remote RTX 4090 server.
Communication: compressed images via portable Wi-Fi, ~100ms overhead.
**Effective inference: 2.5 FPS** during long-horizon real-world navigation.
NMPC trajectory tracker converts waypoints to motor commands.

Key result: **zero additional fine-tuning on real-world data** — sim-to-real transfer is direct. Model generalises to unseen objects (bike, outdoor scenes) despite training only on indoor HM3D/MP3D.

---

## 7. Ablation Studies — What Actually Matters

### 7.1 AdaCoT Strategies (Table 6, HM3D OVON val unseen)

| Strategy | ObjNav SR | EVT SR | ImgNav SR | rCoT (%) |
|----------|-----------|--------|-----------|----------|
| w/o CoT | 36.2 | 62.7 | 56.3 | 0 |
| Dense CoT (every step) | **25.3** | 59.8 | 19.6 | 100 |
| Fixed interval k=5 | 42.5 | 68.5 | 48.2 | 20 |
| Fixed interval k=20 | 39.7 | 66.2 | 51.3 | 5 |
| **Adaptive CoT (ours)** | **50.1 ★** | **67.6 ★** | **60.8 ★** | **2.1** |

**Critical finding:** Dense CoT *hurts* performance (25.3 vs 36.2 without CoT on ObjNav). Reasoning at every step degrades performance — the model gets distracted or introduces noise into the action prediction. Adaptive triggering at 2.1% of steps dominates all fixed schedules.

This confirms: **CoT is useful at decision junctions, harmful if used uniformly.**

### 7.2 Memory Modalities (Table 7)

| Memory Mode | ObjNav SR | Track TR | ImgNav SR |
|-------------|-----------|----------|-----------|
| w/o Memory | 15.4 | 59.1 | 21.0 |
| Visual-only | 45.2 | 70.6 | 57.9 |
| Language-only | 18.8 | 55.2 | 23.3 |
| **VLingMem (ours)** | **50.1 ★** | **73.5 ★** | **60.8 ★** |

**Most striking:** w/o Memory collapses to 15.4 SR — the model loops and gets stuck catastrophically without persistent context. Visual-only helps substantially (45.2) but language-only is almost as bad as no memory (18.8). The cross-modal combination is essential.

**Why language > visual alone:** Language memory is semantically compressed and LLM-native. Visual-only memory accumulates feature compression artifacts. Combined, they reinforce each other.

---

## 8. Limitations (From Paper + My Reading)

1. **Dataset scale:** Nav-AdaCoT-2.9M requires Qwen2.5-VL-72B inference on 2.9M samples — expensive annotation pipeline. 472K CoT labels at 72B inference cost is significant.

2. **Real-world speed:** 2.5 FPS effective (300ms model + 100ms communication) constrains dynamic scenarios. For fast AMR navigation (>1 m/s) this may be too slow.

3. **Waypoint action space:** Output (x, y, θ) requires a downstream trajectory tracker (NMPC in their case). For our Nav2 integration, we need to convert these to Nav2-compatible goals or cmd_vel — design challenge.

4. **Indoor scene bias:** Training data is predominantly indoor HM3D/MP3D (household/office). Warehouse layouts (wide aisles, forklifts, shelving units) are out-of-distribution.

5. **No LiDAR:** VLingNav is purely camera-based (RGB). Our Nova Carter has 3D LiDAR — potentially better but requires architectural extension.

6. **Fixed CoT trigger threshold:** The 2.1% activation rate emerges from training, not from a tunable threshold. Our C1 (entropy-gated trigger) is more interpretable and controllable — a deliberate design advantage.

---

## 9. Critical Analysis — What This Means for VLA4AMR

### The AdaCoT → C1 connection:
VLingNav's key result (Dense CoT *hurts*) is the strongest empirical justification for our entropy-gated approach. We don't *train* the trigger — we derive it analytically from token probability entropy. This has advantages:
- No Nav-AdaCoT-scale labelling effort (we can't afford 2.9M annotations at Qwen-72B cost)
- Interpretable threshold (θ is a tunable hyperparameter)
- Can be adapted to different uncertainty regimes without retraining

**Risk:** Our entropy trigger may fire at different rates than AdaCoT's 2.1%. If it fires too frequently, performance degrades (Dense CoT finding). Need to calibrate θ empirically.

### The VLingMem → not directly applicable:
VLingMem requires the LLM to generate `<summary>` tokens — it's tightly coupled to the autoregressive text generation loop. Our VLA setup (OpenVLA-OFT base with parallel decoding) doesn't naturally generate text summaries during navigation. We'd need to add a separate text generation head or redesign the architecture.

**Our C3 (ConfidenceGatedHandoff)** is a simpler, complementary approach: instead of episodic memory within the VLA, we use AMCL covariance + obstacle detection as an external memory proxy about localisation confidence. This is more practical for Nav2 integration.

### The RL stage → C5 design:
Their PPO + hybrid rollout is directly applicable to C5. Key design choices to adopt:
- Expert-guided rollout with k=15 stuck-detection threshold
- λ=0.01 (small RL weight) to prevent catastrophic forgetting
- Hybrid buffer: on-policy successes + expert-corrected recoveries
- Our reward: R = α·success − β·collision + γ·path_efficiency (matches their intuition)

### Numbers to quote in paper:
- AdaCoT activates at 2.1% of steps → our justification for entropy gating (not dense CoT)
- Dense CoT drops ObjNav SR from 36.2 → 25.3 → motivates our adaptive approach
- w/o Memory collapses to 15.4 SR → motivates any form of memory (ours via AMCL/map)
- 79.1% ObjNav SR → SOTA baseline we compare against (different domain but useful reference)

### The most important sentence in this paper for VLA4AMR:
> *"Remarkably, it accomplishes this while maintaining an exceptionally low reasoning frequency (r_CoT = 2.1%), far more efficient than even the sparse fixed-interval method."*

This tells us that smart triggering — not always-on reasoning — is the right design principle.

---

## 10. Open Questions After Reading

- Can we replicate Nav-AdaCoT-style CoT annotations at small scale (~5K episodes) for our warehouse domain using a smaller model (Qwen2.5-VL-32B or 14B)?
- Our entropy threshold θ — how do we calibrate it? Can we use the 2.1% activation rate as a target and tune θ to match it on a held-out validation set?
- VLingMem generates summaries during inference. Can we use a lightweight rule-based summary (AMCL pose + last detected objects) as a cheap proxy without full LLM generation?
- Their real-world platform is a quadruped (Go2) at 2.5 FPS. Nova Carter is a wheeled AMR — can it tolerate 2.5 FPS or do we need higher frequency? Our ROS2 pipeline targets 10 Hz.
- Expert-guided rollout uses a shortest-path planner as the expert (they have ground-truth maps in simulation). In Isaac Sim warehouse, we can do this with Nav2 in map mode — this is directly feasible for C5.
