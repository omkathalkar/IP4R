# Reading Notes — OpenVLA-OFT
**Paper:** Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success
**Authors:** Moo Jin Kim · Chelsea Finn · Percy Liang (Stanford)
**arXiv:** 2502.19645 · Apr 2025
**Read:** 2026-05-26 · Om Kathalkar
**Relevance:** Technical base for C2 (Mid-Level Action Head) and C4 (Fusion Ablation)

---

## 1. The Problem They Are Solving

VLAs like OpenVLA can zero-shot generalise well but fail badly when deployed on a *new robot or task* without fine-tuning. The fine-tuning strategy is therefore critical—but nobody had systematically studied which design choices matter.

Two concrete failure modes with the baseline (autoregressive OpenVLA):
1. **Speed:** 3–5 Hz inference → too slow for high-frequency control (25–50 Hz target)
2. **Task performance:** Unreliable on bimanual, novel setups even after LoRA fine-tuning

Root cause of the speed problem: OpenVLA's autoregressive decoding generates action tokens one-by-one. For action chunk size K and action dimensionality D, you need K×D sequential decoder forward passes. That bottleneck makes chunking—which would otherwise improve trajectory smoothness—computationally prohibitive.

### Why this paper is needed
Prior work (Kim et al. 2023) showed LoRA works for single-arm low-frequency robots. Recent tokenisation tricks (VQ, cosine transform) give 2–13× gains. But nobody had studied the full design space cleanly. This paper runs *controlled ablations* over three axes simultaneously.

---

## 2. Three Design Axes

The paper frames fine-tuning as choices along three independent axes:

| Axis | Option A (baseline) | Option B |
|------|---------------------|----------|
| Action generation | Autoregressive (causal mask) | Parallel decoding (bidirectional mask) |
| Action representation | Discrete (256-bin tokens) | Continuous (L1 MLP or diffusion) |
| Learning objective | Next-token prediction (CE) | L1 regression / conditional diffusion |

The key insight: these interact, but they can be studied incrementally because parallel decoding is a prerequisite for efficient chunking, and chunking is what makes continuous actions practical.

---

## 3. Architecture

### 3.1 Base Model
**OpenVLA-7B** = Prismatic VLM (SigLIP + DINOv2 dual vision encoders) + Llama-2-7B decoder.
Pretrained on 1M Open X-Embodiment episodes.
Original: predicts 7 discrete robot action tokens autoregressively, each from a 256-bin vocabulary, cross-entropy loss.

### 3.2 Parallel Decoding + Action Chunking

Key modification: replace the causal attention mask with a **bidirectional attention mask** and feed **empty action embeddings** as input tokens. The model maps all input embeddings → predicted output sequence in a *single forward pass*.

This reduces D sequential passes → **1 pass**, regardless of action dimensionality.

For chunking (K timesteps, D-dim actions):
- Insert K×D empty action embeddings into the decoder input
- Model predicts all K×D actions simultaneously
- Throughput gain: K-fold (minimal latency increase since all K are in one pass)
- Execute full chunk before re-querying

> **Critical observation:** Parallel decoding may theoretically be less expressive than autoregressive (since future tokens can't influence past tokens during generation). But empirically, the paper sees *no performance degradation*—and the bidirectional attention actually helps capture action dependencies.

### 3.3 Continuous Action Head (L1)

Replace decoder output embedding layer with a **4-layer MLP action head**:
- Input: final decoder layer hidden states
- Output: continuous action values (normalised to [−1, +1])
- Loss: **mean L1** between predicted and ground-truth actions

Discrete alternative: 256-bin discretisation of normalised actions → softmax over vocabulary → cross-entropy loss (loses fine-grained precision).

Diffusion alternative: conditional denoising (50 steps at train, DDIM at test with 1–50 steps). Expressive but slow.

**L1 vs Diffusion finding:** Comparable task performance, but L1 is **26× faster** (no denoising steps). For a 7B model this is the right tradeoff.

### 3.4 FiLM for Language Grounding (OFT+)

Problem discovered on ALOHA: with multiple camera viewpoints, policies learn spurious visual correlations and ignore language. Fix: **Feature-wise Linear Modulation (FiLM)**.

$$\text{FiLM}(\mathbf{F}|\gamma, \beta) = \hat{\mathbf{F}} = (1 + \gamma) \odot \mathbf{F} + \beta$$

- Compute average of language embeddings **x** from task description
- Project to scaling vector γ and shift vector β (each D_{ViT}-dimensional)
- Apply element-wise to *all patch embeddings* in each ViT block (after self-attention, before FFN)
- This makes γ and β influence all patches uniformly — not individual patch-by-patch modulation

**Key detail:** Individual patch modulation was tried and failed (poor language following). Spatially-agnostic FiLM (same γ, β across all patches per block) is what works, consistent with how FiLM works in conv nets.

FiLM is only used for ALOHA (multi-camera), not LIBERO (single camera). The spurious correlation problem doesn't manifest with one viewpoint.

### 3.5 Additional Inputs
For multi-view setups: up to 3 camera images processed through shared dual-encoder (SigLIP+DINOv2), yielding 256 patch embeddings per view. Robot proprio state (joint angles, gripper) passed through a separate projection network → concatenated as one additional embedding. Total input to decoder: up to 256×3 + 1 = 769 embeddings.

---

## 4. Experiments

### 4.1 LIBERO Simulation Benchmark
**Setup:** Franka Panda arm, 4 task suites (Spatial, Object, Goal, Long), 500 episodes per suite (10 tasks × 50 eps). Fine-tune on each suite independently with 500 demonstrations.
**Training:** 50–150K gradient steps, batch 64–128, 8× A100/H100.

**Table I Results (Modified dataset — unsuccessful demos filtered):**

| Method | Spatial | Object | Goal | Long | **Avg** |
|--------|---------|--------|------|------|---------|
| Diffusion Policy (scratch) | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| OpenVLA (fine-tuned) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| + PD+AC | 91.3 | 92.7 | 90.5 | 86.5 | 90.2 |
| **OpenVLA-OFT (PD+AC+Cont-L1)** | **97.6** | **98.3** | **96.2** | **90.7** | **95.3** ★ |
| π₀ (fine-tuned) | 96.8 | 98.4 | 95.8 | 85.2 | 94.2 |

**Key findings:**
- PD+AC alone: +14% absolute over baseline — temporal dependencies captured, compounding errors reduced
- Continuous L1: +5% over discrete — higher precision, avoids discretisation artefacts
- OpenVLA-OFT beats π₀ (which has much larger pretraining + flow matching) with a *simpler* recipe
- LIBERO-Long gains are largest: chunking is most beneficial for long-horizon tasks

### 4.2 LIBERO Inference Efficiency
**Setup:** 100 queries on NVIDIA A100, 224×224 image + LIBERO language instruction.

| Config | Throughput (Hz) ↑ | Latency (s) ↓ |
|--------|-------------------|----------------|
| OpenVLA baseline | 4.2 | 0.2396 |
| + PD | 15.9 | 0.0629 |
| + PD+AC (K=8) | 108.8 | 0.0735 |
| **OpenVLA-OFT (PD+AC+Cont-L1)** | **109.7 ★** | **0.0729** |
| + Additional inputs (wrist, proprio) | 71.4 | 0.1120 |

Diffusion (50 steps): 4.2 Hz at Ttrain=Ttest=50. With DDIM (Ttest=1): 109.4 Hz but success rate collapses to 0 on Long suite.

**Critical observation:** 26× throughput gain from PD+AC comes almost entirely from removing the K-fold sequential passes. The L1 head adds minimal overhead.

### 4.3 ALOHA Real-Robot Tasks
**Setup:** Two ViperX 300 S arms, 3 cameras (top + 2 wrist), 14-D joint angles, 25 Hz control. 4 tasks: fold shorts (20 demos), fold shirt (30), scoop X into bowl (45), put X into pot (300).

**Table III Inference (100 queries, three 224×224 images + proprio + language):**

| Method | Throughput (Hz) | Latency (s) |
|--------|-----------------|-------------|
| OpenVLA | 1.8 | 0.543 |
| **OpenVLA-OFT+** | **77.9** | **0.321** |
| RDT-1B | 84.1 | 0.297 |
| Diffusion Policy | 267.4 | 0.090 |
| π₀ | 291.6 | 0.086 |
| ACT | 432.8 | **0.058 ★** |

OFT+ achieves 77.9 Hz despite 7.5B params — comparable to RDT-1B (1.2B) because it generates all 25-step actions in a single forward pass.

**Task performance (avg % completion):**

| Method | Avg Score |
|--------|-----------|
| ACT (scratch) | 72.3 |
| Diffusion Policy | 77.5 |
| RDT-1B | 78.4 |
| π₀ (fine-tuned) | 83.9 |
| **OpenVLA-OFT+** | **87.8 ★** |

**FiLM ablation:** Without FiLM on language-dependent tasks (scoop X, put X into pot) → success rate drops to **33% (chance level)**. FiLM is not just helpful — it is *essential* for language following on multi-camera setups.

---

## 5. Ablation Studies — What Actually Matters

### Ordering of contributions (cumulative):
1. PD+AC → +14% (biggest single gain)
2. Continuous L1 → +5% on top
3. FiLM → essential for language following on ALOHA; neutral on LIBERO

### Chunking size K=8 vs K=25:
- LIBERO uses K=8 (matches Diffusion Policy baseline for fair comparison)
- ALOHA uses K=25 (execute full chunk then re-query)
- Larger K reduces re-querying frequency but increases latency slightly

### Why diffusion is not worth it here:
- Comparable quality to L1 at K=50 denoising steps
- At Ttest < 10 steps: accuracy degrades significantly
- At Ttest = 1: fails completely on Long (collapses to 0%)
- L1 with single forward pass matches 50-step diffusion → no justification for diffusion cost

---

## 6. Limitations (from paper + my reading)

1. **Multimodal demonstrations:** L1 regression learns the median action when multiple valid actions exist. Diffusion can capture multimodal distributions. For warehouse navigation with tight corridors, this may matter.

2. **Pre-training representations:** Beneficial (5.2% drop if ablated), but the study is fine-tuning focused. Whether OFT scales to pretraining-level benefit is untested.

3. **FiLM inconsistency:** Works for ALOHA (bimanual, multi-camera), not needed for LIBERO (single camera). Why? Possibly because LIBERO lacks the spurious visual correlation problem. *Our AMR setting: single front camera + possible occlusion → unclear if FiLM needed.*

4. **Domain gap:** Tabletop manipulation only. No wheeled AMR, no Nav2, no map-based reasoning.

---

## 7. Critical Analysis — What This Means for VLA4AMR

### What transfers directly:
- **Parallel decoding + action chunking (C2):** K=8 gives 26× throughput. For AMR navigation targeting 10 Hz, even a conservative K=4 should be sufficient. The bidirectional attention trick is architecture-clean and can be applied to any VLA.
- **L1 MLP head (C2):** Replace discrete action tokens (e.g. directional command vocabulary) with a 4-layer MLP outputting continuous (direction, distance, context) tuples. Expected +5% and faster inference.
- **FiLM (C4):** Our warehouse has a single front camera + LiDAR. FiLM may be less critical than for ALOHA. But if we add a wrist/rear camera later, FiLM is non-negotiable.

### What does NOT transfer:
- **Action dimensionality assumption:** Their action head predicts 7D (manipulation joints) or 14D (bimanual). Our action is semantically different: (direction ∈ [−π, π], distance ∈ ℝ₊, context ∈ categorical). Need to redesign the MLP head output.
- **No reasoning:** OFT is purely reactive — no CoT, no memory. This is the gap that C1 (AdaCoT) and C3 (handoff) address.
- **No navigation stack integration:** Their policy outputs raw joint angles. Ours must output Nav2-compatible semantic waypoints.

### The FiLM ablation finding (FiLM → 33% without it) is *our strongest argument for C4:*
If FiLM is that critical for ALOHA language following, and our AMR needs to execute "turn left at the yellow shelf, avoid the forklift" — then our C4 fusion ablation comparing token concat vs FiLM vs cross-attention becomes critical. Token concat may fail for the same spurious correlation reason.

### Numbers to quote in paper:
- 26× throughput gain (PD+AC)
- 109.7 Hz on A100 → extrapolate to ~X Hz on RTX 5000 Pro (48GB, Blackwell — likely faster)
- 97.1% LIBERO avg, 87.8% ALOHA avg — state of the art with a simple recipe

---

## 8. Open Questions After Reading

- Does K=8 chunking translate to navigation? In manipulation, K=8 joint angle targets form a smooth trajectory. In navigation, K=8 waypoints might over-commit to a path before replanning — need to study.
- Is FiLM strictly necessary in our warehouse setting (single front camera, LiDAR occupancy) or does AMCL confidence substitute for language grounding?
- Can we apply OFT to OpenVLA-7B *without* repretraining, just fine-tuning on our warehouse dataset? (The paper says yes for manipulation — we need to verify for navigation.)
- BridgeData-V2 has ~15K episodes. OFT paper used 500 demos per task suite and achieved 95%+. Our data scale should be more than sufficient.
