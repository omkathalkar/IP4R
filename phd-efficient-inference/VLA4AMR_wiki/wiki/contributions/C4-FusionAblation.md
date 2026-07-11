# C4 — Fusion Ablation

**Type:** contribution
**Status:** active
**Last updated:** 2026-06-20
**Related:** [[OpenVLA-OFT]], [[C2-MidLevelActionHead]], [[bw03_c4_fusion]], [[C6-EvaluationProtocol]]

## Summary

Compares three strategies for injecting a language goal embedding into OpenVLA-7B's
visual token stream for warehouse AMR navigation. The ablation is motivated by the
OpenVLA-OFT finding that removing FiLM from ALOHA drops success rate to 33% (chance).
We ask: which fusion strategy best supports language-directed navigation in our setting?

## Three fusion strategies

### A. Token concat (baseline)
Goal instruction is tokenised and prepended to the image tokens in the VLA prompt:
`"In: What action should the robot take to {TASK_INSTR}? Out:"`
No architectural change — this is what the base OpenVLA-7B already does.
Added parameters: **0**.

### B. FiLM conditioning
Project goal embedding → (γ, β) ∈ R^{4096}, apply spatially-agnostic modulation to all
visual patch tokens before the first LLM transformer block:

```
v̂ = (1 + γ) ⊙ v + β
```

Key constraint from OpenVLA-OFT: individual patch modulation (different γ/β per patch position)
**failed** — only spatially-shared γ/β works. Implemented in `FiLMConditioner`.
Added parameters: **~67M** (two Linear layers: 4096→8192→8192).

### C. Cross-attention
Insert a multi-head cross-attention layer after the vision encoder projection:
visual patch tokens = Q; goal text tokens = K, V.
More expressive than FiLM (goal can selectively attend to patches) but adds latency.
Implemented in `CrossAttentionFuser` (8 heads, residual + LayerNorm).
Added parameters: **~67M** (four Linear layers + norm).

## Architecture hookup

OpenVLA-7B (prismatic VLM):
- Vision encoder: SigLIP-So400M → 256 patch tokens (14×14 at 224px), dim=1152
- Projection: linear → LLM space, dim=4096
- LLM: LLaMA-7B, 32 transformer blocks, hidden_dim=4096

Fusion is applied **after projection, before the first LLM block** — on the 256 visual tokens
in LLM hidden space. This avoids modifying the ViT internals.
Goal embedding: mean-pool of tokenised goal text → LLaMA embed_tokens → mean over sequence.

Hook mechanism: `FusionHook` registers a `register_forward_hook` on `embed_tokens` and
intercepts the first 256 tokens (visual patch slice) to apply FiLM or cross-attention.

## Hypothesis

| Strategy | Expected strength | Expected weakness |
|---|---|---|
| Token concat | Simple, no new params | Goal may be drowned by 256 visual tokens |
| FiLM | Proven on ALOHA; cheap | Spatially blind (same γ/β everywhere) |
| Cross-attention | Goal can select patches | More params, higher latency |

Prediction: FiLM ≥ cross-attn > token-concat for warehouse AMR with text goals.
Reasoning: token concat has the same spurious visual correlation risk as the ALOHA no-FiLM
condition. Cross-attn may over-fit with limited warehouse training data.

## Experimental plan

1. **Dataset**: [[Isaac-Synthetic]] (~5K episodes), each episode = (image, text_goal, action_7d)
   Text goals: ~10 templates (e.g. "Navigate to shelf {X}", "Turn right at the aisle", ...)
2. **Fine-tuning**: LoRA (rank 32) on OpenVLA-7B, each strategy trained for 3 epochs
3. **Eval (open-loop)**: RMSE on held-out action sequences — [[C6-EvaluationProtocol]]
4. **Eval (closed-loop)**: task completion rate in Isaac Sim (10 goal → 10 start combos)
5. **Latency**: measure per-inference dt on RTX PRO 5000 Blackwell for all three

## Current status (BW03)

- [x] Scaffold written: `bw03_c4_fusion.py` — `FiLMConditioner`, `CrossAttentionFuser`, `FusionHook`
- [x] Smoke test passes (CPU, random tensors): shape assertions ✓
- [x] Isaac-Synthetic dataset generation scripts written: `bw03_isaac_collect.py` + `bw03_c4_collect_v2.py`
- [x] LoRA fine-tuning script written: `bw03_c4_train.py` (run in openvla env)
- [x] Open-loop RMSE eval script written: `bw03_c4_eval.py` (run in openvla env)
- [x] Isaac-Synthetic **v2** dataset generated: 5 000 samples, 0 skipped, 0 OOB (2026-06-19)
  - v1 (2026-06-18) was discarded: robot spawned boxed-in, all frames were crashes
  - v2 fix: teleport to safe spawn (0,-8,0) + per-rep reset via flag handshake
- [x] **All 3 strategies trained** (2026-06-19, openvla env, RTX PRO 5000 Blackwell)
  - Checkpoints at `~/VLA4AMR/checkpoints/c4/{token_concat,film,cross_attn}/lora/`
  - FiLM weights: `film/film.pt` (193 MB); Cross-attn weights: `cross_attn/xattn.pt` (129 MB)
- [x] **Open-loop RMSE eval complete** (2026-06-20, 500 test samples) — results in `~/VLA4AMR/checkpoints/c4/eval_results.json`
- [x] **Strategy decision made: token_concat → C2 and C5** (2026-06-20) — see [[decision-c4-strategy]]
- [ ] Closed-loop SR evaluation in Isaac Sim (optional — C6)

## Training results (2026-06-19) — cross-entropy val loss, 3 epochs

| Strategy | Epoch 1 val | Epoch 2 val | Epoch 3 val | Best val loss |
|---|---|---|---|---|
| Token concat | 0.7879 | 0.4019 | **0.1669** | **0.1669** |
| FiLM | 0.8026 | 0.5515 | 0.3152 | 0.3152 |
| Cross-attention | 0.8180 | 0.6799 | 0.4066 | 0.4066 |

**Surprise:** token_concat dominates — opposite of the hypothesis (FiLM ≥ cross_attn > token_concat).
Possible explanation: with only 5K samples, FiLM and cross-attn (~67M extra params each) overfit relative
to token_concat (0 extra params). Open-loop RMSE eval needed to confirm.

## Open-loop RMSE results (2026-06-20, 500 test samples)

| Strategy | Total RMSE | Nav RMSE | Dim0 lx | Dim5 az | Closed-loop SR | Latency (ms) |
|---|---|---|---|---|---|---|
| Token concat | **0.1697** | **0.3174** | 0.1794 | 0.4115 | TBD | ~200 (baseline) |
| FiLM | 0.1997 | 0.3735 | 0.1143 | 0.5157 | TBD | TBD |
| Cross-attention | 0.1962 | 0.3671 | **0.0801** | 0.5129 | TBD | TBD |

**Key findings:**
- Token concat wins on total RMSE and nav RMSE — consistent with training loss ranking
- Cross-attn has the best lx (forward speed) prediction (0.0801 vs 0.1794) despite worst overall training loss
- All strategies struggle on az (angular velocity, dim5): 0.41–0.52 RMSE — turning is the hard part
- Dims 1–4 and 6 are always zero (differential drive), correctly predicted by all strategies

## Open questions

- Is FiLM strictly necessary for our single-front-camera AMR setting, or does the single viewpoint
  eliminate the spurious correlation problem (as it did for LIBERO in OpenVLA-OFT)?
- How many Isaac-Synthetic episodes are needed for FiLM vs cross-attn to overfit — what's the
  minimum data regime where FiLM's parameter efficiency matters?
- Do we apply fusion at every LLM layer (like FiLM in conv nets) or only before block 0?

## Sources

- [[OpenVLA-OFT]] — FiLM conditioning, ALOHA ablation (33% without FiLM), spatially-agnostic finding
- [[OpenVLA-OFT-reading-notes]] — critical analysis, C4 implications
- `bw03_c4_fusion.py` — scaffold implementation (FiLMConditioner, CrossAttentionFuser, FusionHook)
