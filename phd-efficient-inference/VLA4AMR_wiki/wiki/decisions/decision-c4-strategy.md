# Decision: Token Concat as fusion strategy for C2 and C5

**Type:** decision
**Status:** active
**Last updated:** 2026-06-20
**Related:** [[C4-FusionAblation]], [[C2-MidLevelActionHead]], [[C5-RLFinetuning]], [[Isaac-Synthetic]]

## Decision

Use **token concat** (baseline, zero extra parameters) as the fusion strategy for C2 and C5.

## Evidence

From [[C4-FusionAblation]] open-loop RMSE eval on 500 held-out Isaac-Synthetic samples:

| Strategy | Total RMSE | Nav RMSE |
|---|---|---|
| Token concat | **0.1697** | **0.3174** |
| Cross-attention | 0.1962 | 0.3671 |
| FiLM | 0.1997 | 0.3735 |

Token concat wins on both metrics. Training val loss ranking matched: token_concat=0.1669, film=0.3152, cross_attn=0.4066.

## Rationale

Data-scarce regime (5K samples) favours the zero-param baseline. FiLM and cross-attn each add ~67M parameters — insufficient data to learn meaningful goal conditioning before the LoRA capacity is saturated. The OpenVLA-OFT FiLM finding (33% without FiLM on ALOHA) was for a multi-camera tabletop manipulation setting; our single-front-camera AMR setting is simpler and may genuinely not need spatial goal conditioning at this data scale.

**Paper framing:** present this as a finding, not a failure. "In data-scarce industrial warehouse settings, architectural complexity does not help — a key result for practitioners."

## Cross-attn lx anomaly

Cross-attn achieves the best forward-speed prediction (lx RMSE=0.0801 vs token_concat=0.1794) despite worse overall RMSE. Hypothesis: cross-attn learns to attend to the floor-level visual tokens most relevant to straight-line driving, but lacks enough data to generalise to turning. Worth one sentence in the paper.

## Implications

- **C2 (MidLevelActionHead):** use token concat prompt format — goal instruction prepended to image tokens as text. No architectural changes to OpenVLA backbone.
- **C5 (RLFinetuning):** PPO rollouts use token concat inference — same as current bw03_inference_server.py. No fusion hook needed.
- **Future work:** revisit FiLM with BridgeData-V2 + HM3D (>20K samples) — at larger data scale FiLM may recover its advantage.
