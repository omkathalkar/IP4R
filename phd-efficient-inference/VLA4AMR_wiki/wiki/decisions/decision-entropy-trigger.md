# Decision: Entropy gating over learned AdaCoT

**Type:** decision
**Status:** active
**Last updated:** 2026-05-23
**Related:** [[C1-AdaCoT]], [[VLingNav]]

## Decision
Use entropy-gated trigger (H > θ) for AdaCoT rather than learned trigger from VLingNav.

## Rationale
VLingNav learned trigger requires Nav-AdaCoT-2.9M annotated samples. We have ~28K episodes total. Entropy is computable from any VLA model's output distribution — no additional training data required. Tuning θ per environment via ablation is practical at our scale.

## Risk
Entropy may not perfectly correlate with "need to reason". Mitigation: ablation study in BW04 comparing fixed interval vs entropy at multiple θ values.
