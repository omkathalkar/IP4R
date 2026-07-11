# Decision: Entropy gate threshold θ = 2.795 nats for C1 AdaCoT

**Type:** decision
**Status:** active
**Last updated:** 2026-06-20
**Related:** [[C1-AdaCoT]], [[bw04_c1_adacot]]

## Decision

Use **θ = 2.795 nats** as the entropy gate threshold for C1 AdaCoT in the warehouse AMR setting.

## Evidence

Two ablation sweeps on 450 robot-front frames from `bw03_tour_v3` in Isaac Sim 6.0.0.1:

Sweep 1 (θ ∈ {0.3, 0.5, 0.7}): all fired 100% — real warehouse H ∈ [1.88, 2.89], well above 0.7.

Sweep 2 (θ ∈ {2.664, 2.719, 2.795, 2.830}):

| θ | Activation rate | Latency |
|---|---|---|
| 2.664 | 10.0% | 245ms |
| 2.719 | 5.1% | 237ms |
| **2.795** | **3.1%** | **233ms** |
| 2.830 | 2.0% | 231ms |

## Rationale

- 3.1% is the closest to VLingNav's reported 2.1% optimal activation rate
- +10ms latency overhead over no-CoT baseline (223ms) — negligible at 5Hz
- CoT fires only on top-3% highest-entropy steps: intersections, occlusions, ambiguous goals
- When CoT fires, mean entropy drops from 2.401 → 2.095 — reasoning provides genuine disambiguation, not noise
- θ=2.795 is interpretable: it corresponds to the p97 percentile of the warehouse frame entropy distribution

## Environment note

θ is environment-specific. For new environments or scenes, recalibrate by running `no_cot` mode on representative frames and picking the p97 percentile of the entropy distribution.
