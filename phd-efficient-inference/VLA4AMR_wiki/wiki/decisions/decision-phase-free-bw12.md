# Decision: Phase-Free Training for BW12

**Type:** decision
**Status:** active
**Last updated:** 2026-07-01
**Related:** [[C1-AdaCoT]], [[Nav-AMR-WH]], [[decision-qwen-bw11]], [[overview]]

## Decision

Remove the `Phase:` field from the user prompt in BW12 training and replace it with a rolling `Memory:` field derived from prior CoT `<summary>` tokens in the same episode.

**BW11 prompt:**
```
Task: {instruction}
Phase: {phase_name}
Output action:
```

**BW12 prompt:**
```
Task: {instruction}
Memory: {last_summary | "Start of episode."}
Output action:
```

## Why

BW11 offline eval revealed that the `Phase:` token was **load-bearing**: the model strongly conditioned turning behaviour on it. In live deployment this is a problem — the nav stack must provide an accurate phase label (approach / turn_right / turn_left / depart) at every frame. This requires a separate phase classifier or heuristic, adding a fragile dependency.

The root cause: training data has a perfect correlation between `Phase: turn_right` and high angular velocity. The model exploits this shortcut rather than learning to turn from visual cues (intersection geometry, corridor end).

**Memory as a better signal:** The `<summary>` tokens from CoT frames capture scene state ("approaching T-junction ahead, right corridor visible") — a richer and naturally accumulated signal than a discrete phase label. The model can use this to maintain context across frames without external annotations.

## Results

BW12 val evaluation on 4,936 samples:
- Turning tasks: ang MAE = **0.0003–0.0028** (near-perfect) — without any Phase: hint
- Overall ang MAE = 0.0124 (vs BW11: 0.0105 with Phase: — small regression acceptable given the dependency is gone)
- Best val_loss = 0.0542 (vs BW11: 0.0551) — marginally better convergence
- 100% parse rate, 0 failures

## How to apply

All future training and inference scripts (BW13+) should use the phase-free memory prompt. The rolling memory chain must be built at inference time by extracting `<summary>` tokens from the model's own CoT output and passing them forward to the next frame's `Memory:` field.

At episode start, initialise memory as `"Start of episode."`.
