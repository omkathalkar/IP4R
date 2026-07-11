# Decision: OFT recipe (L1 + parallel + chunking) over diffusion

**Type:** decision
**Status:** active
**Last updated:** 2026-05-23
**Related:** [[OpenVLA-OFT]], [[C2-MidLevelActionHead]]

## Decision
Use L1 regression head (OFT recipe) rather than diffusion for action prediction.

## Rationale
From [[OpenVLA-OFT]] ablation: L1 and diffusion achieve comparable task success, but L1 is 26× faster (no denoising steps). For real-time AMR navigation (target 10Hz+), 26× throughput advantage is critical. Diffusion also has slower training convergence. On RTX 5000 Pro 48GB in BF16, OFT runs at 109.7 Hz — well into real-time territory.
