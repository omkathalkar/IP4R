# Decision: BW20 — Hybrid A*/VLA Nav Stack Design

**Type:** decision
**Status:** complete
**Last updated:** 2026-08-07
**Related:** [[C3-ConfidenceGatedHandoff]], [[decision-ipc-two-process]], [[bw17-warehouse-dataset]], [[simulator-machine]]

## Summary

Four interconnected design decisions made during BW20 (2026-08-07) hybrid nav stack implementation.
Each is documented separately because it was non-obvious, had competing alternatives, and is
relevant to the paper's method section.

---

## Decision 1 — Output-magnitude gate instead of AMCL-covariance trigger

**Chosen:** VLA magnitude `||(lin_vel, ang_vel)||` as the primary confidence signal.

**Rejected:** AMCL pose covariance threshold (original C3 plan).

**Why:**
The original C3 design (BW06 notes) used AMCL covariance `σ_pose > σ_thresh` as the handoff
trigger. This was invalidated by two findings:

1. In Isaac Sim, AMCL is not available in the isaac6 Python process (ROS topic, not Python API).
   The only pose estimate is dead-reckoned from cmd_vel (see [[C3-ConfidenceGatedHandoff]] §Dead reckoning).

2. More fundamentally: AMCL covariance measures localisation uncertainty, not VLA competence.
   A VLA can be confident and wrong (OOD scene, novel texture), or uncertain but
   correct (ambiguous visual scene but in-distribution). What we actually want to measure is
   whether the VLA is producing non-trivial, directionally consistent output — which is exactly
   what magnitude captures.

**Calibration evidence (Phase 7 Ada HPC eval):**
- Trained FlowVLA-BW v3 on in-distribution warehouse frames: magnitude ≈ 0.46
- Same model on OOD scene (VLN-PE indoor): magnitude ≈ 0.0001 (Phase 8 paper checkpoint eval)
- Threshold 0.15 gives 4600× separation — no false positives observed in Phase 7

**Paper framing:** "We gate on VLA output magnitude rather than external localisation uncertainty,
making C3 sensor-agnostic and deployable on AMRs without operational SLAM."

---

## Decision 2 — Replay-based Ada eval instead of Isaac Sim on Ada

**Chosen:** BW17 DynaNav replay frames on Ada HPC (no simulator).

**Rejected:** Isaac Sim on Ada HPC nodes.

**Why:**
Ada HPC (`u22` partition) nodes run headless Ubuntu 22.04 with no GDM session. Isaac Sim 6.0.0.1
requires an Xorg session with NVIDIA Vulkan ICD (`VK_ICD_FILENAMES`). EGL headless mode is
untested on Ada's RTX 2080 Ti nodes and the isaac6 conda env has no EGL fallback path.

Additionally, replay-based eval is more *repeatable* for paper numbers: the same 20 BW17 windows
evaluated by any method gives directly comparable ADE/FDE/heading-error without simulation variance.
Isaac Sim is kept for the qualitative demo (Phase 6, on cvit-car-simulator) rather than
quantitative reporting.

**Boundary:** Isaac Sim = demo video on cvit-car-simulator. Ada HPC = quantitative paper metrics.

---

## Decision 3 — Fixed 4-string vocabulary instead of free-text instructions

**Chosen:** `WaypointToVLAInput` maps all headings to exactly 4 training vocabulary strings.

**Rejected:** Natural language generation (e.g., "Turn 35° left toward waypoint (12.4, 8.3)").

**Why:**
FlowVLA-BW v3 was trained on 4 instruction strings only (from the intern's `warehouse_capture`
teleoperation sessions). Any string outside these 4 is OOD for the text encoder
(`embed_tokens().mean(0)` in `vla_inference_worker.py`). The model produces near-zero output
for OOD instructions — which is exactly what the magnitude gate detects.

The 4-string constraint is therefore a feature during the confidence gate calibration phase:
it provides a clean binary (in-vocab → magnitude ≈ 0.46, out-of-vocab → magnitude ≈ 0.0001)
that lets us set the threshold at 0.15 with zero overlap.

**Risk:** `"slow"` string maps to the same geometry as `"forward"` but with different VLA action.
Observed: the model learned to output lower lin_vel for `"Navigate carefully and slow down"` — this
was validated in FlowVLA-BW v3 training (100% DirAcc for turns, val MAE=0.0044).

**Future work (BW20+ dataset):** with ≥150 balanced episodes and operator-typed instructions,
retrain FlowVLA-BW v4 on a richer vocabulary — then update `VOCAB` in `waypoint_to_vla_input.py`.

---

## Decision 4 — Dual-signal gate (magnitude AND progress) instead of magnitude-only

**Chosen:** `used_vla = mag_ok AND progress_ok` (both signals required).

**Rejected:** `used_vla = mag_ok` (magnitude only).

**Why:**
Magnitude alone catches OOD inference (near-zero output). But there is a second failure mode:
the VLA spins the robot in place — magnitude is high (e.g., ang=0.9), but the robot makes
no progress toward the waypoint. This happens when the model gets stuck predicting large turns
on ambiguous visual features.

Progress gate: net displacement toward the waypoint over a 20-step (4s) window must exceed 0.1m.
This is generous — pure pursuit at min_lin=0.1 m/s makes 0.1×4=0.4m in 4s — so the threshold
is set well below fallback performance to avoid false positives during legitimate turn maneuvers.

**Test coverage:** `test_gate_fallback_on_no_progress` and `test_gate_trusts_strong_vla` in
`tests/test_confidence_gate.py` verify both failure modes and the happy path.

**In Ada headless eval (magnitude-only mode):**
Progress cannot be computed from single-frame replay data (no sequential position history).
`ConfidenceGate(progress_window=1, progress_min_m=0.0)` disables the progress check.
The Phase 7 eval therefore understates the gate's real-world rejection rate.

## Sources

- `nav_stack/control/confidence_gate.py` — implementation
- `nav_stack/tests/test_confidence_gate.py` — 20 tests covering all gate behaviours
- [[C3-ConfidenceGatedHandoff]] — full architecture and eval
- Phase 7 Ada HPC results: `/home2/om.kathalkar/logs/phase7/`
