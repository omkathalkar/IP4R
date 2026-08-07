# C3 — Confidence-Gated Nav2 Handoff

**Type:** contribution
**Status:** active
**Last updated:** 2026-08-07
**Related:** [[C1-AdaCoT]], [[C2-MidLevelActionHead]], [[VLingNav]], [[Language-as-Cost]], [[decision-bw20-hybrid-navstack]], [[bw17-warehouse-dataset]], [[simulator-machine]]

## Summary

C3 implements a hybrid A\*/VLA navigation stack where a global A\* planner provides waypoints and a
frozen VLA (FlowVLA-BW v3) provides local control commands. A dual-signal confidence gate decides
per-step whether to trust the VLA or fall back to a pure-pursuit geometric controller. This replaces
the original AMCL-covariance trigger design with a simpler, provably-testable output-magnitude gate.

## Architecture

```
start_xy ──► A* planner ──► waypoints[0..N]
                                  │
                                  ▼
              robot_pose ──► WaypointToVLAInput
                                  │ instruction_str (4-vocab)
                                  ▼
              camera_frame ──► VLA inference ──► (vla_lin, vla_ang)
                                  │
                                  ▼
                           ConfidenceGate
                          ┌────────────────────────────────────┐
                          │  mag_ok  = ||(lin,ang)|| ≥ 0.15   │
                          │  prog_ok = net_progress ≥ 0.1m/W  │
                          │  used_vla = mag_ok AND prog_ok     │
                          └────────────────────────────────────┘
                                  │                 │
                           [used_vla=T]       [used_vla=F]
                                  │                 │
                              VLA cmd          PurePursuit cmd
                                  └────────┬────────┘
                                           ▼
                                   /cmd_vel → robot
```

**Key property:** the gate is a function of VLA outputs only — it requires no external localisation
signal, no AMCL, no obstacle detector. This makes it trivially testable offline and deployable
without a full ROS stack.

## Modules

| Module | File | Purpose |
|--------|------|---------|
| A\* global planner | `planning/astar_planner.py` | 8-connected A\*, Douglas-Peucker simplification, min 0.5m waypoint spacing |
| Occupancy grid | `grid/generate_occupancy_grid.py` | Parametric warehouse approximation (30×20m, 0.05m/cell, robot radius 0.4m) or ROS PGM |
| VLA input mapper | `interface/waypoint_to_vla_input.py` | Converts rel_heading → one of 4 FlowVLA-BW v3 vocab strings |
| Confidence gate | `control/confidence_gate.py` | Dual-signal gate: magnitude + net progress over sliding window |
| Pure pursuit | `control/pure_pursuit.py` | Fallback geometric controller: lookahead 1.2m, max_lin=0.35, max_ang=1.0 |
| VLA local executor | `executor/vla_local_executor.py` | High-level step loop: WP advance, stuck detection, ExecutorState |
| Isaac Sim episode | `sim/run_episode_simulator.py` | Closed-loop sim runner (isaac6 env, GPU 1 Blackwell) |
| VLA worker | `sim/vla_inference_worker.py` | Separate process (tic-vla env, GPU 0 4060 Ti), IPC file protocol |
| Headless eval | `sim/run_episode_headless.py` | Replay-based eval on BW17 DynaNav data — no Isaac Sim required |
| SLURM job | `sim/run_episode_ada.slurm` | 20-episode array job on Ada HPC (partition u22, 1 GPU per task) |
| Sync script | `sim/sync_to_ada.sh` | rsync data + checkpoint + code from simulator → Ada |
| Eval suite | `eval/run_eval_suite.py` | 3-column paper table: Hybrid vs Pure VLA vs Pure A\* |

**File tree:** `~/Desktop/nav_stack/` on cvit-car-simulator; mirrored to `/home2/om.kathalkar/nav_stack/` on Ada.

**Test coverage:** 92 tests across 4 test files, all passing (pytest, tic-vla conda env).

## FlowVLA-BW v3 vocabulary constraint

FlowVLA-BW v3 was trained on exactly 4 instruction strings. Any other string is OOD and produces
near-zero magnitude (≈0.0001). The gate's mag check enforces this implicitly.

| Key | Instruction string | When used |
|-----|--------------------|-----------|
| `forward` | "Drive forward through the warehouse aisle" | |heading| < 20° |
| `turn_left` | "Turn left to navigate the warehouse corridor" | heading > +20° |
| `turn_right` | "Turn right at the intersection" | heading < −20° |
| `slow` | "Navigate carefully and slow down" | within 0.8m of final WP |

`WaypointToVLAInputOffline.step()` computes `rel_heading = atan2(wy-ry, wx-rx) − robot_theta`
and maps it to one of the four keys above. The string is cached — IPC writes
`current_instruction.txt` only when the bucket changes (waypoint transition).

## Confidence gate calibration

| Signal | Threshold | Calibration source |
|--------|-----------|--------------------|
| VLA magnitude `||(lin,ang)||` | ≥ 0.15 | Trained VLA: ~0.46, OOD/non-functional: ~0.0001 |
| Net progress toward WP | ≥ 0.1 m over window W=20 steps | Pure-pursuit fallback provides ≥0.33×0.2=0.066m/step → window fills in 1.5s |
| Progress window W | 20 steps (4s at 5 Hz) | Reset at each waypoint advance |

**Benefit of doubt:** when the progress window is not yet full (first 20 steps of a new waypoint),
`progress_ok = True` — the gate does not penalise the VLA before it has had a chance to make progress.

In Ada headless eval (single-frame replay), the gate runs in magnitude-only mode
(`progress_window=1, progress_min_m=0.0`) — no sequential position history is available.

## Novelty claim

1. **Architecture novelty:** First hybrid A\*/VLA stack where the handoff criterion is derived purely
   from VLA output magnitude — no external sensor (AMCL covariance, obstacle detector, GPS) required.
   Enables deployment on AMRs with degraded localisation.

2. **Gap vs Language-as-Cost [[Language-as-Cost]]:** That paper uses a VLM as an additional semantic
   cost term in Nav2's costmap — Nav2 is always in control. C3 performs a *full handoff*: VLA
   replaces Nav2's cmd_vel when confident. The switch criterion design (dual-signal gate, calibrated
   threshold) is novel.

3. **Gap vs VLingNav [[VLingNav]]:** VLingNav's C3-analog is a covariance monitor requiring SLAM.
   Our gate requires only the VLA's own output — no SLAM, no map quality metric.

## IPC protocol (Isaac Sim ↔ VLA worker)

Directory: `/tmp/navstack_ipc/`

| File | Direction | Meaning |
|------|-----------|---------|
| `frame.jpg` | Isaac → VLA | Current camera frame (448×448) |
| `frame.ready` | Isaac → VLA | Sentinel: new frame available |
| `current_instruction.txt` | Isaac → VLA | Active instruction string (one of 4 vocab) |
| `action.json` | VLA → Isaac | `{"lin_vel": f, "ang_vel": f, "step": n, "instruction": s}` |
| `action.ready` | VLA → Isaac | Sentinel: action ready to consume |
| `quit` | Isaac → VLA | Shutdown signal |

IPC is necessary because isaac6 (Python 3.12, CUDA 12) and tic-vla (Python 3.10, CUDA 11.8) are
incompatible in the same process. See [[decision-ipc-two-process]].

## Evaluation (Phase 7/8)

### Phase 7 — Ada HPC headless batch eval

- Dataset: BW17 DynaNav test split (20 windows, SLURM array)
- Metrics per window: heading error (°), ADE (m), FDE (m) vs GT future, VLA magnitude, gate decision
- Results file: `/home2/om.kathalkar/logs/phase7/<window>_result.json`

### Phase 8 — C3 comparison table

Script: `eval/run_eval_suite.py`

Three columns computed from Phase 7 results:
- **Hybrid (ours):** gated output (`fin_lin, fin_ang`)
- **Pure VLA:** bare VLA without gate (`vla_lin, vla_ang`)
- **Pure A\*:** A\* plan → PurePursuit (no VLA), recomputed at eval time (no GPU needed)

Output: `eval_comparison.tex` (LaTeX booktabs table, ready for paper §IV), `eval_comparison.json`.

## Dead reckoning (Isaac Sim only)

Isaac Sim 6.0.0.1's XformCache does not return physics-engine positions, and the default RTAB-Map
odometry in `carter_warehouse_navigation.usd` publishes to `/chassis/odom` which is not accessible
from within the Isaac Python process. Position is tracked by integrating cmd_vel each step:

```python
dt = 1.0 / SIM_HZ
robot_x += math.cos(robot_theta) * lin * dt
robot_y += math.sin(robot_theta) * lin * dt
robot_theta += ang * dt
```

This provides accurate enough pose tracking for waypoint advance checks (threshold 1.0m) over the
duration of a single warehouse episode (~60–300 steps).

## Open questions

- BW20 dataset collection: v3 dataset has 64% forward bias and only 3 synthetic instruction strings.
  Need ≥150 episodes with operator-typed per-segment instructions (see log [2026-08-04]).
- Phase 9 TIC-VLA baseline: run Phase 7 headless eval with TIC-VLA paper checkpoint to populate
  the `eval_comparison.tex` fourth column — direct comparison for C2 vs C3 in the paper.
- Closed-loop Isaac Sim success rate: currently measured as "robot reaches within 1.0m of goal" on
  a single (3,1)→(22,15) episode. Need ≥10 start/goal pairs for a meaningful success rate number.

## Sources

- `nav_stack/` codebase on cvit-car-simulator (`~/Desktop/nav_stack/`)
- [[decision-bw20-hybrid-navstack]] — design decisions for the hybrid stack
- [[decision-ipc-two-process]] — why file-based IPC between isaac6 and tic-vla
- [[bw17-warehouse-dataset]] — BW17 DynaNav replay data used in Phase 7 eval
- [[simulator-machine]] — cvit-car-simulator GPU config (4060 Ti + Blackwell)
