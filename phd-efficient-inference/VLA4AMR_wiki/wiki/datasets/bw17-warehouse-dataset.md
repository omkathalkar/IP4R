# BW17 Warehouse Dataset

**Type:** dataset
**Status:** complete
**Last updated:** 2026-07-11
**Related:** [[Isaac-Synthetic]], [[IsaacSim]], [[decision-bw18-ticvla-bw17]], [[C6-EvaluationProtocol]]

## Summary

Synthetic warehouse navigation dataset generated in Isaac Sim 6.0.0.1 for BW18 TIC-VLA training.
9 task types, 560 episodes total, converted to TIC-VLA DynaNav JSON format with sliding-window
trajectory segmentation, Qwen2.5-VL instruction augmentation, and decision-frame CoT annotation.
Final output: 14,360 DynaNav_json sample directories on the simulator.

## Episode statistics

| Split | Episodes | Windows | DynaNav dirs |
|-------|----------|---------|--------------|
| Train | 392      | 2,510   | 10,040       |
| Val   | 84       | 540     | 2,160        |
| Test  | 84       | 540     | 2,160        |
| **Total** | **560** | **3,590** | **14,360** |

## Task types (9)

| Task | Description | Episodes |
|------|-------------|----------|
| `aisle_fwd` | Straight forward through warehouse aisle | 80 |
| `aisle_fwd_slow` | Same aisle, slower speed | 40 |
| `aisle_u_turn` | Travel south then U-turn at Y≈−4.5m dead end (kinematic, no wall contact) | 80 |
| `cross_turn_left` | Cross-aisle left turn at intersection | 80 |
| `cross_turn_right` | Cross-aisle right turn at intersection | 80 |
| `cross_turn_left_slow` | Same left turn, reduced speed | 40 |
| `cross_turn_right_slow` | Same right turn, reduced speed | 40 |
| `obj_goal` | Navigate to a target object | 80 |
| `obstacle_slalom` | Navigate around dynamic obstacles | 40 |

## Collection parameters

| Parameter | Value |
|-----------|-------|
| Frame rate | 10 Hz |
| Image resolution | 336 × 336 JPEG |
| Action: linear velocity | m/s |
| Action: angular velocity | rad/s |
| Episode length | task-dependent (40–200 frames) |
| Scene | `warehouse_multiple_shelves.usd` (Isaac Sim 6.0) |
| Robot | Nova Carter (differential drive) |
| `CUDA_VISIBLE_DEVICES` | 0 (RTX 5000 Pro) |
| Conda env | `isaac6` |

## Pipeline scripts

| Step | Script | Description |
|------|--------|-------------|
| 1. Collect | `bw17_collect.py` | Isaac Sim headless collection, decision-frame marking |
| 2. Window | `bw17_window.py` | Sliding-window JSON (STRIDE=3, HISTORY=5, HORIZON=30) |
| 3. Instruct | `bw17_instruct.py` | 4 instruction variants/window via Qwen2.5-VL-7B |
| 4. Annotate | `bw17_annotate.py` | CoT at decision frames via Qwen2.5-VL-7B |
| 5. Verify | `bw17_verify.py` | 8-check sanity suite (ALL PASSED) |
| 6. Convert | `bw17_convert.py` | → TIC-VLA DynaNav_json format |

## Sliding window format (post-bw17_window)

```
HISTORY  = 5    # delayed frames before current
HORIZON  = 30   # future waypoints (3 s at 10 Hz)
STRIDE   = 3    # window step (0.3 s)
```

Each window JSON contains:
- `waypoints`: (30, 2) cumulative FLU offsets from current frame
- `is_decision_window`: true at task phase transitions
- `instructions`: list of 4 phrasings (added by bw17_instruct)
- `cot`: chain-of-thought string (added by bw17_annotate; `""` on non-decision windows)

## FLU coordinate convention

Forward-Left-Up, body frame. Heading=0 → facing +Y (north).

```python
def world_to_flu(dx, dy, heading):
    flu_x = dx * sin(heading) + dy * cos(heading)   # forward
    flu_y = -dx * cos(heading) + dy * sin(heading)  # left
    return flu_x, flu_y
```

## TIC-VLA DynaNav_json format

One subdirectory per (window × instruction-variant) = 4 subdirs per window.
Each subdir contains exactly 1 JSON file → TICVLADataset_VLM's every-5th-file filter keeps all samples.

```
{window_id}_i{0..3}/
  t_000000.json     — history + current + future + instruction_file + cot
  instruction.txt   — one instruction phrasing
  cot.txt           — CoT text (only if is_decision_window)
```

JSON schema:
```json
{
  "timestamp": 0,
  "history": [{"img": "path.jpg", "offset": [0,0,0], "orientation": [0,0,0,1]}, ...],
  "current": {"img": "path.jpg", "orientation": [0,0,0,1]},
  "future":  [{"offset": [fx, fy, 0], "orientation": [0,0,0,1]}, ...],
  "instruction_file": "instruction.txt",
  "cot": "cot.txt"
}
```

## Dataset location (simulator)

```
~/Desktop/bw17_dynav/
  train/DynaNav_json/    10,040 sample dirs
  val/DynaNav_json/      2,160 sample dirs
  test/DynaNav_json/     2,160 sample dirs
```

Raw collection output (preserved): `~/Desktop/bw17_dataset/`

## Verification results (bw17_verify.py)

- Duplicate IDs: 0
- Required fields: all present (3,590/3,590)
- Image paths: 21,540 verified (HISTORY×3,590 + current×3,590)
- Waypoints shape: (30,2) ✓, no stuck episodes (mean final magnitude 1.895 ± 0.572 m)
- Instructions: 4 items per window ✓
- CoT on decision windows: all annotated ✓
- Per-task balance: within spec ✓
- Report: `~/Desktop/bw17_dynav/verify_report.json`

## GPU notes

- **Collection** (`isaac6` env): `DISPLAY=:1 CUDA_VISIBLE_DEVICES=0`
- **Instruct/Annotate** (`openvla` env): `--device cuda:0` = RTX PRO 5000 Blackwell (47.3 GiB)
  - GPU order is REVERSED vs nvidia-smi in openvla env
- **Training** (`tic-vla` env): `CUDA_VISIBLE_DEVICES=0` = RTX PRO 5000 Blackwell

## Open questions

- Does 3,590 windows (14,360 after 4× augmentation) provide enough diversity for BW18 action head?
- CoT quality on `obj_goal` task — fewest episodes, most variable trajectories.

## Sources

- `bw17_collect.py` through `bw17_convert.py` — generation pipeline scripts
- [[IsaacSim]] — simulation platform
- [[decision-bw18-ticvla-bw17]] — training decision rationale
