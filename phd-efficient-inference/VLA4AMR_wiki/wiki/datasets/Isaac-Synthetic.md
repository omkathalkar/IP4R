# Isaac-Synthetic Dataset

**Type:** dataset
**Status:** complete
**Last updated:** 2026-06-19
**Related:** [[C4-FusionAblation]], [[IsaacSim]], [[NovaCarter]], [[C6-EvaluationProtocol]]

## Summary

Synthetic warehouse navigation dataset generated in Isaac Sim 6.0.0.1 using the
`carter_warehouse_navigation.usd` scene and Nova Carter robot. Each sample is a
(front_camera_image, text_goal, action_7d) triple captured during scripted trajectories.
Used as fine-tuning data for the [[C4-FusionAblation]] experiment.

## Dataset properties

| Property | Value |
|---|---|
| Total samples | ~5 000 |
| Image resolution | 640 × 480 JPEG |
| Action dimensions | 7 (OpenVLA format) |
| Goal templates | 10 |
| Trajectory types | straight, left-turn, right-turn, wide-turn, reverse, U-turn |
| Scene | carter_warehouse_navigation.usd (NVIDIA Isaac 6.0) |
| Robot | Nova Carter (front Hawk stereo camera, left) |

## Action format

OpenVLA 7D action: `[linear_x, 0, 0, 0, 0, angular_z, 0]`

- Dim 0: linear_x (forward/backward velocity, m/s, range ~[−0.18, 0.33])
- Dims 1–4, 6: always 0 (differential drive, no lateral/vertical/gripper)
- Dim 5: angular_z (yaw rate, rad/s, range ~[−0.63, 0.63])

Action statistics (v2 dataset):
- mean: [0.137, 0, 0, 0, 0, −0.015, 0]
- std:  [0.119, 0, 0, 0, 0,  0.345, 0]
- q01:  [−0.177, …, −0.621, 0]
- q99:  [0.319,  …,  0.622, 0]

## Goal templates (10)

All trajectories start from safe spawn (0, -8, 0) facing +X after per-rep teleport reset.

| ID | Text goal | lx (m/s) | az (rad/s) | Motion type |
|---|---|---|---|---|
| 0 | Navigate to the east storage area | 0.30 | 0.00 | straight fast |
| 1 | Navigate carefully through the corridor | 0.12 | 0.00 | straight slow |
| 2 | Back up to clear the forklift path | −0.15 | 0.00 | reverse |
| 3 | Turn left to reach the northern aisle | 0.20 | +0.22 | gentle left curve |
| 4 | Navigate to the charging dock on the right | 0.20 | −0.22 | gentle right curve |
| 5 | Navigate to the left loading bay | 0.15 | +0.42 | tight left arc |
| 6 | Navigate to the right dispatch area | 0.15 | −0.42 | tight right arc |
| 7 | Turn to face the receiving station | 0.08 | +0.60 | sharp left spin |
| 8 | Turn right to the exit gate | 0.08 | −0.60 | sharp right spin |
| 9 | Move to the forklift pickup station | 0.25 | −0.15 | forward-right diagonal |

## File layout

```
~/VLA4AMR/datasets/isaac_synthetic/
  images/            — JPEG frames named {frame_id:05d}.jpg
  episodes.jsonl     — one JSON per line
  stats.json         — per-dim action statistics for tokenizer normalization
```

### episodes.jsonl format

```json
{
  "frame_id":  1234,
  "image":     "01234.jpg",
  "text_goal": "Navigate to shelf A on the left",
  "action_7d": [0.31, 0.0, 0.0, 0.0, 0.0, 0.24, 0.0],
  "traj_type": 0,
  "rep_idx":   7,
  "step":      3
}
```

### stats.json format

```json
{
  "mean": [0.20, 0.0, 0.0, 0.0, 0.0, 0.11, 0.0],
  "std":  [0.24, 0.0, 0.0, 0.0, 0.0, 0.40, 0.0],
  "min":  [...], "max": [...], "q01": [...], "q99": [...]
}
```

## Generation

Scripts (two-terminal architecture):
- **Terminal 1**: `bw03_isaac_collect.py` (isaac6 env, Python 3.12) — Isaac Sim + ROS2 bridge + reset server
- **Terminal 2**: `bw03_c4_collect_v2.py` (system Python 3.12 + ROS2 Jazzy) — camera subscriber + dataset writer

Key design choices:
- Robot **teleported to safe spawn (0, -8, 0)** before every rep via pause→set xformOp→resume
  (default spawn at (-6,-1) is boxed in: forklift 5m ahead, shelf wall 1m north)
- Safe zone confirmed by top-down map extracted from USD geometry (`bw03_map_gen.py`)
- **Reset after every rep** via `~/Desktop/reset_robot.flag` handshake (Terminal 1 watches, deletes flag on teleport, Terminal 2 waits for deletion)
- 10 trajectory types × 50 repetitions × 10 steps = 5 000 samples
- ±0.03 velocity noise per repetition for visual diversity
- Frames captured via ROS2 `/front_stereo_camera/left/image_raw` topic (fisheye, Nova Carter front Hawk)

## Train / Val / Test split

| Split | Fraction | Samples |
|---|---|---|
| Train | 90% | ~4 500 |
| Val | 5% | ~250 |
| Test | 5% (random_split seed=42) | ~250 |

Split is handled in `bw03_c4_train.py` / `bw03_c4_eval.py` at load time.

## Open questions

- Does 5K samples give enough diversity for FiLM vs cross-attn to separate? May need to scale.
- Are the scripted actions (with noise) sufficiently representative of real navigation commands?
- Images generated with `RayTracedLighting` renderer — confirm visual quality is reasonable before training.

## Sources

- [[IsaacSim]] — simulation platform (6.0.0.1, carter_warehouse_navigation.usd)
- [[NovaCarter]] — robot platform (front Hawk stereo camera)
- `bw03_c4_dataset.py` — generation script
