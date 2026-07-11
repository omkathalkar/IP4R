# Nav-AMR-WH Dataset (bw11_dataset)

**Type:** dataset
**Status:** active
**Last updated:** 2026-06-30
**Related:** [[C1-AdaCoT]], [[IsaacSim]], [[NovaCarter]], [[C6-EvaluationProtocol]], [[Isaac-Synthetic]]

## Summary

Nav-AMR-WH is the primary training dataset for VLA4AMR BW11+. It merges two Isaac Sim collection campaigns (bw10_dataset + bw10_dataset_turns) into a unified HuggingFace DatasetDict with AdaCoT-style chain-of-thought annotations. Inspired by VLingNav's Nav-AdaCoT-2.9M dataset but domain-specific to industrial warehouse AMR.

## Dataset properties

| Property | Value |
|---|---|
| Total episodes | 640 (400 straight + 240 turning) |
| Total frames | 50,971 |
| Train frames | 46,035 |
| Val frames | 4,936 |
| Train/val split | 90/10 stratified by task_type at episode level |
| Image resolution | 336×336 PNG (egocentric, eye-level 0.8m) |
| Action format | lin_vel (m/s), ang_vel (rad/s) — 2D direct control |
| CoT frames | 6,720 (13.2% of train) |
| CoT format | `<think>...</think><summary>...</summary>` (VLingNav AdaCoT style) |
| Scene | warehouse_multiple_shelves.usd (Isaac Sim 6.0.0.1) |
| Robot | Nova Carter (ego camera, no stereo) |
| Storage | `~/Desktop/bw11_dataset/hf_dataset/` (HuggingFace arrow) |

## Task types

| Task | Episodes | Description | ang_vel profile |
|------|----------|-------------|-----------------|
| aisle_fwd | 200 | Straight aisle traversal | 0.0 rad/s |
| aisle_fwd_slow | — | Reduced-speed straight traversal | 0.0 rad/s |
| cross_turn_left | 84 | Approach + 90° left turn + depart | +1.5708 rad/s during turn |
| cross_turn_right | 84 | Approach + 90° right turn + depart | −1.5708 rad/s during turn |
| obj_goal | 36+36 | Navigate toward a specific shelf landmark | Variable |
| obstacle_slalom | 36+36 | Swerve around parked forklift | ±0.6 rad/s |

## Phase structure

Each episode is split into phases recorded in `actions.csv`:
- `approach` — straight driving toward junction/goal
- `turn_left` / `turn_right` — 90° turn at constant reduced speed (0.5 m/s)
- `depart` — straight driving after turn
- `navigation` — general (obj_goal, slalom)

**Critical:** The `Phase:` field in the user prompt is load-bearing for the trained model. Without correct phase, the model defaults to straight driving even during turn frames.

## Normalization stats (train split)

| Velocity | q01 | q99 | mean | std |
|----------|-----|-----|------|-----|
| lin_vel | 0.400 | 1.000 | 0.870 | 0.220 |
| ang_vel | −1.571 | +1.571 | ~0.000 | 0.440 |

## CoT annotation

Annotated by Qwen2.5-VL-7B-Instruct running locally on RTX PRO 5000 Blackwell (48GB, bfloat16, eager attention).

- **Script:** `~/VLA4AMR/code/bw10_cot_annotate_local.py`
- **Target ratio:** 16% of frames (matches VLingNav 16.4%)
- **Actual ratio:** 13.2% after quality filtering
- **Filter pass rate:** bw10_dataset = 99.8%; bw10_dataset_turns = 73.1% (lower because turn frames require direction keywords)
- **CoT format:** `<think>reasoning</think><summary>scene description</summary>`
- `think_on=false` frames (clear straight path): only `<summary>` written, `<think>` omitted

## Files

```
~/Desktop/bw11_dataset/
  hf_dataset/
    train/          — 46,035 samples (HuggingFace arrow)
    val/            — 4,936 samples
  norm_stats.json   — q01/q99/mean/std for lin_vel, ang_vel
  dataset_info.json — episode counts, task distribution, CoT stats
```

Each sample schema:
```json
{
  "image_path": "/path/to/frame_000040.png",
  "instruction": "Drive forward through the aisle and make a right turn at the end",
  "lin_vel": 0.5,
  "ang_vel": 1.5708,
  "cot_text": "<think>...</think><summary>...</summary>",
  "has_cot": true,
  "phase": "turn_right",
  "task_type": "cross_turn_right",
  "episode_id": "ep_000000",
  "frame_idx": 40,
  "dataset_source": "bw10_dataset_turns"
}
```

## Generation scripts

| Script | Env | Purpose |
|--------|-----|---------|
| `bw10_collect_tasks.py` | isaac6 | Collect episodes in Isaac Sim (6 task types) |
| `bw10_cot_annotate_local.py` | openvla | Annotate CoT with local Qwen2.5-VL-7B |
| `bw11_convert.py` | openvla | Merge + HF convert + norm stats |

## Known issues

- bw10_dataset (400 eps): angular velocity bug caused first collection pass to use 1.5° instead of 90° turns. Fixed by adding `SIM_DT` factor to `_turn_steps()`. 400 episodes are effectively straight-driving only.
- bw10_dataset_turns (240 eps): bug fixed — correct 90° turns.
- 5 episodes in bw10_dataset had stale `--dry-run` CoT annotations which were cleaned up before conversion.

## Open questions

- obj_goal ang MAE = 0.0904 on val — needs more variable-angular training examples
- TartanGround pre-training (BW12) could improve generalisation to non-warehouse environments

## Sources

- [[IsaacSim]] — warehouse_multiple_shelves.usd, Nova Carter camera setup
- [[C1-AdaCoT]] — AdaCoT annotation format and CoT ratio rationale
- [[VLingNav]] — Nav-AdaCoT-2.9M annotation pipeline inspiration
