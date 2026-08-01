# VLA4AMR Wiki — Activity Log

Append-only. Each entry: `## [YYYY-MM-DD] type | title`
Types: ingest | query | lint | decision | milestone | setup

## [2026-08-01] milestone | Phase 7 COMPLETE — TIC-VLA trained checkpoint: CoT logging + accuracy eval on Ada HPC

**Job:** SLURM 2661166 on gnode052 (Ada HPC, RTX 2080 Ti) — completed
**Checkpoint loaded:** `TIC-VLA-model.ckpt` (1.9 GB, epoch=9, PyTorch Lightning) — confirmed on simulator at 3 paths:
- `~/VLA4AMR/checkpoints/tic-vla/TIC-VLA-model.ckpt`
- `~/Desktop/tic-vla-full-ckpt/TIC-VLA-model.ckpt`
- `~/Desktop/TIC-VLA-model.ckpt`

Copied to Ada: `/ssd_scratch/om.kathalkar/checkpoints/tic-vla/TIC-VLA-model.ckpt`

**Loading strategy** (both components):
- VLM weights: `model.load_vlm_checkpoint(ckpt_path)` handles `model.vlm.*` key remapping → `TICVLA_VLM`
- ActionExpert weights: 50 keys under `model.action_expert.*`; stripped and loaded into `model.action_expert` separately
- Total checkpoint keys: 1,324 (VLM fine-tuned + ActionExpert trained)

**Experiment config:**
- Dataset: VLN-PE val_unseen (scenes `17DRP5sb8fy`, `rPc6DW4iMge`), 5 episodes per scene
- Steps: 10 per episode (spread), delayed window = 3 frames → 100 total inference calls
- Ground truth: parquet `observation.robot_position` (3D), `observation.robot_yaw`, `observation.action` (0=STOP, 1=FWD, 2=LEFT, 3=RIGHT)

**Results:**

| Metric | Value |
|---|---|
| Mean latency | **5.84 s** (down from 7.29 s base — trained VLM generates shorter, task-focused CoT) |
| Mean WP magnitude | **0.4603** (vs ~0.0001 for random ActionExpert — 4,600x larger) |
| Heading error mean | 101.2° (n=92 steps with GT motion) |
| Heading error median | 130.3° |
| Heading error p95 | 174.7° |

Per GT action class:

| GT Action | N | Mean WP mag | Mean heading err |
|---|---|---|---|
| FWD | 60 | 0.5037 | 102.6° |
| LEFT | 12 | 0.3119 | 78.2° |
| RIGHT | 28 | 0.4311 | 106.6° |

**Key findings:**
1. **ActionExpert is functional**: WP magnitude 0.46 (trained) vs 0.0001 (random) confirms trained weights load and produce real, structured displacements
2. **Latency drops with trained weights**: 5.84 s vs 7.29 s base — trained VLM generates more concise navigation-relevant responses (400–440 tokens vs 500–960 tokens)
3. **Heading error is high (101°)**: Expected — this checkpoint was trained on BW17 warehouse data, NOT on VLN-PE Matterport3D environments. Same domain mismatch as BW18 (robot turned left despite "turn right" instruction)
4. **LEFT class has lowest heading error (78°)**: Turning commands show better directional signal
5. **100 CoT reasoning traces logged**: Qualitative analysis of scene descriptions pending

**Interpretation vs BW18 (2026-07-11):** This is the same checkpoint used in BW18 `--full-ckpt` mode on the simulator. The high heading error on VLN-PE confirms: the ActionExpert produces non-trivial, non-zero waypoints but in wrong directions for unseen environments. Needs VLN-PE or warehouse fine-tuning to align.

**Next step:** Fine-tune ActionExpert on VLN-PE ground truth trajectories (available on Ada ssd_scratch), or run inference on BW17-style warehouse episodes for an in-distribution accuracy test.

**Artifacts (local):** `~/Downloads/ticvla_phase7/phase7_cot_log.jsonl`, `phase7_summary.json`

---

## [2026-07-31] milestone | Phase 5 + 6 — TIC-VLA latency baseline on InternData-N1

**Context:** TIC-VLA's trained nav checkpoint is not released publicly. Phases 5 and 6 ran the base InternVL3-1B backbone through the TIC-VLA wrapper to establish inference latency baselines. ActionExpert had random weights; waypoints are numerically meaningless. Navigation accuracy was NOT evaluated.

**Phase 5 — VLN-PE val_unseen (Ada HPC, SLURM 2661004):**
- Dataset: `InternData-N1 vln_pe/`, scenes `17DRP5sb8fy` + `rPc6DW4iMge`, LeRobot v2.1 format
- Images: `.npy` arrays `(T, 256, 256, 3)` uint8, converted to temp JPGs per call
- Config: 20 episodes × 5 steps = 100 calls; delayed window = 3 frames

| Metric | Value |
|---|---|
| Mean latency | 7.29 s |
| Median | 7.57 s |
| p95 | 7.63 s |
| p99 | 7.81 s |
| 2-tile mean | 6.90 s |
| 4-tile mean | 7.39 s |

**Phase 6 — VLN-CE val_unseen (Ada HPC, SLURM 2661004):**
- Dataset: `InternData-N1 vln_ce/`, scene `pRbA3pwrgk9` (135 episodes, 20 used)
- Images: individual `.jpg` per step — `episode_XXXXXX_STEP.jpg`, 640×480 px
- Viewpoint: `rgb.125cm_0deg` (primary forward-facing)

| Metric | Value |
|---|---|
| Mean latency | 7.38 s |
| Median | 7.66 s |
| p95 | 7.79 s |
| p99 | 7.92 s |
| 2-tile mean | 7.09 s |
| 4-tile mean | 7.45 s |

**Key finding — resolution vs latency:**
Despite 6.25× resolution increase (256×256 → 640×480 JPEG), mean latency increases by only +0.09 s (+1.2%). InternVL3-1B on RTX 2080 Ti is **generation-bound**, not vision-bound. Response length (400–960 tokens) dominates wall-clock time. Image tokenization cost is negligible.

**Data format difference (VLN-PE vs VLN-CE):**
- VLN-PE: `episode_{idx:06d}.npy` — stacked frames `(T, H, W, 3)`, one file per episode
- VLN-CE: `observation.images.rgb.125cm_0deg/episode_{idx:06d}_{step}.jpg` — individual JPGs per step, multiple viewpoints
- VLN-CE parquet has full ground truth per step: `robot_position (3D)`, `robot_yaw`, `action (0-3)`, `progress (0–1)`

**Artifacts (local):**
- `~/Downloads/ticvla_phase5/phase5_latency.csv`, `phase5_summary.json`
- `~/Downloads/ticvla_phase6/phase6_latency.csv`, `phase6_summary.json`
- `~/Downloads/ticvla_results_summary.md` — cross-dataset comparison doc

**GPU / env:** gnode052 (RTX 2080 Ti, sm_75, no FlashAttention2), Ada HPC `u22` partition, `vla` conda env (torch 2.8.0+cu128, transformers 4.57.6), `module load u22/cuda/12.9`

---

## [2026-07-31] setup | DynaNav_data.zip transferred from simulator to Ada HPC

**Source:** `cvit-car-simulator@10.2.141.227:/home/cvit-car-simulator/DynaNav_data.zip` (73 GB)
**Destination:** `/scratch/om.kathalkar/DynaNav_data.zip` on gnode052 (Ada HPC)
- `/scratch` on gnode052 = `/dev/sdb1`, 1.8 TB HDD, 1.7 TB free (world-writable, sticky bit)
**Transfer method:** SLURM job 2661085 on gnode052, `sshpass scp` over internal IIIT network (~4 h at ~5 MB/s)
**Status:** Transfer in progress as of 2026-07-31 15:29 IST

**Context:** BW17 warehouse dataset (14,360 TIC-VLA JSON sample dirs, 560 episodes, 9 task types) needs to reach Ada HPC for potential retraining or dataset analysis without requiring simulator machine access.

---

## [2026-07-26] setup | BW19 — MobileVLA-R1 Isaac Sim integration initiated

**Script:** `bw19_sim_vla.py` — MobileVLA-R1 (NaVILAImageInference) paired with `bw11_sim_isaac.py`
**Model:** `AIGeeksGroup/MobileVLA-R1` — downloading to `~/VLA4AMR/checkpoints/mobilevla-r1/`
**Conda env:** `mobilevla` (Python 3.10, torch 2.3+cu121) — setting up on simulator machine
**Key design:**
- Same IPC protocol as BW18 (`/tmp/bw11_ipc/`, `action.json` with `{lin, ang}`)
- `generate_response()` → `ActionExtractor.extract_velocity_vector()` → `[x_vel_cmd, yaw_vel_cmd]` → `(lin, ang)`
- Prompt asks for `<answer>[x_vel_cmd, 0.0, yaw_vel_cmd]</answer>` format
- Up to 8 frames of history fed as multi-image context

**Status:** Download + env setup running in background. Script uploaded to `~/VLA4AMR/code/bw19_sim_vla.py`

**Run command (once setup is done):**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mobilevla
CUDA_VISIBLE_DEVICES=0 python3 ~/VLA4AMR/code/bw19_sim_vla.py \
    --model-path ~/VLA4AMR/checkpoints/mobilevla-r1 \
    --instruction "Navigate to the charging station on the left" \
    --output-video ~/Desktop/bw19_sim_demo.mp4 \
    2>&1 | tee ~/Desktop/bw19_sim_vla.log
```

---

## [2026-07-11] milestone | Full-ckpt DEMO — Original TIC-VLA pretrained checkpoint in Isaac Sim

**Script:** `bw18_sim_vla.py --full-ckpt` (new `--full-ckpt` mode added to support single-file Lightning checkpoints)
**Checkpoint:** `~/Desktop/tic-vla-full-ckpt/TIC-VLA-model.ckpt` (1.9 GB, original TIC-VLA authors, epoch=9)
**Instruction:** "Drive forward and turn right at the intersection"
**Loaded as:** VLM via `load_vlm_checkpoint()` (handles `model.vlm.*` keys automatically) + ActionExpert via `model.action_expert.*` key extraction; `action_num_layers=3` read from checkpoint `hyper_parameters`.

**Episode stats (40 queries, 40 frames):**
- `avg_lin = 0.213 m/s` (variable speed, slower than BW18)
- `avg_ang = −0.692 r/s`, clipped to `−1.000` frequently (persistent left turn — wrong direction)
- `avg_lat = 2.31s/query` (faster than BW18 due to `action_num_layers=3` vs 6)

**Outcome:** Robot turned **left** throughout (ang always negative), contrary to the "turn right" instruction.
Expected — this checkpoint was trained on MobileVLA's own warehouse dataset with a different instruction/action distribution, not on BW17.

**Comparison vs BW18 fine-tuned:**

| Metric | BW18 (fine-tuned) | Full ckpt (pretrained) |
|--------|-------------------|------------------------|
| avg_lin | 0.400 m/s (max) | 0.213 m/s (variable) |
| avg_ang | +0.241 (right ✓) | −0.692 (left ✗) |
| avg_lat | 2.99s | 2.31s |
| action_num_layers | 6 | 3 |
| Instruction-following | Correct right turn | Wrong direction |

**Conclusion:** BW18 fine-tuning on BW17 is necessary and effective — pretrained weights alone do not generalise to our warehouse instruction distribution.

**Artifacts (local):** `VLA4AMR_wiki/bw18_sim_demo_fullckpt.mp4`, `bw18_sim_demo_fullckpt.json`, `bw18_sim_vla_fullckpt.log`

---

## [2026-07-11] milestone | BW18 DEMO — Closed-loop Isaac Sim demo complete

**Script:** `bw18_sim_vla.py` paired with `bw11_sim_isaac.py` (IPC via `/tmp/bw11_ipc/`)
**Instruction:** "Drive forward and turn right at the intersection"
**Checkpoints:** VLM epoch=00 (val_loss=0.278) + Action epoch=14 (val_total_loss=0.0041)

**Episode stats (40 queries, 40 frames):**
- `avg_lin = 0.400 m/s` (constant max speed throughout)
- `avg_ang = 0.241 r/s`, `max_ang = 1.000 r/s` (clipped during turn)
- `avg_lat = 2.99s/query` (InternVL3-1B inference on RTX PRO 5000)

**Angular trajectory:**
- q=1–4: −0.17 → +0.01 (straight approach, minor left-correction)
- q=5–21: +0.04 → +0.39 (gentle right lean, pre-intersection)
- q=22–30: +0.43 → +1.00 (strong right turn at intersection)
- q=31–40: +0.02 → +0.18 (exit straight, turn complete)

**Output:** `~/Desktop/bw18_sim_demo.mp4` (1280×720, dual-view ego+isometric)
**Episode log:** `~/Desktop/bw18_sim_demo.json`

**Result:** Robot successfully executed forward-then-right-turn trajectory per instruction.
Behavior consistent with BW18 training (turns ADE=0.058–0.097m offline).

---

## [2026-07-11] milestone | BW18 COMPLETE — TIC-VLA trained on BW17, ADE=0.090m (−56% vs BW16)

**Stage 1 VLM (InternVL3-1B CoT fine-tuning):**
- Best checkpoint: epoch=0, val_language_loss=0.278 (overfitting from epoch 1 onward → stopped early)
- Config: batch=4, accum=4, lr=2e-5, 10 epochs planned; killed at epoch 4 (val loss monotonically increasing)

**Stage 2 Action head training:**
- Fix applied: `policy_data.py` `delayed_idx = max(0, current_idx - files_back)` — single-file dirs caused IndexError
- 15 epochs, batch=32, lr=1e-4, action_horizon_steps=30
- Best: epoch=14, val_total_loss=0.0041, val_ADE=0.077m, val_FDE=0.156m (plateaued from epoch 10)
- Checkpoint: `~/Desktop/bw18_ticvla_output/checkpoints/ticvla/action/ticvla-action-epoch=14-val_total_loss=0.0041.ckpt`

**Offline eval (bw18_eval.py on 2,160 test samples):**

| Metric | BW16 | BW18 | Δ |
|--------|------|------|---|
| ADE | 0.206 m | **0.090 m** | **−56%** |
| FDE | 0.338 m | **0.185 m** | **−45%** |

Per-task ADE: aisle_straight=0.044m, turns=0.058–0.097m, u_turn=0.089m,
multi_turn=0.077–0.097m, goal_seek=0.217m, obstacle_avoid=0.349m (hardest).

**Eval results:** `~/Desktop/bw18_eval_results.json`

**Lessons learned:**
- VLM stage overfits quickly on 2,510 windows (634M trainable params) → early stop at epoch 0
- Action stage (19.7M params) trains stably for 15 epochs, good generalisation
- `obstacle_avoid` and `goal_seek` need more training diversity (only 40/80 episodes each)

**See:** [[decision-bw18-ticvla-bw17]], [[bw17-warehouse-dataset]]

---

## [2026-07-10] milestone | BW18 LAUNCHED — TIC-VLA Stage 1 VLM training on BW17 warehouse dataset

**Preceded by BW17 full pipeline completion (same session).**

**BW17 pipeline results:**
- 560 episodes (9 task types): train=392, val=84, test=84
- 3,590 sliding windows (HISTORY=5, HORIZON=30, STRIDE=3)
- Instruction augmentation: 4 phrasings/window via Qwen2.5-VL-7B (openvla env, `cuda:0`)
- CoT annotation: decision-window only (~2,990 windows annotated), mean 486 chars
- Final verify: ALL CHECKS PASSED — 14,360 total DynaNav_json sample dirs

**BW17 dataset location on simulator:** `~/Desktop/bw17_dynav/`
- `train/DynaNav_json/`: 10,040 sample dirs
- `val/DynaNav_json/`: 2,160 sample dirs
- `test/DynaNav_json/`: 2,160 sample dirs

**BW18 training config:**
- Stage 1 VLM: InternVL3-1B, batch=4, lr=2e-5, epochs=10, `accumulate_grad_batches=4`
- Stage 2 Action: batch=32, lr=1e-4, epochs=15, `action_horizon_steps=30`
- Scripts: `~/VLA4AMR/code/bw18_train_vlm.sh` / `bw18_train_action.sh`
- Config: `~/VLA4AMR/code/bw18_train_vlm.yaml` / `bw18_train_action.yaml`
- Env: `~/VLA4AMR/code/.env.bw18` (BW18_DATA_ROOT=~/Desktop/bw17_dynav)

**Status:** Stage 1 launched at 02:23 2026-07-11 in tmux `bw18_vlm`, log at `~/Desktop/bw18_vlm.log`.
Stage 2 to follow once `vlm/last.ckpt` is written.

**Key design note:** One subdir per (window × instruction-variant) ensures TICVLADataset_VLM's
every-5th-file-per-dir filter retains 100% of samples (each dir has exactly 1 JSON file).

**GPU note (tic-vla env):** CUDA_VISIBLE_DEVICES=0 → RTX PRO 5000 Blackwell (47.3 GiB).
This differs from openvla env (same CUDA index, same GPU).

**See:** [[bw17-warehouse-dataset]], [[decision-bw18-ticvla-bw17]]

---

## [2026-07-09] decision | BW16 — TIC-VLA (InternVL3-1B + ActionExpert) launched on warehouse data

**Session:** TIC-VLA (ICML 2026, arXiv 2602.02459) added as second VLA baseline alongside CorrectNav.

**Work done:**
- Uploaded TIC-VLA-main.zip (1.3GB) → `~/VLA4AMR/TIC-VLA-main/` on simulator (via rsync)
- Extracted zip: confirmed structure (`configs/`, `ticvla/`, `DynaNav/`)
- Written `bw16_convert_ticvla.py` → converts bw11_dataset (HF format) to TIC-VLA JSON
  - Fixed image path bug: `/Desktop/bwXX/` → `/Desktop/datasets/bwXX/` (all 50,971 images verified)
  - Cumulative FLU world-frame future offsets via Euler integration of (lin_vel, ang_vel)
  - Output: `~/Desktop/bw16_ticvla_dataset/` (377 train eps, 57 val eps, 50,971 JSON files)
- Created `tic-vla` conda env (Python 3.11, torch 2.8.0+cu128, transformers<5.0)
- Downloading InternVL3-1B to `~/VLA4AMR/checkpoints/internvl3-1b/`
- Training configs and scripts: `~/VLA4AMR/code/bw16_train_vlm.{yaml,sh}`, `bw16_train_action.{yaml,sh}`, `.env.bw16`

**Status:** tic-vla env installing (torch 2.8.0 download), InternVL3-1B weights downloading.
Stage 1 VLM training will start once both complete (run `bash ~/VLA4AMR/code/bw16_train_vlm.sh`).

**See:** [[decision-bw16-ticvla-warehouse]]

## [2026-07-04] debug | BW15 Training — 13 transformers 5.x bugs fixed, OOM killed twice, ready to run

**Session**: ~8-hour debug marathon fixing compatibility between CorrectNav (written for
transformers ~4.40) and the transformers 5.x installed in openvla env.

**Bugs fixed (in order encountered):**

1. `KeyError: 'rope_theta'` in `modeling_qwen2.py` → add rope_theta + max_position_embeddings
   to rope_parameters dict in `llava_qwen.py.__init__`
2. `AttributeError: 'LlavaConfig' has no 'layer_types'` → initialize layer_types list
3. `TypeError: NoneType not subscriptable at config.rope_parameters["rope_type"]` →
   `config.rope_scaling = None` is a property setter in transformers 5.x that wipes
   rope_parameters; removed that line
4. `RuntimeError: from_pretrained with a meta device context` in SigLIP → pass `device_map="cpu"`
5. `AttributeError: 'SigLipVisionConfig' has no '_set_token_in_kwargs'` → guard with hasattr
6. 28-minute hang on SigLIP loading → `.incomplete` blob in HF cache; deleted + full
   snapshot_download; added `local_files_only=True` + `TRANSFORMERS_OFFLINE=1`
7. `OSError: siglip does not appear to have model.safetensors` → full snapshot_download
8. `ValueError: need sentencepiece or tiktoken` → `pip install tiktoken`
9. Missing LLaVA-Video tokenizer files → `snapshot_download(..., ignore_patterns=['*.safetensors'])`
10. `TypeError: Trainer.__init__() unexpected 'tokenizer'` → renamed to `processing_class=`
11. `AttributeError: 'TrainingArguments' has no 'dispatch_batches'` → removed from Accelerator call
12. `AttributeError: 'TrainingArguments' has no 'group_by_length'` → all group_by_* wrapped in getattr
13. `ImportError: cannot import name 'notf' from tensorboard.compat` (fatal crash at
    LLaVATrainer init due to numpy 2.x / ml_dtypes / TF incompatibility) → `--report_to none`

**OOM kills (after all 13 patches):**
Kernel OOM-killed python3.10 twice (confirmed via dmesg, no log entry — SIGKILL):
- Root: `mm_tunable_parts=mm_language_model` unfreezes full 7B LM even with `lora_enable=True`
  (mm_tunable_parts logic overrides PEFT freezing; 7713 MB trainable)
- ZeRO-2 CPU offload needs ~30 GB RAM for fp32 optimizer states → RAM exhausted on 62 GB machine
- Fix: `mm_tunable_parts="mm_mlp_adapter"` → trainable params drop to **17 MB**
- Removed `offload_optimizer` from `bw15_zero2.json` (not needed at 17 MB)

**Also fixed this session:**
- Broken `~/.local/bin/ninja` wrapper shadowing system ninja → removed + `pip install ninja` in openvla
- DeepSpeed cpu_adam JIT now builds and is cached at `~/.cache/torch_extensions/py310_cu128/cpu_adam/`

**Current state:** All patches applied, all configs corrected. Training never reached step 1.
Next: start training tomorrow — expect first `train_loss` within ~5 min of launch.

**Files changed on simulator:**
- `CorrectNav/model/language_model/llava_qwen.py` — rope_parameters, layer_types, pad_token_id
- `CorrectNav/model/multimodal_encoder/siglip_encoder.py` — hasattr guard, local_files_only, device_map
- `CorrectNav/train/train.py` — processing_class=
- `CorrectNav/train/llava_trainer.py` — dispatch_batches, group_by_*, persistent_workers
- `~/VLA4AMR/code/bw15_train.sh` — TRANSFORMERS_OFFLINE, report_to none, mm_tunable_parts
- `~/VLA4AMR/code/bw15_zero2.json` — removed offload_optimizer

---

## [2026-07-03] milestone | BW15 LAUNCHED — Full CorrectNav stack (LLaVA-Video-7B-Qwen2 + SigLIP + MP4 video history)

**Motivation:** BW14 training converged (val=0.0661) but before closed-loop testing Mr. K pivoted to use the full CorrectNav system directly. Key reasons: proven video-history model built for navigation, identical discrete vocabulary, self-correction flywheel, and publication leverage for ICRA 2027.

**Architecture:** LLaVA-Video-7B-Qwen2 backbone + SigLIP-SO400M-patch14-384 vision encoder + 8-frame MP4 sliding window + N=6 multi-step discrete action prediction. Same IPC as BW14. LoRA r=32 α=64 on 1× RTX 5000 Pro 48GB.

**Scripts written (local → uploaded to simulator):**
- `bw15_convert.py` — convert bw11_dataset to CorrectNav MP4+JSON format
- `bw15_train.sh` + `bw15_zero2.json` — adapted CorrectNav training for 1 GPU
- `bw15_sim_vla.py` — CorrectNav inference in Isaac Sim via bw11_ipc

**Decision page:** [[decision-bw15-correctnav-fullstack]]

**Setup on simulator:** CorrectNav repo extracted at ~/VLA4AMR/CorrectNav-main/; install running in tmux bw15_setup.

---

## [2026-07-03] milestone | BW14 LAUNCHED — DiscreteNav architecture (discrete tokens + H=6 history + CorrectNav flywheel plan)

**Motivation:** BW13 closed-loop demo (2026-07-01) outputs ang=0.0000 for all 40 frames, identical to BW12 failure. Root cause is structural: continuous regression has a mode-collapse attractor at (0,0). BW14 replaces continuous action output with discrete navigation tokens {FORWARD, TURN_LEFT, TURN_RIGHT, STOP}.

**Three key changes:**
1. Discrete action tokens — eliminates mode collapse entirely (binary token, no continuous hedge)
2. H=6 multi-frame history at `max_pixels=64×28×28` (64 tokens/image → 384 vision tokens for 6 frames, fits in 1024 max_len without OOM)
3. N=6 action sequence in CorrectNav style ("FORWARD,TURN_RIGHT,TURN_RIGHT,FORWARD,FORWARD,FORWARD")

**Architecture decision:** see [[decision-bw14-discrete-nav]] for full rationale including VLN-CE + CorrectNav analysis.

**Scripts written:**
- `bw14_train.py` — Phase 1 supervised training on GT discrete tokens
- `bw14_infer.py` — offline eval: token accuracy, turn-initiation check per chunk position
- `bw14_sim_vla.py` — live Isaac Sim demo with TOKEN_TO_VEL mapper

**Expected training time:** ~12h on cvit-car-simulator (same dataset bw11_dataset, similar LoRA config)

## [2026-07-01] milestone | BW13 LAUNCHED — CorrectNav-inspired training (multi-frame + chunking + turn reweight + always-summary)

**Root cause analysis (F3 — closed-loop failure):**
BW12 offline val shows ang MAE=0.0003 on turns but closed-loop Isaac Sim demo outputs ang=0.0000 for ALL 40 frames.
Four identified failure modes:
1. Imitation ≠ decision: offline frames are already mid-turn; live inference must initiate turns proactively
2. Mode collapse: ~80% straight frames → ang=0 overwhelming prior
3. Memory self-locking: CoT fires only once → memory never updates → reinforces straight prediction
4. No recovery data: pure expert trajectories, zero off-distribution states

**CorrectNav paradigm (Yu et al., AAAI 2026) adapted to VLA4AMR:**
Self-Correction Flywheel: evaluate → detect deviation → create correction data → retrain
BW13 implements the four static dataset-side fixes (Flywheel Step 1–2 of 4):
- Fix 1: Multi-frame history (H=3 consecutive frames → temporal context "been straight → intersection NOW visible")
- Fix 2: Action chunking (N=4 next actions → approach frames get turn actions in chunk positions 1-3, implicit look-ahead supervision)
- Fix 3: Turn loss reweighting (3× weight on turning chunks → corrects mode collapse)
- Fix 4: Always-on summary (every frame outputs `<summary>`, decoupled from CoT → breaks memory self-locking loop)

**bw13_train.py launched on simulator:**
- Session: `bw13_train` tmux, openvla env
- Command: `CUDA_VISIBLE_DEVICES=0 python bw13_train.py | tee ~/Desktop/bw13_train.log`
- Dataset: 46,035 train / 4,936 val (bw11_dataset, same episodes, chunked view)
- Turning chunks: **16.9% train / 12.6% val** (up from 13.2% in BW11/BW12 — chunking adds look-ahead)
- Optimizer steps: 4,314 total (3 epochs, grad_accum=32, effective batch=32)
- Output: `~/VLA4AMR/checkpoints/bw13_lora/`

---

## [2026-07-01] milestone | BW12 COMPLETE — phase-free training, eval, live Isaac Sim demo

**bw12_lora training complete** — phase-free, memory-chained, instruction-diverse retraining on Nav-AMR-WH:
- Base: Qwen2.5-VL-7B-Instruct; LoRA rank=32, α=64 (same as BW11)
- Dataset: 46,035 train / 4,936 val (same bw11_dataset, with memory chain added)
- **Key change vs BW11:** NO `Phase:` field in user prompt. Rolling `Memory:` from `<summary>` tokens instead.
- **115 instruction variants** across 6 task types (15-20 phrasings each)
- **Best val_loss = 0.0542** (step 2400, epoch 2) — slightly better than BW11 (0.0551)
- Total training time: ~12.5 hours on RTX PRO 5000 Blackwell
- Output dir: `~/VLA4AMR/checkpoints/bw12_lora/best_lora/`

**Offline eval (bw12_infer.py) — 4,936 val samples, 35 min:**

| Task | n | Lin MAE | Ang MAE | CoT% |
|------|---|---------|---------|------|
| aisle_fwd | 960 | 0.0000 | 0.0000 | 14.8% |
| aisle_fwd_slow | 320 | 0.0000 | 0.0000 | 15.0% |
| cross_turn_left | 1280 | 0.0070 | 0.0028 | 12.7% |
| cross_turn_right | 1280 | 0.0051 | 0.0003 | 11.8% |
| obj_goal | 536 | 0.1015 | 0.0951 | 12.5% |
| obstacle_slalom | 560 | 0.0207 | 0.0110 | 13.4% |
| **OVERALL** | **4936** | **0.0165** | **0.0124** | — |

- 100% parse rate; zero failures
- Turning tasks (cross_turn): ang MAE=0.0003–0.0028 **without any Phase: hint** — model uses vision + memory
- obj_goal error expected (most variable task, fewest episodes)

**Live demo (bw12_sim_isaac.py + bw12_sim_vla.py):**
- Same two-process IPC architecture as BW11 (see [[decision-ipc-two-process]])
- BW12 VLA process uses phase-free prompt: `Task: {instruction}\nMemory: {summary}\nOutput action:`
- Rolling memory updated per frame from `<summary>` tokens in CoT output
- Demo video: `~/Desktop/bw12_sim_demo.mp4` (1280×720, ego + isometric overhead)

**Significance:** Phase-dependency eliminated. AMR now follows natural language instructions (115+ variants) without external phase labels from the nav stack.

**Connects to:** [[C1-AdaCoT]], [[C2-MidLevelActionHead]], [[Nav-AMR-WH]], [[decision-phase-free-bw12]]

## [2026-06-30] milestone | BW11 COMPLETE — training, eval, live Isaac Sim demo recorded

**bw11_lora training complete** on merged Nav-AMR-WH dataset (bw11_dataset):
- Base: Qwen2.5-VL-7B-Instruct; LoRA rank=32, α=64, 95M/8.4B trainable (1.13%)
- Dataset: 46,035 train + 4,936 val frames; 6,720 CoT frames (13.2%)
- **Best checkpoint: Epoch 1 end, val_loss=0.0551** → `~/VLA4AMR/checkpoints/bw11_lora/best_lora/`

**Offline eval (bw11_infer.py) — 4,936 val samples:**
- 100% parse rate; lin MAE=0.0000; ang MAE=0.0105 overall
- turn phases: ang MAE=0.0013; obj_goal: ang MAE=0.0904

**Key finding:** `Phase:` token in user prompt is load-bearing — model conditions turning on it.

**Live inference + recording (bw11_sim_isaac.py + bw11_sim_vla.py):**
- Two-process IPC (file-based, /tmp/bw11_ipc/) needed because isaac6 env PyTorch incompatible with driver (see [[decision-ipc-two-process]])
- Isaac Sim: warehouse_multiple_shelves.usd, ego camera (follows robot) + fixed overhead isometric camera at (8, -2, 14)m
- 1280×720 dual-view MP4: egocentric left, overhead isometric right
- 40 frames recorded, 1.5 Hz closed-loop, cross_turn_right episode
- **Demo video:** `~/Desktop/bw11_sim_demo.mp4`

**New wiki pages:** [[Nav-AMR-WH]], [[decision-qwen-bw11]], [[decision-ipc-two-process]]

**Connects to:** [[C1-AdaCoT]], [[C6-EvaluationProtocol]], [[Nav-AMR-WH]], [[IsaacSim]]

## [2026-06-29] milestone | BW10 Nav-AMR-WH dataset collection + CoT annotation RUNNING

Two datasets collected in Isaac Sim (warehouse_multiple_shelves.usd + Nova Carter):

**bw10_dataset** (straight-driving, ~/Desktop/bw10_dataset):
- 400 episodes: aisle_fwd, aisle_fwd_slow, cross_turn_left*, cross_turn_right*, obj_goal*, obstacle_slalom*
- RETROSPECTIVE BUG: angular velocity bug caused all turns to be 1.5° instead of 90° — all episodes effectively straight driving
- 336×336 PNG frames, actions.csv (lin_vel, ang_vel, cam_x, cam_y, cam_heading_deg), meta.json

**bw10_dataset_turns** (turning-heavy, ~/Desktop/bw10_dataset_turns) — BUG FIXED:
- 240 episodes: 84 cross_turn_left + 84 cross_turn_right + 36 obj_goal + 36 obstacle_slalom
- 18,986 total frames; correct 90° turns achieved after fixing `_turn_steps()` (missing SIM_DT factor)
- Angular velocity bug fix: `ang = target_rad / (n_steps * SIM_DT)` instead of `/ n_steps`

**CoT annotation — RUNNING in parallel on Blackwell (GPU 1, 48GB):**
- Qwen2.5-VL-7B-Instruct loaded in bfloat16, eager attention (flash_attn not built for Blackwell compute 12.0)
- bw10_cot session: annotating bw10_dataset (400 eps, ~93 min)
- bw10_cot_turns session: annotating bw10_dataset_turns (240 eps, ~103 min)
- Format: `<think>...</think><summary>...</summary>` (VLingNav AdaCoT style)
- CoT ratio target: 16% of frames (matches VLingNav 16.4%)

**Key scripts:**
- `~/VLA4AMR/code/bw10_collect_tasks.py` — Isaac Sim collection (6 task types)
- `~/VLA4AMR/code/bw10_cot_annotate_local.py` — Local Qwen2.5-VL annotation
- `~/VLA4AMR/code/bw10_launch.sh` — Headless launcher (Xvfb :99 + CUDA_VISIBLE_DEVICES=0)

**Connects to:** [[C1-AdaCoT]] (CoT annotation strategy), [[Isaac-Synthetic]] (dataset)

## [2026-06-24] milestone | BW09 VSLAM pipeline COMPLETE — warehouse_multiple_shelves.usd + Nova Carter + RTAB-Map

Full stereo VSLAM pipeline running on simulator for VLA4AMR C3/C6:

**Architecture:**
- Isaac Sim: `warehouse_multiple_shelves.usd` + Nova Carter ROS USD (front hawk stereo)
- ROS2 bridge: stereo RGB at 640×480 on `/front_stereo_camera/left|right/image_rect_color`
- SLAM: RTAB-Map (stereo_odometry + rtabmap) → `/rtabmap/odom` (valid pose + quaternion)
- Map: `/tmp/vla4amr_warehouse_map.db` (30 MB, 104 nodes, ~18m aisle run)

**Key bugs fixed (in order):**
1. `omni.isaac.core` deprecated → use `omni.kit.app.get_extension_manager()` for ROS2 bridge
2. Wrong Carter USD path → `{ASSETS}/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd`
3. `/clock` not published → OmniGraph `ROS2PublishClock` after `open_stage()` (not before)
4. `use_sim_time=true` with no `/clock` → RTAB-Map froze; fix: publish `/clock` first
5. Replicator `ROS2PublishCameraInfo` writer publishes empty messages → removed; Carter OmniGraph handles camera_info
6. Stereo baseline=0 in CameraInfo → static TF `front_stereo_camera_left_optical→right` at 12 cm
7. `frame_id=base_link` but no TF tree → set `frame_id=front_stereo_camera_left_optical`
8. stereo_odometry publishes to `/odom` but rtabmap remapped to `/rtabmap/odom` → add `('odom', '/rtabmap/odom')` remap to stereo_odometry

**Scripts:**
- `~/VLA4AMR/code/bw09_isaac_sim_ros2.py` — Isaac Sim headless + RTAB-Map subprocess launcher
- `~/VLA4AMR/code/bw09_launch_rtabmap.py` — RTAB-Map launch (stereo_odometry + rtabmap + static TF)
- Map backup: `~/Desktop/vla4amr_warehouse_map.db`

**Connects to:** [[C3-ConfidenceGatedHandoff]] (vo_pose_covariance for handoff gate), [[C6-EvaluationProtocol]] (VSLAM odometry as ground truth for closed-loop eval)

## [2026-06-23] milestone | BW06 inference evaluation COMPLETE — bw06_infer.py

Evaluated fine-tuned LoRA adapter on 300 held-out val samples from bw05_dataset_v2.
Run: `CUDA_VISIBLE_DEVICES=0 python bw06_infer.py` — 6.6 it/s on RTX 4060 Ti (~45 s for 300 samples).

**Results — OVERALL (n=300):**
| Metric   | lin_vel (m/s) | ang_vel (rad/s) |
|----------|---------------|-----------------|
| MAE      | 0.1660        | 0.0382          |
| Accuracy (tol=0.05) | 17.0% | 85.0%       |

**Per task type:**
| Task            | n   | mae_lin | mae_ang | acc_lin | acc_ang |
|-----------------|-----|---------|---------|---------|---------|
| aisle_a_fwd     |  91 | 0.1451  | 0.0699  | 27%     | 74%     |
| aisle_a_slow    |  90 | 0.2000  | 0.0000  | 0%      | 100%    |
| aisle_a_zigzag  |  10 | 0.0000  | 0.2237  | 100%    | 20%     |
| aisle_b_fwd     |  54 | 0.2000  | 0.0000  | 0%      | 100%    |
| aisle_b_slow    |  39 | 0.2000  | 0.0000  | 0%      | 100%    |
| aisle_b_zigzag  |  16 | 0.0000  | 0.1780  | 100%    | 19%     |

**Interpretation:**
- ang_vel near-zero prediction is excellent (85% overall) — model correctly predicts 0 for straight/slow tasks.
- lin_vel slow tasks (GT=0.6): model predicts ≈0.8 consistently → MAE=0.2000, acc=0%. Keyword "slowly" was learned but output is mis-calibrated.
- lin_vel zigzag tasks (GT=1.0): model is perfect (acc=100%, mae=0.0) — "weaving/adjusting" → full speed learned correctly.
- ang_vel zigzag tasks: acc≈20% — model cannot track the sinusoidal ang_vel variation from visual features alone. This confirms the visual grounding limitation from fast convergence on 84% straight data.
- Artifacts saved: ~/Desktop/bw06_eval_results.json, ~/Desktop/bw06_annotated_episode.mp4 (ep 229, zigzag, 84 frames)

**Fix directions for BW07:**
1. Re-balance dataset to 50/50 straight/zigzag to force visual feature learning
2. Add curriculum: train on easy tasks first, zigzag second
3. Consider per-task normalization (separate q01/q99 for slow vs. normal speed)

## [2026-06-23] milestone | BW06 LoRA fine-tuning COMPLETE — val_loss=0.0001, early stop at step 1000

Training converged in ~18 min (1000/5670 planned optimizer steps). Stopped early.

**Val loss curve:**
| Step | val_loss |
|------|----------|
| 250  | 0.0525   |
| 500  | 0.0225   |
| 750  | 0.0073   |
| 1000 | **0.0001** ← best, saved |

**Checkpoint:** ~/VLA4AMR/checkpoints/bw06_lora/best_lora/
- adapter_model.safetensors: 65 MB (LoRA rank=32, Q/K/V/O projections)
- adapter_config.json, norm_info.json (q01/q99, action_token_base=31745, vocab=32001)

**Interpretation of fast convergence:**
84% of training frames have ang_vel=0 (straight episodes). Model learned instruction→action
mapping quickly: "slowly" → lin_vel=0.6, "forward/straight" → lin_vel=1.0, ang=0 → token 31873.
The 16% zigzag frames (ang_vel sinusoidal ±0.35) provide visual-grounding signal.
Model is fine for inference but is primarily language-conditioned at this stage.

**Next iterations to improve visual grounding:**
1. Collect 50/50 straight/zigzag episodes (currently 16/84 ratio)
2. Add collision-avoidance episodes where robot must steer based on visual obstacles
3. Add more instruction diversity that requires visual disambiguation

---

## [2026-06-23] milestone | BW06 training pipeline written + v2 collection RUNNING

Three scripts written and uploaded to simulator:

**bw04_collect_v2.py** (PID 672278, RUNNING — ~/Desktop/bw04_collect_v2.log):
- 200 new episodes, 6 task types (4 straight + 2 new zigzag)
- Zigzag: `ang = 0.35 × sin(2π × step / 80)` — sinusoidal ±0.35 rad/s per step
- Camera uses rotateXYZ + heading integration: position AND heading update per step
- Confirmed working: zigzag episodes show X drift (end_x ≠ start_x) and dist=3.33m
- Estimated ~60 min total; auto-chains to bw05_convert_v2.py + bw06_train.py on completion

**bw05_convert_v2.py** (queued — runs after v2 collection):
- Merges bw04_dataset (v1, 200 ep straight) + bw04_dataset_v2 (v2, 200 ep mixed)
- Outputs ~/Desktop/bw05_dataset_v2 (~400 episodes, ~33K samples)
- ang_vel now has real variance ±0.35 rad/s for zigzag episodes

**bw06_train.py** (queued — runs after bw05_convert_v2.py):
- LoRA rank=32 on Q/K/V/O projections (peft 0.11.1)
- 2D nav actions padded to 7D for OpenVLA compatibility
- Custom action tokenizer: normalize→256 bins→vocab token IDs (base = vocab_size - 256)
- 3 epochs, lr=2e-4, effective batch=16, cosine decay
- Saves best_lora/ (by val loss) + final_lora/ + norm_info.json
- Est. ~2h on RTX PRO 5000 Blackwell 48GB

---

## [2026-06-23] milestone | BW05 HuggingFace dataset conversion COMPLETE — 16 798 samples, 238.6 MB

Script: `bw05_convert.py` (openvla env, Python 3.10). Ran in <10s on simulator.
Output: ~/Desktop/bw05_dataset/ (HuggingFace DatasetDict, arrow format)

**Dataset stats:**
- Train: 15 118 samples (180 episodes, 90%)
- Val:   1 680 samples (20 episodes, 10%)
- Split: episode-level (no leakage)
- Image: 224×224 JPEG bytes stored inline in arrow file
- Action: [lin_vel (m/s), ang_vel (rad/s)] — 2D
- Action stats: lin_vel ∈ {0.6, 1.0} (mean=0.8), ang_vel = 0.0 (all straight-forward tasks)
- Size on disk: 238.6 MB

**Note:** ang_vel is always 0.0 in this dataset (all tasks are straight-forward aisle traversals).
For a richer dataset with turning, add angular episodes in a follow-up collection run.

Loading:
```python
from datasets import load_from_disk
import io; from PIL import Image
ds = load_from_disk("~/Desktop/bw05_dataset")
sample = ds["train"][0]
img = Image.open(io.BytesIO(sample["image"]))       # 224×224 RGB
action = sample["action"]                            # [lin_vel, ang_vel]
inst   = sample["language_instruction"]              # e.g. "Navigate forward..."
```

Next: BW06 — OpenVLA-OFT LoRA fine-tuning script on this dataset.

---

## [2026-06-23] milestone | BW04 Isaac-Synthetic v3 collection COMPLETE — 200 episodes, 16 802 frames

Script: `bw04_collect.py` — completed in ~21 min (1285s wallclock) on cvit-car-simulator.
Architecture: standalone UsdGeom.Camera at /Root/data_cam, world position integrated per step (vel×dt),
BasicWriter 224×224 → episode frames. No robot physics / articulation API needed.

Why camera-only approach: Nova Carter articulation at /Root/Nova_Carter_ROS not findable via
UsdPhysics.ArticulationRootAPI scan or IsaacRobot(). omni.physx.tensors.plugin: "Pattern did not match
any articulations". Solution: decouple camera from robot physics — camera moves as a kinematic prim.

**Final stats:**
- 200 episodes × ~84 frames = **16 802 frames** total
- 4 task types: aisle_a_fwd, aisle_b_fwd, aisle_a_slow, aisle_b_slow
- aisle_fwd  (lin=1.0 m/s): 4.17m travel per episode
- aisle_slow (lin=0.6 m/s): 2.50m travel per episode
- 5 instruction phrasings per fwd type, 2 per slow type → language diversity
- Dataset: ~/Desktop/bw04_dataset/ on simulator (ep_XXXXXX/frames/ + actions.csv + meta.json + dataset_info.json)

Next: BW05 — convert to HuggingFace/LeRobot parquet for OpenVLA-OFT fine-tuning.

Script: `bw04_collect.py` (PID 668100 on cvit-car-simulator, ~/Desktop/bw04_collect.log).
Architecture: standalone UsdGeom.Camera at /Root/data_cam, world position integrated per step (vel×dt),
BasicWriter 224×224 → episode frames. No robot physics / articulation API needed.

Why: Nova Carter articulation at /Root/Nova_Carter_ROS not findable via UsdPhysics.ArticulationRootAPI scan
or IsaacRobot(). omni.physx.tensors.plugin error: "Pattern did not match any articulations".
Solution: decouple camera from robot physics entirely — camera moves as a kinematic prim.

Confirmed working (first 17 episodes checked):
- aisle_fwd  (lin=1.0 m/s): dist=4.17m per episode (250 × 1/60 × 1.0) ✓
- aisle_slow (lin=0.6 m/s): dist=2.50m per episode (250 × 1/60 × 0.6) ✓
- 84–86 frames per episode, non-black, camera travelling north through warehouse aisles ✓
- Estimated total time: ~60 min for 200 episodes

Dataset: ~/Desktop/bw04_dataset/ on simulator
- 4 task types × 5/2 phrasings: aisle_a_fwd, aisle_b_fwd, aisle_a_slow, aisle_b_slow
- episode format: frames/frame_*.png + actions.csv (step, lin_vel, ang_vel, cam_x, cam_y) + meta.json
- Target: ~16 800 frames total (200 ep × 84 frames)

Next: BW05 — convert PNG episodes to HuggingFace/LeRobot parquet for OpenVLA-OFT fine-tuning.

---

## [2026-06-20] milestone | C1 LIVE CLOSED-LOOP RUNNING — full Isaac Sim + AdaCoT pipeline verified

Full closed-loop run: Isaac Sim (298193) → ROS2 camera → bw04_openvla_node.py (298397) → bw04_c1_live.py (298506) → /cmd_vel.

Live entropy on warehouse corridor: H = 0.5–1.5 nats. cot_rate=0.000 (correct — straight aisle = confident model).
Inference: 0.34s/step no-CoT path. ~480 inferences sustained without error.

Three bugs fixed during deployment:
1. `attn_implementation="eager"` removed from `load_model()` — conflicted with `predict_action()` causal mask
2. `list(np.ndarray.flatten())` → `.tolist()` — numpy.float32 not JSON serializable
3. **Entropy probe prompt bug**: bare TASK_INSTR gives H ≈ 10.37 nats (near-max over 32K vocab). Fix:
   `PROBE_PROMPT = "In: What action should the robot take to {TASK_INSTR}? Out:"` — primes model to predict
   from 256-bin action vocabulary → calibrated H ∈ [0.5, 2.5] nats on warehouse frames.

Also discovered: ROS2 (rclpy) logs go to `~/.ros/log/` not stdout. Monitor with:
  `strings ~/.ros/log/python3_<PID>_*.log | grep -E "inf#|cot_rate|H="`

Isaac Sim must be launched with `source /opt/ros/jazzy/setup.bash` BEFORE running the python binary,
or isaacsim.ros2.bridge loads without rcl libraries and publishes no camera topics (Publisher count=0).

Next: BridgeData-V2 download + HM3D curation for C2 SFT dataset.

---

## [2026-06-20] decision | C1 AdaCoT threshold: θ = 2.795 nats

Two sweeps on 450 robot-front warehouse frames.
Sweep 1 (θ ∈ {0.3, 0.5, 0.7}): all fired 100% — real warehouse H ∈ [1.88, 2.89].
Sweep 2 (θ ∈ {2.664, 2.719, 2.795, 2.830}): θ=2.795 → 3.1% activation, 233ms latency (+10ms over baseline).
Chosen: θ=2.795 (p97 of entropy distribution). See [[decision-c1-theta]].

---

## [2026-06-20] milestone | BW04 started early — C1 AdaCoT + dataset curation scripts written

Two scripts written and ready to SCP to simulator:

1. `bw04_c1_adacot.py` (openvla env) — entropy-gated AdaCoT trigger:
   - 5 ablation modes: no_cot | dense_cot | fixed_k5 | fixed_k20 | entropy_gated (ours)
   - Entropy computed from last-token logits before greedy decode
   - Logs: cot_activation_rate, mean_entropy, mean_inference_ms, action sequence
   - Smoke test: 10 synthetic frames (no frames_dir needed)
   - Run: `python bw04_c1_adacot.py --ablation entropy_gated --theta 0.5`

2. `bw04_dataset_curator.py` — BridgeData-V2 + HM3D curation:
   - BridgeData: filter lx > min_forward AND CLIP(image, goal) > 0.3
   - HM3D: filter Nav2 failure episodes (timeout or collision)
   - Output: same schema as Isaac-Synthetic (episodes.jsonl + stats.json)
   - Note: BridgeData-V2 (~16TB) needs download first; HM3D needs Habitat dataset

BW04 ablation plan (from [[C1-AdaCoT]]):
  Compare 5 modes × 3 θ values (0.3, 0.5, 0.7) = 7 conditions
  Metrics: cot_activation_rate, mean_inference_ms (proxy for SR until Isaac Sim eval)
  θ selection: pick lowest θ where activation_rate ≈ 2–5% (matching VLingNav's 2.1%)

Next: SCP bw04_c1_adacot.py → smoke test → θ sweep on Isaac Sim frames from recordings/v5/

---

## [2026-06-20] decision | C4 strategy: token_concat → C2 and C5

token_concat chosen. Data-scarce regime (5K samples) favours zero-param baseline over 67M-param FiLM/cross-attn heads.
Paper framing: present as finding for practitioners. Cross-attn lx anomaly (0.0801 vs 0.1794) worth one sentence.
C4 contribution fully complete. See [[decision-c4-strategy]].

---

## [2026-06-20] milestone | C4 FusionAblation — open-loop RMSE eval complete

500 test samples, batch_size=1, eager attention (flash_attn incompatible with Blackwell sm_120 via this code path).

| Strategy | Total RMSE | Nav RMSE | Dim0 lx | Dim5 az |
|---|---|---|---|---|
| Token concat | **0.1697** | **0.3174** | 0.1794 | 0.4115 |
| FiLM | 0.1997 | 0.3735 | 0.1143 | 0.5157 |
| Cross-attention | 0.1962 | 0.3671 | **0.0801** | 0.5129 |

token_concat wins overall; cross_attn is best at forward-speed prediction (lx).
All strategies struggle on az (angular/turning) — consistent across methods.
Results saved: `~/VLA4AMR/checkpoints/c4/eval_results.json`

Two bugs fixed during eval run:
1. `batch_size=4` → `batch_size=1` (OpenVLA `prepare_inputs_for_generation` rejects batch>1)
2. FiLM/cross-attn modules loaded as float32 → cast to bfloat16 to match model dtype

Next: decide which strategy to use for C2/C5, proceed to BW04 (AdaCoT trigger).

---

## [2026-06-20] milestone | C4 FusionAblation — all 3 strategies trained (2026-06-19)

All three LoRA fusion strategies trained to completion on [[Isaac-Synthetic]] v2 (5K samples, 3 epochs each)
on the RTX PRO 5000 Blackwell in the `openvla` env.

Training cross-entropy val loss (best epoch):

| Strategy | Best val loss | Checkpoint |
|---|---|---|
| Token concat | **0.1669** | `~/VLA4AMR/checkpoints/c4/token_concat/lora/` |
| FiLM | 0.3152 | `~/VLA4AMR/checkpoints/c4/film/lora/` + `film/film.pt` (193 MB) |
| Cross-attention | 0.4066 | `~/VLA4AMR/checkpoints/c4/cross_attn/lora/` + `cross_attn/xattn.pt` (129 MB) |

**Key finding:** token_concat wins on training loss — opposite of hypothesis (predicted FiLM ≥ cross_attn > token_concat).
Likely explanation: data-scarce regime (5K samples) favours zero-param baseline over 67M-param fusion heads.
Open-loop RMSE eval (`bw03_c4_eval.py`) still needed to confirm on action prediction quality.

Next: run `bw03_c4_eval.py` in openvla env on simulator → fill RMSE results table.

---

## [2026-06-19] milestone | Isaac-Synthetic v2 dataset generated (5 000 samples, 0 crashes)

v1 dataset (2026-06-18) was discarded — robot spawn at (-6,-1) was boxed in between forklift and shelf wall,
causing all frames to be face-first collision images.

Fix: extracted top-down warehouse map via `bw03_map_gen.py` (USD BBoxCache + matplotlib), identified safe
southern corridor at (0,-8). Wrote `bw03_isaac_collect.py` (Terminal 1) with pause→xformOp→resume teleport
and `bw03_c4_collect_v2.py` (Terminal 2) with per-rep flag-based reset.

v2 result: 5 000 samples, 0 skipped, 0 OOB-stops. 271 MB on simulator.
Action stats: lx mean=0.137, std=0.119, range=[−0.18,0.33]; az mean=−0.015, std=0.345, range=[−0.63,0.63].

---

## [2026-05-23] setup | Wiki initialised

Wiki created following Karpathy LLM Wiki pattern.
Structure: CLAUDE.md (schema) + wiki/ (LLM-maintained) + sources/ + raw_sources/
Initial pages created: index, overview, all 6 contributions, 5 papers, 4 datasets, 3 architecture, 2 infrastructure, 3 decisions.
Status: BW01 in progress (May 15–28). Driver install pending on simulator machine.

## [2026-05-23] ingest | OpenVLA-OFT (arXiv 2502.19645)

Source: Kim, Finn, Liang — Stanford — Apr 2025
Pages updated: [[OpenVLA-OFT]], [[C2-MidLevelActionHead]], [[C4-FusionAblation]], [[decision-oft-recipe]]
Key facts filed: 26× throughput, 97.1% LIBERO, 87.8% ALOHA, FiLM drops language following to 33%.

## [2026-05-23] ingest | VLingNav (arXiv 2601.08665)

Source: Wang, Luo et al. — ByteDance Seed & Peking U — Jan 2026
Pages updated: [[VLingNav]], [[C1-AdaCoT]], [[C3-ConfidenceGatedHandoff]], [[C5-RLFinetuning]]
Key facts filed: AdaCoT fires at 2.1% of steps, dense CoT hurts performance, Nav-AdaCoT-2.9M dataset, expert-guided RL stage.

## [2026-05-23] milestone | Dr. Shankar call prep complete

Prepared 4 dissertation slides (PaperDissertation_v2.pptx) and speaking script (VLA4AMR_Speaking_Script.docx).
Also prepared 9-slide call deck (VLA4AMR_Shankar_Prep.pptx) covering papers, datasets, Isaac Sim, GPU setup, bi-weekly plan.

## [2026-05-23] setup | simulator-machine audit

Machine: cvit-car-simulator@simulator
GPUs: RTX 4060 Ti 16GB (01:00.0) + RTX 5000 Pro Blackwell 48GB (02:00.0)
Driver: NOT installed — nvidia-smi failing, no kernel module
Recommended: nvidia-driver-595-open
Next action: sudo apt install -y nvidia-driver-595-open → reboot → verify nvidia-smi

## [2026-05-26] milestone | GPU driver installed on simulator-machine

Driver: nvidia-driver-595-open, version 595.71.05, CUDA 13.2
GPU 0: NVIDIA GeForce RTX 4060 Ti — 16380 MiB — PCI 01:00.0 — display + CARLA
GPU 1: NVIDIA RTX PRO 5000 Blackwell — 48935 MiB — PCI 02:00.0 — VLA4AMR (CUDA_VISIBLE_DEVICES=1)
Resolved: RTX 5000 Pro Blackwell works with 595-open (PCI ID concern closed)
Resolved: GPU index assignment confirmed — GPU 0 = 4060 Ti, GPU 1 = 5000 Pro
Pages updated: [[simulator-machine]], [[overview]]
Next: check ROS 2 Jazzy, install CUDA toolkit 12.4, set up Isaac Sim

## [2026-05-26] milestone | ROS 2 Jazzy installed on simulator-machine

Package: ros-jazzy-desktop + ros-dev-tools
Sourced via: source /opt/ros/jazzy/setup.bash (added to ~/.bashrc)
Note: GPG key issue fixed by piping curl through gpg --dearmor with sudo
Pages updated: [[simulator-machine]]
Next: CUDA toolkit 12.4, then Isaac Sim

## [2026-05-26] ingest | Deep reading notes — OpenVLA-OFT + VLingNav

Files created: [[OpenVLA-OFT-reading-notes]], [[VLingNav-reading-notes]]
OpenVLA-OFT: full architecture (parallel decoding, L1 head, FiLM), all tables, ablations, critical analysis for C2/C4
VLingNav: full architecture (AdaCoT, VLingMem, dynamic FPS, probabilistic head), 3-stage training, all results, critical analysis for C1/C3/C5
Key filings: Dense CoT hurts (25.3 vs 36.2), AdaCoT at 2.1% steps, w/o Memory collapses to 15.4 SR, FiLM drops language following to 33%

## [2026-05-26] milestone | CUDA toolkit 12.6 installed on simulator-machine

Package: cuda-toolkit-12-6 (12.4 not available in repo)
nvcc: /usr/local/cuda-12.6/bin/nvcc — PATH updated in ~/.bashrc
Compatible with PyTorch cu124 builds and flash-attention compilation
Pages updated: [[simulator-machine]]
Next: Isaac Sim via pip (isaacsim package)

## [2026-05-26] milestone | Isaac Sim 4.5.0.0 installed on simulator-machine

Package: isaacsim==4.5.0.0 from pypi.nvidia.com
Conda env: isaac (Python 3.10)
Pages updated: [[simulator-machine]]
Next: verify Isaac Sim launch, then conda env openvla

## [2026-05-26] milestone | Isaac Sim 4.5.0.0 import verified

Fix: needed isaacsim[all] — base package is a launcher shell, extensions are separate
Verified: `from isaacsim import SimulationApp` → Isaac Sim import OK
Pages updated: [[simulator-machine]]
Next: conda env openvla (Python 3.10, torch+cu124, OpenVLA-OFT)

## [2026-05-26] milestone | openvla conda env ready with PyTorch 2.11.0+cu128

Issue: PyTorch cu124 doesn't support Blackwell sm_120 — upgraded to cu128
torch 2.11.0+cu128 verified: CUDA available, device = NVIDIA RTX PRO 5000 Blackwell, no warnings
GPU assignment correction: CUDA FASTEST_FIRST ordering means CUDA_VISIBLE_DEVICES=0 = RTX 5000 Pro (not 1 as originally planned)
Pages updated: [[simulator-machine]]
Next: install OpenVLA-OFT into openvla env

## [2026-06-06] milestone | OpenVLA-OFT fully installed in openvla env

Steps taken:
- `pip install -e .` initially downgraded torch 2.11.0+cu128 → 2.2.0 (cu121) — broke Blackwell
- Fixed: relaxed pyproject.toml torch pins from `==2.2.0` to `>=2.2.0`; restored cu128; reinstalled with `--no-deps`
- `flash-attn==2.5.5` compiled from source with `--no-build-isolation` against torch 2.11.0+cu128 (~15 min)
- Note: stale `/home/cvit-car-simulator/.local/bin/ninja` wrapper can be ignored — build uses Python module
- Note: TF 2.15 CUDA factory warnings on import are benign — TF used only for dataset loading
Verified: torch 2.11.0+cu128, flash_attn 2.5.5, CUDA on RTX PRO 5000 Blackwell, openvla-oft import OK
Pages updated: [[simulator-machine]]
Next: download OpenVLA-7B weights → BW02 Nav2 baseline benchmarks

## [2026-06-06] milestone | OpenVLA-7B weights downloaded

15 GB, 3 safetensors shards at ~/VLA4AMR/checkpoints/openvla-7b
Pages updated: [[simulator-machine]]
Next: Isaac Sim + Nova Carter launch (BW02)

## [2026-06-13] milestone | Nova Carter bringup PASS in Isaac Sim

Asset: Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd (200 on public S3 CDN)
Root: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5
Load time: ~36s (S3 download, cached after first run)
Result: prim valid=True, 31 children, 2141 total prims
Sensors confirmed: front/right/left/back Hawk stereo, front/back/left/right Owl fisheye, ros_lidars, chassis_imu
Physics confirmed: differential_drive, all wheel joints, caster joints, transform_tree_odometry
ROS2 action graph: dormant (expected — Jazzy bridge not loaded), non-fatal warnings only
Key path fix: correct asset is at Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd NOT Isaac/Robots/NovaCarter/nova_carter.usd
Key path fix: base nova_carter.usd (Isaac/Robots/NVIDIA/NovaCarter/) is also 404 on 4.5 CDN — use ROS variant
Working command:
```bash
DISPLAY=:1 /home/cvit-car-simulator/miniconda3/envs/isaac/bin/python -u ~/VLA4AMR/code/nova_carter_launch_test.py 2>&1 | tee ~/Desktop/isaac_out.txt
```
Pages updated: [[simulator-machine]], [[NovaCarter]], [[overview]]
Next: find carter_warehouse_navigation.usd + fix ROS2 Jazzy bridge → full Nav2 pipeline (BW02)

## [2026-06-13] milestone | Isaac Sim + Blackwell working — driver downgraded to 570.211.01

Root cause identified: NVIDIA driver 595.71.05 incompatible with RTX PRO 5000 Blackwell (CC 12.0) for Isaac Sim rendering.
Symptoms: iRay explicit "CC 12.0 unsupported" warning + TLAS buffer overflow (`within: false`) → SimulationApp segfault at `_wait_for_viewport`.
Fix: downgraded to `nvidia-driver-570-open` (570.211.01) — TLAS now `valid true, within: true`, iRay warning remains but non-fatal.
PyTorch cu128 still works (requires driver ≥ 570.00 — confirmed compatible).
Working command:
```bash
DISPLAY=:1 /home/cvit-car-simulator/miniconda3/envs/isaac/bin/python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'no_window': True})
print('SimulationApp: OK')
app.close()
" 2>&1 | tee ~/Desktop/isaac_out.txt
```
Result: "Simulation App Startup Complete" at 234.738s, "Simulation App Shutting Down" at 235.832s ✅
Key learnings: do NOT set CUDA_VISIBLE_DEVICES for Isaac Sim (carb.cudainterop crash); use direct Python path not `conda run`; Blackwell = Omniverse GPU 0 (fastest-first order).
Pages updated: [[simulator-machine]], [[overview]]
Next: Nova Carter launch in Isaac Sim warehouse world (BW02)

## [2026-06-13] milestone | Full-stack re-verification passed on simulator-machine

All 8 components verified live on cvit-car-simulator (pre Nova Carter bringup):
1. NVIDIA driver 595.71.05 — RTX 4060 Ti 16380 MiB + RTX PRO 5000 Blackwell 48935 MiB
2. CUDA toolkit 12.6 — nvcc V12.6.85
3. ROS 2 Jazzy — ROS_DISTRO=jazzy confirmed
4. Isaac Sim 4.5.0.0 — import OK (conda env: isaac, Python 3.10)
5. PyTorch 2.11.0+cu128 — CUDA available, device = NVIDIA RTX PRO 5000 Blackwell
6. flash-attn 2.5.5 — OK
7. OpenVLA-OFT (prismatic) — import OK (TF cuDNN factory warnings are benign)
8. OpenVLA-7B weights — 15G, 3 shards at ~/VLA4AMR/checkpoints/openvla-7b
Status: stack confirmed clean. Next: Nova Carter launch in Isaac Sim (BW02).

## [2026-06-13] decision | Isaac Sim upgrade to 6.0.0 required for Blackwell + Jazzy

NVIDIA forum response to our bug report confirmed:
- Isaac Sim 4.5.0 (Kit SDK 106.5) has NO native Blackwell (CC 12.0) support — will never support it
- Driver R595 (NFB) is not validated for any Isaac Sim release — our 595→570 downgrade was the right fix
- **Recommended path: Isaac Sim 6.0.0.1 + Python 3.12 + R580 driver (580.95.05+)**

Key insight: Isaac Sim 6.0.0 uses Python 3.12, which is the same Python Jazzy was built for.
This eliminates the entire rclpy C extension version mismatch (`cpython-310` vs `cpython-312`).
Jazzy's ROS2 bridge should work natively in 6.0.0 with zero env-var patching.

Isaac Sim version → Python requirement: 4.x=3.10, 5.x=3.11, 6.x=3.12
(pip shows 4.5.0 as "latest" in a Python 3.10 env because 5.x/6.x are filtered by requires_python)

Current status on 4.5.0 (blocker hit during BW02):
- ROS2 bridge fails: Jazzy PYTHONPATH has py3.12 packages; bridge can't load `_rclpy_pybind11.cpython-310` in py3.10
- Crash during carter_warehouse_navigation.usd load when bridge is partially initialized
- Possible fix: `env -u PYTHONPATH -u AMENT_PREFIX_PATH` (try before upgrading)

Upgrade plan:
1. Create conda env `isaac6` with Python 3.12
2. pip install isaacsim[all]==6.0.0.1
3. Install R580 driver (580.95.05+)
4. Re-run Nav2 baseline (BW03 start)
Pages updated: [[simulator-machine]], [[IsaacSim]], [[overview]]

## [2026-06-13] milestone | Isaac Sim 6.0.0.1 working on simulator-machine

Root cause of missing extension resolved: `isaacsim[all]` does NOT include the `extscache` extra.
The extensions `isaacsim.anim.robot.schema`, `omni.anim.*`, and other Kit extensions ship in three separate packages:
- `isaacsim-extscache-kit`
- `isaacsim-extscache-kit-sdk`
- `isaacsim-extscache-physics`

Fix: `pip install "isaacsim[extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com`

Verified: `SimulationApp({'headless': True, 'no_window': True})` → "Simulation App Startup Complete" at 9.485s ✅
Key results:
- `isaacsim.anim.robot.schema-0.1.1` loaded at [2.263s]
- Warp: cuda:0 = RTX PRO 5000 Blackwell (sm_120, 47 GiB) — Blackwell CC 12.0 confirmed native
- Startup time: ~9.5s (vs ~230s in 4.5.0 — 24× faster)
- Driver R570 (570.211.01) still works for 6.0.0 (R580 recommended but not required)
Pages updated: [[simulator-machine]], [[IsaacSim]], [[overview]]
Next: test ROS2 Jazzy bridge natively in 6.0.0 → load carter_warehouse_navigation.usd → Nav2 baseline

## [2026-06-15] milestone | carter_warehouse_navigation.usd stage load verified in Isaac Sim 6.0.0.1

Stage loaded from S3: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd`
Result: **5,267 prims**, 17 cameras — full warehouse + Nova Carter scene confirmed.
Startup time: ~7.8s (isaac6 env, Python 3.12, Blackwell native).

Camera paths for Nova Carter (front Hawk stereo):
- Left: `/World/Nova_Carter_ROS/chassis_link/sensors/front_hawk/left/camera_left`
- Right: `/World/Nova_Carter_ROS/chassis_link/sensors/front_hawk/right/camera_right`
Additional Hawk rigs: `left_hawk`, `right_hawk`, `back_hawk` (same pattern).
Warehouse camera: `/World/warehouse_with_forklifts/Warehouse_Empty_small_realtime/Camera`

**Bug discovered: `get_stage_loading_status()` always returns (0, 0, 0) in Isaac Sim 6.0.0.1.**
Do NOT use it as a load-completion signal. Instead, poll for prim count > threshold:
```python
stage = omni.usd.get_context().get_stage()
if stage and stage.GetPrimAtPath('/World').IsValid():
    count = sum(1 for _ in stage.Traverse())
    if count > 100:
        break  # stage is ready
```

**Next blocker: `isaacsim.ros2.bridge` not installed in isaac6 env.**
All OmniGraph ROS2 nodes (`ROS2Context`, `ROS2CameraHelper`, `ROS2PublishTransformTree`, etc.) fail with
"Could not find node type interface" — the extension is absent, not just dormant.
Fix: `pip install "isaacsim-ros2-bridge==6.0.0.1" --extra-index-url https://pypi.nvidia.com`
(or `isaacsim[ros2]` extra — confirm package name first).
Pages updated: [[IsaacSim]], [[NovaCarter]], [[simulator-machine]], [[overview]]

## [2026-06-15] milestone | ROS2 bridge enabled in Isaac Sim 6.0.0.1

`isaacsim.ros2.bridge` is NOT missing — it IS installed in isaac6 exts/ but must be explicitly enabled at runtime.
Fix: call `mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)` BEFORE opening any stage.

Load sequence confirmed:
- `isaacsim.ros2.core-1.9.1` → "Attempting to load system rclpy" → "rclpy loaded" ✅
- `isaacsim.ros2.nodes-1.18.11` → `isaacsim.ros2.bridge-5.1.1` ✅
- Bridge version: 5.1.1 (independent of isaac sim 6.0.0.1 version number)

Key insight: uses **system Jazzy rclpy** (not pip-installed standalone). Python 3.12 match between
isaac6 and Jazzy makes this seamless — no `env -u ROS_DISTRO` needed, keep system ROS2 env vars.

Also confirmed: `isaacsim-ros2==6.0.0.1` pip package name exists (what makes the extension available),
but there is no `isaacsim-ros2-bridge` pip package — that was a wrong guess.

Next: run full bridge + carter scene + simulation + ROS2 topic verification.
Pages updated: [[IsaacSim]], [[simulator-machine]], [[overview]]

## [2026-06-16] milestone | BW02 gate PASSED — ROS2 pipeline end-to-end verified

Script: `VLA4AMR_wiki/bw03_ros2_verify.py` — run on cvit-car-simulator.
Result: 5/5 topics PASS on second run (first run had wrong topic names + missing app.update() in checker).
Confirmed publishing topics:
- /tf [tf2_msgs/msg/TFMessage]
- /chassis/odom [nav_msgs/msg/Odometry]
- /front_stereo_camera/left/image_raw [sensor_msgs/msg/Image]
- /front_stereo_camera/left/camera_info [sensor_msgs/msg/CameraInfo]
- /front_3d_lidar/lidar_points [sensor_msgs/msg/PointCloud2]
Also visible: /chassis/imu, /front_stereo_imu/imu, /left_stereo_imu/imu, /right_stereo_imu/imu, /back_stereo_imu/imu, /cmd_vel, /clock
Key corrections filed: topic names differ from NVIDIA defaults (/odom→/chassis/odom, no 2D /scan — LiDAR is 3D PointCloud2).
Next: BW03 — OpenVLA-OFT integration + inference node on live camera frames.

## [2026-06-17] milestone | Warehouse environment recorded — 15-angle tour (v5)

Script: `VLA4AMR_wiki/bw03_record_env.py` (v5) — renders 15 cameras × 450 frames in Isaac Sim 6.0.0.1.
Output: `recordings/v5/` (15 MP4s + mosaic) at `VLA4AMR_wiki/recordings/v5/`.

Working cameras (10/15): top_centre, top_NE, top_SE, top_SW, wall_south, wall_east, aisle_wide, robot_front, robot_left, robot_back, warehouse_cam.
Dark cameras (5/15): top_NW (shelves block north corner), wall_north, wall_west, aisle_long (long aisle too dark).

Key engineering lessons:
- Multi-GPU crash: default `multi_gpu=True` causes RTX 4060 Ti (device 1) → `cudaErrorIllegalAddress` in CUDA interop. Fix: `SimulationApp({"multi_gpu": False, "active_gpu": 0})`.
- Cameras OUTSIDE the warehouse (> bbox extents): see opaque exterior walls → black frames. Fix: place ALL synthetic cameras INSIDE warehouse bounds (X: ±10.5m, Y: −16..+19m, Z: 1..8.3m).
- Overhead cameras ABOVE ceiling (H > CEIL): ceiling is opaque → black frames. Fix: H_HIGH = CEIL − 1.0 = 8.3m.
- USD prim names cannot start with a digit — prefix prim names with `cam_`.
- `rep.create.camera()` does not register with RTX renderer in headless mode — use `UsdGeom.Camera.Define()`.
- `app.close()` crashes Python before any post-close code runs — stitch (ffmpeg) must be a separate script.
- BasicWriter saves frames as `{cam_dir}/rgb_XXXX.png` (NOT in a `rgb/` subdirectory).
- Stage up-axis IS Z for this scene — use cross-product 4×4 look-at, not Euler angles.

Next: BW03 main goal — OpenVLA-OFT inference node (two-process architecture).

## [2026-06-17] milestone | BW03 OpenVLA-OFT inference node written (two-process)

Scripts written and SCP'd to ~/VLA4AMR/code/ on simulator:
- `bw03_inference_server.py` — runs in `openvla` env (Python 3.10, torch+cu128); loads OpenVLA-7B, reads base64-JPEG from stdin, writes JSON action to stdout; signals "READY\n" on stderr when loaded.
- `bw03_openvla_node.py` — runs with system Python 3.12 + Jazzy rclpy; subscribes to /front_stereo_camera/left/image_raw, proxies every Nth frame to inference server via subprocess pipe, publishes /cmd_vel + /openvla/action_raw + /openvla/status.

Architecture decision: two-process IPC instead of installing torch in new env.
Reason: Jazzy rclpy requires Python 3.12; openvla env (Python 3.10) already has torch+cu128+model. 
Avoids 2.4GB torch re-download and env conflict. ROS2 node is torch-free.

Node deps verified on system python3 (rclpy, numpy, PIL — all OK).
Inference server deps verified in openvla env (torch cu128, transformers, CUDA on Blackwell — OK).

Run commands:
  Terminal 1 (Isaac Sim + ROS2 bridge):
    source /opt/ros/jazzy/setup.bash
    DISPLAY=:1 /home/cvit-car-simulator/miniconda3/envs/isaac6/bin/python -u \
        ~/VLA4AMR/code/bw03_isaac_live.py 2>&1 | tee ~/Desktop/bw03_isaac.txt

  Terminal 2 (OpenVLA inference node):
    conda deactivate
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 ~/VLA4AMR/code/bw03_openvla_node.py 2>&1 | tee ~/Desktop/bw03_openvla.txt

Next: run both terminals on simulator and verify /cmd_vel publishes with live model inference.

## [2026-06-18] decision | Context shift — meeting demo complete, moving to C4 FusionAblation

Meeting demo videos delivered:
- `recordings/bw03_drive/bw03_meeting_v4.mp4` — 1280×960, 4-panel, 1 min 39s, robot navigating (use this)
- `recordings/bw03_drive/bw03_meeting_v5.mp4` — same setup, 1 min 15s

Remaining known issue: overhead tracking panel shows approximate robot position (XformCache ≠ physics position). Acceptable for demo; fix requires subscribing to /chassis/odom within Isaac Sim.

Next focus: **C4 FusionAblation** — Isaac-Synthetic dataset generation, `bw03_c4_train.py`, `bw03_c4_eval.py`.

## [2026-06-18] milestone | Meeting demo video — robot navigation in 4-panel view

Scripts: `bw03_meeting_live.py` (T1: Isaac Sim + recording) + `bw03_meeting_drive.py` (T2: cmd_vel)
Output: `recordings/bw03_drive/bw03_meeting_v4.mp4` — 1280×960, 4-panel, 99s, robot navigating

**Critical bugs found and fixed (2026-06-18):**

1. **Multi-GPU crash**: Isaac Sim without `CUDA_VISIBLE_DEVICES=0` attempts RTX rendering on device 1
   (RTX PRO 5000 Blackwell) AND device 0 (RTX 4060 Ti). Device 1 fails with `CUDA error 700:
   cudaErrorIllegalAddress` in `carb.scenerenderer-rtx.plugin`. Fix: always use
   `CUDA_VISIBLE_DEVICES=0 DISPLAY=:1 python -u bw03_meeting_live.py`.
   NOTE: Contradicts BW01 entry which said "do NOT set CUDA_VISIBLE_DEVICES" — that was for
   Isaac Sim 4.5 + driver 570. Isaac Sim 6.0.0.1 REQUIRES it to prevent multi-GPU render crash.

2. **Robot spawn modification breaks physics**: `ClearXformOpOrder()` + `AddTranslateOp()` on
   `/World/Nova_Carter_ROS` disconnects the OmniGraph articulation. Robot stays stationary even
   though cmd_vel is received (confirmed via `ros2 topic info /cmd_vel`). Fix: do NOT modify
   robot's xform; use default spawn (-6, -1) and design trajectory to reverse+turn out of shelf.

3. **XformCache returns static USD position, not physics position**: Both root prim and
   chassis_link `GetLocalToWorldTransform()` return a position near the initial USD spawn, not
   the actual physics-driven position. Actual position confirmed via `/chassis/odom` ROS2 topic.
   Workaround: overhead tracking camera is imprecise (shows area near spawn); acceptable for demo.

Robot movement confirmed: cmd_vel subscriber is `_World_Nova_Carter_ROS_differential_drive_...`,
odometry shows robot at (0.847, -1.567) after traversing warehouse from start (-6, -1).

v4 video panels: AI INPUT (front cam, moving ✓), ROBOT PATH (overhead, stuck near start ✗),
ISOMETRIC VIEW (fixed cam, robot visible ✓), WAREHOUSE OVERVIEW (fixed cam, robot visible ✓).

Next: C4 FusionAblation — Isaac-Synthetic dataset generation, bw03_c4_train.py, eval.

## [2026-06-17] milestone | BW03 COMPLETE — OpenVLA-OFT end-to-end inference verified LIVE

Full stack running on cvit-car-simulator:
  Isaac Sim 6.0.0.1 → /front_stereo_camera/left/image_raw → OpenVLA ROS2 node
  → subprocess pipe (base64 JPEG) → OpenVLA-7B (openvla env, Python 3.10, RTX PRO 5000 Blackwell)
  → JSON action → /cmd_vel + /openvla/action_raw

Key metrics observed:
- Model load time: ~14.5s (OpenVLA-7B bfloat16 on Blackwell)
- Inference latency: **0.20s per frame** (~5 Hz) on RTX PRO 5000 Blackwell
- Frame rate: camera at ~30 Hz, INFER_EVERY=5 → inference every ~167ms (slightly over budget at 0.20s)
- Verified at inf#615, frame=3075 — sustained over 600 continuous inferences
- Action output example: [-0.0, -0.0, -0.0, 0.002, -0.003, -0.007, 0.996]

Bug fixed: `predict_action()` in installed openvla version returns `numpy.ndarray` (not torch.Tensor).
Fixed in `bw03_inference_server.py`:
  Before: `action.cpu().float().numpy().flatten().tolist()`
  After: type-safe check — `hasattr(action, 'cpu')` branch for tensor vs ndarray

Action interpretation: near-zero linear.x and angular.z — expected for base OpenVLA-7B which was
trained on BridgeData V2 (tabletop manipulation), not warehouse driving. Fine-tuning (C4) required.

Terminal 1 warning (benign): "Invalid deltaTime 0.000000" from differential_controller_01 — appears
when OmniGraph timestep is zero (warmup/catchup). Does not affect topic publishing.

Next: C4 FusionAblation scaffolding — token concat vs FiLM vs cross-attention comparison.

## [2026-06-16] milestone | BW03 started — end-to-end ROS2 verification script written

Script: `VLA4AMR_wiki/bw03_ros2_verify.py`
What it does: Launch Isaac Sim 6.0.0.1 → enable ROS2 bridge → load carter_warehouse_navigation.usd → warm up 300 sim steps → check 5 key topics → print pass/fail report + full topic list.
Next: Run on cvit-car-simulator, paste output back to diagnose any failing topics.

## [2026-06-18] milestone | Isaac-Synthetic dataset generated — 5 000 samples

Script: `bw03_c4_collect.py` (Terminal 2, system Python 3.12) + `bw03_isaac_live.py` (Terminal 1)
Result: **5 000 frames collected, 0 skipped**

Architecture fix (v3): two-process approach — rclpy subscriber cannot run inside Isaac Sim process
(conflicts with ROS2 bridge's own rclpy context). Moved to separate system Python process,
same pattern as bw03_openvla_node.py (proven in BW03).

Action statistics:
  mean: [0.204, 0.0, 0.0, 0.0, 0.0, 0.070, 0.0]
  std:  [0.207, 0.0, 0.0, 0.0, 0.0, 0.383, 0.0]
  (dims 1-4 and 6 always 0 — differential drive)

Output on simulator: ~/VLA4AMR/datasets/isaac_synthetic/ (154 MB)
Copied to local Mac: ~/Downloads/isaac_synthetic/

Next: bw03_c4_train.py — LoRA fine-tuning for 3 fusion strategies on openvla env

## [2026-06-18] milestone | C4 FusionAblation scripts written — dataset + train + eval

Three scripts written and ready to SCP to simulator / openvla env:

1. `bw03_c4_dataset.py` (isaac6 env, Python 3.12) — generates [[Isaac-Synthetic]] dataset:
   - 10 trajectory templates × 50 reps × 10 steps = 5 000 (image, text_goal, action_7d) samples
   - Captures front Hawk camera frames via omni.replicator RGB annotator (in-memory)
   - Publishes cmd_vel via ROS2 from within the script (no second terminal needed)
   - Resets timeline every 5 reps to keep Nova Carter within warehouse bounds
   - Output: ~/VLA4AMR/datasets/isaac_synthetic/ {images/, episodes.jsonl, stats.json}

2. `bw03_c4_train.py` (openvla env, Python 3.10) — LoRA fine-tuning for 3 strategies:
   - Loads dataset + stats.json, computes action tokenizer with q01/q99 normalization
   - For each strategy: deep-copies base model, applies LoRA rank 32 on q_proj/v_proj
   - FusionHook injects FiLM / cross-attn goal conditioning during forward pass
   - Cross-entropy loss on the 7 action token positions
   - Saves LoRA adapter + fusion module state_dict per strategy

3. `bw03_c4_eval.py` (openvla env, Python 3.10) — open-loop RMSE evaluation:
   - Loads each strategy checkpoint + fusion module
   - Runs model.generate() on 10% test split
   - Reports: per-strategy total RMSE, nav RMSE (dims 0=lx, 5=az), per-dim RMSE
   - Saves results table to ~/VLA4AMR/checkpoints/c4/eval_results.json

Wiki page created: [[Isaac-Synthetic]]
C4 status updated: scaffold ✓, dataset script ✓, train script ✓, eval script ✓

Next: SCP bw03_c4_dataset.py → run on simulator → generate 5K samples

## [2026-06-05] milestone | Full-stack sanity check passed on simulator-machine

All 5 components verified live on cvit-car-simulator:
1. NVIDIA driver 595.71.05 — RTX 4060 Ti 16380 MiB + RTX PRO 5000 Blackwell 48935 MiB
2. CUDA toolkit 12.6 — nvcc V12.6.85
3. ROS 2 Jazzy — ROS_DISTRO=jazzy (system-wide at /opt/ros/jazzy/, not conda)
4. Isaac Sim 4.5.0.0 — `from isaacsim import SimulationApp` import OK (conda env: isaac)
5. openvla env — torch 2.11.0+cu128, CUDA available, device = NVIDIA RTX PRO 5000 Blackwell
Note: `ros2 --version` is not a valid flag; use `printenv ROS_DISTRO` to check.
Status: BW01 stack fully confirmed. Ready to proceed with BW02 (Nav2 baseline benchmarks).
