# Decision: BW15 — Full CorrectNav Stack (LLaVA-Video backbone)

**Type:** decision
**Status:** active
**Last updated:** 2026-07-04
**Related:** [[decision-bw14-discrete-nav]], [[Nav-AMR-WH]], [[IsaacSim]], [[C6-EvaluationProtocol]]

## Summary

BW15 replaces our custom Qwen2.5-VL-7B training stack with the full CorrectNav
system (AAAI 2025): LLaVA-Video-7B-Qwen2 backbone + SigLIP-SO400M vision encoder
+ MP4 video-history input + multi-step discrete action prediction, adapted to
run on a single GPU and our Isaac Sim warehouse environment.

## Context

After BW14 achieved discrete-token training (val=0.0661) but before closed-loop
testing, Mr. K decided to pivot to the CorrectNav architecture directly. Reasons:

1. **Proven video-history model**: CorrectNav is purpose-built for multi-frame
   navigation (8-frame sliding MP4 window), avoiding our custom collation/OOM hacks.
2. **Same discrete vocabulary**: "Move forward / Turn left / Turn right / Stop" —
   identical to BW14 tokens, so dataset and IPC remain compatible.
3. **Self-correction flywheel**: CorrectNav's Phase 2 mechanism (detect deviation
   → collect corrective episodes → retrain) is directly applicable to our Isaac Sim loop.
4. **Publication leverage**: Using a published system strengthens the experimental
   baseline narrative for ICRA 2027.

## Architecture changes from BW14

| Aspect            | BW14                          | BW15 CorrectNav                          |
|-------------------|-------------------------------|------------------------------------------|
| Base model        | Qwen2.5-VL-7B-Instruct        | LLaVA-Video-7B-Qwen2 (CorrectNav)        |
| Vision encoder    | Qwen2.5-VL native             | SigLIP-SO400M-patch14-384                |
| Video format      | 6 PIL images (multi-image)    | per-step MP4 (8-frame sliding window)    |
| Action format     | `ACTIONS: FORWARD,TURN_LEFT`  | `Final Answer: Move forward,Turn right`  |
| Training          | bw14_train.py (custom LoRA)   | CorrectNav/train/train_mem.py (DeepSpeed)|
| Max frames        | H=6                           | H=8 (frames_upbound=8)                  |
| Attention         | eager (Blackwell-safe)        | eager (overriding flash_attention_2)     |

## 1-GPU adaptation

CorrectNav ships with an 8-GPU ZeRO-3 training script. Our adaptation:

- `GPU_NUM=1`, `CUDA_VISIBLE_DEVICES=0` (RTX 5000 Pro 48GB)
- LoRA r=32 α=64 on language model (replaces full fine-tune → fits in GPU memory)
- ZeRO stage 2 (no cross-GPU sharding needed, but helps gradient clipping/stability)
- `GRADIENT_ACCUMULATION_STEPS=16` (global batch = 1×1×16 = 16, matches original 8×1×2)
- `model_max_length=8192` (was 32768, sufficient for 8 SigLIP frames + text)
- `frames_upbound=8` (our MAX_HISTORY)
- Remove `--torch_compile` (single-GPU instability)
- `mm_tunable_parts="mm_mlp_adapter"` (freeze SigLIP vision tower AND LLM base weights — see fix below)
- `report_to none` (TensorBoard broken: numpy 2.x / TF incompatibility in openvla env)
- ZeRO-2 without `offload_optimizer` (CPU offload not needed once trainable params are small)

## Dataset conversion (bw15_convert.py)

bw11_dataset → CorrectNav format:
- Input: `~/Desktop/bw11_dataset/hf_dataset/` (46,035 train / 4,936 val frames)
- For each episode frame `step_idx`: build MP4 of last 8 frames at 384×384, 1fps
- Ground-truth label: `"Final Answer: Move forward,Turn left,..."` (N=6 lookahead)
- Prompt: CorrectNav's multi-step prompt with warehouse instruction substituted
- Output: `~/Desktop/bw15_dataset/videos/`, `warehouse_train.json`, `warehouse_val.json`

## Isaac Sim integration (bw15_sim_vla.py)

- Tries `CorrectNav.model.builder.load_pretrained_model` first; falls back to AutoModel
- `attn_implementation="eager"` in both paths
- Rolling 8-frame deque; executes 6 tokens then re-queries
- Same IPC mechanism as bw13/bw14 (`/tmp/bw11_ipc/` flag files)
- TOKEN_TO_VEL: `move forward → (0.35, 0.0)`, `turn left → (0.25, +0.7)`, etc.

## Training plan

1. **Phase 1**: Fine-tune on bw15_dataset (bw15_train.sh, ~3 epochs)
   - Start from `lmms-lab/LLaVA-Video-7B-Qwen2` base
   - Output: `~/Desktop/bw15_checkpoints/`
2. **Phase 2 (CorrectNav flywheel)**: Run in Isaac Sim, detect deviations,
   collect corrective episodes, convert with bw15_convert.py, retrain
   - Deviation detection: compare Isaac Sim robot pose vs planned path ≥0.5m error
   - Correction label: oracle action to re-align within 3 steps

## transformers 5.x compatibility patches (applied 2026-07-03 to 2026-07-04)

CorrectNav was written for transformers ~4.40. Running against transformers 5.x required
the following surgical patches on the simulator. All patches are live on cvit-car-simulator
at `~/VLA4AMR/CorrectNav-main/`.

### `CorrectNav/model/language_model/llava_qwen.py`

`LlavaQwenForCausalLM.__init__` must bootstrap three attributes that Qwen2Model now
requires before `super().__init__()` is called:

```python
# 1. pad_token_id (Qwen2 needs non-None)
if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
    config.pad_token_id = getattr(config, "eos_token_id", 0)
    if isinstance(config.pad_token_id, list):
        config.pad_token_id = config.pad_token_id[0]

# 2. rope_parameters dict with rope_type + rope_theta + max_position_embeddings
if not hasattr(config, "rope_parameters") or config.rope_parameters is None:
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        config.rope_parameters = {
            "rope_type": "default",
            "rope_theta": getattr(config, "rope_theta", 1000000.0),
            "max_position_embeddings": getattr(config, "max_position_embeddings", 32768),
        }
    else:
        config.rope_parameters = dict(rope_scaling)
        if "rope_type" not in config.rope_parameters:
            config.rope_parameters["rope_type"] = rope_scaling.get("type", "default")

# 3. layer_types list ("full_attention" vs "sliding_attention" per layer)
if not hasattr(config, "layer_types") or config.layer_types is None:
    _use_sw = getattr(config, "use_sliding_window", False)
    _sw = getattr(config, "sliding_window", None) if _use_sw else None
    _max_wl = getattr(config, "max_window_layers", 0)
    _n_layers = getattr(config, "num_hidden_layers", 28)
    config.layer_types = [
        "sliding_attention" if _sw is not None and i >= _max_wl else "full_attention"
        for i in range(_n_layers)
    ]

Qwen2ForCausalLM.__init__(self, config)
config.model_type = "llava_qwen"
# NOTE: do NOT set config.rope_scaling = None here — in transformers 5.x that
# property setter internally wipes rope_parameters, undoing fix #2 above.
```

### `CorrectNav/model/multimodal_encoder/siglip_encoder.py`

Three patches:
1. **`_set_token_in_kwargs` guard** (method removed in newer transformers):
   `if hasattr(cls, "_set_token_in_kwargs"): cls._set_token_in_kwargs(kwargs)`
2. **`local_files_only=True`** on `get_config_dict` call to avoid network hang
   (symptom: 28-minute stall on a `.incomplete` blob trying to re-fetch from HF)
3. **`device_map="cpu"`** on `SigLipVisionModel.from_pretrained` — transformers 5.x
   raises "meta device context" error when `device_map=None` + ZeRO-2

### `CorrectNav/train/train.py`

Line ~1694: `tokenizer=` → `processing_class=` (renamed in transformers 5.x Trainer API):
```python
trainer = LLaVATrainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
```

### `CorrectNav/train/llava_trainer.py`

Three patches:
1. **`create_accelerator_and_postprocess`**: remove `dispatch_batches` arg (dropped in
   transformers 5.x), use `getattr` for `split_batches` and `deepspeed_plugin`
2. **`_get_train_sampler`**: all `self.args.group_by_*` → `getattr(self.args, 'group_by_*', False)`
   (four variants: `group_by_length`, `group_by_modality_length`,
   `group_by_modality_length_auto`, `group_by_varlen`)
3. **`dataloader_persistent_workers`**: wrapped in `getattr(..., False)`

### Previously patched files (earlier sessions)

- `CorrectNav/model/multimodal_resampler/qformer.py`: `apply_chunking_to_forward`,
  `find_pruneable_heads_and_indices`, `prune_linear_layer` — try/except stubs
- `trl/extras/__init__.py`: `BestOfNSampler` import wrapped in try/except
- `trl/models/__init__.py`: SD model imports wrapped in try/except
- `llava_trainer.py`: `ALL_LAYERNORM_LAYERS`, `is_accelerate_available`,
  `GradientAccumulationPlugin` import fallbacks

### Tokenizer / model weight downloads

LLaVA-Video-7B-Qwen2 HF snapshot initially lacked tokenizer files. Fix:
```python
snapshot_download('lmms-lab/LLaVA-Video-7B-Qwen2', ignore_patterns=['*.safetensors'])
```
Also required: `pip install tiktoken` (Qwen tokenizer backend).
SigLIP cache had an `.incomplete` blob causing network hangs — deleted it and
ran `snapshot_download('google/siglip-so400m-patch14-384')` to complete.

### Environment fixes

- `pip install ninja` in openvla conda env — broken Python wrapper at
  `~/.local/bin/ninja` was shadowing `/usr/bin/ninja` (system ninja 1.11.1);
  wrapper removed, pip ninja installed inside env. DeepSpeed cpu_adam JIT now works.
- `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` added to `bw15_train.sh` to
  prevent HF network calls during training (caused hangs post-download).

## Training status (as of 2026-07-04)

**Not yet running.** Training was OOM-killed by the kernel twice (RAM OOM, not GPU OOM):

- Root cause: `mm_tunable_parts=mm_language_model` unfreezes the full 7B LM even
  with `lora_enable=True` — the `mm_tunable_parts` logic runs after PEFT and calls
  `requires_grad_(True)` on all LM params, overriding LoRA freezing.
- Result: 7713 MB trainable → ZeRO-2 CPU offload needs ~30 GB RAM for fp32 optimizer
  states → machine's ~62 GB RAM exhausted → OOM kill (no error in log, kernel SIGKILL).
- **Fix**: `mm_tunable_parts="mm_mlp_adapter"` only (LoRA adapters stay trainable via
  PEFT; only MLP projector is explicitly unfrozen). Trainable params drop to ~17 MB.
- CPU offload removed from `bw15_zero2.json` — not needed at 17 MB trainable.
- `report_to tensorboard` → `report_to none` — TensorBoard callback crashed at
  `LLaVATrainer.__init__` due to numpy 2.x / ml_dtypes / TF incompatibility in openvla.

**Ready to run**: All patches applied, configs corrected. Will start tomorrow (2026-07-05).

## Open questions

- With `mm_tunable_parts="mm_mlp_adapter"`, only ~17 MB of params train — are the
  LoRA adapters in the LLM actually updating? Need to verify by checking whether PEFT
  keeps LoRA adapter `requires_grad=True` independently of `mm_tunable_parts`. If not,
  add `mm_lora_adapter` to tunable parts or inspect the CorrectNav `make_only_tunable_params`
  function to understand the interaction.
- SigLIP-SO400M has never been exposed to warehouse imagery — will domain gap
  require unfreezing the vision tower (add `mm_vision_tower` to `mm_tunable_parts`)?
- LR 1e-4: confirm convergence; may need 2e-4 if val_loss plateaus early.
- CorrectNav pre-trained weights (PKU disk) vs training from LLaVA-Video-7B-Qwen2
  base: if PKU weights load without auth, prefer them (VLN pretraining helps).
- TensorBoard disabled (`report_to none`) — need an alternative for loss curves.
  Options: (a) fix numpy/ml_dtypes conflict via `pip install 'numpy<2'`; (b) use wandb.

## Sources

- [[decision-bw14-discrete-nav]] — the pivot decision and context
- CorrectNav paper (arXiv 2508.10416), read via CorrectNav-main/README.md
- bw15_train.sh, bw15_convert.py, bw15_sim_vla.py — implementation
