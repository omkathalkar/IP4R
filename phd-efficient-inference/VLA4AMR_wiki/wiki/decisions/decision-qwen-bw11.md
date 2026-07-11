# Decision: Qwen2.5-VL-7B for BW11 Training (replacing OpenVLA-7B)

**Type:** decision
**Status:** active
**Last updated:** 2026-06-30
**Related:** [[C1-AdaCoT]], [[Nav-AMR-WH]], [[overview]]

## Decision

Switch base model from OpenVLA-7B (LLaMA-2 backbone) to **Qwen2.5-VL-7B-Instruct** for BW11 training and all subsequent work.

## Why

1. **Native multi-image / vision-language support.** Qwen2.5-VL handles image+text natively in chat format. OpenVLA uses a 7D action tokenizer that conflicts with language-format CoT output — you cannot easily emit `<think>...</think>` alongside the 7D action tokens.
2. **Text-format action output.** BW11 trains the model to output `ACTION: lin=x ang=y` as free text. Qwen's causal LM head supports this natively. OpenVLA's action head is a separate discrete token head.
3. **Same annotation model.** Qwen2.5-VL-7B is the same model family used for CoT annotation (`bw10_cot_annotate_local.py`), so annotation quality and training distribution are aligned.
4. **LoRA compatibility.** PEFT LoRA works cleanly on Qwen2.5-VL's `q/k/v/o/gate/up/down_proj` modules. No special action tokenizer surgery needed.
5. **Blackwell compatibility.** Qwen2.5-VL with `attn_implementation="eager"` runs on RTX PRO 5000 Blackwell (compute 12.0). flash_attention_2 binary was not built for compute 12.0 in the openvla env.

## Trade-off

- **Loss:** Action prediction is free-text (`ACTION: lin=x ang=y`) rather than discretised 7D bins. Slightly more error-prone if parsing fails — but 100% parse rate on 4,936 val samples confirms reliability.
- **Gain:** CoT text generation works natively. lin MAE=0.0000, ang MAE=0.0105 on val — far better than BW06 OpenVLA results (lin MAE=0.166).

## How to apply

Use Qwen2.5-VL-7B-Instruct as base for all future fine-tuning. Action format: `ACTION: lin={:.3f} ang={:.3f}`. Parse with regex `ACTION\s*:\s*lin\s*=\s*([+-]?\d+\.?\d*)\s+ang\s*=\s*([+-]?\d+\.?\d*)`.
