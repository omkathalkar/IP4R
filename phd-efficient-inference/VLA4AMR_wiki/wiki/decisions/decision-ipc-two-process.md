# Decision: Two-Process IPC for Isaac Sim + VLA Inference

**Type:** decision
**Status:** active
**Last updated:** 2026-06-30
**Related:** [[IsaacSim]], [[simulator-machine]], [[C1-AdaCoT]]

## Decision

Use **file-based IPC between two separate processes** for closed-loop Isaac Sim recording with VLA inference, rather than a single process.

- **Process A** (`bw11_sim_isaac.py`, isaac6 env): Isaac Sim, camera capture, robot control
- **Process B** (`bw11_sim_vla.py`, openvla env): VLA inference, video stitching

## Why

The isaac6 conda env's PyTorch (bundled with IsaacSim 6.0.0.1) requires a newer NVIDIA driver than what's installed on the simulator (driver 12080). Calling `torch._C._cuda_init()` in isaac6 fails with "driver too old" even after IsaacSim initialises the CUDA context — the bundled torch uses a different CUDA runtime path.

The openvla env's PyTorch works fine because it was installed against the actual driver version. There is no single conda env where both Isaac Sim and the VLA model's PyTorch can co-exist with the current driver.

## IPC protocol

Shared directory `/tmp/bw11_ipc/` with flag files:
- `ego_frame.png` / `iso_frame.png` — latest frames (written by isaac process)
- `phase.txt` — current phase name
- `action.json` — `{"lin": x, "ang": y}` (written by VLA process)
- `frame_ready` — new frame available (isaac sets, VLA clears)
- `action_ready` — new action available (VLA sets, isaac clears)
- `vla_ready` — VLA process booted (VLA sets, isaac waits on)
- `done` — recording finished (isaac sets)

## Performance

- VLA inference latency: ~0.67s/frame on Blackwell (1.5 Hz closed-loop)
- Isaac Sim runs at 60 Hz, VLA queried every 6 steps (10 Hz nominal, 1.5 Hz actual due to inference time)
- No frame drops: isaac waits up to 5s for VLA response before timing out

## How to apply

Always launch isaac process first (it must boot and reach warmup before VLA can connect). Use two separate tmux sessions. If upgrading driver or conda env resolves the PyTorch conflict, can merge back into a single process.
