# simulator-machine

**Type:** infrastructure
**Status:** active
**Last updated:** 2026-07-04
**Related:** [[openvla-env]], [[IsaacSim]], [[NovaCarter]], [[overview]]

## Summary

Primary development machine for VLA4AMR. Machine name: `cvit-car-simulator@simulator`.
Two GPUs: RTX 4060 Ti (CARLA simulator work) + RTX 5000 Pro Blackwell 48GB (VLA4AMR work).
Dr. Shankar's instructions: use the right GPU, back up to Mobility Server, create dedicated folder.

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 9 7950X — 16-core, 32-thread, 2.8 GHz |
| RAM | 62 GB total (32 GB actual, extension in-process) |
| GPU 0 | RTX 4060 Ti 16GB — PCI 01:00.0 — CARLA simulator |
| GPU 1 | RTX 5000 Pro Blackwell 48GB — PCI 02:00.0 — VLA4AMR |
| Disk | 915 GB total, 259 GB free (extension in-process) |
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Conda | 25.3.1 at ~/miniconda3 |
| Existing envs | base, carla, nerfstudio |

## GPU assignment

- `CUDA_VISIBLE_DEVICES=0` → RTX PRO 5000 Blackwell for VLA4AMR work
- `CUDA_VISIBLE_DEVICES=1` → RTX 4060 Ti for CARLA work
- **Warning:** nvidia-smi uses PCI order (GPU 0 = 4060 Ti, GPU 1 = 5000 Pro) but CUDA uses FASTEST_FIRST order (GPU 0 = 5000 Pro, GPU 1 = 4060 Ti). Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` to match nvidia-smi, or remember the flip.

## Installation status

| Step | Status | Notes |
|------|--------|-------|
| Ubuntu 24.04 | ✅ | Already installed |
| NVIDIA driver | ✅ | 570.211.01 (nvidia-driver-570-open), CUDA 12.6 — downgraded 2026-06-13 (595.71.05 incompatible with Blackwell CC 12.0 in Isaac Sim) |
| CUDA toolkit 12.6 | ✅ | Installed 2026-05-26, nvcc at /usr/local/cuda-12.6/bin/nvcc, PATH updated |
| ROS 2 Jazzy | ✅ | Installed 2026-05-26, `source /opt/ros/jazzy/setup.bash`, ROS_DISTRO=jazzy |
| Isaac Sim 4.5.0 | ✅ | conda env: isaac (Python 3.10) — deprecated, use isaac6 |
| Isaac Sim 6.0.0.1 | ✅ | conda env: isaac6 (Python 3.12), `isaacsim[all,extscache]` from pypi.nvidia.com, SimulationApp confirmed 2026-06-13 |
| Conda env: openvla | ✅ | Python 3.10, torch 2.11.0+cu128 (cu124 dropped — Blackwell needs sm_120), verified 2026-05-26 — see env notes below |
| OpenVLA-OFT | ✅ | Installed 2026-06-06 via `pip install -e . --no-deps` + `flash-attn==2.5.5`; pyproject.toml torch pin relaxed to `>=2.2.0` to protect cu128 build |
| OpenVLA-7B weights | ✅ | 15 GB, 3 safetensors shards, ~/VLA4AMR/checkpoints/openvla-7b, downloaded 2026-06-06 |

## Isaac Sim launch — working config (2026-06-13)

**Driver requirement:** `nvidia-driver-570-open` (570.211.01). Driver 595.x crashes with Blackwell CC 12.0 (TLAS overflow).

**Working command:**
```bash
DISPLAY=:1 /home/cvit-car-simulator/miniconda3/envs/isaac/bin/python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'no_window': True})
print('SimulationApp: OK')
app.close()
" 2>&1 | tee ~/Desktop/isaac_out.txt
```

**Key rules:**
- `DISPLAY=:1` required — Xorg runs at :1 (check with `echo $DISPLAY`)
- `no_window: True` required — otherwise URDF build_ui crashes headless
- Never set `CUDA_VISIBLE_DEVICES` for Isaac Sim — carb.cudainterop will crash
- Use direct Python path (`miniconda3/envs/isaac/bin/python`) not `conda run` — `conda run` triggers omni.kit_app pre-import error
- Blackwell = Omniverse GPU 0 (fastest-first), 4060 Ti = GPU 1 — opposite of nvidia-smi PCI order
- iRay "CC 12.0 unsupported" warning is expected and non-fatal with driver 570
- First launch: ~230s shader compilation (subsequent launches are faster)

## Folder structure (planned)

```
~/VLA4AMR/
├── wiki/          ← this wiki (backup to Mobility Server)
├── code/          ← all scripts (sync with Mobility Server)
├── data/          ← datasets (symlink to larger disk when extended)
├── checkpoints/   ← model weights
└── logs/          ← experiment logs
```

## Next commands (in order)

```bash
# Step 1–4 DONE (2026-05-26): driver 595.71.05, CUDA 13.2, both GPUs confirmed

# Step 5 DONE (2026-05-26): ROS 2 Jazzy installed

# Step 6 — install CUDA toolkit (12.4 for torch+cu124 compatibility)
# Note: driver reports CUDA 13.2 but torch+cu124 needs toolkit 12.4
# Install both and let torch pin to 12.4

# Step 7 — Isaac Sim via Omniverse Launcher
```

## Mobility Server backup

Per Dr. Shankar: all code and data must be backed up to Mobility Server.
Set up rsync or git remote to Mobility Server for `~/VLA4AMR/` directory.

## Isaac Sim 6.0.0.1 install (conda env: isaac6) — DONE 2026-06-13

```bash
conda create -n isaac6 python=3.12 -y
conda activate isaac6

# CRITICAL: must install both [all] AND [extscache] — [all] alone is missing anim/schema extensions
pip install "isaacsim[all]==6.0.0.1" --extra-index-url https://pypi.nvidia.com
pip install "isaacsim[extscache]==6.0.0.1" --extra-index-url https://pypi.nvidia.com
# extscache installs: isaacsim-extscache-kit, isaacsim-extscache-kit-sdk, isaacsim-extscache-physics
```

**Working launch command (isaac6):**
```bash
DISPLAY=:1 /home/cvit-car-simulator/miniconda3/envs/isaac6/bin/python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'no_window': True})
print('SimulationApp 6.0.0: OK', flush=True)
app.close()
" 2>&1 | tee ~/Desktop/isaac6_out.txt
```

**Key differences from 4.5.0:**
- Python 3.12 (matches Jazzy) — ROS2 bridge works natively
- Startup: ~9.5s (vs ~230s in 4.5.0)
- Blackwell sm_120 supported natively (no driver downgrade below 570 needed)
- Warp cuda:0 = RTX PRO 5000 Blackwell, cuda:1 = RTX 4060 Ti (same FASTEST_FIRST ordering)
- Driver R570 (570.211.01) works; R580 recommended but not required

## openvla env — known issues (updated 2026-07-04)

| Issue | Status | Fix |
|-------|--------|-----|
| transformers 5.x vs CorrectNav 4.40 | ✅ Patched | 5 files patched — see [[decision-bw15-correctnav-fullstack]] |
| TensorBoard broken (numpy 2.x / ml_dtypes / TF) | ⚠️ Workaround | `--report_to none` in training script; TF built against numpy 1.x, env has numpy 2.2.6 |
| ninja pip wrapper shadowing `/usr/bin/ninja` | ✅ Fixed | Removed `~/.local/bin/ninja` wrapper; `pip install ninja` inside openvla env |
| DeepSpeed cpu_adam JIT | ✅ Working | Built and cached at `~/.cache/torch_extensions/py310_cu128/cpu_adam/cpu_adam.so` |
| tiktoken missing | ✅ Fixed | `pip install tiktoken` (Qwen2 tokenizer backend) |
| HF model downloads | ✅ Complete | LLaVA-Video-7B-Qwen2 full snapshot + SigLIP-SO400M at `~/.cache/huggingface/hub/` |

**RAM constraint**: Machine has ~62 GB RAM. With `mm_tunable_parts=mm_language_model`, ZeRO-2
CPU optimizer offload needs ~30 GB RAM → kernel OOM kills python3.10 (confirmed via dmesg).
Fixed by restricting tunable parts to `mm_mlp_adapter` (~17 MB trainable).

**CRITICAL**: Do not use `offload_optimizer` in `bw15_zero2.json` when trainable params
are small — pointless overhead and contributed to RAM pressure.

## CorrectNav install (openvla env)

```bash
cd ~/VLA4AMR/CorrectNav-main
pip install -e . --no-deps   # already done
pip install tiktoken          # for Qwen2 tokenizer
pip install ninja             # for DeepSpeed JIT ops
```

Files at: `~/VLA4AMR/CorrectNav-main/`
Training script: `~/VLA4AMR/code/bw15_train.sh`
DeepSpeed config: `~/VLA4AMR/code/bw15_zero2.json`

## Open questions

- What is the Mobility Server address/protocol? (confirm with Dr. Shankar)
- ~~Is ROS 2 Jazzy already installed?~~ **Resolved 2026-05-26:** Installed (ros-jazzy-desktop + ros-dev-tools)
- ~~Will RTX 5000 Pro appear as GPU 1 or GPU 0 after driver install?~~ **Resolved 2026-05-26:** GPU 0 = RTX 4060 Ti, GPU 1 = RTX PRO 5000 Blackwell — matches planned assignment
- ~~Will nvidia-driver-595-open support RTX 5000 Pro Blackwell?~~ **Resolved 2026-05-26:** Yes (but 595 NFB branch not validated — must use R570 or R580)
- ~~ROS2 bridge Jazzy incompatibility in 4.5.0?~~ **Root cause 2026-06-13:** Python version mismatch (4.5.0=py3.10, Jazzy=py3.12). Fixed in 6.0.0 (py3.12 matches Jazzy).
- ~~**Isaac Sim 6.0.0.1 headless from SSH crashes**~~ **Root cause (fully resolved 2026-08-04):** 5 bugs, all fixed. See log entry [2026-08-04]. Short form:
  1. `DISPLAY=:1` not `:0` (Xorg at `:1`)
  2. `XAUTHORITY=/run/user/1000/gdm/Xauthority` (SSH lacks X auth)
  3. No `CUDA_VISIBLE_DEVICES` for Isaac (carb.cudainterop crashes; use `active_gpu=0` in SimulationApp instead)
  4. Enable `isaacsim.ros2.bridge` AFTER `open_stage()`, not before (pre-wired ROS2 OmniGraph nodes crash if bridge loads during scene init)
  5. `cv2.namedWindow` in headless mode — wrap in try/except

## Isaac Sim 6.0.0.1 — GPU / display config reference (updated 2026-08-03)

| Variable | Value | Why |
|---|---|---|
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | Aligns CUDA order with Vulkan/nvidia-smi (4060 Ti=0, Blackwell=1) |
| `CUDA_VISIBLE_DEVICES` | `0` for Isaac, `1` for VLA | Isaac on 4060 Ti, VLA inference on Blackwell |
| `VK_ICD_FILENAMES` | `/usr/share/vulkan/icd.d/nvidia_icd.json` | Force NVIDIA ICD; blocks AMD iGPU in SSH/headless Vulkan enumeration |
| `DISPLAY` | `:1` | Xorg session (GPU-accelerated) — **:1 not :0**; confirmed via `/tmp/.X11-unix/X1` |
| `XAUTHORITY` | `/run/user/1000/gdm/Xauthority` | X auth cookie for GNOME Xorg session — SSH sessions don't inherit this |
| `DBUS_SESSION_BUS_ADDRESS` | `unix:path=/run/user/1000/bus` | ROS2 bridge needs DBUS for daemon comms |
| `XDG_RUNTIME_DIR` | `/run/user/1000` | Required by ROS2 bridge extension |
| `XDG_SESSION_TYPE` | `x11` | Tells GNOME stack this is X11 session |
| `QT_QPA_PLATFORM` | `xcb` | Force Qt to X11 over Wayland |
| SimulationApp `active_gpu` | `0` | Selects 4060 Ti as Vulkan renderer |
| SimulationApp `headless` | `True` | No Kit window — safe from GNOME terminal |
| SimulationApp `no_window` | `True` | Suppresses Kit's own render surface |
| `rep.orchestrator.step()` | Before every `rgb_ann.get_data()` | Sync OmniGraph image pipeline; prevents race in `libomni.graph.image.core.plugin.so` |
