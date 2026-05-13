# FlashRT Analysis and Customization Guide

This document explains what this workspace is for, the main techniques it uses,
how inference flows through the codebase, and the easiest way to start adapting
it for your own model or robot workload.

## 1. What This Project Does

FlashRT is a realtime CUDA inference engine for small-batch, latency-sensitive
AI workloads. Its main supported use case is VLA (vision-language-action)
robot control, with production-style frontends for Pi0, Pi0.5, Pi0-FAST, and
GROOT. The repository also contains Qwen3/Qwen3.6 LLM pipelines and experimental
world-model-related CUDA components.

The project is not shaped like TensorRT, vLLM, or a general graph compiler.
Instead, it is a hand-composed kernel runtime:

- Python selects a model frontend based on `config`, `framework`, and GPU type.
- The frontend loads checkpoint weights and allocates fixed GPU buffers.
- Calibration computes FP8 or FP4 scaling values for the checkpoint and input
  shape.
- A static CUDA graph captures the model forward path.
- Repeated inference calls replay the graph with new image or prompt data.

The target is predictable realtime latency rather than high-throughput
multi-user batching.

## 2. Main Techniques Used

### Hand-Written CUDA Kernels

The low-level implementation lives mainly under `csrc/` and is exposed through
compiled Python extensions in `flash_rt/`, especially:

- `flash_rt_kernels.so`: norm, activation, quantization, RoPE, GEMM wrappers,
  fused decoder operations, attention helpers, and utility kernels.
- `flash_rt_fa2.so`: vendored FlashAttention-2 path for RTX-class builds.
- `flash_rt_fp4.so`: NVFP4-specific kernels for SM100+ hardware.
- `libfmha_fp16_strided.so`: Thor/Hopper-style FMHA support.

Pipeline files call these kernels through pointer-based interfaces so CUDA graph
capture can replay the same operations without Python allocation overhead.

### Static CUDA Graph Replay

FlashRT preallocates buffers, captures a fixed forward pass, then replays it.
The key design rule is that captured forwards pass raw device pointers and small
Python primitives, not newly allocated tensors.

This is why the first inference can be slower: it may load weights, calibrate,
and capture the graph. Later calls are much faster because they reuse the same
captured graph.

### FP8 and NVFP4 Quantization

Most realtime VLA paths use FP8 activation and weight handling. Some Pi0.5 paths
support NVFP4 on SM100+ hardware. Calibration is handled by shared code in:

- `flash_rt/core/quant/calibrator.py`
- frontend-specific `_calibrate()` or `calibrate()` methods

Calibration values are cached under `~/.flash_rt/calibration/`, keyed by
checkpoint and sequence shape.

### Hardware Dispatch

The public API does not ask users to choose internal pipeline classes. It detects
the current GPU and resolves a frontend through:

- `flash_rt/api.py`
- `flash_rt/hardware/__init__.py`

Supported dispatch names are currently:

- `thor`: Jetson AGX Thor, SM110
- `rtx_sm120`: RTX 5090 / Blackwell consumer, SM120
- `rtx_sm89`: RTX 4090 / Ada, SM89

The dispatch table maps `(config, framework, arch)` to a concrete frontend
class such as `flash_rt.frontends.torch.pi05_rtx.Pi05TorchFrontendRtx`.

### Declarative Weight Loading

Weights are mapped from checkpoint tensors into runtime attributes through
declarative specs rather than scattered ad hoc loading code. Important files:

- `flash_rt/executors/weight_loader.py`
- `flash_rt/executors/torch_weights.py`
- `flash_rt/executors/jax_weights.py`
- model-specific specs such as `flash_rt/frontends/torch/_pi05_thor_spec.py`

For new models, the weight spec is usually one of the first files to implement.

### Attention Backend Protocol

Attention shapes are declared as sites, then routed through hardware-specific
backends:

- `flash_rt/hardware/backend.py`
- `flash_rt/hardware/thor/attn_backend.py`
- `flash_rt/hardware/rtx/attn_backend.py`

This keeps model pipelines from directly depending on every hardware attention
implementation.

## 3. Repository Map

Use this map to orient yourself before editing:

| Path | Purpose |
|---|---|
| `flash_rt/api.py` | Stable public API: `load_model()` and `VLAModel.predict()` |
| `flash_rt/hardware/__init__.py` | GPU detection and frontend dispatch table |
| `flash_rt/frontends/torch/` | Torch checkpoint frontends and weight specs |
| `flash_rt/frontends/jax/` | JAX/Orbax frontends and weight cache paths |
| `flash_rt/models/` | Model pipeline code that sequences kernels |
| `flash_rt/hardware/{rtx,thor}/` | Hardware-specific attention backends and shared primitives |
| `flash_rt/core/` | Calibration, CUDA graph, buffers, precision specs, common utilities |
| `flash_rt/executors/` | Weight loading and checkpoint transform helpers |
| `flash_rt/configs/` | Model metadata YAML files |
| `csrc/` | Main CUDA/C++ kernel source |
| `flash_wm/` | BAGEL/world-model-related CUDA and FP4/FP8 components |
| `examples/` | Runnable quickstarts, LIBERO evals, and OpenAI-compatible server examples |
| `training/` | LoRA, RL, value-function, and JAX/PyTorch training utilities |
| `docs/` | Architecture, install, calibration, kernel, and extension docs |
| `tests/` | Precision, calibration, integration, and benchmark checks |

## 4. Runtime Pipeline Flow

The normal VLA inference path looks like this:

```text
User code
  |
  | import flash_rt
  | model = flash_rt.load_model(...)
  v
flash_rt/api.py
  |
  | validate config/framework
  | detect or use requested hardware
  v
flash_rt/hardware/__init__.py
  |
  | resolve (config, framework, arch)
  v
Frontend class
  |
  | load checkpoint weights
  | build attention backend
  | allocate buffers
  | prepare tokenizer/prompt state
  | calibrate FP8/FP4 if needed
  | capture CUDA graph
  v
Model pipeline
  |
  | call CUDA kernels with stable device pointers
  v
Compiled extension modules
  |
  | execute fused kernels, GEMMs, attention, quantization
  v
Action output
```

On the first `predict()` call, the frontend may run prompt setup,
calibration, and graph capture. On later calls with the same prompt, it mostly
copies new input data into existing buffers and replays the graph.

## 5. Easy Start-With Method

### Step 1: Install and Build

Use a fresh Python environment, install editable, clone CUTLASS, then build:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[torch]"

git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass

cmake -B build -S .
cmake --build build -j
```

For JAX paths, use the JAX dependency set described in `docs/INSTALL.md`.

### Step 2: Verify Import and Kernels

```bash
python -c "
import flash_rt, torch
print('flash_rt:', flash_rt.__version__)
print('cuda capability:', torch.cuda.get_device_capability())
from flash_rt import flash_rt_kernels
print('kernels import: ok')
"
```

### Step 3: Run the Quickstart

Use your checkpoint directory:

```bash
python examples/quickstart.py \
    --checkpoint /path/to/checkpoint \
    --config pi05 \
    --framework torch
```

Useful variants:

```bash
# Force fresh calibration
python examples/quickstart.py --checkpoint /path/to/checkpoint --recalibrate

# Benchmark steady-state graph replay
python examples/quickstart.py --checkpoint /path/to/checkpoint --benchmark 20

# Select hardware explicitly for debugging
python examples/quickstart.py --checkpoint /path/to/checkpoint --hardware rtx_sm120
```

Pi0 and Pi0.5 need the PaliGemma tokenizer. Download it once:

```bash
bash scripts/download_paligemma_tokenizer.sh
```

### Step 4: Use the 3-Line API

```python
import flash_rt

model = flash_rt.load_model(
    checkpoint="/path/to/checkpoint",
    config="pi05",
    framework="torch",
)

actions = model.predict(
    images=[base_img, wrist_img],
    prompt="pick up the red block",
)
```

The first call may calibrate and capture. Later calls can omit `prompt` to reuse
the previous prompt:

```python
actions = model.predict(images=[base_img, wrist_img])
```

## 6. How to Customize for Your Own Purpose

### If You Only Want to Change Deployment Behavior

Start with:

- `examples/quickstart.py`
- `flash_rt/api.py`
- `docs/stable_api.md`

Common changes:

- Change `config` to `pi0`, `pi05`, `groot`, or `pi0fast`.
- Change `num_views` for one, two, or three camera inputs.
- Use `hardware="thor"`, `hardware="rtx_sm120"`, or `hardware="rtx_sm89"` for
  explicit backend testing.
- Use `model.calibrate(observations)` for dataset calibration when supported.
- Use `model.recalibrate()` after fine-tuning or moving to a new visual domain.

### If You Want to Adapt an Existing Model

Start from the closest existing implementation:

- Pi0.5-like model: `flash_rt/models/pi05/` and `flash_rt/frontends/torch/pi05_*`
- Pi0-like model: `flash_rt/models/pi0/` and `flash_rt/frontends/torch/pi0_*`
- GROOT-like model: `flash_rt/models/groot/` and `flash_rt/frontends/torch/groot_*`
- Autoregressive action tokens: `flash_rt/models/pi0fast/` and related frontends

Typical edit order:

1. Update or add a YAML config in `flash_rt/configs/`.
2. Update the weight spec for the checkpoint naming and tensor layout.
3. Adjust attention site specs if head counts, sequence lengths, or KV layout
   changed.
4. Modify the model pipeline to match the new forward pass.
5. Update the frontend to allocate the right buffers and call calibration.
6. Add a dispatch row in `flash_rt/hardware/__init__.py`.
7. Add precision or smoke tests under `tests/`.

### If You Want to Add a New Model

Read these in order:

1. `docs/adding_new_model.md`
2. `flash_rt/frontends/torch/_template/README.md`
3. `docs/stable_api.md`
4. `docs/calibration.md`
5. `docs/kernel_fusion.md`
6. `docs/plugin_model_template.md`

The recommended implementation path is:

1. Copy the frontend template.
2. Build `weights_spec.py` first.
3. Add attention site definitions.
4. Translate the model forward into a pipeline file.
5. Wire the frontend around the pipeline.
6. Register the frontend in `_PIPELINE_MAP`.
7. Compare output against the original PyTorch/JAX model before optimizing.

For one `(framework, hardware)` target, expect most of the work to be in the
pipeline and frontend files.

### If You Want to Serve an LLM

The stable `load_model()` API is VLA-focused. Qwen3/Qwen3.6 code exists in:

- `flash_rt/models/qwen3/`
- `flash_rt/models/qwen36/`
- `flash_rt/frontends/torch/qwen3_rtx.py`
- `flash_rt/frontends/torch/qwen36_rtx.py`
- `examples/qwen36_openai_server.py`
- `docs/qwen36_nvfp4.md`
- `docs/qwen36_usage.md`

Start from `examples/qwen36_openai_server.py` rather than the VLA quickstart.

## 7. Practical Customization Checklist

Use this checklist when starting a new adaptation:

- Confirm the target GPU maps to a supported architecture.
- Confirm the checkpoint format: safetensors for Torch, Orbax for JAX.
- Run `examples/quickstart.py` with a known-supported model before editing.
- Find the closest existing frontend and pipeline.
- List every source checkpoint tensor and map it in a weight spec.
- Identify every distinct attention shape and add site specs.
- Keep all forward-pass temporary buffers preallocated.
- Avoid allocations, CPU transfers, or sync calls inside captured forwards.
- Run first-light correctness against a reference model.
- Recalibrate after changing weights, model shape, or deployment domain.
- Add tests for precision, calibration, and dispatch.

## 8. Key Docs Already in This Repo

- `README.md`: project overview, performance claims, model support, build entry.
- `USAGE.md`: API parameters and usage patterns.
- `docs/INSTALL.md`: native and Docker installation details.
- `docs/architecture.md`: deeper explanation of the eight main components.
- `docs/adding_new_model.md`: full model-integration guide.
- `docs/stable_api.md`: public API contract.
- `docs/calibration.md`: FP8 calibration mechanics.
- `docs/kernel_fusion.md`: available kernels and fusion guidance.
- `docs/kernel_catalog.md`: kernel inventory.
- `docs/plugin_model_template.md`: external plugin pattern.
- `training/README.md`: training and fine-tuning workflows.

## 9. Recommended First Custom Project

The lowest-risk first customization is not adding a brand-new architecture. Start
by taking a known-supported model and changing only deployment inputs:

1. Build and run Pi0.5 with `examples/quickstart.py`.
2. Replace the random image inputs with your own camera frames.
3. Run a few prompts and record action shape/range.
4. Build a small script around `flash_rt.load_model()` and `model.predict()`.
5. Add dataset calibration with real observations.
6. Only after that, modify model weights or add a new frontend.

This path keeps the CUDA graph, calibration, and attention pieces stable while
you learn the public API and deployment assumptions.
