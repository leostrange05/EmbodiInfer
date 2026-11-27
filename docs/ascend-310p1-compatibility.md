# Ascend 310P1 Compatibility Notes

## 1. Tested environment

Platform: OrangePi AI Station
Architecture: Linux aarch64
OS: Ubuntu 22.04 LTS

Hardware:

- NPU: Ascend 310P1
- Device count: 1

Software:

- Python: 3.12.15
- PyTorch: 2.9.0+cpu
- torch_npu: 2.9.0
- LeRobot: 0.5.1
- Transformers: 5.3.0
- CANN runtime and OPP: 8.5.0
- Runtime: Docker, based on Ubuntu 22.04

Development container: `embodiinfer-ascend-cann85`.

## PI0.5 backend profile and code ownership

The optional `backend_profile="ascend310p1"` is the explicit PI0.5 policy
configuration for the validated 310P1 TorchAir path. It selects eager policy
and vision attention, TorchAir denoise and vision graphs, and prepared Expert,
Prefix FFN and vision weights. Prefix padding compaction stays disabled unless
`npu_prefix_bucket=576` is explicitly requested. The profile is opt-in;
existing calls without `backend_profile` retain their previous defaults.

```python
import torch_npu  # Register the NPU device before constructing the engine.
from embodiinfer import EngineConfig, EngineCore, make_policy

policy = make_policy(
    "pi05", checkpoint="/path/to/pi05_libero",
    load_device="cpu", low_cpu_mem_usage=True,
    backend_profile="ascend310p1",
    action_horizon=10, default_num_steps=10,
)
core = EngineCore(
    policy,
    EngineConfig(device="npu", max_batch_size=1, use_cuda_graph=False),
)
```

This is a configuration example, not a complete observation-to-action program.
The model is loaded on CPU and `EngineCore` moves it to NPU before entering eval
mode. The profile requires the actual execution device to be an Ascend310P1
with FP32 master weights. An incompatible device or contradictory execution
mode raises an error; it does not silently select an NVIDIA or generic NPU
backend. Optional features can be explicitly disabled with their existing flags.
The profile cannot be combined with CUDA `preset` / `optimizations`; legacy
`compile_backend` and individual NPU options remain supported without the profile.

`backend_profile` is policy-specific. It does not imply support for other Ascend
models, other vendors, or automatic device detection. New backends should add
their own explicit profile and validation rather than treating every `npu`
device as Ascend. The HTTP serving factory accepts the same policy option in
`policy_kwargs` within the existing adapter configuration.

The branch's changes since main are organized by ownership:

- `policies/pi05/` owns model, attention/precision compatibility, OpenPI
  checkpoint conversion and the PI0.5 serving/processor contracts.
- `policies/pi05/inference/` owns PI0.5 backend profiles, TorchAir scheduling,
  and guarded Prefix compaction; `backend/torch/npu_linear.py` owns NPU linear
  operations, and `models/prepared_linear.py` owns reusable derived weights.
- `policies/base.py` and `engine/serve/http_server.py` contain shared device and
  HTTP compatibility changes; CUDA implementations remain in their own paths.
- `tests/` contains compatibility, policy lifecycle and HTTP coverage. LIBERO
  environments and evaluation scripts are downstream, not part of this package.

## 2. Verified functionality

The following checks have passed in the environment above:

- PyTorch and torch_npu initialization and NPU availability
- FP16 matrix multiplication
- Eager attention using MatMul and Softmax
- SigLIP vision-tower eager attention during real pi0.5 inference
- NPU uniform RNG with seed reproducibility
- NPU Box-Muller normal-noise generation
- Native FP32 PI0.5 sinusoidal time embedding
- Pinned CPU-to-NPU asynchronous copies
- Real pi0.5 checkpoint single-request inference without monkey patches
- CPU and NPU action-output comparison with matched test inputs

The real pi0.5 action output has shape `(50, 32)`, contains only finite
values when checked on CPU, and passes the recorded CPU/NPU comparison
criterion `allclose(atol=0.01, rtol=0.01)`. The comparison used the same
checkpoint, fixed synthetic observation, action schedule, and explicit RNG
seeds. The largest recorded action difference was 0.00163; the relative RMSE
was 0.00366.

## 3. Compatibility behavior and limitations

### 3.1 Attention and SigLIP vision tower

Native PyTorch SDPA fails on the tested 310P1 environment with an NPU
operator or format compatibility error. Select `attention="eager"` for the
EmbodiInfer attention path.

When `vision_attention="eager"` is selected, PI0.5 also configures the loaded
Hugging Face SigLIP vision tower with its public
`set_attn_implementation("eager")` API. This is not a CPU fallback: the
standard eager attention calculation runs on NPU using MatMul and Softmax.
It aligns the Hugging Face submodule with the policy-level selection.

Eager attention is the verified correctness path. It can be slower than a
working SDPA or vendor-fused implementation, so a future performance effort
may evaluate those alternatives only after full PI0.5 semantic and numerical
regression testing.

### 3.2 Random normal generation

The tested runtime fails for direct NPU normal operations:

    torch.randn(..., device="npu")
    torch.randn_like(npu_tensor)

The failure is associated with `StatelessRandomNormalV2`. The supported
NPU `torch.rand` operation is therefore transformed into standard normal
noise with the Box-Muller formula. This is the default implementation for
`FlowVLAPolicy.new_noise` on NPU and keeps both random-number generation and
the transform on device.

Set the following variable only to force the diagnostic or recovery CPU path:

    EMBODIINFER_ASCEND_CPU_NOISE=1

The CPU fallback remains limited to `FlowVLAPolicy.new_noise`; it does not
replace PyTorch RNG globally. CPU and NPU generator streams are not assumed
to be numerically identical.

### 3.3 PI0.5 time embedding

The reference PI0.5 sinusoidal implementation uses a float64 reciprocal path
that is not supported by this NPU runtime. PI0.5 therefore computes an
algebraically equivalent FP32 formulation on NPU by default. The tested
embedding difference against the CPU float64 reference was at most
0.000432.

Set the following variable only to force the CPU reference implementation:

    EMBODIINFER_ASCEND_CPU_TIME_EMBEDDING=1

### 3.4 CPU-to-NPU copies

A pageable CPU tensor copied with `non_blocking=True` fails on this runtime.
`Pi05Batch.to("npu")` now pins CPU input tensors before submitting an
asynchronous NPU copy, and this is the default PI0.5 batch-transfer path.
The full synchronous and asynchronous PI0.5 action outputs passed
`allclose(atol=0.01, rtol=0.01)`.

Set the following variable only to force synchronous copies:

    EMBODIINFER_ASCEND_SYNC_COPY=1

### 3.5 IsFinite operator and management interface

An isolated NPU `torch.isfinite()` check still fails in the current runtime.
Validation copies the final action output to CPU before checking finiteness.
Production PI0.5 inference does not depend on the failing check.

`npu-smi` reports a DCMI management-plane error on this device, while ACL and
PyTorch NPU computation complete successfully. Treat this as a management
interface issue unless it begins to affect compute workloads.

### 3.6 CANN environment initialization

Before running Python in the container, source the CANN environment without
enabling shell `set -u`. Append to `PYTHONPATH`; do not replace it, because
CANN packages such as `tbe` must remain visible:

    source /opt/ascend85/cann/set_env.sh
    export PYTHONPATH=/workspace/EmbodiInfer:/workspace/tests/ascend:$PYTHONPATH

## 4. Validated inference configuration

Use these settings for the verified path:

- device: `npu`
- EmbodiInfer attention: `eager`
- PI0.5 vision attention: `eager`
- default NPU Box-Muller noise
- default NPU FP32 time embedding
- default pinned asynchronous batch copy
- `ASCEND_LAUNCH_BLOCKING` unset

CUDA graphs and CUDA-specific fused optimizations are not part of this path.

## 5. Functional performance observation

A real pi0.5 single-request forward pass completed in approximately 110
seconds, including target-device execution in this development environment.
This is a functional observation, not a steady-state benchmark. Initialization,
operator compilation, warmup, exact checkpoint revision, image inputs, decode
steps, and timing boundaries must be controlled before making a performance
claim.

## 6. Reproduction

The deployment workspace is outside this repository. On the OrangePi host,
run the prepared test with the CANN environment initialized:

    source /opt/ascend85/cann/set_env.sh
    export PYTHONPATH=/workspace/EmbodiInfer:/workspace/tests/ascend:$PYTHONPATH
    cd /workspace/tests/ascend
    /workspace/.home/envs/pi05-npu-py312-cann85/bin/python pi05_real_npu.py

The Ascend smoke and capability scripts currently reside outside the Git
repository. Preserve their output logs with the exact environment and
checkpoint details, or promote stable hardware-gated coverage into the
repository when CI or device access is available.

## 7. Remaining work

- Investigate an upstream CANN or torch_npu fix for native NPU `randn`
- Evaluate SDPA or vendor-fused attention only after full PI0.5 regression
- Investigate the NPU `torch.isfinite()` failure
- Diagnose the `npu-smi` DCMI management-plane error
- Add hardware-gated Ascend regression coverage to the repository
- Establish a reproducible container definition and steady-state benchmark
