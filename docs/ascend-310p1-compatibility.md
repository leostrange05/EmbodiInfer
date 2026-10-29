# Ascend 310P1 Compatibility Notes

## 1. Tested environment

Platform: OrangePi AI Station
Architecture: Linux aarch64
OS: Ubuntu 22.04 LTS

Hardware:
- NPU: Ascend 310P1
- Device count: 1

Software:
- Python: 3.11.17
- PyTorch: 2.7.1
- torch_npu: 2.7.1
- CANN Toolkit: 8.3.RC1
- Runtime: Docker, based on Ubuntu 22.04

Development image:
`embodiinfer/ascend-torch-dev:2.7.1-cann8.3`

Development container:
`embodiinfer-ascend-torch-dev`

## 2. Verified functionality

The following tests have passed:

- PyTorch / torch_npu initialization
- NPU availability detection
- FP16 matrix multiplication
- Eager attention using MatMul + Softmax
- CPU-to-NPU random tensor transfer
- MockFlowVLA single-request inference
- MockFlowVLA batched inference (batch size 8)
- CPU RNG regression
- End-to-end inference without global monkey patches

MockFlowVLA outputs:

- Single request: action chunk (50, 7)
- Batch of 8: eight action chunks, each (50, 7)

These results validate the synthetic MockFlowVLA execution
path, not real pi0.5 checkpoint inference.

## 3. Known compatibility issues

### 3.1 Scaled Dot Product Attention

Native PyTorch SDPA fails on the tested 310P1 environment
with an NPU operator/format compatibility error.

Workaround:

    attention="eager"

The existing EmbodiInfer EagerAttention implementation
uses PyTorch MatMul and Softmax operations.

This fallback has been verified for MockFlowVLA.
Pi0.5 attention semantics and masks require separate tests.

### 3.2 Random normal generation

The following operations fail when executed directly
on the tested NPU:

    torch.randn(..., device="npu")
    torch.randn_like(npu_tensor)

The failure is associated with:

    StatelessRandomNormalV2

Generating random tensors on CPU and transferring them
to NPU succeeds.

An opt-in fallback is implemented in:

    embodiinfer/policies/base.py

Enable it with:

    EMBODIINFER_ASCEND_CPU_NOISE=1

The fallback is limited to FlowVLAPolicy.new_noise().
It does not replace PyTorch RNG globally.

CPU and NPU generator streams are not assumed to be
numerically equivalent.

### 3.3 IsFinite operator compilation

An isolated NPU torch.isfinite() test failed during
online compilation due to a missing C standard header:

    fatal error: 'stdio.h' file not found

The current development container lacks libc6-dev.

For validation, transfer the tensor to CPU before
performing torch.isfinite().

This is a compiler-environment issue under investigation,
not evidence that all IsFinite implementations are
unsupported by Ascend 310P1.

### 3.4 CANN environment initialization

Direct invocation:

    docker exec <container> python ...

may fail to import torch_npu because libhccl.so
is not on the dynamic library search path.

Invoking Python through Bash works because the
configured BASH_ENV initializes the CANN environment.

Use the host-side helper script:

    ~/embodiinfer-ascend/scripts/npu_exec.sh

This helper is part of the local deployment workspace,
not currently part of the EmbodiInfer repository.

## 4. Validated inference configuration

Device:

    npu

Attention:

    eager

Noise fallback:

    EMBODIINFER_ASCEND_CPU_NOISE=1

Execution:

    ASCEND_LAUNCH_BLOCKING unset

CUDA Graphs and CUDA-specific fused optimizations
are not used in this verified path.

## 5. Functional performance observations

Normal asynchronous mode:

- Single request: approximately 27.20 s
- Batch size 8: approximately 26.90 s
- Total process wall time: approximately 235 s

These are preliminary measurements of a synthetic model.

They are not steady-state benchmarks:
initialization, operator compilation, warmup and
application-level timing boundaries have not been
fully separated.

Do not compare these values directly with published
CUDA benchmark results.

## 6. Reproduction

The current local deployment workspace is:

    ~/embodiinfer-ascend

The following commands are run on the OrangePi host.

Basic NPU check:

    ./scripts/npu_exec.sh \
      python -c 'import torch,torch_npu; print(torch.npu.is_available())'

MockFlowVLA inference:

    ./scripts/npu_exec.sh \
      env -u ASCEND_LAUNCH_BLOCKING \
      EMBODIINFER_ASCEND_CPU_NOISE=1 \
      python /workspace/tests/ascend/quickstart_npu_eager.py

The referenced smoke-test scripts currently reside
outside the Git repository. They must be preserved
separately or moved into the repository for fully
self-contained reproduction.

## 7. Remaining work

- Add automated Ascend regression tests
- Validate EagerAttention mask and numerical semantics
- Resolve missing C/C++ compilation headers
- Establish a stable, reproducible Docker image
- Investigate Python 3.12 and LeRobot compatibility
- Load a real pi0.5 checkpoint
- Validate pi0.5 prefix encoding and denoising
- Benchmark steady-state inference latency
