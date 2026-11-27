"""PI0.5 Ascend310P1 graph configuration and schedule cache identity."""

from __future__ import annotations

from collections.abc import Callable

import torch


def execution_signature() -> tuple:
    """Identify the NPU arithmetic modes used to derive a cached schedule."""
    return (
        torch.get_float32_matmul_precision(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.is_autocast_enabled("npu"),
        str(torch.get_autocast_dtype("npu")),
        torch.npu.matmul.allow_hf32,
        torch.npu.conv.allow_hf32,
    )


def _compile(region: Callable[..., torch.Tensor], *, frozen_parameter: bool = True) -> Callable[..., torch.Tensor]:
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    class Pi05NpuConfig(CompilerConfig):
        def as_dict(self) -> tuple[dict, dict]:
            local, global_options = super().as_dict()
            local["ge.exec.precision_mode"] = "cube_fp16in_fp32out"
            return local, global_options

    config = Pi05NpuConfig()
    config.experimental_config.frozen_parameter = frozen_parameter
    return torch.compile(
        region,
        backend=torchair.get_npu_backend(compiler_config=config),
        fullgraph=True,
        dynamic=False,
    )


def compile_denoise(step: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Compile the measured static denoise region with FP32 Cube outputs."""
    return _compile(step)


def compile_image(
    encoder: Callable[[torch.Tensor], torch.Tensor], *, frozen_parameter: bool = True
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Compile one camera's encoder, preserving per-camera execution and FP32 outputs."""
    return _compile(encoder, frozen_parameter=frozen_parameter)


def compile_prefix_ffn() -> Callable[..., torch.Tensor]:
    """Compile the shared 310P prefix FFN with explicit prepared weights."""
    import torch_npu
    from torch.nn import functional as F

    def packed_ffn(x, gate, up, down):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        value = F.gelu(torch_npu.npu_linear(x, gate), approximate="tanh") * torch_npu.npu_linear(x, up)
        return torch_npu.npu_linear(value, down).reshape(*shape[:-1], down.shape[0])

    return _compile(packed_ffn, frozen_parameter=False)
