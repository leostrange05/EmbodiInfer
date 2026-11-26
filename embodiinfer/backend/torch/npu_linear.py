"""Optional Ascend prepared-weight linear operations."""

from __future__ import annotations

import torch


def prepare_weight(weight: torch.Tensor) -> torch.Tensor:
    """Prepare an immutable FP16 FRACTAL_NZ copy of an NPU master weight."""
    import torch_npu

    return torch_npu.npu_format_cast(weight.detach().half(), 29)


def linear(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Apply a prepared weight and optional FP32 bias, preserving the input rank."""
    import torch_npu

    shape = inputs.shape
    result = torch_npu.npu_linear(inputs.reshape(-1, shape[-1]), weight)
    if bias is not None:
        result = result + bias
    return result.reshape(*shape[:-1], weight.shape[0])
