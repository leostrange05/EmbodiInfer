"""Optional Ascend prepared-weight linear operations."""

from __future__ import annotations

import torch


def prepare_weight(weight: torch.Tensor) -> torch.Tensor:
    """Prepare an immutable FP16 FRACTAL_NZ copy of an NPU master weight."""
    import torch_npu

    return torch_npu.npu_format_cast(weight.detach().half(), 29)


def linear(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply a prepared weight while retaining FP32 input/output interfaces."""
    import torch_npu

    shape = inputs.shape
    result = torch_npu.npu_linear(inputs.reshape(-1, shape[-1]), weight)
    return result.reshape(*shape[:-1], weight.shape[0])
