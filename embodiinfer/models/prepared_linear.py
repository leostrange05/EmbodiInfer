"""Inference-only derived weights with checkpoint-compatible master parameters."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..backend.torch import npu_linear


class NpuPreparedLinear(nn.Linear):
    """Keep the original FP32 parameter and cache an FP16/NZ inference copy.

    The derived tensor is excluded from state_dict. Training, autograd and
    non-FP32 execution use the master parameter through the ordinary linear.
    Call prepare() before compiling an inference region.
    """

    def __init__(self, source: nn.Linear) -> None:
        nn.Module.__init__(self)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer("_prepared_weight", None, persistent=False)
        self._prepared_key: tuple | None = None
        self.training = source.training

    def clear(self) -> None:
        """Release the derived copy after refit, loading or migration."""
        self._prepared_weight = None
        self._prepared_key = None

    def prepare(self) -> None:
        """Refresh the derived weight outside the compiled inference region."""
        weight = self.weight
        if weight.device.type != "npu" or weight.dtype != torch.float32 or self.bias is not None:
            self.clear()
            return
        version = None if torch.is_inference(weight) else weight._version
        key = (id(weight), version, weight.data_ptr(), weight.device, weight.dtype)
        if key != self._prepared_key:
            with torch.no_grad():
                self._prepared_weight = npu_linear.prepare_weight(weight)
            self._prepared_key = key

    def _apply(self, fn: Any, recurse: bool = True) -> NpuPreparedLinear:
        self.clear()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        self.clear()
        super()._load_from_state_dict(*args, **kwargs)

    def train(self, mode: bool = True) -> NpuPreparedLinear:
        if mode:
            self.clear()
        return super().train(mode)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Use prepared weights for FP32 NPU inference, with an autograd fallback."""
        if (
            self.training
            or torch.is_grad_enabled()
            or inputs.device.type != "npu"
            or inputs.dtype != torch.float32
            or self.weight.dtype != torch.float32
            or self.bias is not None
        ):
            return F.linear(inputs, self.weight, self.bias)
        if not torch.compiler.is_compiling():
            self.prepare()
        if self._prepared_weight is None:
            raise RuntimeError("Prepare NPU linear weights before compiling inference")
        return npu_linear.linear(inputs, self._prepared_weight)
