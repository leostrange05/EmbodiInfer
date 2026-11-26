"""CPU contracts for optional NPU-derived linear weights."""

import torch
import sys
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F

from embodiinfer.models.prepared_linear import NpuPreparedLinear
from embodiinfer.backend.torch import npu_linear


def test_preserves_parameters_state_dict_and_cpu_gradients():
    torch.manual_seed(123)
    source = nn.Linear(4, 3, bias=True)
    prepared = NpuPreparedLinear(source)
    assert prepared.weight is source.weight
    assert prepared.bias is source.bias
    assert set(prepared.state_dict()) == {"weight", "bias"}
    inputs = torch.randn(2, 4, requires_grad=True)
    result = prepared(inputs)
    torch.testing.assert_close(result, F.linear(inputs, source.weight, source.bias), rtol=0, atol=0)
    result.sum().backward()
    assert inputs.grad is not None and source.weight.grad is not None and source.bias.grad is not None


def test_cpu_eval_uses_master_and_prepare_does_not_require_torch_npu():
    torch.manual_seed(321)
    source = nn.Linear(4, 3, bias=False)
    prepared = NpuPreparedLinear(source).eval()
    inputs = torch.randn(2, 4)
    with torch.no_grad():
        prepared.prepare()
        result = prepared(inputs)
    assert prepared._prepared_weight is None
    torch.testing.assert_close(result, F.linear(inputs, source.weight), rtol=0, atol=0)


def test_cache_is_not_serialized_and_is_cleared_on_lifecycle_changes():
    layer = NpuPreparedLinear(nn.Linear(4, 3, bias=False)).eval()
    state = layer.state_dict()
    layer._prepared_weight = torch.ones(3, 4)
    layer._prepared_key = ("sentinel",)
    assert "_prepared_weight" not in layer.state_dict()
    layer.load_state_dict(state)
    assert layer._prepared_weight is None and layer._prepared_key is None
    layer._prepared_weight = torch.ones(3, 4)
    layer._prepared_key = ("sentinel",)
    layer.train()
    assert layer._prepared_weight is None and layer._prepared_key is None
    layer._prepared_weight = torch.ones(3, 4)
    layer._prepared_key = ("sentinel",)
    layer.to(dtype=torch.float64)
    assert layer._prepared_weight is None and layer._prepared_key is None


def test_optional_bias_preparation_preserves_parameters_and_cpu_fallback():
    source = nn.Linear(4, 3, bias=True)
    layer = NpuPreparedLinear(source, allow_bias=True).eval()
    x = torch.randn(2, 5, 4)
    assert layer.weight is source.weight and layer.bias is source.bias
    with torch.no_grad():
        layer.prepare()
        torch.testing.assert_close(layer(x), F.linear(x, source.weight, source.bias), atol=0, rtol=0)
    assert layer._prepared_weight is None
    assert set(layer.state_dict()) == {"weight", "bias"}


def test_prepared_backend_adds_bias_before_restoring_shape(monkeypatch):
    backend = SimpleNamespace(npu_linear=lambda x, w: F.linear(x, w))
    monkeypatch.setitem(sys.modules, "torch_npu", backend)
    x = torch.randn(2, 5, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)
    expected = (F.linear(x.reshape(-1, 4), weight) + bias).reshape(2, 5, 3)
    torch.testing.assert_close(npu_linear.linear(x, weight, bias), expected, atol=0, rtol=0)
    torch.testing.assert_close(npu_linear.linear(x, weight), F.linear(x, weight), atol=0, rtol=0)
