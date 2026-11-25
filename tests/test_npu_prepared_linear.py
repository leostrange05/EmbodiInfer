"""CPU contracts for optional NPU-derived linear weights."""

import torch
from torch import nn
from torch.nn import functional as F

from embodiinfer.models.prepared_linear import NpuPreparedLinear


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
