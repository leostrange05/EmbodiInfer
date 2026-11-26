"""Cached NPU artifacts must be invalidated by the policy lifecycle."""

import pytest
import torch

from embodiinfer.models.prepared_linear import NpuPreparedLinear


@pytest.mark.parametrize("event", ["refit", "load", "train", "move"])
@pytest.mark.parametrize("tower", ["prefix", "vision"])
def test_policy_invalidates_prepared_and_compiled_state(make_pi05_policy, event, tower):
    policy = make_pi05_policy(device="cpu")
    if tower == "prefix":
        mlp = policy._prefix_tower.layers[0].mlp
        mlp.gate_proj = prepared = NpuPreparedLinear(mlp.gate_proj)
    else:
        model = policy._m.paligemma_with_expert.paligemma.model
        model.vision_tower = torch.nn.Sequential(torch.nn.Linear(4, 3))
        model.vision_tower[0] = prepared = NpuPreparedLinear(model.vision_tower[0], allow_bias=True)
    state = policy.state_dict()

    prepared._prepared_weight = torch.ones_like(prepared.weight)
    prepared._prepared_key = ("cached",)
    policy._npu_compiled_prefix_ffn = object()
    policy._npu_compiled_denoise = object()
    policy._npu_compiled_image = object()
    def eager_image(image):
        return image
    policy._npu_eager_image = eager_image

    if event == "refit":
        policy.on_refit(1)
    elif event == "load":
        policy.load_state_dict(state)
    elif event == "train":
        policy.train()
    else:
        policy.to(dtype=torch.float32)

    assert prepared._prepared_weight is None and prepared._prepared_key is None
    assert policy._npu_compiled_prefix_ffn is None
    assert policy._npu_compiled_denoise is None
    assert policy._npu_compiled_image is None
    assert policy._m.paligemma_with_expert.embed_image is eager_image
