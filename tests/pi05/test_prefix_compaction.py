"""Regression tests for the opt-in pi0.5 prefix compaction contract."""

import pytest
import torch

from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05.inference.prefix_compaction import compact_pi05_prefix
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy


def _inputs(valid: int):
    torch.manual_seed(7)
    hidden = torch.randn(1, 712, 8)
    pad = torch.arange(712).unsqueeze(0) < valid
    attention = torch.zeros(1, 712, dtype=torch.bool)
    return hidden, pad, attention


def test_compacts_only_padding_and_preserves_valid_attention():
    hidden, pad, attention = _inputs(528)
    short, short_pad, short_attention = compact_pi05_prefix(hidden, pad, attention, bucket=576)
    assert short.shape == (1, 576, 8)
    assert torch.equal(short, hidden[:, :576])
    assert torch.equal(short_pad, pad[:, :576])
    assert torch.equal(short_attention, attention[:, :576])
    # Valid prefix queries and downstream suffix queries attend to the same keys.
    query = torch.randn(1, 9, 8)
    before = (query @ hidden.transpose(-1, -2)).masked_fill(~pad[:, None, :], -torch.inf).softmax(-1) @ hidden
    after = (query @ short.transpose(-1, -2)).masked_fill(~short_pad[:, None, :], -torch.inf).softmax(-1) @ short
    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-6)


def test_boundary_and_unsafe_masks_fall_back():
    hidden, pad, attention = _inputs(576)
    assert compact_pi05_prefix(hidden, pad, attention, bucket=576)[0].shape[1] == 576
    for valid in (0, 577, 712):
        items = _inputs(valid)
        result = compact_pi05_prefix(*items, bucket=576)
        assert all(a is b for a, b in zip(result, items))
    items = _inputs(528)
    items[1][0, 30] = False
    result = compact_pi05_prefix(*items, bucket=576)
    assert all(a is b for a, b in zip(result, items))


def test_unsupported_shapes_and_bucket_fall_back():
    items = _inputs(528)
    assert all(a is b for a, b in zip(compact_pi05_prefix(*items, bucket=512), items))
    batched = tuple(t.expand(2, *t.shape[1:]) for t in items)
    assert all(a is b for a, b in zip(compact_pi05_prefix(*batched, bucket=576), batched))
    bad_attention = (items[0], items[1], items[2][:, :-1])
    assert all(a is b for a, b in zip(compact_pi05_prefix(*bad_attention, bucket=576), bad_attention))


@pytest.mark.parametrize("bucket", [True, 0, 512, 712, "576"])
def test_unsupported_bucket_is_rejected_before_checkpoint_load(bucket):
    with pytest.raises(ValueError, match="must be None or 576"):
        Pi05Policy(VLAPolicyConfig(), checkpoint=None, npu_prefix_bucket=bucket)


def test_compaction_requires_torchair():
    with pytest.raises(ValueError, match="requires compile_backend"):
        Pi05Policy(VLAPolicyConfig(), checkpoint=None, npu_prefix_bucket=576)
