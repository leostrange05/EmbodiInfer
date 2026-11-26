"""Reject unsupported combinations before loading a PI0.5 checkpoint."""

import pytest

from embodiinfer.policies.config import VLAPolicyConfig
from embodiinfer.policies.pi05.modeling_pi05 import Pi05Policy


@pytest.mark.parametrize("option", ["npu_prefix_ffn_graph", "npu_prepared_vision"])
def test_prepared_graph_features_require_torchair(option):
    with pytest.raises(ValueError, match="require TorchAir"):
        Pi05Policy(VLAPolicyConfig(), checkpoint=None, **{option: True})


def test_prepared_vision_requires_compiled_vision():
    with pytest.raises(ValueError, match="requires compiled"):
        Pi05Policy(
            VLAPolicyConfig(), checkpoint=None, attention="eager", compile_backend="torchair",
            npu_prepared_vision=True, npu_compile_vision=False,
        )


@pytest.mark.parametrize("bucket", [True, 0, 512, 712, "576"])
def test_prefix_bucket_rejects_unsupported_values(bucket):
    with pytest.raises(ValueError, match="must be None or 576"):
        Pi05Policy(VLAPolicyConfig(), checkpoint=None, npu_prefix_bucket=bucket)


def test_prefix_bucket_requires_torchair():
    with pytest.raises(ValueError, match="requires compile_backend"):
        Pi05Policy(VLAPolicyConfig(), checkpoint=None, npu_prefix_bucket=576)
