"""PI0.5 backend profile selection remains explicit and device-scoped."""

from types import SimpleNamespace

import pytest
import torch

from embodiinfer import EngineConfig, EngineCore, make_policy
from embodiinfer.policies.pi05 import modeling_pi05
from embodiinfer.policies.pi05.inference.backend_profiles import resolve_backend_profile


def test_default_and_explicit_legacy_options_remain_unchanged():
    defaults = resolve_backend_profile(None)
    assert defaults["attention"] == "sdpa"
    assert defaults["compile_backend"] == "none"
    assert defaults["npu_prefix_ffn_graph"] is False
    assert defaults["npu_prepared_vision"] is False
    legacy = resolve_backend_profile(None, attention="eager", compile_backend="torchair")
    assert legacy["attention"] == "eager" and legacy["compile_backend"] == "torchair"


def test_ascend_profile_and_optional_feature_override():
    options = resolve_backend_profile("ascend310p1", npu_prefix_ffn_graph=False)
    assert options["attention"] == options["vision_attention"] == "eager"
    assert options["compile_backend"] == "torchair" and options["native_inference"] is False
    assert options["npu_prepared_ffn"] and options["npu_compile_vision"] and options["npu_prepared_vision"]
    assert options["npu_prefix_ffn_graph"] is False


@pytest.mark.parametrize(
    "options", [{"attention": "sdpa"}, {"compile_backend": "inductor"}, {"native_inference": True}]
)
def test_ascend_profile_rejects_conflicting_execution_modes(options):
    with pytest.raises(ValueError, match="requires"):
        resolve_backend_profile("ascend310p1", **options)


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown Pi05 backend_profile"):
        resolve_backend_profile("other_npu")


def test_public_builder_passes_resolved_options_and_keeps_compaction_opt_in(monkeypatch):
    captured = {}

    class FakePolicy:
        def __init__(self, config, **kwargs):
            captured.update(kwargs)
            self._lerobot = SimpleNamespace(
                config=SimpleNamespace(max_action_dim=32, chunk_size=10, num_inference_steps=10)
            )

    monkeypatch.setattr(modeling_pi05, "Pi05Policy", FakePolicy)
    policy = make_policy("pi05", checkpoint="unused", backend_profile="ascend310p1")
    assert policy._backend_profile == "ascend310p1"
    assert captured["compile_backend"] == "torchair" and captured["attention"] == "eager"
    assert captured["npu_prefix_ffn_graph"] and captured["npu_prepared_vision"]
    assert captured["npu_prefix_bucket"] is None
    modeling_pi05._build_pi05(checkpoint="unused", backend_profile="ascend310p1", npu_prefix_bucket=576)
    assert captured["npu_prefix_bucket"] == 576


def test_profile_and_existing_optimizations_are_mutually_exclusive():
    with pytest.raises(ValueError, match="cannot be combined"):
        modeling_pi05._build_pi05(checkpoint="unused", backend_profile="ascend310p1", preset="strict")


def test_profile_rejects_cpu_instead_of_silently_falling_back():
    fake = SimpleNamespace(
        _backend_profile="ascend310p1",
        _m=SimpleNamespace(action_in_proj=SimpleNamespace(weight=torch.zeros(1))),
    )
    with pytest.raises(ValueError, match="requires an Ascend310P1 NPU"):
        modeling_pi05.Pi05Policy._validate_backend_profile_device(fake)


def test_engine_rejects_profile_on_actual_cpu_policy(make_pi05_policy):
    policy = make_pi05_policy(device="cpu")
    policy._backend_profile = "ascend310p1"
    with pytest.raises(ValueError, match="requires an Ascend310P1 NPU"):
        EngineCore(policy, EngineConfig(device="cpu", use_cuda_graph=False))
    with pytest.raises(ValueError, match="requires an Ascend310P1 NPU"):
        policy.encode_prefix(None)


def test_profile_rejects_other_ascend_device(monkeypatch):
    weight = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.float32)
    fake = SimpleNamespace(
        _backend_profile="ascend310p1",
        compile_backend="torchair",
        _m=SimpleNamespace(action_in_proj=SimpleNamespace(weight=weight)),
    )
    fake._validate_backend_profile_device = lambda: modeling_pi05.Pi05Policy._validate_backend_profile_device(
        fake
    )
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(get_device_name=lambda device: "Ascend910B"), raising=False
    )
    with pytest.raises(ValueError, match="targets Ascend310P1"):
        modeling_pi05.Pi05Policy._prepare_npu_inference(fake)
