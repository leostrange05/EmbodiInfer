"""Policy-level execution profiles; operator implementations stay in their backends."""

from __future__ import annotations

from typing import Any

_LEGACY_DEFAULTS: dict[str, Any] = {
    "attention": "sdpa",
    "vision_attention": "sdpa",
    "compile_backend": "none",
    "native_inference": False,
    "npu_prepared_ffn": True,
    "npu_compile_vision": True,
    "npu_prefix_ffn_graph": False,
    "npu_prepared_vision": False,
}

# Only configurations actually supported by a policy belong here. A future
# hardware profile can select its own options without changing other defaults.
_PROFILES: dict[str, dict[str, Any]] = {
    "ascend310p1": {
        "attention": "eager",
        "vision_attention": "eager",
        "compile_backend": "torchair",
        "native_inference": False,
        "npu_prepared_ffn": True,
        "npu_compile_vision": True,
        "npu_prefix_ffn_graph": True,
        "npu_prepared_vision": True,
    },
}
_REQUIRED = frozenset({"attention", "vision_attention", "compile_backend", "native_inference"})


def resolve_backend_profile(profile: str | None, **options: Any) -> dict[str, Any]:
    """Resolve an optional named profile, preserving explicitly supplied optional features.

    ``None`` preserves the existing Pi05 defaults. The named 310P1 profile owns
    its execution modes, while individual NPU optimizations may be disabled.
    Device validation occurs when the policy enters evaluation mode.
    """
    if profile is not None and profile not in _PROFILES:
        raise ValueError(f"Unknown Pi05 backend_profile {profile!r}; choose from {tuple(_PROFILES)}")
    unknown = options.keys() - _LEGACY_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown Pi05 backend profile options: {sorted(unknown)}")
    defaults = _LEGACY_DEFAULTS if profile is None else _PROFILES[profile]
    resolved = dict(defaults)
    for name, value in options.items():
        if value is None:
            continue
        if profile is not None and name in _REQUIRED and value != defaults[name]:
            raise ValueError(f"backend_profile={profile!r} requires {name}={defaults[name]!r}")
        resolved[name] = value
    return resolved
