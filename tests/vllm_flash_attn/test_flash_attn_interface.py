import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def flash_attn_interface(monkeypatch: pytest.MonkeyPatch):
    platform_module = ModuleType("vllm.platforms")
    platform_module.current_platform = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform_module)

    module_path = (
        Path(__file__).parents[2]
        / "vllm"
        / "vllm_flash_attn"
        / "flash_attn_interface.py"
    )
    spec = importlib.util.spec_from_file_location(
        "flash_attn_interface_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.FA4_AVAILABLE = True
    return module, platform_module


@pytest.mark.parametrize("capability", [90, 100, 110, 120])
def test_fa4_supported_capability_families(
    flash_attn_interface, capability, monkeypatch
):
    module, platform_module = flash_attn_interface
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    platform_module.current_platform = SimpleNamespace(
        is_device_capability_family=lambda family: family == capability
    )

    assert module._is_fa4_supported() == (True, None)


def test_fa4_rejects_unsupported_capability_family(
    flash_attn_interface, monkeypatch
):
    module, platform_module = flash_attn_interface
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    platform_module.current_platform = SimpleNamespace(
        is_device_capability_family=lambda family: False
    )

    supported, reason = module._is_fa4_supported()

    assert not supported
    assert "12.x" in reason


def test_fa4_requires_cutlass_dsl_runtime(flash_attn_interface, monkeypatch):
    module, _ = flash_attn_interface
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)

    assert module._is_fa4_supported() == (
        False,
        "FA4 requires the nvidia-cutlass-dsl runtime",
    )