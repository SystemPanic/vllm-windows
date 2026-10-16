# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Availability probes must reject unusable optional router dependencies."""

import builtins
from types import SimpleNamespace

import pytest

from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16


@pytest.mark.parametrize("missing", ["cutlass", "quack.compile_utils", "fcntl"])
def test_missing_router_dependency_disables_kernel(monkeypatch, missing):
    """A transitive POSIX import must fall back before model warmup."""
    imports = []
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name not in {"cutlass", "cutlass.cute", "quack.compile_utils"}:
            return real_import(name, *args, **kwargs)
        imports.append(name)
        if name == missing or (name == "quack.compile_utils" and missing == "fcntl"):
            raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)
        return SimpleNamespace(make_fake_tensor=object())

    monkeypatch.setattr(ll_bf16, "_cutedsl_available", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert not ll_bf16.is_available()
    first_probe = imports.copy()
    assert not ll_bf16.is_available()
    assert imports == first_probe


def test_usable_router_dependencies_keep_kernel_available(monkeypatch):
    """Checking Quack must retain the optimized path when it imports."""
    imports = []
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name not in {"cutlass", "cutlass.cute", "quack.compile_utils"}:
            return real_import(name, *args, **kwargs)
        imports.append(name)
        return SimpleNamespace(make_fake_tensor=object())

    monkeypatch.setattr(ll_bf16, "_cutedsl_available", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ll_bf16.is_available()
    assert "quack.compile_utils" in imports
    first_probe = imports.copy()
    assert ll_bf16.is_available()
    assert imports == first_probe
