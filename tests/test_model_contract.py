"""Dtype probe/guard contract: measurement never raises, assert refuses mismatch."""

import pytest

pytest.importorskip("torch")

import torch

from cyberft.errors import DependencyError
from cyberft.model import assert_trainable_dtypes, probe_trainable_dtypes


class _Param:
    def __init__(self, dtype, grad_dtype, trainable=True, numel=8):
        self.dtype = dtype
        self.grad_dtype = grad_dtype
        self.requires_grad = trainable
        self._numel = numel

    def numel(self):
        return self._numel


class _Model:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return list(self._params)


def test_probe_reports_zero_mismatch_when_clean():
    model = _Model(
        [
            ("lora_A", _Param(torch.float16, torch.float16)),
            ("lora_B", _Param(torch.float16, torch.float16)),
            ("frozen", _Param(torch.float32, torch.float32, trainable=False)),
        ]
    )
    report = probe_trainable_dtypes(model, torch.float16)
    assert report["trainable_count"] == 2
    assert report["mismatched_count"] == 0
    assert report["mismatched"] == []
    assert report["trainable_params"] == 16


def test_probe_flags_dtype_and_grad_dtype_mismatches():
    model = _Model(
        [
            ("ok", _Param(torch.float16, torch.float16)),
            ("upcast", _Param(torch.float32, torch.float32)),
            ("grad_only", _Param(torch.float16, torch.float32)),
        ]
    )
    report = probe_trainable_dtypes(model, torch.float16)
    assert report["mismatched_count"] == 2
    assert sorted(report["mismatched"]) == ["grad_only", "upcast"]


def test_assert_passes_when_clean():
    model = _Model([("a", _Param(torch.bfloat16, torch.bfloat16))])
    assert_trainable_dtypes(model, torch.bfloat16)  # must not raise


def test_assert_raises_naming_offenders():
    model = _Model(
        [
            ("ok", _Param(torch.float16, torch.float16)),
            ("bad_adapter", _Param(torch.float32, torch.float32)),
        ]
    )
    with pytest.raises(DependencyError, match="bad_adapter"):
        assert_trainable_dtypes(model, torch.float16)
