# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import pytest
import torch
from pathlib import Path

_PATH = Path(__file__).parents[1] / 'src/mcore_bridge/model/modules/kernels/ple_kernels.py'
_SPEC = importlib.util.spec_from_file_location('ple_workspace_under_test', _PATH)
kernels = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kernels)


@pytest.mark.parametrize('total', [0, 1, 19, 257])
@pytest.mark.parametrize('groups', [2, 4])
@pytest.mark.parametrize('budget', [128, 1 << 30])
def test_norm_weight_gradient_matches_full_product(monkeypatch, total, groups, budget):
    torch.manual_seed(43)
    width = groups * 16
    gated = torch.randn(total, width)
    dnormed = torch.randn_like(gated)
    rstd = torch.rand(total, groups)
    expected = (dnormed * (gated.reshape(total, groups, 16) * rstd[..., None]).reshape(total, width)).sum(0)
    monkeypatch.setattr(kernels, '_PLE_BACKWARD_WORKSPACE_BYTES', budget)
    actual = kernels._ple_norm_weight_gradient(gated, dnormed, rstd, groups, torch.float32)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=2e-5)


def test_gate_partial_workspace_is_bounded():
    width = 10240
    rows = kernels._ple_backward_chunk_rows(width)
    assert 3 * rows * width * 4 <= kernels._PLE_BACKWARD_WORKSPACE_BYTES
    assert rows < 262144


@pytest.mark.skipif(not torch.cuda.is_available() or not kernels.HAVE_TRITON, reason='CUDA and Triton required')
@pytest.mark.parametrize('rows,seq_len', [(1, 37), (2, 257)])
def test_chunked_gate_backward_matches_single_chunk(monkeypatch, rows, seq_len):
    torch.manual_seed(47)
    groups, channels = 4, 16
    total, width = rows * seq_len, groups * channels
    shapes = [(total, width), (total, width), (total, channels), (width, ), (width, ), (width, ), (width, 1, 4)]
    values = [torch.randn(shape, device='cuda') * 0.2 for shape in shapes]
    single = [value.clone().requires_grad_() for value in values]
    chunked = [value.clone().requires_grad_() for value in values]
    gradient = torch.randn(total, width, device='cuda')
    monkeypatch.setattr(kernels, '_PLE_BACKWARD_WORKSPACE_BYTES', 1 << 30)
    expected = kernels.ple_gate_conv_triton(*single, groups, 1e-6, 3, seq_len)
    expected_grads = torch.autograd.grad(expected, single, gradient)
    monkeypatch.setattr(kernels, '_PLE_BACKWARD_WORKSPACE_BYTES', 4096)
    actual = kernels.ple_gate_conv_triton(*chunked, groups, 1e-6, 3, seq_len)
    actual_grads = torch.autograd.grad(actual, chunked, gradient)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference, atol=2e-4, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA peak-memory measurement required')
def test_norm_weight_gradient_peak_memory_is_bounded(monkeypatch):
    torch.manual_seed(53)
    total, groups, channels = 4096, 4, 128
    width = groups * channels
    gated = torch.randn(total, width, device='cuda')
    dnormed = torch.randn_like(gated)
    rstd = torch.rand(total, groups, device='cuda')
    peaks, results = [], []
    for budget in [1 << 30, 128 * 1024]:
        monkeypatch.setattr(kernels, '_PLE_BACKWARD_WORKSPACE_BYTES', budget)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = kernels._ple_norm_weight_gradient(gated, dnormed, rstd, groups, torch.float32)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - baseline)
        results.append(result)
    torch.testing.assert_close(results[1], results[0], atol=1e-4, rtol=2e-4)
    assert peaks[1] < peaks[0] / 2, peaks
