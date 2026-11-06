# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from mcore_bridge.model.modules import hyper_connection_gated as hc


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('rows', [1, 17, 257])
@pytest.mark.parametrize('streams', [2, 4])
def test_low_precision_mix_forward_and_gradients_match_eager(dtype, rows, streams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(37)
    width = 64
    values = [torch.randn(rows, streams * width, device=device, dtype=dtype) for _ in range(2)]
    actual_inputs = [x.clone().requires_grad_() for x in values]
    reference_inputs = [x.clone().requires_grad_() for x in values]
    actual = hc._mix_and_reduce(*actual_inputs, streams, width)
    up, residual = reference_inputs
    gates = up.sigmoid().reshape(rows, streams, width)
    expected = (gates * residual.reshape(rows, streams, width)).mean(-2)
    gradient = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual, actual_inputs, gradient)
    expected_grads = torch.autograd.grad(expected, reference_inputs, gradient)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_low_precision_down_mix_preserves_division_rounding(dtype):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(31)
    values = torch.randn(257, 64, device=device, dtype=dtype)
    actual_input = values.clone().requires_grad_()
    reference_input = values.clone().requires_grad_()
    actual = hc._mix_elementwise(actual_input, 4)
    expected = F.silu(reference_input / 4)
    gradient = torch.randn_like(expected)
    actual_grad = torch.autograd.grad(actual, actual_input, gradient)[0]
    expected_grad = torch.autograd.grad(expected, reference_input, gradient)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_fp32_helpers_keep_compiled_dispatch(monkeypatch):
    calls = []

    def elementwise(x, streams):
        calls.append('down')
        return F.silu(x / streams)

    def reduce(up, residual, streams, width):
        calls.append('up')
        return (up.sigmoid().unflatten(-1, (streams, width)) * residual.unflatten(-1, (streams, width))).mean(-2)

    monkeypatch.setattr(hc, '_mix_elementwise_compiled', elementwise)
    monkeypatch.setattr(hc, '_mix_and_reduce_compiled', reduce)
    x = torch.randn(3, 16)
    hc._mix_elementwise(x, 4)
    hc._mix_and_reduce(x, x, 4, 4)
    assert calls == ['down', 'up']
