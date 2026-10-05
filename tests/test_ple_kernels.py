"""PLE fused numerics and opt-in GPU regressions beyond the int32 boundary."""
import importlib.util
import math
import os
import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

_KERNEL_PATH = Path(__file__).parents[1] / 'src/mcore_bridge/model/modules/kernels/ple_kernels.py'
_spec = importlib.util.spec_from_file_location('ple_kernels_under_test', _KERNEL_PATH)
kernels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kernels)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not kernels.HAVE_TRITON, reason='PLE kernels require CUDA and Triton')


@pytest.mark.parametrize('rows,seq_len', [(1, 19), (2, 13)])
def test_ple_forward_backward_match_torch(rows, seq_len):
    torch.manual_seed(42)
    n, c, k, dilation, eps = 2, 16, 4, 3, 1e-6
    t, w = rows * seq_len, n * c
    shapes = [(t, w), (t, w), (t, c), (w, ), (w, ), (w, ), (w, 1, k)]
    inputs = [(torch.randn(shape, device='cuda') * 0.2).requires_grad_() for shape in shapes]
    query, key, value, wk, wq, wc, conv = inputs

    def norm(x, weight):
        groups = x.reshape(t, n, c)
        normalized = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + eps)
        return normalized * (1 + weight.reshape(n, c))

    score = (norm(key, wk) * norm(query, wq)).sum(-1) / math.sqrt(c)
    gate = torch.sigmoid(score.sign() * score.abs().clamp_min(1e-6).sqrt())
    gated = (gate[..., None] * value[:, None, :]).reshape(t, w)
    normalized = norm(gated, wc).reshape(rows, seq_len, w).transpose(1, 2)
    convolved = F.conv1d(
        F.pad(normalized, (dilation * (k - 1), 0)), conv, dilation=dilation, groups=w).transpose(1, 2).reshape(t, w)
    expected = gated + F.silu(convolved)
    actual = kernels.ple_gate_conv_triton(*inputs, n, eps, dilation, seq_len)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-4, rtol=2e-3)


@pytest.mark.skipif(
    os.environ.get('MCORE_BRIDGE_TEST_LONG_PLE') != '1',
    reason='Set MCORE_BRIDGE_TEST_LONG_PLE=1 on a GPU with at least 80 GiB free')
@pytest.mark.parametrize('stage', ['gate_fwd', 'gate_bwd', 'norm_fwd', 'norm_bwd', 'conv_fwd', 'conv_bwd'])
def test_ple_256k_offsets_cross_int32_boundary(stage):
    # Actual model width: the first overflow is token 209715, channel 2048.
    t, n, c, w = 262144, 4, 2560, 10240
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 80 * 1024**3:
        pytest.skip('Long PLE regression requires at least 80 GiB free GPU memory')

    def zeros(shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype, device='cuda')

    def empty(shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device='cuda')

    def check(tensor, value=0):
        torch.cuda.synchronize()
        for index in [0, 209714, 209715, t - 1]:
            row = tensor[index]
            torch.testing.assert_close(row, torch.full_like(row, value), atol=1e-6, rtol=1e-6)

    weights = zeros(w, torch.bfloat16)
    x = zeros((t, w), torch.bfloat16)
    block = kernels.triton.next_power_of_2(c)
    if stage == 'gate_fwd':
        value = torch.ones((t, c), device='cuda', dtype=torch.bfloat16)
        out, gate, rk, rq = empty((t, w)), empty((t, n)), empty((t, n)), empty((t, n))
        kernels._ple_gate_fwd_kernel[(t * n, )](
            x, x, value, weights, weights, out, gate, rk, rq, t, N=n, C=c, EPS=1e-6, SQRTC=math.sqrt(c), BLOCK_C=block)
        check(out, 1 / (1 + math.exp(-0.001)))
        check(gate, 1 / (1 + math.exp(-0.001)))
    elif stage == 'gate_bwd':
        dg, value, stats = zeros((t, w)), zeros((t, c), torch.bfloat16), torch.ones((t, n), device='cuda')
        dk, dq = empty((t, w), torch.bfloat16), empty((t, w), torch.bfloat16)
        dv, dwk, dwq = empty((t, w)), empty((t, w)), empty((t, w))
        kernels._ple_gate_bwd_kernel[(t * n, )](
            dg,
            x,
            x,
            value,
            weights,
            weights,
            stats,
            stats,
            stats,
            dk,
            dq,
            dv,
            dwk,
            dwq,
            t,
            N=n,
            C=c,
            SQRTC=math.sqrt(c),
            BLOCK_C=block)
        for out in [dk, dq, dv, dwk, dwq]:
            check(out)
    elif stage == 'norm_fwd':
        out, stats = empty((t, w)), empty((t, n))
        kernels._ple_norm_fwd_kernel[(t * n, )](x, weights, out, stats, t, N=n, C=c, EPS=1e-6, BLOCK_C=block)
        check(out)
        check(stats, 1000)
    elif stage == 'norm_bwd':
        stats, out = torch.ones((t, n), device='cuda'), empty((t, w))
        kernels._ple_norm_bwd_kernel[(t * n, )](x, weights, stats, x, out, t, N=n, C=c, BLOCK_C=block)
        check(out)
    else:
        lo, hi = kernels._uniform_seg_bounds(t, t, 'cuda')
        conv = zeros((w, 4), torch.bfloat16)
        out, pre = empty((t, w)), zeros((t, w))
        grid = (t, kernels.triton.cdiv(w, 256))
        if stage == 'conv_fwd':
            kernels._ple_conv_fwd_kernel[grid](x, x, conv, lo, out, pre, t, w, K=4, DIL=3, BLOCK_W=256)
            check(out)
            check(pre)
        else:
            dw, residual = zeros((w, 4)), empty((t, w))
            kernels._ple_conv_bwd_kernel[grid](
                x, pre, x, conv, lo, hi, out, dw, residual, t, w, K=4, DIL=3, BLOCK_W=256)
            check(out)
            check(residual)
            torch.testing.assert_close(dw, torch.zeros_like(dw), atol=0, rtol=0)
