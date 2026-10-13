"""QSA sparse attention numerical and long-context address regressions."""
import importlib.util
import os
import pytest
import torch
from pathlib import Path

pytest.importorskip('triton')
_spec = importlib.util.spec_from_file_location(
    'qsa_sparse_under_test',
    Path(__file__).parents[1] / 'src/mcore_bridge/model/modules/kernels/qsa_block_sparse_attn.py')
kernels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kernels)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='QSA kernels require CUDA')


@pytest.mark.parametrize('packed', [False, True])
def test_qsa_forward_backward_match_dense_attention(packed):
    torch.manual_seed(123)
    t, h, d = 39, 2, 32
    q = (torch.randn(t, h, d, device='cuda', dtype=torch.bfloat16) * 0.2).requires_grad_()
    k = (torch.randn(t, 1, d, device='cuda', dtype=torch.bfloat16) * 0.2).requires_grad_()
    v = torch.randn_like(k, requires_grad=True)
    pos = torch.arange(t, device='cuda')
    sel = torch.zeros(t, (t + 3) // 4, device='cuda', dtype=torch.uint8)
    sel[:, 0] = 1
    sel[pos, pos // 4] = 1
    lo = torch.where(pos >= 19, 19, 0).int() if packed else torch.zeros_like(pos, dtype=torch.int32)
    hi = pos.int()
    zero = torch.zeros_like(lo)
    actual = kernels.qsa_block_sparse_attention_triton(q, k, v, sel, lo, hi, zero, zero, d**-0.5, 4)
    allowed = sel[:, pos // 4].bool() & (pos[None, :] >= lo[:, None]) & (pos[None, :] <= hi[:, None])
    scores = torch.einsum('thd,shd->hts', q.float(), k.expand(-1, h, -1).float()) * d**-0.5
    probs = scores.masked_fill(~allowed[None], float('-inf')).softmax(-1)
    expected = torch.einsum('hts,shd->thd', probs, v.expand(-1, h, -1).float()).to(q.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.008)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (q, k, v), grad)
    expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol=0.04, atol=0.015)


@pytest.mark.skipif(
    os.environ.get('MCORE_BRIDGE_TEST_LONG_QSA') != '1',
    reason='Set MCORE_BRIDGE_TEST_LONG_QSA=1 for the 16 GiB bitmap regression')
def test_qsa_256k_bitmap_forward_backward_cross_int32_boundary():
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < 32 * 1024**3:
        pytest.skip('Long QSA regression requires at least 32 GiB free GPU memory')
    t, d = 262144, 16
    q = torch.zeros(t, 1, d, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    v = torch.ones_like(q, requires_grad=True)
    # Every query selects the first four keys. The bitmap row stride remains
    # the real 256K model stride, so rows >=32768 require >int32 addressing.
    sel = torch.zeros(t, t // 4, device='cuda', dtype=torch.uint8)
    sel[:, 0] = 1
    lo = torch.zeros(t, device='cuda', dtype=torch.int32)
    hi = torch.full_like(lo, 3)
    out = kernels.qsa_block_sparse_attention_triton(q, k, v, sel, lo, hi, lo, lo, d**-0.5, 4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    out.sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(q.grad, torch.zeros_like(q), rtol=0, atol=0)
    torch.testing.assert_close(k.grad, torch.zeros_like(k), rtol=0, atol=0)
    expected = torch.zeros_like(v)
    expected[:4] = t / 4
    torch.testing.assert_close(v.grad, expected, rtol=0, atol=0)
