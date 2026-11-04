"""QSA sparse attention numerical and long-context address regressions."""
import os
import pytest
import torch

pytest.importorskip('triton')

from mcore_bridge.model.modules.kernels import qsa_kernels as qk  # noqa: E402
from mcore_bridge.model.modules.kernels.qsa_block_sparse_attn import qsa_block_sparse_attention  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='QSA kernels require CUDA')


def _block_spans(num_blocks, block_size, total, device='cuda'):
    first = torch.arange(num_blocks, dtype=torch.int32, device=device) * block_size
    last = (first + block_size).clamp_max(total) - 1
    return first, last


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
    first, last = _block_spans(sel.shape[1], 4, t)
    actual = qsa_block_sparse_attention(q, k, v, sel, lo, hi, zero, zero, first, last, d**-0.5, 4)
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


def _packed_indexer_rows(lengths, ratio, topk=2):
    """Indexer-shaped rows on packed documents: whole document-relative blocks with
    holes (0, 2, 4, ...) plus the query's causal tail."""
    cu = torch.tensor([0, *lengths]).cumsum(0)
    total = int(cu[-1])
    starts = cu[:-1]
    seg = torch.zeros(total, dtype=torch.long)
    seg[starts[1:]] = 1
    seg = seg.cumsum(0)
    positions = torch.arange(total) - starts[seg]
    rows = []
    for t in range(total):
        doc_start = int(starts[seg[t]])
        p = int(positions[t])
        visible = (p + 1) // ratio
        blocks = list(range(0, visible, 2))[:topk]
        toks = [doc_start + b * ratio + j for b in blocks for j in range(ratio)]
        qn = (p + 1) // ratio
        toks += list(range(doc_start + qn * ratio, min(doc_start + qn * ratio + ratio, t + 1)))
        rows.append(sorted({x for x in toks if doc_start <= x <= t}))
    idx = torch.full((total, max(len(r) for r in rows)), -1, dtype=torch.int64, device='cuda')
    for t, row in enumerate(rows):
        idx[t, :len(row)] = torch.tensor(row, device='cuda')
    return idx, cu.int().cuda()


def _dense_reference(q, k, v, idx, scale):
    total, hq, _ = q.shape
    rep = hq // k.shape[1]
    allowed = torch.zeros(total, total, dtype=torch.bool, device=q.device)
    for t, row in enumerate(idx):
        allowed[t, row[row >= 0]] = True
    kk = k.repeat_interleave(rep, dim=1).float()
    vv = v.repeat_interleave(rep, dim=1).float()
    scores = torch.einsum('thd,shd->hts', q.float(), kk) * scale
    probs = scores.masked_fill(~allowed[None], float('-inf')).softmax(-1)
    return torch.einsum('hts,shd->thd', probs, vv).to(q.dtype)


@pytest.mark.parametrize('lengths,ratio', [
    ([3, 20], 4),
    ([61, 128], 4),
    ([1, 3, 20, 37, 9], 8),
    ([5, 61, 6], 3),
    ([5, 61, 6], 5),
])
def test_qsa_off_grid_packed_matches_dense(lengths, ratio):
    """Off-grid packed documents must attend exactly their selection (fwd + bwd)."""
    torch.manual_seed(7)
    idx, cu = _packed_indexer_rows(lengths, ratio)
    total, d, hq = idx.shape[0], 32, 2
    hkv = 1
    scale = d**-0.5
    q0 = (torch.randn(total, hq, d, device='cuda', dtype=torch.bfloat16) * 0.2)
    k0 = (torch.randn(total, hkv, d, device='cuda', dtype=torch.bfloat16) * 0.2)
    v0 = (torch.randn(total, hkv, d, device='cuda', dtype=torch.bfloat16) * 0.2)
    go = torch.randn(total, hq, d, device='cuda', dtype=torch.bfloat16)

    q, k, v = q0.clone().requires_grad_(), k0.clone().requires_grad_(), v0.clone().requires_grad_()
    out = qk.qsa_sparse_attention_thd(q, k, v, idx, cu, scale, ratio)
    grads = torch.autograd.grad(out, (q, k, v), go)

    q, k, v = q0.clone().requires_grad_(), k0.clone().requires_grad_(), v0.clone().requires_grad_()
    ref = _dense_reference(q, k, v, idx, scale)
    ref_grads = torch.autograd.grad(ref.float(), (q, k, v), go.float())

    torch.testing.assert_close(out.float(), ref.float(), rtol=0.03, atol=0.01)
    for a, e in zip(grads, ref_grads):
        torch.testing.assert_close(a.float(), e.float(), rtol=0.04, atol=0.02)


def test_qsa_sbhd_flattening_keeps_each_sample_on_its_own_grid():
    """sbhd flattening treats each sample as its own sequence."""
    torch.manual_seed(11)
    s, b, d, hq, hkv, ratio = 20, 2, 32, 2, 1, 4
    # per-sample rows on the sample's own grid, with holes
    idx = torch.full((b, s, 16), -1, dtype=torch.int64, device='cuda')
    for sample in range(b):
        for p in range(s):
            visible = (p + 1) // ratio
            blocks = list(range(0, visible, 2))[:2]
            toks = [blk * ratio + j for blk in blocks for j in range(ratio)]
            qn = (p + 1) // ratio
            toks += list(range(qn * ratio, min(qn * ratio + ratio, p + 1)))
            row = sorted({x for x in toks if x <= p})
            idx[sample, p, :len(row)] = torch.tensor(row, device='cuda')

    q = torch.randn(s, b, hq, d, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(s, b, hkv, d, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(s, b, hkv, d, device='cuda', dtype=torch.bfloat16)
    out = qk.qsa_sparse_attention(q, k, v, idx, d**-0.5, ratio)

    idx_f = torch.where(idx >= 0, idx + torch.arange(b, device='cuda').view(b, 1, 1) * s, idx.new_full(
        (), -1)).reshape(b * s, -1)
    q_f = q.permute(1, 0, 2, 3).reshape(b * s, hq, d)
    k_f = k.permute(1, 0, 2, 3).reshape(b * s, hkv, d)
    v_f = v.permute(1, 0, 2, 3).reshape(b * s, hkv, d)
    ref = _dense_reference(q_f, k_f, v_f, idx_f, d**-0.5)
    out_f = out.permute(1, 0, 2, 3).reshape(b * s, hq, d)
    torch.testing.assert_close(out_f.float(), ref.float(), rtol=0.03, atol=0.01)


def test_qsa_core_attention_shim_resolves_packed_cu():
    """forward must resolve cu_seqlens from packed_seq_params for packed inputs."""
    from mcore_bridge.model.modules.kernels import QSASparseCoreAttention

    class _Config:

        context_parallel_size = 1
        indexer_compress_ratio = 4

    class _PackedSeqParams:

        cu_seqlens_q = torch.tensor([0, 3, 23], dtype=torch.int32)
        cu_seqlens_q_padded = None

    layer = QSASparseCoreAttention(core_attention=None, config=_Config())
    idx, _ = _packed_indexer_rows([3, 20], 4)
    total, d, hq, hkv = idx.shape[0], 32, 2, 1
    torch.manual_seed(3)
    q = torch.randn(total, hq, d, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(total, hkv, d, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(total, hkv, d, device='cuda', dtype=torch.bfloat16)
    out = layer(q, k, v, idx, packed_seq_params=_PackedSeqParams())
    ref = _dense_reference(q, k, v, idx, d**-0.5)
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.03, atol=0.01)


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
    first, last = _block_spans(sel.shape[1], 4, t)
    out = qsa_block_sparse_attention(q, k, v, sel, lo, hi, lo, lo, first, last, d**-0.5, 4)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    out.sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(q.grad, torch.zeros_like(q), rtol=0, atol=0)
    torch.testing.assert_close(k.grad, torch.zeros_like(k), rtol=0, atol=0)
    expected = torch.zeros_like(v)
    expected[:4] = t / 4
    torch.testing.assert_close(v.grad, expected, rtol=0, atol=0)
