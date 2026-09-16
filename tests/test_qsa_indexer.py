# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from mcore_bridge.model.modules.qsa_indexer import _materialize_rope, _rotate_half


def _packed_inputs(idx, doc_lens, hidden_size, d, device):
    """Build a packed (thd) multi-doc input for select_token_indices_thd."""
    T = sum(doc_lens)
    cu = [0]
    for L in doc_lens:
        cu.append(cu[-1] + L)
    cu_seqlens = torch.tensor(cu, dtype=torch.long, device=device)
    hidden_tok = torch.randn(T, hidden_size, device=device)
    freqs = torch.randn(T, 1, 1, d, device=device)
    return hidden_tok, freqs, cu_seqlens


def _make_idx(compress_ratio, budget, device):
    from test_qwen4_exp_units import _make_config

    from mcore_bridge.model.modules.qsa_indexer import QSAIndexer
    cfg = _make_config(compress_ratio=compress_ratio, budget=budget)
    idx = QSAIndexer(cfg).to(device)
    with torch.no_grad():
        idx.index_qk_proj.weight.normal_(0, 0.02)
        idx.q_layernorm.weight.normal_(0, 0.02)
        idx.k_layernorm.weight.normal_(0, 0.02)
    return idx, cfg


def test_qsa_thd_chunked_matches_single_chunk_bitwise(monkeypatch):
    """Chunked (token, block) scoring must be bitwise identical to the un-chunked reference.

    The score -> relu -> head-sum -> mask -> top-k pipeline is row-wise independent, so forcing a
    single chunk (chunk >= T) reproduces the original full-einsum behaviour exactly; a small chunk
    budget exercises the multi-chunk path. Both must return identical indices.
    """
    import mcore_bridge.model.modules.qsa_indexer as qi
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    idx, cfg = _make_idx(compress_ratio=4, budget=32, device=device)
    d = cfg.indexer_head_dim
    # each doc longer than block_topk blocks so the selection is genuinely sparse
    per = 4 * (idx.block_topk + 4)
    hidden_tok, freqs, cu_seqlens = _packed_inputs(idx, [per, per + 4, per + 8], cfg.hidden_size, d, device)

    monkeypatch.setattr(qi, '_QSA_INDEX_SCORE_CHUNK_BYTES', 1 << 62)  # single chunk = un-chunked ref
    one = idx.select_token_indices_thd(hidden_tok, freqs, cu_seqlens, force_materialize=True)
    monkeypatch.setattr(qi, '_QSA_INDEX_SCORE_CHUNK_BYTES', 1024)  # tiny budget = many chunks
    many = idx.select_token_indices_thd(hidden_tok, freqs, cu_seqlens, force_materialize=True)
    assert torch.equal(one, many), 'chunked selection diverged from the un-chunked reference'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='peak-memory regression needs a GPU')
def test_qsa_thd_peak_memory_seq_shape_insensitive():
    """Peak selection memory must not track the packed sequence shape at equal total tokens.

    Pattern A = few long docs (large per-doc NB), pattern B = many short docs; both at the same
    total token count. Before the chunked scoring the peak tracked [T, n_heads, NB] (i.e. the
    s^2/compress_ratio term), so B (larger total NB) peaked far above A. With chunking the peak is
    O(chunk * NB) and the two patterns must be close.
    """
    import mcore_bridge.model.modules.qsa_indexer as qi
    torch.manual_seed(0)
    idx, cfg = _make_idx(compress_ratio=4, budget=32, device='cuda')
    d = cfg.indexer_head_dim
    per = 4 * (idx.block_topk + 8)
    n = 8
    total = per * n
    cases = {
        'few_long': [total // 2, total // 2],  # 2 long docs
        'many_short': [per] * n,  # n short docs, same total tokens
    }
    peaks = {}
    for name, doc_lens in cases.items():
        hidden_tok, freqs, cu_seqlens = _packed_inputs(idx, doc_lens, cfg.hidden_size, d, 'cuda')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        idx.select_token_indices_thd(hidden_tok, freqs, cu_seqlens, force_materialize=True)
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated()
        del hidden_tok, freqs, cu_seqlens
    ratio = peaks['many_short'] / max(1, peaks['few_long'])
    assert ratio < 1.3, (f'peak memory tracks sequence shape (many_short/few_long={ratio:.2f}); peaks={peaks}; '
                         'the (token, block) scoring is not chunked')


def test_materialize_rope_preserves_mrope_batch_dimension():
    """MRoPE positions differ per sample, so ``freq_b`` must stay its own dim.

    The bug this pins: flattening ``[s, b, 1, rot]`` to ``[s, b * rot]`` makes
    ``rot`` come out as ``b * rot``, and every downstream ``[..., :rot]`` slice
    then reads the wrong half -- silently, and only when ``b > 1``.
    """
    seq_len, batch_size, rope_dim = 16, 2, 64
    freqs = torch.randn(seq_len, batch_size, 1, rope_dim)

    cos, sin = _materialize_rope(freqs, seq_len, torch.float32, 1.0)
    expected = freqs.squeeze(2).permute(1, 0, 2)

    # rot must stay rope_dim, not batch_size * rope_dim
    assert cos.shape == (batch_size, seq_len, rope_dim), \
        f'expected [b, s, rot] = {(batch_size, seq_len, rope_dim)}, got {tuple(cos.shape)}'
    torch.testing.assert_close(cos, expected.cos())
    torch.testing.assert_close(sin, expected.sin())

    # Perturbing sample 1 must leave sample 0 untouched: no cross-sample bleed.
    changed_freqs = freqs.clone()
    changed_freqs[:, 1].add_(0.5)
    changed_cos, changed_sin = _materialize_rope(changed_freqs, seq_len, torch.float32, 1.0)

    torch.testing.assert_close(changed_cos[0], cos[0])
    torch.testing.assert_close(changed_sin[0], sin[0])
    assert not torch.equal(changed_cos[1], cos[1]), 'sample 1 should have changed'
    assert not torch.equal(changed_sin[1], sin[1]), 'sample 1 should have changed'


def test_materialize_rope_applies_mscale_and_dtype():
    """``mscale`` mirrors the attention path's attention_scaling; dtype is honoured."""
    freqs = torch.randn(8, 1, 1, 32)
    mscale = 1.7

    cos, sin = _materialize_rope(freqs, 8, torch.bfloat16, mscale)
    ref = freqs.squeeze(2).permute(1, 0, 2)

    assert cos.dtype is torch.bfloat16 and sin.dtype is torch.bfloat16
    torch.testing.assert_close(cos.float(), (ref.cos() * mscale).bfloat16().float())
    torch.testing.assert_close(sin.float(), (ref.sin() * mscale).bfloat16().float())


def test_materialize_rope_truncates_to_seq_len():
    """A longer freq table is sliced to ``seq_len`` (CP hands over full-length tables)."""
    freqs = torch.randn(64, 3, 1, 16)

    cos, _ = _materialize_rope(freqs, 20, torch.float32, 1.0)

    assert cos.shape == (3, 20, 16)
    torch.testing.assert_close(cos, freqs[:20].squeeze(2).permute(1, 0, 2).cos())


def test_rotate_half_matches_reference():
    """``_rotate_half`` is the standard (-x2, x1) split the attention path uses."""
    x = torch.randn(2, 5, 8)

    got = _rotate_half(x)

    x1, x2 = x[..., :4], x[..., 4:]
    torch.testing.assert_close(got, torch.cat((-x2, x1), dim=-1))
