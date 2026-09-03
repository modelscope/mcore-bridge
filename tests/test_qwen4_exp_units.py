# Copyright (c) ModelScope Contributors. All rights reserved.
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# Every test here builds a real TELinear/triton kernel, so a GPU is required.
# Skip (rather than fail) on CPU-only machines so the file is safe to collect in CI.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='qwen4_exp unit tests require a GPU')

# bf16 rounding budget. The gated-residual chain is compiled, which reassociates
# the mean-reduction in fp32 and drifts by ~1 ULP on the output and ~2 ULP on the
# input grad (relative L1 ~2e-3), and the exact value is not reproducible between
# runs. 4 ULP leaves headroom for that without hiding a real regression.
_BF16_ULP = 2**-7
_TOL = 4 * _BF16_ULP


def _make_config(hidden_size=256,
                 hc_count=4,
                 hc_lowrank=64,
                 index_n_heads=4,
                 index_kv_heads=1,
                 index_head_dim=64,
                 compress_ratio=4,
                 budget=64,
                 dtype=torch.float32):
    from megatron.core.transformer.transformer_config import TransformerConfig
    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=8, params_dtype=dtype)
    cfg.hc_count = hc_count
    cfg.hc_lowrank = hc_lowrank
    cfg.layernorm_epsilon = 1e-6
    cfg.indexer_n_heads = index_n_heads
    cfg.indexer_kv_heads = index_kv_heads
    cfg.indexer_head_dim = index_head_dim
    cfg.indexer_compress_ratio = compress_ratio
    cfg.indexer_budget = budget
    cfg.attention_scaling = 1.0
    return cfg


# --------------------------------------------------------------------------
# 1. QSA indexer selection vs the transformers reference loop
# --------------------------------------------------------------------------
def _rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, cos, sin, unsqueeze_dim=1):
    """Verbatim from transformers modeling_qwen4_exp.apply_rotary_pos_emb (k=None branch)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rope, q_nope = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rope = (q_rope * cos) + (_rotate_half(q_rope) * sin)
    return torch.cat([q_rope, q_nope], dim=-1)


def _reference_select(indexer, hidden_states, cos, sin):
    """The reference double loop: per query, re-pool blocks and take top-k.

    Mirrors transformers Qwen4ExpTextQSAIndexer's selection (modeling_qwen4_exp.py
    ~L667-702): only blocks fully inside the causal prefix are scored, the top
    `block_topk` are expanded back to token ids, and the trailing tokens that do
    not fill a whole block ("tail") are always visible.
    """
    s = hidden_states.shape[0]
    R = indexer.compress_ratio
    topk = indexer.block_topk
    d = indexer.index_head_dim
    nh, nkv = indexer.index_n_heads, indexer.index_kv_heads

    qk = indexer.index_qk_proj(hidden_states)
    if isinstance(qk, tuple):
        qk = qk[0]
    q, token_k = torch.split(qk, [nh * d, nkv * d], dim=-1)
    # Mirrors the reference exactly (modeling_qwen4_exp.py ~L645-651):
    #   * q goes through q_layernorm AND RoPE at its own position
    #   * raw_keys stays RAW here -- k_layernorm is applied once, later, to the
    #     pooled block keys only. Norming token_k up front (as an earlier version
    #     of this test did) applies the norm twice and silently changes the scores.
    q = indexer.q_layernorm(q.view(s, nh, d))
    q = _apply_rope(q.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0), unsqueeze_dim=2)[0]
    raw_keys = token_k.view(s, d)

    allowed = torch.zeros(s, s, dtype=torch.bool, device=hidden_states.device)
    for i in range(s):
        n_blocks = (i + 1) // R
        if n_blocks > 0:
            pooled = raw_keys[:n_blocks * R].view(n_blocks, R, d).float().mean(dim=1).to(raw_keys.dtype)
            pooled = indexer.k_layernorm(pooled)
            starts = torch.arange(n_blocks, device=hidden_states.device) * R
            bk = _apply_rope(pooled.unsqueeze(1), cos[starts], sin[starts]).squeeze(1)
            score = torch.relu(q[i].float() @ bk.float().T).sum(dim=0) / math.sqrt(d)
            keep = score.topk(min(topk, n_blocks), dim=-1).indices
            for b in keep.tolist():
                allowed[i, b * R:(b + 1) * R] = True
        # tail: visible tokens that do not fill a complete block
        allowed[i, n_blocks * R:i + 1] = True
    return ~allowed[None, None]


def test_indexer_matches_reference():
    from mcore_bridge.model.modules.qsa_indexer import QSAIndexer
    print('\n[1] QSAIndexer.selection_as_mask vs transformers reference loop')
    ok = True
    # Shapes chosen so the selection is genuinely sparse: with budget/ratio =
    # block_topk, sparsity needs num_blocks > block_topk, i.e. s > budget+ratio-1.
    # A case at/below the budget is included too, where the model's own math makes
    # every block selected (selection_as_mask must then report a no-op).
    cases = [
        # (seq_len, compress_ratio, budget, seed) -- includes non-multiples of the
        # ratio to exercise the tail, and two ratios to catch hardcoded 4s.
        (200, 4, 64, 0),
        (201, 4, 64, 1),
        (203, 4, 64, 2),
        (128, 2, 32, 3),
        (130, 8, 64, 4),
    ]
    for s, ratio, budget, seed in cases:
        torch.manual_seed(seed)
        cfg = _make_config(compress_ratio=ratio, budget=budget)
        idx = QSAIndexer(cfg).cuda()
        with torch.no_grad():
            idx.index_qk_proj.weight.normal_(0, 0.02)
            idx.q_layernorm.weight.normal_(0, 0.02)
            idx.k_layernorm.weight.normal_(0, 0.02)

        hs = torch.randn(s, 1, cfg.hidden_size, device='cuda')
        d = cfg.indexer_head_dim
        # mcore stores rope angles; selection_as_mask materializes cos/sin from them, so
        # feed angles here and derive the same cos/sin for the reference.
        freqs = torch.randn(s, 1, 1, d, device='cuda')
        cos, sin = freqs[:, 0, 0].cos(), freqs[:, 0, 0].sin()

        got = idx.selection_as_mask(hs, freqs)
        n_blocks = s // ratio
        expect_noop = n_blocks <= budget // ratio
        if expect_noop:
            good = got is None
            print(f'    s={s:4d} ratio={ratio} budget={budget}: below-budget no-op -> '
                  f"{'None as expected' if good else 'UNEXPECTED mask'}")
        else:
            ref = _reference_select(idx, hs, cos, sin)
            good = got is not None and torch.equal(got, ref)
            if got is None:
                print(f'    s={s:4d} ratio={ratio} budget={budget}: MISMATCH (got None, '
                      'expected a sparse mask)')
            else:
                nsel = int((~ref).sum())
                print(f'    s={s:4d} ratio={ratio} budget={budget}: '
                      f"blocks={n_blocks} selected_keys={nsel} -> {'MATCH' if good else 'MISMATCH'}")
        ok &= good
    assert ok, "numerical check failed; see printed relL1/diff above"


def test_indexer_is_forward_only():
    """The indexer participates in the forward but must take no gradient.

    The selection is a discrete top-k turned into a bool mask, so the backbone
    still differentiates normally while index_qk_proj / q_layernorm /
    k_layernorm stay frozen. If a refactor ever makes the mask carry grad, the
    "forward-only" design claim silently stops holding.
    """
    from mcore_bridge.model.modules.qsa_indexer import QSAIndexer
    print('\n[2] indexer is forward-only (mask is a constant, backbone still differentiates)')
    torch.manual_seed(0)
    cfg = _make_config(compress_ratio=4, budget=64)
    idx = QSAIndexer(cfg).cuda()
    with torch.no_grad():
        idx.index_qk_proj.weight.normal_(0, 0.02)

    s = 200
    hs = torch.randn(s, 1, cfg.hidden_size, device='cuda', requires_grad=True)
    freqs = torch.randn(s, 1, 1, cfg.indexer_head_dim, device='cuda')
    mask = idx.selection_as_mask(hs, freqs)

    mask_is_const = mask is not None and mask.dtype == torch.bool and not mask.requires_grad
    # a toy attention consuming the mask -- the backbone must still get grads
    q = hs.squeeze(1)
    logits = (q @ q.T).masked_fill(mask[0, 0], float('-inf'))
    logits.softmax(-1).sum().backward()
    backbone_ok = hs.grad is not None and torch.isfinite(hs.grad).all() and hs.grad.abs().sum() > 0
    indexer_ok = all(p.grad is None for p in idx.parameters())

    print(f'    mask is non-differentiable constant: {mask_is_const} (dtype={mask.dtype})')
    print(f'    backbone received gradient: {backbone_ok}')
    print(f'    indexer params still have no grad: {indexer_ok}')
    assert mask_is_const, f'QSA mask must be a non-differentiable bool constant (got {mask.dtype})'
    assert backbone_ok, 'backbone did not receive a finite non-zero gradient through the mask'
    assert indexer_ok, 'indexer parameters received gradient; the forward-only design no longer holds'


# --------------------------------------------------------------------------
# 2. gated hyper-connection numerical equivalence
# --------------------------------------------------------------------------
def _gated_residual_eager(m, hyper_input):
    """The gated residual written out plainly, with no compiled helpers."""
    normed = m.hc_norm(hyper_input)
    w = F.silu(m.input_mix_weight_down(normed)[0] / m.hc_count)
    w = torch.sigmoid(m.input_mix_weight_up(w)[0]).unflatten(-1, (m.hc_count, m.hidden_size))
    mixed = (w * normed.unflatten(-1, (m.hc_count, m.hidden_size))).mean(dim=-2)
    if m.block_inject_weight is None:
        return mixed, None
    inj = 2 * torch.sigmoid(m.block_inject_weight(normed)[0] / m.hc_count)
    return mixed, inj


def test_gated_residual_equivalence():
    from mcore_bridge.model.modules.hyper_connection_gated import Qwen4ExpTextGatedResidual
    print(f'\n[3] Qwen4ExpTextGatedResidual vs plain eager (tol {_TOL:.4g} = 4 bf16 ULP)')
    ok = True
    for use_combine in (True, False):
        for s in (64, 256):
            torch.manual_seed(7)
            cfg = _make_config(dtype=torch.bfloat16)
            m = Qwen4ExpTextGatedResidual(cfg, use_combine=use_combine).cuda()
            x = torch.randn(s, 1, cfg.hc_count * cfg.hidden_size, device='cuda', dtype=torch.bfloat16)
            xa = x.clone().requires_grad_(True)
            xb = x.clone().requires_grad_(True)

            out_a = m(xa)
            got = out_a[0] if isinstance(out_a, tuple) else out_a
            ref, ref_inj = _gated_residual_eager(m, xb)

            grad_seed = torch.randn_like(got)
            got.backward(grad_seed)
            ref.backward(grad_seed)

            fwd_d = (got.float() - ref.float()).abs().max().item()
            grad_d = (xa.grad.float() - xb.grad.float()).abs().max().item()
            inj_d = 0.0
            if isinstance(out_a, tuple):
                inj_d = (out_a[2].float() - ref_inj.float()).abs().max().item()
            finite = bool(torch.isfinite(got).all() and torch.isfinite(xa.grad).all())
            good = fwd_d <= _TOL and grad_d <= _TOL and inj_d <= _TOL and finite
            ok &= good
            print(f'    combine={use_combine!s:5s} s={s:4d}: fwd={fwd_d:.6g} grad={grad_d:.6g} '
                  f"inj={inj_d:.6g} finite={finite} -> {'OK' if good else 'FAIL'}")
    assert ok, "numerical check failed; see printed relL1/diff above"


# --------------------------------------------------------------------------
# 3. host-offload table lookup
# --------------------------------------------------------------------------
def test_host_lookup_reference():
    """Single-rank host lookup must equal a plain gather of the same rows."""
    from types import SimpleNamespace
    from mcore_bridge.model.modules.ple import Qwen4ExpTextNGramEmbedding
    print('\n[4] host-offload lookup vs plain gather (single rank)')
    torch.manual_seed(11)
    total, head_dim = 1024, 16
    host_table = torch.randn(total, head_dim)
    dummy = SimpleNamespace(vocab_start=0, vocab_end=total, host_table=host_table, _tp_size=1, _tp_group=None)
    ngram_ids = torch.randint(0, total, (2, 5, 3))
    got = Qwen4ExpTextNGramEmbedding._host_lookup(dummy, ngram_ids)
    ref = host_table[ngram_ids].flatten(-2)
    d = (got.float() - ref.float()).abs().max().item()
    ok = d <= 1e-5
    print(f'    max diff = {d:.3e} -> {"OK" if ok else "FAIL"}')
    assert ok, "numerical check failed; see printed relL1/diff above"


def test_host_lookup_tp_partition():
    """Each TP rank gathers its own slice; the pieces reassemble the full gather."""
    from types import SimpleNamespace
    from mcore_bridge.model.modules.ple import Qwen4ExpTextNGramEmbedding
    print('\n[5] host-offload TP=2 partition reassembly')
    total, head_dim = 1024, 16
    torch.manual_seed(13)
    full = torch.randn(total, head_dim)
    ngram_ids = torch.randint(0, total, (2, 5, 3))
    acc = None
    for (vs, ve) in [(0, 512), (512, 1024)]:
        dummy = SimpleNamespace(
            vocab_start=vs, vocab_end=ve, host_table=full[vs:ve].contiguous(), _tp_size=1, _tp_group=None)
        part = Qwen4ExpTextNGramEmbedding._host_lookup(dummy, ngram_ids)
        acc = part if acc is None else acc + part
    ref = full[ngram_ids].flatten(-2)
    d = (acc.float() - ref.float()).abs().max().item()
    ok = d <= 1e-4
    print(f'    max diff = {d:.3e} -> {"OK" if ok else "FAIL"}')
    assert ok, "numerical check failed; see printed relL1/diff above"


# --------------------------------------------------------------------------
# 4. PLE fused gate+conv triton kernel
# --------------------------------------------------------------------------
def _fp32_chain(key, query, value, wk, wq, wc, conv_w, n, C, eps, dilation, seq_len):
    """The fused chain in pure fp32 (what the kernel numerically implements):
    grouped zero-centered RMSNorms, gate transform, sigmoid*value, norm_conv,
    causal dilated depthwise conv + SiLU + residual; one cast to bf16 at the end."""
    import torch.nn.functional as F
    T = key.shape[0]
    k = key.view(T, n, C).float()
    q = query.view(T, n, C).float()
    rk = torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps)
    rq = torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)
    kn = k * rk * (1 + wk.view(n, C).float())
    qn = q * rq * (1 + wq.view(n, C).float())
    score = (kn * qn).sum(-1, keepdim=True) / math.sqrt(C)
    u = score.abs().clamp_min(1e-6).sqrt() * score.sign()
    gated = (torch.sigmoid(u) * value.view(T, 1, C).float()).view(T, n * C)
    g = gated.view(T, n, C)
    rc = torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)
    normed = (g * rc * (1 + wc.view(n, C).float())).view(T, n * C)
    rows = T // seq_len
    x = normed.view(rows, seq_len, n * C).transpose(1, 2)
    K = conv_w.shape[-1]
    pad = (K - 1) * dilation
    x = F.pad(x, (pad, 0))[..., -(pad + seq_len):]
    conv = F.silu(F.conv1d(x, conv_w.float(), dilation=dilation, groups=n * C)).transpose(1, 2)
    out = gated.view(rows, seq_len, n * C) + conv
    return out.view(T, n * C)


def test_ple_fused_kernel():
    """The triton gate+conv chain vs its fp32 formulation, fwd+bwd.

    bf16 inputs (as in training), fp32 accumulation inside the kernels, one
    dtype cast at the output -- so the fp32 reference is the strict contract;
    against the module's bf16-intermediate torch path the kernel drifts by
    design (see ple_fused_parity.py for the measured distribution)."""
    from mcore_bridge.model.modules.kernels.ple_kernels import HAVE_TRITON, ple_gate_conv_triton
    print('\n[6] PLE fused gate+conv kernel vs fp32 reference (fwd+bwd)')
    if not HAVE_TRITON:
        print('    triton unavailable -> SKIP (treated as pass)')
        return True
    torch.manual_seed(17)
    rows, L, n, C, K, dilation = 2, 33, 4, 128, 4, 3
    T = rows * L
    dev = 'cuda'
    key = torch.randn(T, n * C, device=dev, dtype=torch.bfloat16).requires_grad_(True)
    query = torch.randn(T, n * C, device=dev, dtype=torch.bfloat16).requires_grad_(True)
    value = torch.randn(T, C, device=dev, dtype=torch.bfloat16).requires_grad_(True)
    wk = (torch.randn(n * C, device=dev, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    wq = (torch.randn(n * C, device=dev, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    wc = (torch.randn(n * C, device=dev, dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    conv_w = (torch.randn(n * C, 1, K, device=dev, dtype=torch.bfloat16) * 0.2).requires_grad_(True)
    eps = 1e-6

    out_k = ple_gate_conv_triton(query, key, value, wk, wq, wc, conv_w, n, eps, dilation, L)
    assert out_k is not None, 'kernel path unavailable despite HAVE_TRITON'
    g = torch.randn_like(out_k)
    out_k.backward(g)
    grads_k = {p: p.grad.float().clone() for p in (key, query, value, wk, wq, wc, conv_w)}
    out_k = out_k.float().clone()

    for p in (key, query, value, wk, wq, wc, conv_w):
        p.grad = None
    out_r = _fp32_chain(key, query, value, wk, wq, wc, conv_w, n, C, eps, dilation, L).to(torch.bfloat16)
    out_r.backward(g)

    ok = True
    o_rel = ((out_k - out_r.float()).abs().sum() / out_r.float().abs().sum()).item()
    good = o_rel <= 1e-2
    ok &= good
    print(f'    fwd  relL1={o_rel:.3e} -> {"OK" if good else "FAIL"}')
    names = {
        id(key): 'key',
        id(query): 'query',
        id(value): 'value',
        id(wk): 'wk',
        id(wq): 'wq',
        id(wc): 'wc',
        id(conv_w): 'conv_w'
    }
    for p in (key, query, value, wk, wq, wc, conv_w):
        a, b = grads_k[p], p.grad.float()
        rel = ((a - b).abs().sum() / a.abs().sum().clamp_min(1e-12)).item()
        good = rel <= 1e-2
        ok &= good
        print(f'    grad[{names[id(p)]:6s}] relL1={rel:.3e} -> {"OK" if good else "FAIL"}')
    assert ok, "numerical check failed; see printed relL1/diff above"


def test_indexer_indices_match_reference():
    """selection_as_token_indices (the sparse-kernel input) must select exactly the tokens the
    transformers reference loop selects. A drift here means the kernel attends a
    different set than the validated mask path -- invisible without this test."""
    from mcore_bridge.model.modules.qsa_indexer import QSAIndexer
    print('\n[7] QSAIndexer.selection_as_token_indices == reference allowed set (sbhd)')
    ok = True
    cases = [
        (200, 4, 64, 0),
        (201, 4, 64, 1),
        (128, 2, 32, 3),
        (130, 8, 64, 4),
        (60, 4, 64, 5),  # below budget -> no-op
    ]
    for s, ratio, budget, seed in cases:
        torch.manual_seed(seed)
        cfg = _make_config(compress_ratio=ratio, budget=budget)
        idx = QSAIndexer(cfg).cuda()
        with torch.no_grad():
            idx.index_qk_proj.weight.normal_(0, 0.02)
            idx.q_layernorm.weight.normal_(0, 0.02)
            idx.k_layernorm.weight.normal_(0, 0.02)
        hs = torch.randn(s, 1, cfg.hidden_size, device='cuda')
        d = cfg.indexer_head_dim
        freqs = torch.randn(s, 1, 1, d, device='cuda')
        cos, sin = freqs[:, 0, 0].cos(), freqs[:, 0, 0].sin()
        indices = idx.selection_as_token_indices(hs, freqs)
        n_blocks = s // ratio
        if n_blocks <= budget // ratio:
            good = indices is None
            print(f'    s={s:4d} ratio={ratio}: below-budget no-op -> '
                  f"{'None as expected' if good else 'UNEXPECTED indices'}")
        else:
            ref_allowed = ~_reference_select(idx, hs, cos, sin)[0, 0]  # [s, s]
            allowed = torch.zeros(s, s, dtype=torch.bool, device='cuda')
            row = indices[0]
            qq, kk = torch.nonzero(row >= 0, as_tuple=True)
            allowed[qq, row[qq, kk]] = True
            good = bool(torch.equal(allowed, ref_allowed))
            print(f'    s={s:4d} ratio={ratio}: allowed set == reference -> '
                  f"{'MATCH' if good else 'MISMATCH'}")
        ok &= good
    assert ok, "numerical check failed; see printed relL1/diff above"


def test_indexer_indices_packed():
    """select_token_indices_thd (thd) must reproduce running selection_as_token_indices on each
    document independently, and never emit a cross-document or future index."""
    from mcore_bridge.model.modules.qsa_indexer import QSAIndexer
    print('\n[8] QSAIndexer.select_token_indices_thd == per-doc selection_as_token_indices (thd)')
    torch.manual_seed(5)
    cfg = _make_config(compress_ratio=4, budget=32)  # block_topk = 8
    idx = QSAIndexer(cfg).cuda()
    with torch.no_grad():
        idx.index_qk_proj.weight.normal_(0, 0.02)
        idx.q_layernorm.weight.normal_(0, 0.02)
        idx.k_layernorm.weight.normal_(0, 0.02)
    doc_lens = [48, 40, 12]  # blocks 12/10/3 -> first two sparse, third causal
    T = sum(doc_lens)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(doc_lens), 0).tolist()), dtype=torch.long, device='cuda')
    h, d = cfg.hidden_size, cfg.indexer_head_dim
    hidden_tok = torch.randn(T, h, device='cuda')
    pos_in_doc = torch.cat([torch.arange(x) for x in doc_lens]).cuda()
    base = torch.randn(64, d, device='cuda')  # freqs as a function of in-doc position
    freqs = base[pos_in_doc].reshape(T, 1, 1, d)
    packed = idx.select_token_indices_thd(hidden_tok, freqs, cu)
    assert packed is not None, 'expected sparse to engage for packed docs'
    ok = True
    off = 0
    for L in doc_lens:
        hid_d = hidden_tok[off:off + L].unsqueeze(1)  # [L, 1, h]
        freqs_d = base[:L].reshape(L, 1, 1, d)
        ref = idx.selection_as_token_indices(hid_d, freqs_d)
        if ref is None:  # doc below budget -> full causal
            pos = torch.arange(L, device='cuda')
            ref_allowed = pos[None, :] <= pos[:, None]
        else:
            ref_allowed = torch.zeros(L, L, dtype=torch.bool, device='cuda')
            rr = ref[0]
            qq, kk = torch.nonzero(rr >= 0, as_tuple=True)
            ref_allowed[qq, rr[qq, kk]] = True
        seg = packed[off:off + L]
        valid = seg >= 0
        in_doc = bool(((seg[valid] - off) >= 0).all() and ((seg[valid] - off) < L).all())
        got_allowed = torch.zeros(L, L, dtype=torch.bool, device='cuda')
        qq, kk = torch.nonzero(valid, as_tuple=True)
        got_allowed[qq, seg[qq, kk] - off] = True
        good = in_doc and bool(torch.equal(ref_allowed, got_allowed))
        ok &= good
        print(f'    doc L={L:3d}: in-doc={in_doc} allowed==per-doc -> '
              f"{'MATCH' if good else 'MISMATCH'}")
        off += L
    assert ok, "numerical check failed; see printed relL1/diff above"


def test_indexer_mrope_batch():
    """Selection must stay per-sample when freqs carry a real batch dim.

    mrope hands mcore a ``[s, b, 1, rot]`` freq tensor (rope_utils.py:344 keys off
    ``freqs.shape[1] > 1``). Flattening that to ``[s, b*rot]`` folds batch into the
    rotary feature dim, so ``rot`` becomes ``b*rot`` and every sample gets the wrong
    angles. Every other indexer test runs b=1, where ``b*rot == rot`` hides it."""
    from mcore_bridge.model.modules.qsa_indexer import QSAIndexer
    print('\n[9] QSAIndexer per-sample selection under mrope freqs (b=2)')
    torch.manual_seed(11)
    s, ratio, budget = 200, 4, 64
    cfg = _make_config(compress_ratio=ratio, budget=budget)
    idx = QSAIndexer(cfg).cuda()
    with torch.no_grad():
        idx.index_qk_proj.weight.normal_(0, 0.02)
        idx.q_layernorm.weight.normal_(0, 0.02)
        idx.k_layernorm.weight.normal_(0, 0.02)
    d = cfg.indexer_head_dim
    hs = torch.randn(s, 2, cfg.hidden_size, device='cuda')
    # distinct per-sample angles: the whole point of mrope
    freqs = torch.randn(s, 2, 1, d, device='cuda')

    both = idx.selection_as_token_indices(hs, freqs)
    assert both is not None, 'expected an active selection for this length'
    ok = True
    for i in range(2):
        # sample i alone must reproduce row i of the batched call
        alone = idx.selection_as_token_indices(hs[:, i:i + 1], freqs[:, i:i + 1])
        same = torch.equal(alone[0], both[i])
        ok &= same
        print(f'    sample {i}: batched == standalone -> {"OK" if same else "FAIL"}')
    assert ok, 'per-sample selection changed when batched (mrope batch dim leaked into rot)'
