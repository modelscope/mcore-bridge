# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
from megatron.core.extensions.transformer_engine import TELinear
from torch import nn


class Qwen4ExpTextRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, sequence_parallel: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype))
        # Replicated across TP; reduce grads across TP when SP is on.
        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        # zero-centered: (1 + w), and Qwen4ExpText casts after scaling
        return (out * (1.0 + self.weight.float())).type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _materialize_rope(freqs: torch.Tensor, seq_len: int, dtype: torch.dtype, mscale: float):
    """Materialize batch-aware RoPE cos/sin from mcore rotary angles.

    ``[s, freq_b, 1, rot] -> [freq_b, s, rot]``. mcore stores angles rather than
    cos/sin, so they are materialized the way ``_patch_apply_rotary_pos_emb``
    does, keeping the indexer's RoPE identical to the attention's.

    Keeping ``freq_b`` as its own dim is what makes MRoPE correct: its positions
    differ per sample, so flattening here would fold batch into the rotary
    feature dim and make ``rot`` come out as ``b * rot``.
    """
    f = freqs[:seq_len].squeeze(2).permute(1, 0, 2)
    cos = (torch.cos(f) * mscale).to(dtype)
    sin = (torch.sin(f) * mscale).to(dtype)
    return cos, sin


class QSAIndexer(nn.Module):
    """QSA block selection: score compressed key blocks, keep the top-k per query.

    refer: transformers ``Qwen4ExpTextQSAIndexer``. That reference packs scoring and
    mask construction into one ``forward`` with a per-query Python loop; here the
    scoring is factored out so the two output encodings provably agree.

    Two layers, not four peers::

        _score_and_topk_blocks(hidden, freqs)      the actual selection
          |                                        -> (top_blocks, keep, n_blocks)
          +-- selection_as_mask(...)               encode as [b, 1, s, s] bool
          +-- selection_as_token_indices(...)      encode as [b, s, K] int64

        select_token_indices_thd(...)              SEPARATE implementation for thd;
                                                   does NOT reuse the scorer

    Both encoders expand the same blocks to tokens and append the query's own
    partial-block tail, so they select an identical key set -- only the wire format
    differs. Which one the layer asks for (``qwen4_exp.py:_qsa_select``)::

        sbhd, CP==1        -> selection_as_mask           -> TE `arbitrary` mask
        thd or CP>1        -> selection_as_token_indices  -> triton sparse kernel
        thd                -> select_token_indices_thd    -> triton sparse kernel

    thd needs its own path because the causal prefix is per-document (``cu_seqlens``)
    rather than ``(arange(s) + 1) // R``, and because TE rejects an ``arbitrary``
    mask once ``qkv_format == 'thd'``. Indices also scale as O(s*K) instead of the
    mask's O(s^2), which is what makes very long sequences feasible.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.compress_ratio = config.indexer_compress_ratio
        self.token_budget = config.indexer_budget
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = TELinear(
            input_size=config.hidden_size,
            output_size=(self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            parallel_mode='duplicated',
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
        )
        self.q_layernorm = Qwen4ExpTextRMSNorm(
            self.index_head_dim,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        self.k_layernorm = Qwen4ExpTextRMSNorm(
            self.index_head_dim,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        setattr(self.index_qk_proj.weight, 'sequence_parallel', config.sequence_parallel)

    def forward(self, *args, **kwargs):
        raise RuntimeError('QSAIndexer selects via selection_as_mask / selection_as_token_indices / '
                           'select_token_indices_thd, not forward.')

    @torch.no_grad()
    def _score_and_topk_blocks(self, hidden_states: torch.Tensor, freqs: torch.Tensor):
        """The selection itself: score every (query, block) pair and keep the top-k.

        Shared by ``selection_as_mask`` / ``selection_as_token_indices`` so both
        encodings describe the same choice.

        Args:
            hidden_states: ``[s, b, h]`` (mcore layout), pre-attention input --
                the same tensor the reference indexer consumes.
            freqs: mcore rotary frequencies ``[s, 1, 1, rot_dim]``. mcore stores
                angles rather than cos/sin, so they are materialized here the way
                ``_patch_apply_rotary_pos_emb`` does (``cos(freqs) * mscale``),
                keeping the indexer's RoPE identical to the attention's.

        Returns:
            ``None`` when selection is a no-op (the causal prefix never exceeds
            the budget), else ``(top_blocks, keep, n_blocks)`` where
            ``top_blocks``/``keep`` are ``[b, s, k]`` (``k =
            min(block_topk, max_blocks)``) and ``n_blocks`` is ``[s]``.
        """
        s, b, _ = hidden_states.shape
        R = self.compress_ratio
        max_blocks = s // R
        if max_blocks <= self.block_topk:
            return None

        device = hidden_states.device
        # ---- project to indexer q/k ----
        # [s, b, h] -> [s, b, (nh + nkv) * d]
        qk = self.index_qk_proj(hidden_states)[0]
        q, token_k = torch.split(
            qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
        # -> [b, s, nh, d] / [b, s, d]
        q = q.view(s, b, self.index_n_heads, self.index_head_dim).permute(1, 0, 2, 3)
        raw_keys = token_k.view(s, b, self.index_kv_heads, self.index_head_dim).permute(1, 0, 2, 3).squeeze(2)
        q = self.q_layernorm(q)

        # ---- materialize cos/sin from mcore freqs ----
        # freqs is [s, freq_b, 1, rot]; mrope makes dim 1 the real batch
        # (rope_utils.py:344), and _materialize_rope keeps it so it broadcasts.
        mscale = self.config.attention_scaling
        cos, sin = _materialize_rope(freqs, s, q.dtype, mscale)
        rot = cos.shape[-1]

        def apply_rope(t, cos_, sin_):
            t_rope, t_pass = t[..., :rot], t[..., rot:]
            t_rope = (t_rope * cos_) + (_rotate_half(t_rope) * sin_)
            return torch.cat((t_rope, t_pass), dim=-1)

        # queries rotate at their own position: cos [bf, s, rot] -> [bf, s, 1, rot]
        q = apply_rope(q, cos.unsqueeze(2), sin.unsqueeze(2))

        # ---- pool every block once (shared across queries) ----
        usable = max_blocks * R
        key_groups = raw_keys[:, :usable].view(b, max_blocks, R, self.index_head_dim)
        pooled = key_groups.float().mean(dim=2).to(raw_keys.dtype)
        pooled = self.k_layernorm(pooled)
        # blocks rotate at their first token's position
        starts = torch.arange(max_blocks, device=device) * R
        block_keys = apply_rope(pooled, cos[:, starts], sin[:, starts])  # [b, nb, d]

        # ---- score all (query, block) pairs ----
        scores = torch.einsum('bqhd,bkd->bqhk', q.float(), block_keys.float())
        scores = torch.relu(scores).sum(dim=2) / math.sqrt(self.index_head_dim)  # [b, s, nb]

        # ---- restrict to blocks fully inside the causal prefix ----
        n_blocks = (torch.arange(s, device=device) + 1) // R  # [s]
        block_ids = torch.arange(max_blocks, device=device)
        scores = scores.masked_fill((block_ids[None, :] >= n_blocks[:, None])[None], float('-inf'))

        k = min(self.block_topk, max_blocks)
        top_blocks = scores.topk(k, dim=-1).indices  # [b, s, k]
        keep = top_blocks < n_blocks[None, :, None]  # drop the -inf padding slots
        return top_blocks, keep, n_blocks

    @torch.no_grad()
    def selection_as_mask(self, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Encode the selection as a bool mask, or ``None`` when it is a no-op.

        Returns:
            ``[b, 1, s, s]`` bool mask where True marks a *masked-out* key (TE's
            ``arbitrary`` convention), or ``None`` if every visible key is
            selected -- in which case the caller keeps the plain causal path and
            pays nothing.

        Only the causal, unpacked layout is handled; callers must not invoke this
        for packed/THD or context-parallel inputs (see the guard in the layer).
        """
        core = self._score_and_topk_blocks(hidden_states, freqs)
        # when seq lengths less than budget
        if core is None:
            return None
        top_blocks, keep, n_blocks = core
        s, b, _ = hidden_states.shape
        R = self.compress_ratio
        device = hidden_states.device

        # ---- top-k blocks -> token mask ----
        tok = (top_blocks.unsqueeze(-1) * R + torch.arange(R, device=device)).flatten(-2)
        keep_tok = keep.unsqueeze(-1).expand(-1, -1, -1, R).flatten(-2)

        allowed = torch.zeros((b, s, s + 1), dtype=torch.bool, device=device)
        allowed.scatter_(-1, torch.where(keep_tok, tok, torch.full_like(tok, s)).long(), True)
        allowed = allowed[..., :s]
        # tail: visible tokens after the last complete block are always attended
        pos = torch.arange(s, device=device)
        tail = (pos[None, :] >= (n_blocks * R)[:, None]) & (pos[None, :] <= pos[:, None])
        allowed |= tail[None]
        # causal safety net (the block expansion never crosses it, but keep the
        # invariant explicit so a future layout change fails loudly instead of
        # silently attending to the future)
        allowed &= (pos[None, :] <= pos[:, None])[None]
        return ~allowed.unsqueeze(1)  # True == masked out

    @torch.no_grad()
    def selection_as_token_indices(self, hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Encode the selection as token indices for the sparse kernel.

        Returns:
            ``[b, s, K]`` int64 (``K = block_topk*R + R``) with ``-1`` marking
            unused slots, or ``None`` when selection is a no-op. Indices live in
            per-sample sequence space, matching ``qsa_sparse_attention``'s sbhd
            branch. Top-block tokens and the query's own partial-block tail are
            disjoint, so the kernel attends each selected key exactly once.

        Same layout constraints as ``selection_as_mask`` (causal, unpacked).
        """
        core = self._score_and_topk_blocks(hidden_states, freqs)
        if core is None:
            return None
        top_blocks, keep, n_blocks = core
        s, b, _ = hidden_states.shape
        R = self.compress_ratio
        device = hidden_states.device
        k = top_blocks.shape[-1]
        arange_r = torch.arange(R, device=device)

        # top-k blocks expanded to tokens: [b, s, k, R] -> [b, s, k*R]; dropped
        # (-inf) slots carry -1 so the kernel skips them.
        tok = top_blocks.unsqueeze(-1) * R + arange_r
        top_idx = tok.flatten(-2)
        top_keep = keep.unsqueeze(-1).expand(-1, -1, -1, R).reshape(b, s, k * R)
        top_idx = torch.where(top_keep, top_idx, top_idx.new_full((), -1))

        # tail: the query's own partial block, causally truncated: [b, s, R]
        pos = torch.arange(s, device=device)
        tail_idx = (n_blocks * R)[:, None] + arange_r[None, :]  # [s, R]
        tail_idx = torch.where(tail_idx <= pos[:, None], tail_idx, tail_idx.new_full((), -1))
        tail_idx = tail_idx[None].expand(b, s, R)

        return torch.cat([top_idx, tail_idx], dim=-1).to(torch.int64)

    @torch.no_grad()
    def select_token_indices_thd(self, hidden_tok: torch.Tensor, freqs: torch.Tensor,
                                 cu_seqlens: torch.Tensor) -> torch.Tensor:
        """QSA selection for packed (thd) inputs, indices in pack space.

        A standalone implementation: it does *not* call
        ``_score_and_topk_blocks``, because under packing the causal prefix is
        per-document (derived from ``cu_seqlens``) rather than a single
        ``(arange(s) + 1) // R``, and the batch dim is already flattened into ``T``.

        Args:
            hidden_tok: ``[T, h]`` packed tokens (the dummy batch dim squeezed out).
            freqs: rotary angles ``[T, ...]``; each row already encodes that token's
                *in-document* position (mcore builds them from per-doc position ids
                under padding_free), so they are reused as-is here.
            cu_seqlens: ``[D+1]`` int document boundaries (``cu[0] == 0``,
                ``cu[-1] == T``).

        Returns:
            ``[T, K]`` int64 pack-space indices (``K = block_topk*R + R``), ``-1``
            unused; every index stays inside its query's document and causal
            prefix. ``None`` when selection is a no-op for every document, in which
            case TE's packed causal kernel reproduces the selection exactly.
        """
        T, _ = hidden_tok.shape
        R = self.compress_ratio
        device = hidden_tok.device
        doc_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()  # [D]
        D = doc_lens.numel()
        full_blocks = doc_lens // R  # complete R-blocks per document
        # No-op when no document's causal prefix can exceed the budget.
        if int(full_blocks.max().item()) <= self.block_topk:
            return None

        # ---- per-token document id / in-doc position ----
        token_doc = torch.repeat_interleave(torch.arange(D, device=device), doc_lens)  # [T]
        pos_in_doc = torch.arange(T, device=device) - cu_seqlens[token_doc].long()  # [T]

        # ---- project to indexer q/k ----
        qk = self.index_qk_proj(hidden_tok)[0]
        q, token_k = torch.split(
            qk, [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim], dim=-1)
        q = q.view(T, self.index_n_heads, self.index_head_dim)
        raw_keys = token_k.view(T, self.index_kv_heads, self.index_head_dim).squeeze(1)  # [T, d]
        q = self.q_layernorm(q)

        # ---- rope on q at its own (in-doc) position ----
        # Deliberately not _materialize_rope: thd has no batch dim to preserve
        # (samples are already flattened into T), so the trailing dims are folded
        # into rot here instead of being kept separate.
        mscale = self.config.attention_scaling
        f = freqs.reshape(freqs.shape[0], -1)[:T]
        cos = (torch.cos(f) * mscale).to(q.dtype)
        sin = (torch.sin(f) * mscale).to(q.dtype)
        rot = cos.shape[-1]

        def apply_rope(t, cos_, sin_):
            t_rope, t_pass = t[..., :rot], t[..., rot:]
            t_rope = (t_rope * cos_) + (_rotate_half(t_rope) * sin_)
            return torch.cat((t_rope, t_pass), dim=-1)

        q = apply_rope(q, cos.unsqueeze(1), sin.unsqueeze(1))  # [T, nh, d]

        # ---- pool complete blocks in pack space (shared across queries) ----
        NB = int(full_blocks.sum().item())
        token_in_full = pos_in_doc < (full_blocks * R)[token_doc]  # [T]
        block_in_doc = pos_in_doc // R  # [T]
        block_offset = torch.cumsum(full_blocks, 0) - full_blocks  # exclusive [D]
        global_block = block_offset[token_doc] + block_in_doc  # [T]

        pooled_sum = torch.zeros(NB, self.index_head_dim, device=device, dtype=torch.float32)
        gb = global_block[token_in_full]
        pooled_sum.index_add_(0, gb, raw_keys[token_in_full].float())
        pooled = (pooled_sum / R).to(raw_keys.dtype)  # exactly R tokens per full block
        pooled = self.k_layernorm(pooled)  # [NB, d]

        # blocks rotate at their first token's in-doc position
        block_doc = torch.repeat_interleave(torch.arange(D, device=device), full_blocks)  # [NB]
        block_in_doc_idx = torch.arange(NB, device=device) - block_offset[block_doc]  # [NB]
        first_pack = cu_seqlens[block_doc].long() + block_in_doc_idx * R  # [NB]
        block_keys = apply_rope(pooled, cos[first_pack], sin[first_pack])  # [NB, d]

        # ---- score every (token, block) pair ----
        scores = torch.einsum('thd,kd->thk', q.float(), block_keys.float())
        scores = torch.relu(scores).sum(dim=1) / math.sqrt(self.index_head_dim)  # [T, NB]

        # ---- restrict to same-document, causally-before blocks ----
        q_nblocks = (pos_in_doc + 1) // R  # [T]
        valid = (block_doc[None, :] == token_doc[:, None]) & \
            (block_in_doc_idx[None, :] < q_nblocks[:, None])  # [T, NB]
        scores = scores.masked_fill(~valid, float('-inf'))

        # ---- top-k blocks -> token indices ----
        k = min(self.block_topk, NB)
        top_blocks = scores.topk(k, dim=-1).indices  # [T, k] into [0, NB)
        keep = valid.gather(1, top_blocks)  # [T, k]
        arange_r = torch.arange(R, device=device)
        base = cu_seqlens[block_doc[top_blocks]].long() + block_in_doc_idx[top_blocks] * R  # [T, k]
        top_idx = (base.unsqueeze(-1) + arange_r).flatten(-2)  # [T, k*R]
        top_keep = keep.unsqueeze(-1).expand(-1, -1, R).reshape(T, k * R)
        top_idx = torch.where(top_keep, top_idx, top_idx.new_full((), -1))

        # ---- tail: the query's own partial block, causally truncated ----
        tail_base = cu_seqlens[token_doc].long() + q_nblocks * R  # [T]
        tail_idx = tail_base.unsqueeze(-1) + arange_r[None, :]  # [T, R]
        token_pos = torch.arange(T, device=device)
        tail_idx = torch.where(tail_idx <= token_pos[:, None], tail_idx, tail_idx.new_full((), -1))

        return torch.cat([top_idx, tail_idx], dim=-1).to(torch.int64)
