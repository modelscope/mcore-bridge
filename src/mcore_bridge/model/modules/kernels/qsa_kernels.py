# Copyright (c) ModelScope Contributors. All rights reserved.
"""QSA sparse attention wrappers around the vendored tensor-core triton kernel.

The kernel itself lives in ``qsa_block_sparse_attn.py`` (vendored from miles). This
file owns the glue: the packed block form, sbhd<->thd flattening, context
parallelism, and the ``core_attention`` shim.

The indexer selects whole blocks on each document's own grid (``doc_start + m * R``),
which need not sit on the global ``token // R`` grid. The packed path therefore builds
``blk_base``/``tok_base``/``block_first``/``block_last`` from ``cu_seqlens`` in
``_PackedBlockLayout``; ``qsa_sparse_attention_from_indices`` is global-grid only.

WHY A KERNEL AT ALL (the decision this file encodes)
    QSA needs a per-query key set, which as an attention mask is TE's
    ``arbitrary`` type. TE refuses that under packing:
    ``dot_product_attention.py:1347`` asserts ``"padding" in attn_mask_type``
    whenever ``qkv_format == "thd"``, so a bool mask cannot express QSA there.
    Context parallelism is blocked for a different reason -- block pooling needs
    keys from other CP ranks, which the mask path never gathers.

    So the layer picks by data shape, unless disabled via the ``QSA_SPARSE_KERNEL``
    env var (then: full attention under thd/CP>1, bool mask otherwise):

        thd (padding_free) or CP>1  ->  this kernel
        sbhd and CP==1              ->  the indexer's bool mask on TE

    The mask path is not a degraded fallback in that last case: both paths run
    the same ``QSAIndexer._score_and_topk_blocks`` top-block selection, so they are
    mathematically equivalent there, and the mask path measures *faster*
    (8192: 8.4 s/it vs 25.9 s/it) because TE keeps the whole thing fused.
"""
import torch

from mcore_bridge.utils import get_env_args, get_logger

from .qsa_block_sparse_attn import qsa_block_sparse_attention

logger = get_logger()

QSA_SPARSE_KERNEL_ENV = 'QSA_SPARSE_KERNEL'

try:
    import triton  # noqa: F401  (import guard for the vendored kernel)
    HAVE_TRITON = True
except Exception:  # pragma: no cover - triton absent
    HAVE_TRITON = False


def use_qsa_sparse_kernel() -> bool:
    return get_env_args(QSA_SPARSE_KERNEL_ENV, bool, True)


def qsa_sparse_supported(head_dim: int) -> bool:
    """Whether the sparse kernel can run for this head dim.

    Triton must be importable, ``head_dim`` must be a power of two (the kernel tiles
    the head with ``tl.arange`` blocks), and the kernel must not be disabled via
    ``QSA_SPARSE_KERNEL=0``.
    """
    if not use_qsa_sparse_kernel():
        return False
    if not HAVE_TRITON or head_dim <= 0 or (head_dim & (head_dim - 1)):
        return False
    if not torch.cuda.is_available():
        # Triton-Ascend compiles this kernel with the CUDA tile sizes and fails
        # (UB/Cc overflow). Do not install it there. Packing and CP then take the
        # explicit QSA_SPARSE_KERNEL=0 full-attention fallback instead of dying
        # inside the compiler. The bool-mask path (CP == 1, not packed) is unchanged.
        logger.warning_once('The QSA sparse kernel is CUDA-only and is disabled on this device. '
                            f'Set {QSA_SPARSE_KERNEL_ENV}=0 to fall back to full attention under '
                            'packing or context parallelism. With CP == 1 and packing off, QSA '
                            'keeps the bool-mask path.')
        return False
    return True


def _cp_query_global_positions(seq_len: int, cp_size: int, cp_rank: int, device) -> torch.Tensor:
    """This CP rank's logical token positions under mcore zigzag sharding.

    Matches ``split_cp_inputs`` / ``rope_utils.get_pos_emb_on_this_cp_rank``: the
    sequence is viewed as ``2 * cp_size`` chunks and the rank owns the
    ``cp_rank``-th front chunk plus its mirrored back chunk.
    """
    chunk = seq_len // (2 * cp_size)
    front = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device, dtype=torch.int64)
    back_chunk = 2 * cp_size - cp_rank - 1
    back = torch.arange(back_chunk * chunk, (back_chunk + 1) * chunk, device=device, dtype=torch.int64)
    return torch.cat((front, back), dim=0)


def _cp_query_global_positions_thd(cu_seqlens: torch.Tensor, cp_size: int, cp_rank: int, device) -> torch.Tensor:
    """Local packed-token positions per sample under zigzag thd CP sharding.

    Each sample is padded to a multiple of ``2 * cp_size`` by the data pipeline;
    the rank owns the ``cp_rank``-th front chunk and the mirrored back chunk of
    every sample (the same partition ``split_cp_inputs`` applies per sample).
    """
    cu = cu_seqlens.to(device=device, dtype=torch.int64)
    starts, ends = cu[:-1], cu[1:]
    half = (ends - starts) // (2 * cp_size)  # per-sample chunk length
    # front chunk cp_rank; back chunk (2*cp-1-cp_rank), whose start is
    # ends - (cp_rank+1)*half since it is the (cp_rank+1)-th chunk from the end
    seg_starts = torch.stack((starts + cp_rank * half, ends - (cp_rank + 1) * half), dim=1).reshape(-1)
    seg_lens = torch.stack((half, half), dim=1).reshape(-1)
    nz = seg_lens > 0
    seg_starts, seg_lens = seg_starts[nz], seg_lens[nz]
    seg_ids = torch.repeat_interleave(torch.arange(seg_lens.numel(), device=device), seg_lens)
    offsets = torch.arange(int(seg_lens.sum().item()), device=device)
    offsets = offsets - torch.repeat_interleave(torch.cumsum(seg_lens, 0) - seg_lens, seg_lens)
    return seg_starts.index_select(0, seg_ids) + offsets


def _cp_gathered_to_logical_order(seq_len: int, cp_size: int, device) -> torch.Tensor:
    """Index restoring rank-major gathered chunks to logical order: applying
    ``gathered[idx]`` yields the same layout as ``_undo_attention_load_balancing``
    (the inverse of ``split_cp_inputs``)."""
    chunk = seq_len // (2 * cp_size)
    order = [2 * i for i in range(cp_size)] + [2 * cp_size - 2 * i - 1 for i in range(cp_size)]
    return torch.cat([torch.arange(c * chunk, (c + 1) * chunk, device=device) for c in order])


def _align_cu_seqlens(cu, total: int):
    """Return ``cu`` ending at ``total``, accepting a missing leading 0 or a batch
    offset; ``None`` when it cannot describe ``total`` tokens."""
    cu = cu.reshape(-1).to(dtype=torch.long)
    if cu.numel() > 0 and int(cu[0]) != 0:
        if int(cu[-1]) == total:
            cu = torch.cat([cu.new_zeros(1), cu])
        elif int(cu[-1]) - int(cu[0]) == total:
            cu = cu - cu[0]
    if not (cu.numel() > 1 and int(cu[-1]) == total):
        return None
    return cu


def _packed_cu_seqlens(packed_seq_params, total: int):
    """Resolve the ``cu_seqlens`` matching ``total`` packed tokens; padded first (it is
    the pack space CP gathers into)."""
    if packed_seq_params is None:
        raise ValueError('QSA packed attention needs packed_seq_params.cu_seqlens_q to place every '
                         f'sequence on its own block grid, but none was passed for {total} tokens.')
    for name in ('cu_seqlens_q_padded', 'cu_seqlens_q'):
        cu = getattr(packed_seq_params, name, None)
        if cu is None:
            continue
        aligned = _align_cu_seqlens(cu, total)
        if aligned is not None:
            return aligned
    raise ValueError(f'QSA cannot align cu_seqlens with the {total} tokens it is about to attend: '
                     f'cu_seqlens_q={getattr(packed_seq_params, "cu_seqlens_q", None)}, '
                     f'cu_seqlens_q_padded={getattr(packed_seq_params, "cu_seqlens_q_padded", None)}.')


class _PackedBlockLayout:
    """Per-sequence block grid: block ``m`` covers ``[start + m*R, start + (m+1)*R)``
    and ids form one global numbering of size ``sum(ceil(len_d / R))``; the exact
    per-block token spans feed the kernel's tile map.
    """

    def __init__(self, cu_seqlens, compress_ratio: int, device):
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        lengths = cu[1:] - cu[:-1]
        if lengths.numel() == 0 or bool((lengths < 0).any()):
            raise ValueError(f'QSA block layout needs non-negative sequence lengths, got cu={cu.tolist()}')
        blocks_per_seq = -(-lengths // compress_ratio)
        self.num_blocks = int(blocks_per_seq.sum().item())
        if self.num_blocks == 0:
            raise ValueError('QSA block layout needs at least one non-empty sequence.')
        seq_block_start = torch.cat([blocks_per_seq.new_zeros(1), blocks_per_seq.cumsum(0)[:-1]])
        seq_token_start = cu[:-1]
        # per token: its sequence start and that sequence's first block id
        seg = torch.repeat_interleave(torch.arange(lengths.numel(), device=device), lengths)
        self.token_start = seq_token_start.index_select(0, seg)
        self.token_block_start = seq_block_start.index_select(0, seg)
        # per block: exact inclusive token span
        block_seq = torch.repeat_interleave(torch.arange(lengths.numel(), device=device), blocks_per_seq)
        block_local = torch.arange(self.num_blocks, device=device) - seq_block_start.index_select(0, block_seq)
        self.block_first_token = seq_token_start.index_select(0, block_seq) + block_local * compress_ratio
        self.block_last_token = torch.minimum(self.block_first_token + compress_ratio, cu[1:].index_select(
            0, block_seq)) - 1


def _block_form(indices, layout: _PackedBlockLayout, block_size: int):
    """Rows -> ``(sel, lo, hi, blk_base, tok_base, block_first, block_last)``: a
    token's block id is ``blk_base + (token - tok_base) // R`` and ``lo``/``hi`` is
    ``[sequence start, query position]`` (block membership supplies the holes)."""
    device = indices.device
    total = indices.shape[0]
    tok_base = layout.token_start.to(torch.int32)
    blk_base = layout.token_block_start.to(torch.int32)
    sel = torch.zeros(total, layout.num_blocks, dtype=torch.uint8, device=device)
    valid = indices >= 0
    if bool(valid.any()):
        rows = torch.arange(total, device=device).unsqueeze(1).expand_as(indices)
        blk = blk_base.unsqueeze(1).long() + (indices - tok_base.unsqueeze(1)).div(block_size, rounding_mode='floor')
        sel[rows[valid], blk[valid].long().clamp_(0, layout.num_blocks - 1)] = 1
    lo = tok_base
    hi = torch.arange(total, dtype=torch.int32, device=device)
    return (sel, lo, hi, blk_base, tok_base, layout.block_first_token.to(torch.int32),
            layout.block_last_token.to(torch.int32))


def qsa_sparse_attention_thd(q, k, v, indices, cu_seqlens, scale, block_size):
    """``q`` [T, Hq, D], ``k``/``v`` [S, Hkv, D], ``indices`` [T, K] (-1 pad).

    ``cu_seqlens`` gives every sequence's start (``_PackedBlockLayout``). ``block_size``
    must be the indexer's ``compress_ratio``: a different value silently changes which
    keys are attended.
    """
    if not HAVE_TRITON or q.device.type == 'cpu':
        raise RuntimeError('QSA sparse attention requires triton and accelerator tensors '
                           f'(HAVE_TRITON={HAVE_TRITON}, device={q.device.type}). This path is only '
                           'selected for packing (thd) or CP>1, where no dense fallback is correct.')
    if q.shape[-1] & (q.shape[-1] - 1):
        raise RuntimeError(f'QSA sparse attention needs a power-of-two head dim, got {q.shape[-1]}.')
    if q.shape[0] != k.shape[0]:
        # The kernel takes its key bound from the query count (T, Hq, D = q.shape,
        # then `offs_k < T`), so unequal lengths would silently drop every key past
        # len(q). Callers must equalise first -- _forward_cp does this by scattering
        # the local query shard into a full-length buffer.
        raise ValueError(f'QSA sparse attention needs len(q) == len(k), got {q.shape[0]} vs {k.shape[0]}.')
    cu = _align_cu_seqlens(cu_seqlens, q.shape[0]) if cu_seqlens is not None else None
    if cu is None:
        raise ValueError(f'QSA packed attention needs cu_seqlens whose last entry is {q.shape[0]}, '
                         f'got {None if cu_seqlens is None else cu_seqlens.tolist()}.')
    layout = _PackedBlockLayout(cu, block_size, q.device)
    sel, lo, hi, blk_base, tok_base, block_first, block_last = _block_form(indices.contiguous(), layout, block_size)
    return qsa_block_sparse_attention(q, k, v, sel, lo, hi, blk_base, tok_base, block_first, block_last, scale,
                                      block_size)


def qsa_sparse_attention(q, k, v, indices, scale, block_size, cu_seqlens=None):
    """QSA sparse attention over mcore layouts.

    sbhd: ``q`` [s, b, Hq, D], ``k``/``v`` [s, b, Hkv, D], ``indices``
        [b, s, K] in per-sample sequence space (each sample is one sequence).
    thd:  ``q`` [T, Hq, D], ``k``/``v`` [T, Hkv, D], ``indices`` [T, K] in
        pack space, with ``cu_seqlens`` giving the document boundaries.

    Raises rather than returning ``None``: the caller only reaches here for
    packing or CP>1, and for those a dense fallback would silently diverge from
    sparse inference.
    """
    if q.dim() == 3:
        return qsa_sparse_attention_thd(q, k, v, indices, cu_seqlens, scale, block_size)
    if q.dim() != 4:
        raise ValueError(f'qsa_sparse_attention expected 3D (thd) or 4D (sbhd) q, got {tuple(q.shape)}')
    s, b, hq, d = q.shape
    sk = k.shape[0]
    q_f = q.permute(1, 0, 2, 3).reshape(b * s, hq, d)
    k_f = k.permute(1, 0, 2, 3).reshape(b * sk, *k.shape[2:])
    v_f = v.permute(1, 0, 2, 3).reshape(b * sk, *v.shape[2:])
    # batch-major flattening: token t = r * sk + p, one sequence per sample
    off = torch.arange(b, device=q.device).view(b, 1, 1) * sk
    idx_f = torch.where(indices >= 0, indices + off, indices.new_full((), -1)).reshape(b * s, -1)
    cu = torch.arange(b + 1, device=q.device, dtype=torch.int64) * sk
    out_f = qsa_sparse_attention_thd(q_f, k_f, v_f, idx_f, cu, scale, block_size)
    return out_f.view(b, s, hq, d).permute(1, 0, 2, 3)


class QSASparseCoreAttention(torch.nn.Module):
    """Drop-in ``core_attention`` that runs the QSA sparse triton kernel.

    The selection indices ride in on the ``attention_mask`` argument (int64) --
    the one positional that mcore threads unchanged through both the plain and
    the activation-checkpointed core-attention paths. Carrying them there (rather
    than on a module attribute) means recompute under
    ``recompute_modules=['core_attn']`` sees the same indices, and a stacked
    microbatch cannot overwrite the selection between forward and its backward.

    Any bool/None mask (or non-int tensor) falls through to the wrapped
    core_attention, keeping the no-op / short-sequence path on TE's fused kernel.

    Context parallelism (``cp_comm_type=all_gather``): q stays on the local CP
    shard while k/v are gathered across the CP group and restored to logical
    order; indices (full-sequence) are sliced to this rank's query rows. The
    gather's backward reduce-scatters dk/dv to the local shards.
    """

    def __init__(self, core_attention, config, softmax_scale=None):
        super().__init__()
        self.core_attention = core_attention
        # None -> resolved to 1/sqrt(head_dim) at call time, matching TE.
        self.softmax_scale = softmax_scale
        self.config = config
        # The kernel tests selection membership per block of compress_ratio
        # consecutive tokens, so this must be the same value the indexer expanded
        # its top-k blocks with -- see the contract in qsa_block_sparse_attn.py.
        self.block_size = config.indexer_compress_ratio
        if not self.block_size:
            raise ValueError('QSASparseCoreAttention needs config.indexer_compress_ratio to size the '
                             f'kernel block grid, got {self.block_size!r}.')

    def forward(self,
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=None,
                attention_bias=None,
                packed_seq_params=None,
                **kwargs):
        if attention_mask is not None and attention_mask.dtype in (torch.int32, torch.int64):
            scale = self.softmax_scale if self.softmax_scale is not None else query.shape[-1]**-0.5
            cp_size = self.config.context_parallel_size
            # Under CP the local query is only a shard, so _forward_cp resolves its own cu.
            cu = _packed_cu_seqlens(packed_seq_params, query.shape[0]) \
                if query.dim() == 3 and cp_size == 1 else None
            if cp_size > 1:
                out = self._forward_cp(query, key, value, attention_mask, scale, packed_seq_params, cu)
            else:
                out = qsa_sparse_attention(query, key, value, attention_mask, scale, self.block_size, cu_seqlens=cu)
            # mcore's sbhd core attention returns heads flattened ([s, b, h*d]);
            # the kernel yields [s, b, h, d]. thd stays [t, h, d] -- mcore
            # reshapes it itself right after (see attention.py thd branch).
            if out.dim() == 4:
                out = out.reshape(out.shape[0], out.shape[1], -1)
            return out
        return self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            **kwargs)

    def _forward_cp(self, query, key, value, indices, scale, packed_seq_params, cu_q):
        from megatron.core import mpu
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        device = query.device
        thd = query.dim() == 3
        if thd:
            # the gathered k/v live in the padded pack space, so the query
            # positions must come from the padded cu as well
            if cu_q is None:
                # the gathered length is the local shard times the group size
                cu_q = _packed_cu_seqlens(packed_seq_params, key.shape[0] * cp_size)
            q_pos = _cp_query_global_positions_thd(cu_q, cp_size, cp_rank, device)
            # rank-major gathered positions -> permutation back to global packed
            # order (same construction as mcore DSA's packed kv reorder)
            gathered_pos = torch.cat([_cp_query_global_positions_thd(cu_q, cp_size, r, device) for r in range(cp_size)])
            kv_reorder = torch.argsort(gathered_pos)
        else:
            sq, b = query.shape[0], query.shape[1]
            q_pos = _cp_query_global_positions(sq * cp_size, cp_size, cp_rank, device)
            kv_reorder = _cp_gathered_to_logical_order(sq * cp_size, cp_size, device)
        # gather k/v across CP with a DIFFERENTIABLE all-gather (backward is a
        # reduce-scatter, so dk/dv reach the local shards) and undo the zigzag
        # shard into full logical order. A raw torch.distributed.all_gather is
        # not differentiable -- reconstruct_tensor_cp is not usable here.
        cp_group = mpu.get_context_parallel_group()
        kv_reorder_t = kv_reorder

        def _gather_full(t):
            g = gather_from_sequence_parallel_region(t, tensor_parallel_output_grad=True, group=cp_group)
            return g.index_select(0, kv_reorder_t)

        key_full = _gather_full(key)
        value_full = _gather_full(value)
        if thd and int(cu_q[-1]) != key_full.shape[0]:
            raise ValueError(f'QSA thd CP: cu_seqlens ends at {int(cu_q[-1])} but the gathered keys have '
                             f'{key_full.shape[0]} tokens; packed_seq_params do not match this shard.')
        # The kernel derives the key bound from the query count (T, Hq, D = q.shape,
        # then `offs_k < T`), so it structurally requires len(q) == len(k). Under CP
        # the queries are a 1/cp_size shard while k/v are now full length, so scatter
        # the local queries back into a full-length buffer, run, and take our rows
        # out again. The padding rows carry an all-`-1` selection, which the kernel
        # skips, so they cost tile launches but produce nothing.
        if thd:
            local_idx = indices[q_pos]
            out_full = qsa_sparse_attention(
                *self._scatter_q_to_full(query, key_full, value_full, local_idx, q_pos),
                scale,
                self.block_size,
                cu_seqlens=cu_q)
            return out_full.index_select(0, q_pos)
        # sbhd: token-space kernel on the batch-major flattening (t = r*sk + p)
        local_idx = indices[:, q_pos]
        sk = key_full.shape[0]
        k_f = key_full.permute(1, 0, 2, 3).reshape(b * sk, key_full.shape[2], key_full.shape[3])
        v_f = value_full.permute(1, 0, 2, 3).reshape(b * sk, value_full.shape[2], value_full.shape[3])
        off = torch.arange(b, device=device).view(b, 1, 1) * sk
        idx_f = torch.where(local_idx >= 0, local_idx + off, local_idx.new_full((), -1)).reshape(sq * b, -1)
        q_f = query.permute(1, 0, 2, 3).reshape(sq * b, query.shape[2], query.shape[3])
        # batch-major token ids of this rank's rows: sample r contributes q_pos + r*sk
        rows = (q_pos[None, :] + torch.arange(b, device=device).view(b, 1) * sk).reshape(-1)
        # each sample is one sequence of length sk in the flattened buffer
        cu = torch.arange(b + 1, device=device, dtype=torch.int64) * sk
        out_f = qsa_sparse_attention(
            *self._scatter_q_to_full(q_f, k_f, v_f, idx_f, rows), scale, self.block_size, cu_seqlens=cu)
        out_f = out_f.index_select(0, rows)
        return out_f.view(b, sq, query.shape[2], query.shape[3]).permute(1, 0, 2, 3)

    @staticmethod
    def _scatter_q_to_full(q, k, v, indices, rows):
        """Place ``q``/``indices`` rows at ``rows`` inside a len(k)-row buffer.

        index_copy keeps this differentiable: backward gathers the same rows, so the
        padded positions contribute no gradient.
        """
        n = k.shape[0]
        q_full = q.new_zeros((n, *q.shape[1:]))
        q_full = q_full.index_copy(0, rows, q)
        idx_full = indices.new_full((n, indices.shape[1]), -1)
        idx_full = idx_full.index_copy(0, rows, indices)
        return q_full, k, v, idx_full
