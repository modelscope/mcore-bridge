# Copyright (c) ModelScope Contributors. All rights reserved.
"""QSA sparse attention wrappers around the vendored tensor-core triton kernel.

The kernel itself lives in ``qsa_block_sparse_attn.py``, vendored verbatim from
miles PR #2777 (commit 0f5dff4). This file owns only the glue mcore needs:
sbhd<->thd flattening, context parallelism, and the ``core_attention`` shim.

WHY A KERNEL AT ALL (the decision this file encodes)
    QSA needs a per-query key set, which as an attention mask is TE's
    ``arbitrary`` type. TE refuses that under packing:
    ``dot_product_attention.py:1347`` asserts ``"padding" in attn_mask_type``
    whenever ``qkv_format == "thd"``, so a bool mask cannot express QSA there.
    Context parallelism is blocked for a different reason -- block pooling needs
    keys from other CP ranks, which the mask path never gathers.

    So the layer picks by data shape, with no user-facing switch:

        thd (padding_free) or CP>1  ->  this kernel
        sbhd and CP==1              ->  the indexer's bool mask on TE

    The mask path is not a degraded fallback in that last case: both paths run
    the same ``QSAIndexer._score_and_topk_blocks`` top-block selection, so they are
    mathematically equivalent there, and the mask path measures *faster*
    (8192: 8.4 s/it vs 25.9 s/it) because TE keeps the whole thing fused.
"""
import torch

from .qsa_block_sparse_attn import qsa_sparse_attention_from_indices

try:
    import triton  # noqa: F401  (import guard for the vendored kernel)
    HAVE_TRITON = True
except Exception:  # pragma: no cover - triton absent
    HAVE_TRITON = False


def qsa_sparse_supported(head_dim: int) -> bool:
    """Whether the sparse kernel can run for this head dim.

    Triton must be importable and ``head_dim`` must be a power of two (the
    kernel tiles the head with ``tl.arange`` blocks).
    """
    return HAVE_TRITON and head_dim > 0 and not (head_dim & (head_dim - 1))


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


def _cp_query_global_positions_thd(cu_seqlens: torch.Tensor, cp_size: int, cp_rank: int,
                                   device) -> torch.Tensor:
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



def qsa_sparse_attention_thd(q, k, v, indices, scale, block_size):
    """``q`` [T, Hq, D], ``k``/``v`` [S, Hkv, D], ``indices`` [T, K] (-1 pad).

    Token-space indices: works for packed (thd) inputs directly, and for any
    pre-flattened token dimension.

    ``block_size`` must be the indexer's ``compress_ratio``. The kernel tests
    selection membership per block of that many consecutive tokens, so passing a
    different value silently changes which keys are attended -- see the contract
    note in qsa_block_sparse_attn.py.
    """
    if not HAVE_TRITON or not q.is_cuda:
        raise RuntimeError(
            'QSA sparse attention requires triton and CUDA tensors '
            f'(HAVE_TRITON={HAVE_TRITON}, q.is_cuda={q.is_cuda}). This path is only '
            'selected for packing (thd) or CP>1, where no dense fallback is correct.')
    if q.shape[-1] & (q.shape[-1] - 1):
        raise RuntimeError(f'QSA sparse attention needs a power-of-two head dim, got {q.shape[-1]}.')
    if q.shape[0] != k.shape[0]:
        # The kernel takes its key bound from the query count (T, Hq, D = q.shape,
        # then `offs_k < T`), so unequal lengths would silently drop every key past
        # len(q). Callers must equalise first -- _forward_cp does this by scattering
        # the local query shard into a full-length buffer.
        raise ValueError(
            f'QSA sparse attention needs len(q) == len(k), got {q.shape[0]} vs {k.shape[0]}.')
    return qsa_sparse_attention_from_indices(q, k, v, indices.contiguous(), scale, block_size)


def qsa_sparse_attention(q, k, v, indices, scale, block_size):
    """QSA sparse attention over mcore layouts.

    sbhd: ``q`` [s, b, Hq, D], ``k``/``v`` [s, b, Hkv, D], ``indices``
        [b, s, K] in per-sample sequence space.
    thd:  ``q`` [T, Hq, D], ``k``/``v`` [T, Hkv, D], ``indices`` [T, K] in
        pack space (document boundaries already clamped torch-side).

    Raises rather than returning ``None``: the caller only reaches here for
    packing or CP>1, and for those a dense fallback would silently diverge from
    sparse inference.
    """
    if q.dim() == 3:
        return qsa_sparse_attention_thd(q, k, v, indices, scale, block_size)
    if q.dim() != 4:
        raise ValueError(f'qsa_sparse_attention expected 3D (thd) or 4D (sbhd) q, got {tuple(q.shape)}')
    s, b, hq, d = q.shape
    sk = k.shape[0]
    # Flatten to token space, batch-major: token t = r * sk + p. The per-sample
    # sequence-space indices are offset by the batch start; -1 padding stays -1.
    # The offset is a whole multiple of sk, so block alignment survives it only
    # when sk % block_size == 0; guard rather than corrupt the selection.
    if sk % block_size:
        raise ValueError(
            f'sbhd QSA needs the kv sequence length ({sk}) to be a multiple of '
            f'block_size ({block_size}); otherwise flattening to token space shifts '
            'each sample off the block grid the kernel indexes by.')
    q_f = q.permute(1, 0, 2, 3).reshape(b * s, hq, d)
    k_f = k.permute(1, 0, 2, 3).reshape(b * sk, *k.shape[2:])
    v_f = v.permute(1, 0, 2, 3).reshape(b * sk, *v.shape[2:])
    off = torch.arange(b, device=q.device).view(b, 1, 1) * sk
    idx_f = torch.where(indices >= 0, indices + off, indices.new_full((), -1)).reshape(b * s, -1)
    out_f = qsa_sparse_attention_thd(q_f, k_f, v_f, idx_f, scale, block_size)
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

    Context parallelism (``cp_comm_type=allgather``): q stays on the local CP
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
            raise ValueError(
                'QSASparseCoreAttention needs config.indexer_compress_ratio to size the '
                f'kernel block grid, got {self.block_size!r}.')

    def forward(self, query, key, value, attention_mask, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None, **kwargs):
        if attention_mask is not None and attention_mask.dtype in (torch.int32, torch.int64):
            scale = self.softmax_scale if self.softmax_scale is not None else query.shape[-1]**-0.5
            cp_size = self.config.context_parallel_size
            if cp_size > 1:
                out = self._forward_cp(query, key, value, attention_mask, scale, packed_seq_params)
            else:
                out = qsa_sparse_attention(query, key, value, attention_mask, scale, self.block_size)
            # mcore's sbhd core attention returns heads flattened ([s, b, h*d]);
            # the kernel yields [s, b, h, d]. thd stays [t, h, d] -- mcore
            # reshapes it itself right after (see attention.py thd branch).
            if out.dim() == 4:
                out = out.reshape(out.shape[0], out.shape[1], -1)
            return out
        return self.core_attention(
            query, key, value, attention_mask, attn_mask_type=attn_mask_type,
            attention_bias=attention_bias, packed_seq_params=packed_seq_params, **kwargs)

    def _forward_cp(self, query, key, value, indices, scale, packed_seq_params):
        from megatron.core import mpu
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        device = query.device
        thd = query.dim() == 3
        if thd:
            # the gathered k/v live in the padded pack space, so the query
            # positions must come from the padded cu as well
            cu_q = (packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params.cu_seqlens_q_padded is not None else packed_seq_params.cu_seqlens_q)
            q_pos = _cp_query_global_positions_thd(cu_q, cp_size, cp_rank, device)
            # rank-major gathered positions -> permutation back to global packed
            # order (same construction as mcore DSA's packed kv reorder)
            gathered_pos = torch.cat([
                _cp_query_global_positions_thd(cu_q, cp_size, r, device) for r in range(cp_size)])
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
            g = gather_from_sequence_parallel_region(
                t, tensor_parallel_output_grad=True, group=cp_group)
            return g.index_select(0, kv_reorder_t)

        key_full = _gather_full(key)
        value_full = _gather_full(value)
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
                scale, self.block_size)
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
        out_f = qsa_sparse_attention(
            *self._scatter_q_to_full(q_f, k_f, v_f, idx_f, rows), scale, self.block_size)
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

