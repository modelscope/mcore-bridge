# Copyright (c) ModelScope Contributors. All rights reserved.
"""DeepSeek-V4.1 adapters for NVIDIA Megatron-Core's optional Engram modules."""

import dataclasses
from contextlib import contextmanager

import torch
from torch import Tensor

from ...utils.megatron_utils import split_cp_inputs

try:
    from megatron.core import mpu
    from megatron.core.models.engram.config import EngramConfig
    from megatron.core.models.engram.engram import Engram
    from megatron.core.models.engram.hashing import (
        compress_token_ids,
        shift_right_reset_at_eos,
        slice_hashes_for_sequence_parallel,
    )
    from megatron.core.transformer.transformer_layer import (
        HyperConnectionTransformerLayer,
        TransformerLayer,
    )
    from megatron.core.utils import get_pg_size, nvtx_range_pop, nvtx_range_push
except ImportError:
    EngramConfig = None
    Engram = torch.nn.Module
    HyperConnectionTransformerLayer = None
    TransformerLayer = None


def has_native_engram() -> bool:
    """Return whether the installed Megatron-Core includes the official Engram extension."""
    return EngramConfig is not None


class _RelaxedParallelismView:
    """Read-only view of a transformer config that hides the CP and VPP guards.

    Lets the Engram validators reuse every upstream parallelism check except the two
    DeepSeek-V4.1 has shown safe to drop, without mutating the shared transformer config:

    * ``context_parallel_size``: V4.1 hashes the full sequence locally and then slices its own
      CP interval, so no n-gram window ever crosses a CP rank boundary.
    * ``virtual_pipeline_model_parallel_size``: ``Engram.forward`` is self-contained (it never
      exchanges state across pipeline/VP stages), and its layer placement keys off the
      vp_stage-aware global ``layer_number`` that ``select_pipeline_segment`` assigns to each
      chunk. The hybrid stack's ``__init__`` block-alignment guard rejects any partial-block
      stage at build time, so an Engram-carrying ``D`` layer can never be split across stages.
    """

    context_parallel_size = 1
    virtual_pipeline_model_parallel_size = None

    def __init__(self, transformer_config):
        self._transformer_config = transformer_config

    def __getattr__(self, name):
        return getattr(self._transformer_config, name)


if EngramConfig is not None:

    class DeepseekV41EngramConfig(EngramConfig):
        """Translate 0-based checkpoint hash IDs to 1-based Megatron placement IDs."""

        def __init__(self, *, placement_layer_ids, hash_layer_ids, excluded_token_ids=(), **kwargs):
            self.hash_layer_ids = tuple(hash_layer_ids)
            placement_layer_ids = tuple(placement_layer_ids)
            if len(placement_layer_ids) != len(self.hash_layer_ids):
                raise ValueError('Engram placement and hash layer IDs must have the same length.')
            self.excluded_token_ids = tuple(excluded_token_ids)
            super().__init__(layer_ids=placement_layer_ids, **kwargs)
            hash_multipliers = self.layer_multipliers
            self.layer_multipliers = {
                placement: hash_multipliers[source]
                for placement, source in zip(placement_layer_ids, self.hash_layer_ids)
            }

        def _load_tokenizer_map(self):
            # The artifact and hash multipliers use checkpoint-native 0-based IDs, while the
            # TransformerLayer composition point validates and selects 1-based layer numbers.
            placement_layer_ids = self.layer_ids
            self.layer_ids = self.hash_layer_ids
            try:
                return super()._load_tokenizer_map()
            finally:
                self.layer_ids = placement_layer_ids

        def _validate_parallelism(self, transformer_config, sequence_length):
            # Drop only the upstream CP and VPP rejections (see _RelaxedParallelismView for why
            # both are safe on the V4.1 hybrid path) and keep every other check -- etp==tp, the
            # SP rank-local history length, etc. CP shortens the rank-local slice the SP check
            # compares against, so apply it here before delegating.
            context_parallel_size = transformer_config.context_parallel_size
            if sequence_length is not None:
                # The SP checks compare against a rank-local slice, which CP shortens first.
                sequence_length = sequence_length // context_parallel_size
            super()._validate_parallelism(
                _RelaxedParallelismView(transformer_config), sequence_length)

        def _validate_packed_sequences(self, transformer_config, packed_sequences):
            # DeepseekV41Engram restarts its n-gram windows at every cu_seqlens document
            # boundary, so packed (THD) rows are safe even though the upstream DeepSeek
            # variant advertises resets_windows_at_boundary_token=False. Keep the remaining
            # packed checks (pipeline stages, padding alignment) from super().
            variant_spec = self.variant_spec
            self.variant_spec = dataclasses.replace(
                variant_spec, resets_windows_at_boundary_token=True)
            try:
                super()._validate_packed_sequences(transformer_config, packed_sequences)
            finally:
                self.variant_spec = variant_spec
else:
    DeepseekV41EngramConfig = None


def build_deepseek_v41_engram_config(**kwargs):
    """Build the DeepSeek adapter over NVIDIA's official Engram configuration."""
    if DeepseekV41EngramConfig is None:
        raise RuntimeError(
            'DeepSeek-V4.1 Engram requires NVIDIA Megatron-LM Engram support. '
            'Install the official Engram extension or disable Engram.')
    return DeepseekV41EngramConfig(**kwargs)


def _hash_token_windows(
    token_windows: Tensor,
    tokenizer_remap: Tensor | None,
    multipliers: Tensor,
    table_sizes: Tensor,
    max_ngram_order: int,
    num_hash_heads: int,
    boundary_token_id: int,
    invalid_token_id: int | None = None,
) -> Tensor:
    if token_windows.ndim != 3 or token_windows.shape[-1] != max_ngram_order:
        raise ValueError(
            'Engram token windows must have shape [batch, sequence, max_ngram_order], '
            f'got {token_windows.shape}.')
    tokens = token_windows.to(torch.int64)
    compressed = tokens if tokenizer_remap is None else compress_token_ids(tokens, tokenizer_remap)
    suffixes = []
    blocked = torch.zeros_like(compressed[..., 0], dtype=torch.bool)
    for shift in range(max_ngram_order):
        source = compressed[..., shift]
        if invalid_token_id is not None:
            blocked = blocked | (source == invalid_token_id)
            source = torch.where(blocked, source.new_full((), boundary_token_id), source)
        suffixes.append(source)

    hashes = []
    table_index = 0
    for order in range(2, max_ngram_order + 1):
        mixed = suffixes[0] * multipliers[0]
        for suffix_index in range(1, order):
            mixed = torch.bitwise_xor(mixed, suffixes[suffix_index] * multipliers[suffix_index])
        for _ in range(num_hash_heads):
            hashes.append(torch.remainder(mixed, table_sizes[table_index]))
            table_index += 1
    return torch.stack(hashes, dim=-1)


def _positions_in_segment(cu_seqlens: Tensor | None, sequence_length: int, device) -> Tensor:
    """Distance from each position to the start of the document that contains it.

    Without ``cu_seqlens`` the whole row is one document, so this is just the position
    index and the window is only padded at the row start.
    """
    positions = torch.arange(sequence_length, device=device, dtype=torch.int64)
    if cu_seqlens is None:
        return positions
    boundaries = cu_seqlens.reshape(-1).to(device=device, dtype=torch.int64)
    if int(boundaries[-1]) != sequence_length:
        raise ValueError(
            f'Engram cu_seqlens ends at {int(boundaries[-1])} but the hashed sequence has '
            f'{sequence_length} tokens; cu_seqlens must describe the full packed row.')
    segment_index = torch.searchsorted(boundaries, positions, right=True) - 1
    return positions - boundaries[segment_index]


def _build_ngram_hashes(
    input_ids: Tensor,
    tokenizer_remap: Tensor | None,
    multipliers: Tensor,
    table_sizes: Tensor,
    max_ngram_order: int,
    num_hash_heads: int,
    boundary_token_id: int,
    reset_at_boundary: bool,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    tokens = input_ids.to(torch.int64)
    compressed = tokens if tokenizer_remap is None else compress_token_ids(tokens, tokenizer_remap)
    sequence_length = compressed.shape[1]
    if reset_at_boundary:
        suffixes = [
            shift_right_reset_at_eos(compressed, shift, boundary_token_id)
            for shift in range(max_ngram_order)
        ]
    else:
        # The DeepSeek variant carries no boundary token in the stream, so packed (THD) rows
        # need the document starts from cu_seqlens to keep n-grams inside one document. With
        # cu_seqlens=None this reduces exactly to padding the window at the row start.
        if cu_seqlens is not None and compressed.shape[0] != 1:
            raise ValueError(
                'Engram cu_seqlens-based window reset expects a single packed row, got '
                f'batch size {compressed.shape[0]}.')
        position_in_segment = _positions_in_segment(
            cu_seqlens, sequence_length, compressed.device).unsqueeze(0)
        suffixes = [compressed]
        for shift in range(1, max_ngram_order):
            shifted = torch.nn.functional.pad(
                compressed, (shift, 0), value=boundary_token_id)[:, :sequence_length]
            suffixes.append(
                torch.where(position_in_segment >= shift, shifted,
                            shifted.new_full((), boundary_token_id)))
    return _hash_token_windows(
        torch.stack(suffixes, dim=-1),
        tokenizer_remap=None,
        multipliers=multipliers,
        table_sizes=table_sizes,
        max_ngram_order=max_ngram_order,
        num_hash_heads=num_hash_heads,
        boundary_token_id=boundary_token_id,
        invalid_token_id=-1,
    )


class DeepseekV41Engram(Engram):
    """DeepSeek V4.1 Engram without projection bias or the Qwen short-conv branch."""

    def __init__(self, *args, **kwargs):
        if EngramConfig is None:
            raise RuntimeError('The installed Megatron-Core does not provide Engram.')
        super().__init__(*args, **kwargs)
        # PR #7231's DeepSeek variant predates the released V4.1 checkpoint layout.
        # Keep its initialized weights but remove parameters absent from that checkpoint.
        self.value_projection.register_parameter('bias', None)
        self.key_projection.register_parameter('bias', None)
        self.conv_norm = None
        self.short_conv = None

    def _mask_excluded_tokens(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        live = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in self.engram_config.excluded_token_ids:
            live = live & (input_ids != token_id)
        return torch.where(live, input_ids, input_ids.new_full((), -1)), live

    def _hash_windows(self, token_windows: Tensor) -> Tensor:
        return _hash_token_windows(
            token_windows,
            self.tokenizer_remap,
            self.hash_multipliers,
            self.table_sizes,
            self.engram_config.max_ngram_order,
            self.engram_config.num_hash_heads,
            self.engram_config.hash_boundary_token_id,
            invalid_token_id=-1,
        )

    def _static_inference_hashes(self, input_ids: Tensor, context) -> tuple[Tensor, Tensor]:
        batch_size, sequence_length = input_ids.shape
        shape = (context.max_batch_size, context.max_sequence_length)
        cache = getattr(context, 'engram_token_cache', None)
        if cache is None or cache.device != input_ids.device or cache.shape != shape:
            cache = input_ids.new_full(shape, self.engram_config.boundary_token_id)
            context.engram_token_cache = cache

        batch_start = context.batch_size_offset
        batch_end = batch_start + batch_size
        sequence_start = context.sequence_len_offset
        sequence_end = sequence_start + sequence_length
        if batch_end > shape[0] or sequence_end > shape[1]:
            raise ValueError('Engram inference token cache is too small for the current batch/chunk.')
        masked_ids, live = self._mask_excluded_tokens(input_ids)
        cache[batch_start:batch_end, sequence_start:sequence_end] = masked_ids
        shifts = torch.arange(
            self.engram_config.max_ngram_order, device=input_ids.device, dtype=torch.long)
        positions = torch.arange(
            sequence_start, sequence_end, device=input_ids.device, dtype=torch.long).unsqueeze(-1) - shifts
        gather_positions = positions.clamp_min(0).reshape(1, -1).expand(batch_size, -1)
        windows = cache[batch_start:batch_end].gather(1, gather_positions).view(
            batch_size, sequence_length, self.engram_config.max_ngram_order)
        windows = torch.where(
            positions.unsqueeze(0) >= 0,
            windows,
            windows.new_full((), self.engram_config.boundary_token_id),
        )
        return self._hash_windows(windows), live

    def _dynamic_inference_hashes(self, input_ids: Tensor, context) -> tuple[Tensor, Tensor]:
        if input_ids.shape[0] != 1:
            raise ValueError('Dynamic Engram inference expects flattened input_ids with batch size 1.')
        shape = (context.max_requests, context.max_sequence_length)
        cache = getattr(context, 'engram_token_cache', None)
        if cache is None or cache.device != input_ids.device or cache.shape != shape:
            cache = input_ids.new_full(shape, self.engram_config.boundary_token_id)
            context.engram_token_cache = cache

        total_tokens = input_ids.shape[1]
        active_tokens = min(int(context.active_token_count), total_tokens)
        live = torch.zeros_like(input_ids, dtype=torch.bool)
        hashes = input_ids.new_zeros((1, total_tokens, self.engram_config.num_tables), dtype=torch.long)
        if active_tokens == 0:
            return hashes, live
        request_indices = context.gpu_view.token_to_request_idx[:active_tokens].long()
        token_positions = context.gpu_view.token_to_position_in_request[:active_tokens].long()
        if request_indices.min() < 0 or request_indices.max() >= shape[0]:
            raise ValueError('Dynamic Engram inference received an out-of-range request index.')
        if token_positions.min() < 0 or token_positions.max() >= shape[1]:
            raise ValueError('Dynamic Engram inference received an out-of-range token position.')
        masked_ids, active_live = self._mask_excluded_tokens(input_ids[:, :active_tokens])
        cache[request_indices, token_positions] = masked_ids.squeeze(0)
        shifts = torch.arange(
            self.engram_config.max_ngram_order, device=input_ids.device, dtype=torch.long)
        positions = token_positions.unsqueeze(-1) - shifts
        windows = cache[request_indices.unsqueeze(-1).expand_as(positions), positions.clamp_min(0)]
        windows = torch.where(
            positions >= 0,
            windows,
            windows.new_full((), self.engram_config.boundary_token_id),
        )
        hashes[:, :active_tokens] = self._hash_windows(windows.unsqueeze(0))
        live[:, :active_tokens] = active_live
        return hashes, live

    def _build_hash_ids(self, input_ids: Tensor, inference_context=None,
                        cu_seqlens: Tensor | None = None) -> tuple[Tensor, Tensor]:
        masked_ids, live = self._mask_excluded_tokens(input_ids)
        if inference_context is None:
            hashes = _build_ngram_hashes(
                masked_ids,
                self.tokenizer_remap,
                self.hash_multipliers,
                self.table_sizes,
                self.engram_config.max_ngram_order,
                self.engram_config.num_hash_heads,
                self.engram_config.hash_boundary_token_id,
                self.engram_config.variant_spec.resets_windows_at_boundary_token,
                cu_seqlens=cu_seqlens,
            )
            return hashes, live
        if inference_context.is_static_batching():
            return self._static_inference_hashes(input_ids, inference_context)
        return self._dynamic_inference_hashes(input_ids, inference_context)

    def _cp_local_sequence_length(self, hidden_states: Tensor) -> int:
        """Length of this rank's CP slice, undoing the innermost SP split first."""
        length = hidden_states.shape[0]
        if self.config.sequence_parallel:
            length *= get_pg_size(self.tp_group)
        return length

    def _gather_input_ids_for_context_parallel(self, input_ids: Tensor,
                                              local_sequence_length: int) -> Tensor:
        """Restore the full token sequence so every rank hashes identical n-gram windows.

        The data pipeline hands us either a CP-sharded copy of ``input_ids`` (swift
        ``get_batch_on_this_cp_rank`` splits it for text models) or the full sequence
        (multimodal models keep it whole and split the embeddings instead), so re-align
        only when the lengths disagree. Gathering int64 token IDs is far cheaper than the
        hidden states the surrounding attention already exchanges.
        """
        cp_size = self.config.context_parallel_size
        present = input_ids.shape[1]
        if present == local_sequence_length * cp_size:
            return input_ids
        if present != local_sequence_length:
            raise ValueError(
                f'Engram input_ids length {present} matches neither this CP rank slice '
                f'({local_sequence_length}) nor the full sequence '
                f'({local_sequence_length * cp_size}).')
        shards = [torch.empty_like(input_ids) for _ in range(cp_size)]
        torch.distributed.all_gather(
            shards, input_ids.contiguous(), group=mpu.get_context_parallel_group())
        # Contiguous partitioning gives rank r the block [r * local, (r + 1) * local), so
        # concatenating the gathered shards in rank order rebuilds the original token order.
        return torch.cat(shards, dim=1)

    def _slice_for_context_parallel(self, hashes: Tensor) -> Tensor:
        """Select this rank's CP interval after the hashes were computed globally."""
        if self.config.context_parallel_size == 1:
            return hashes
        return split_cp_inputs(hashes, None, 1, cp_partition_mode='contiguous')

    def forward(self, hidden_states: Tensor, input_ids: Tensor, inference_context=None) -> Tensor:
        if inference_context is None:
            inference_context = getattr(self, '_bridge_inference_context', None)
        packed_seq_params = getattr(self, '_bridge_packed_seq_params', None)
        if hidden_states.ndim != 3:
            raise ValueError(f'Engram hidden_states must be [S,B,H], got {hidden_states.shape}.')
        expected_hidden = self.num_streams * self.hidden_size
        if hidden_states.shape[-1] != expected_hidden:
            raise ValueError(f'Engram expected hidden width {expected_hidden}, got {hidden_states.shape[-1]}.')
        context_parallel = self.config.context_parallel_size > 1
        if context_parallel and inference_context is not None:
            raise ValueError('Engram inference does not support context parallelism.')

        cu_seqlens = None
        if packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', None) == 'thd':
            # cu_seqlens_q stays global: the data pipeline builds it before the CP split.
            cu_seqlens = getattr(packed_seq_params, 'cu_seqlens_q', None)
            cp_partition_mode = getattr(packed_seq_params, 'cp_partition_mode', 'zigzag')
        else:
            # Non-packed CP carries the partition layout on the transformer config instead of on
            # packed_seq_params (see mm_gpt_model's CP data path).
            cp_partition_mode = getattr(self.config, 'cp_partition_mode', 'zigzag')
        if context_parallel and cp_partition_mode != 'contiguous':
            # Both _gather_input_ids_for_context_parallel (rank-order concat) and
            # _slice_for_context_parallel (contiguous slice) assume contiguous CP blocks, so a
            # zigzag layout would silently mis-align the hashes with the local hidden states.
            # Fail loud on every CP path -- packed (THD) and non-packed alike -- matching the
            # DSv4 THD CP forward.
            raise ValueError(
                "Engram with context parallelism requires cp_partition_mode='contiguous', "
                'matching the DSv4 THD CP forward.')

        nvtx_range_push('engram.hash')
        try:
            if context_parallel:
                input_ids = self._gather_input_ids_for_context_parallel(
                    input_ids, self._cp_local_sequence_length(hidden_states))
            hash_ids, live_tokens = self._build_hash_ids(input_ids, inference_context, cu_seqlens)
            live_tokens = live_tokens.unsqueeze(-1)
            # CP is the outer split and SP the inner one, so undo them in that order.
            hash_ids = self._slice_for_context_parallel(hash_ids)
            live_tokens = self._slice_for_context_parallel(live_tokens)
            hash_ids = slice_hashes_for_sequence_parallel(hash_ids, hidden_states.shape[0], self.tp_group)
            live_tokens = slice_hashes_for_sequence_parallel(
                live_tokens, hidden_states.shape[0], self.tp_group).squeeze(-1)
        finally:
            nvtx_range_pop('engram.hash')

        nvtx_range_push('engram.lookup')
        try:
            memory = self.embedding(hash_ids).flatten(start_dim=-2).transpose(0, 1).contiguous()
        finally:
            nvtx_range_pop('engram.lookup')
        streams = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_streams, self.hidden_size)
        shared_value = self.value_projection(memory)
        key = self.key_norm(self.key_projection(memory)).view_as(streams)
        query = self.query_norm(hidden_states).view_as(streams)
        score = (key * query).sum(dim=-1) / self.hidden_size**0.5
        score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
        output = score.sigmoid().unsqueeze(-1) * shared_value.unsqueeze(2)
        output = output * live_tokens.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        return output.reshape(hidden_states.shape)


class _DeepseekV41EngramLayerMixin:

    def _forward_attention(self, *args, **kwargs):
        engram = getattr(self, 'engram', None)
        if engram is None:
            return super()._forward_attention(*args, **kwargs)
        # `_maybe_apply_engram` only forwards hidden_states and input_ids, so stash the
        # per-microbatch context the Engram needs (inference context, THD cu_seqlens) on the
        # module itself for the duration of this attention call.
        previous = getattr(engram, '_bridge_inference_context', None)
        previous_packed = getattr(engram, '_bridge_packed_seq_params', None)
        engram._bridge_inference_context = kwargs.get('inference_context')
        engram._bridge_packed_seq_params = kwargs.get('packed_seq_params')
        try:
            return super()._forward_attention(*args, **kwargs)
        finally:
            engram._bridge_inference_context = previous
            engram._bridge_packed_seq_params = previous_packed


if TransformerLayer is not None:

    class DeepseekV41TransformerLayer(_DeepseekV41EngramLayerMixin, TransformerLayer):
        pass


    class DeepseekV41HyperConnectionTransformerLayer(
            _DeepseekV41EngramLayerMixin, HyperConnectionTransformerLayer):
        pass
else:
    DeepseekV41TransformerLayer = None
    DeepseekV41HyperConnectionTransformerLayer = None


def adapt_deepseek_v41_layer_specs(transformer_layer_spec, engram_config):
    """Attach the V4.1 Engram module and inference-aware layer subclasses."""
    if not has_native_engram():
        raise RuntimeError('The installed Megatron-Core does not provide Engram.')
    from megatron.core.transformer.spec_utils import ModuleSpec

    engram_spec = ModuleSpec(module=DeepseekV41Engram, params={'engram_config': engram_config})
    for layer_spec in transformer_layer_spec.layer_specs:
        if layer_spec.module is HyperConnectionTransformerLayer:
            layer_spec.module = DeepseekV41HyperConnectionTransformerLayer
        elif layer_spec.module is TransformerLayer:
            layer_spec.module = DeepseekV41TransformerLayer
        if not hasattr(layer_spec.submodules, 'engram'):
            raise RuntimeError(
                'The installed Engram extension does not expose TransformerLayerSubmodules.engram.')
        layer_spec.submodules.engram = engram_spec
    return transformer_layer_spec


@contextmanager
def allow_engram_inference(model_config, input_ids, extra_block_kwargs):
    """Bypass PR #7231's inference guard while preserving input_ids propagation."""
    if not getattr(model_config, 'engram_enabled', False):
        yield extra_block_kwargs
        return
    if input_ids is None:
        raise ValueError('Engram requires input token IDs on every pipeline stage.')
    block_kwargs = dict(extra_block_kwargs or {})
    block_kwargs['input_ids'] = input_ids
    model_config.engram_enabled = False
    try:
        yield block_kwargs
    finally:
        model_config.engram_enabled = True
