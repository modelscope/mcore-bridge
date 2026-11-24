# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing
from megatron.core.tensor_parallel import VocabParallelEmbedding
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.utils import (ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint,
                                             sharded_state_dict_default)
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import List, NamedTuple, Optional

from ...utils import get_env_args, get_logger, use_ngram_embedding_cpu_offload
from ...utils.constants import EXPORT_CHUNK_BYTES
from ...utils.megatron_utils import get_num_samples, reconstruct_tensor_cp
from .hyper_connection_gated import Qwen4ExpTextGroupedRMSNorm
from .kernels import gather_ple_rows, ple_gate_conv_triton

_MASK64 = (1 << 64) - 1
# Cap the per-all_reduce GPU buffer when exporting the (potentially huge) ngram
# table.
_EXPORT_CHUNK_BYTES = EXPORT_CHUNK_BYTES
# The gate/conv chain keeps ~16 B per (token x width) alive, so the chunk length follows from a
# 1 GiB budget instead of a token count.
_PLE_CHUNK_BYTES = 1 << 30
_PLE_CHUNK_BYTES_PER_ELEM = 16
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def use_ple_fused_kernel() -> bool:
    # On by default; PLE_FUSED_KERNEL=0 forces the eager chain.
    return get_env_args('PLE_FUSED_KERNEL', bool, True)


def ple_chunk_tokens(width: int) -> int:
    """Tokens of one chunked PLE call, from a byte budget over ``width`` (= hc*hidden)."""
    return max(1, _PLE_CHUNK_BYTES // max(1, width * _PLE_CHUNK_BYTES_PER_ELEM))


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> List[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return multipliers


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    # Mirrors transformers `_find_nth_prime_after`.
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class Qwen4ExpTextNGramEmbedding(nn.Module):

    def __init__(self, config, ple_layer_index: int):
        super().__init__()
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        # (transformers also stores unigram_vocab_size/ngram_vocab_size_base/
        # seed as attributes, consumed only by its deferred _init_weights
        # buffer population; here the buffers are registered at construction,
        # so no copies are kept.)
        head_dim_per_ngram = config.ple_embed_dim // self.ngram_heads
        # mcore-specific fail-loud: these drive the hash math and the checkpoint
        # shard layout, so they must come from the model config (the parser
        # validates them for qwen4_exp); a silently substituted default would
        # corrupt the weight conversion. (transformers reads eos_token_id
        # directly and has no split_ngram_parts: its table is replicated.)
        eos_token_id = config.eos_token_id
        split_ngram_parts = config.split_ngram_parts
        ple_seed = getattr(config, 'ple_seed', None)
        if ple_seed is None:
            raise ValueError('ple_seed must be provided by the model config (the parser derives it from '
                             'text_config.seed); a silently substituted default would desynchronize the '
                             'n-gram hash multipliers from transformers.')
        if eos_token_id is None or split_ngram_parts is None:
            raise ValueError(f'eos_token_id/split_ngram_parts must be provided by the model '
                             f'config (got {eos_token_id!r}/{split_ngram_parts!r}).')
        self.eos_token_id = int(eos_token_id)
        self.split_ngram_parts = int(split_ngram_parts)
        self.head_dim = head_dim_per_ngram  # mcore-specific: the bridge weight conversion reads it off the module.

        # Multipliers (splitmix64 derived, checkpoint-persistent). The seed comes
        # from the config (parser defaults it to Qwen4ExpTextConfig.seed's 1234);
        # the multipliers must stay bit-identical to transformers or the n-gram
        # lookups desynchronize.
        multipliers = _build_layer_multipliers(config.padded_vocab_size, self.ngram_size, self.ple_layer_index,
                                               int(ple_seed))
        self.register_buffer('layer_multipliers', torch.tensor(multipliers, dtype=torch.long), persistent=True)

        # Per-head prime table sizes/offsets (checkpoint-persistent), named as
        # in transformers.
        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _nth_prime_after(config.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size
        self.register_buffer(
            'ngram_heads_vocab_sizes', torch.tensor(self.head_vocab_sizes, dtype=torch.long), persistent=True)
        self.register_buffer('ngram_heads_offsets', torch.tensor(self.head_offsets, dtype=torch.long), persistent=True)
        ngram_vocab_divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(self.total_vocab_size / ngram_vocab_divisor) * ngram_vocab_divisor
        self.padded_vocab_size = padded_vocab_size
        self.cpu_offload = use_ngram_embedding_cpu_offload()
        if self.cpu_offload:
            # Warn rather than raise: whether this is safe depends on the train type,
            # which the model config does not carry (see use_ngram_embedding_cpu_offload).
            get_logger().warning_once('NGRAM_EMBEDDING_CPU_OFFLOAD=1: the n-gram table is host-resident '
                                      'and receives no gradient, so it stays frozen.')
            self._init_host_table(config, padded_vocab_size, head_dim_per_ngram)
        else:
            # mcore-specific: TP-sharded table (a replicated nn.Embedding would be ~80GB).
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                head_dim_per_ngram,
                init_method=torch.nn.init.normal_,
                config=config,
            )

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        # Hash buffers are replicated; the embedding supplies its own TP metadata.
        state_dict = {}
        self._save_to_state_dict(state_dict, '', keep_vars=True)
        state_dict = make_sharded_tensors_for_checkpoint(
            state_dict, prefix, sharded_offsets=sharded_offsets, tp_group=tp_group, dp_cp_group=metadata['dp_cp_group'])
        for name, module in self.named_children():
            state_dict.update(
                sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group))
        return state_dict

    # The PLE ngram embedding table is stored in the checkpoint as F8_E4M3 shards
    # plus a single scalar `weight_scale` (unlike experts, which use blockwise
    # `weight_scale_inv`). The true values are `weight * scale`.
    _NGRAM_SCALE_KEY = 'ple.ple_embedding.ngram_embedding.weight_scale'

    def fill_table_from_hf(self, hf_state_dict):
        """Populate the (local TP partition of the) n-gram table from the HF
        checkpoint shards at load time.
        """
        total = self.padded_vocab_size
        parts = self.split_ngram_parts
        shard_size = (total + parts - 1) // parts
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if self.cpu_offload:
            tp_start, tp_end = self.vocab_start, self.vocab_end
            dtype = self.host_table.dtype
        else:
            emb = self.ngram_embedding
            per_partition = emb.num_embeddings_per_partition
            tp_start = tp_rank * per_partition
            tp_end = min((tp_rank + 1) * per_partition, total)
            dtype = emb.weight.dtype
        # The fp8 shards must be multiplied by the scalar `weight_scale` before
        # being written into the bf16 table. Checkpoints that store the table in
        # bf16 directly have no such key; keep the raw values then (with a
        # warning). Stashed so export_table_to_hf can re-quantize with it.
        scale = None
        if self._NGRAM_SCALE_KEY in hf_state_dict:
            scale = hf_state_dict[self._NGRAM_SCALE_KEY].load().to(torch.float32)
        else:
            get_logger().warning(f'`{self._NGRAM_SCALE_KEY}` not found in the checkpoint; assuming the PLE ngram '
                                 'embedding is already dequantized and loading it as-is.')
        self._ngram_weight_scale = scale
        for i in range(parts):
            key = f'ple.ple_embedding.ngram_embedding.shard_{i}.weight'
            if key not in hf_state_dict:
                continue
            cs, ce = i * shard_size, min((i + 1) * shard_size, total)
            s, e = max(cs, tp_start), min(ce, tp_end)
            if s >= e:
                continue
            weight = hf_state_dict[key].load()
            if scale is not None:
                weight = weight.to(torch.float32) * scale
            if self.cpu_offload:
                self.host_table[s - tp_start:e - tp_start] = weight[s - cs:e - cs].to(
                    dtype=dtype, device=self.host_table.device)
            else:
                emb.weight.data[s - tp_start:e - tp_start] = weight[s - cs:e - cs].to(dtype)

    @torch.no_grad()
    def iter_export_table_to_hf(self, prefix=''):
        """Generator variant of ``export_table_to_hf``: produce one shard at a
        time so the 100GB-scale table never accumulates on the exporting rank --
        the consumer (the safetensors writer) drives the pace and each shard can
        be released once written. Non-zero TP ranks execute the same all_reduces
        but yield nothing. The collectives advance only while the generator is
        being consumed.
        """
        with torch.no_grad():
            total = self.padded_vocab_size
            parts = self.split_ngram_parts
            shard_size = (total + parts - 1) // parts
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_group = parallel_state.get_tensor_model_parallel_group()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            if self.cpu_offload:
                table = self.host_table
                tp_start, tp_end = self.vocab_start, self.vocab_end
                device = torch.cuda.current_device()
            else:
                table = self.ngram_embedding.weight
                tp_start = tp_rank * self.ngram_embedding.num_embeddings_per_partition
                tp_end = min(tp_start + table.shape[0], total)
                device = table.device
            # Inverse of fill_table_from_hf: the checkpoint format is fp8 shards +
            # scalar `weight_scale`, so divide by the scale stashed during loading
            # and cast back to fp8. Without a known scale the values cannot be
            # represented as fp8 + scale; keep the current dtype and warn.
            # Updated BF16 parameters cannot be losslessly requantized with the
            # original checkpoint scale. Only immutable host tables reuse it.
            scale = getattr(self, '_ngram_weight_scale', None) if self.cpu_offload else None
            if scale is None and self.cpu_offload:
                get_logger().warning(f'`{self._NGRAM_SCALE_KEY}` was not seen during loading; exporting the PLE ngram '
                                     'embedding without re-quantizing to fp8.')
            # Reduce on GPU: NCCL has no CPU backend, and the host table is pinned
            # CPU. Each rank scatters its owned rows into shard chunks, sums across
            # TP (rows are disjoint, so sum == gather), then rank 0 assembles the
            # shard on CPU chunk by chunk. Chunking bounds the GPU transient: a
            # shard can be far larger than the free VRAM left while the training
            # state is still resident.
            elem_size = torch.tensor([], dtype=table.dtype).element_size()
            chunk_rows = max(1, _EXPORT_CHUNK_BYTES // max(1, table.shape[-1] * elem_size))
            out_dtype = torch.float8_e4m3fn if scale is not None else table.dtype
            for i in range(parts):
                cs, ce = i * shard_size, min((i + 1) * shard_size, total)
                # Accumulate the shard on CPU chunk by chunk; assigning inside the
                # loop would keep only the last chunk of each shard.
                shard = None
                if tp_rank == 0:
                    shard = torch.zeros(ce - cs, table.shape[-1], dtype=out_dtype, device='cpu')
                for start in range(cs, ce, chunk_rows):
                    end = min(ce, start + chunk_rows)
                    local = torch.zeros(end - start, table.shape[-1], dtype=table.dtype, device=device)
                    s, e = max(start, tp_start), min(end, tp_end)
                    if s < e:
                        local[s - start:e - start] = table[s - tp_start:e - tp_start].to(device)
                    if tp_size > 1:
                        torch.distributed.all_reduce(local, group=tp_group)
                    if tp_rank == 0:
                        # Re-quantize after the all_reduce: fp8 is not a valid accumulation
                        # dtype for NCCL, and the sum must happen in the loaded dtype.
                        if scale is not None:
                            local = (local.to(torch.float32) / scale.to(local.device)).to(torch.float8_e4m3fn)
                        shard[start - cs:end - cs] = local.cpu()
                if tp_rank == 0:
                    yield f'{prefix}ple.ple_embedding.ngram_embedding.shard_{i}.weight', shard
            if tp_rank == 0 and scale is not None:
                yield f'{prefix}{self._NGRAM_SCALE_KEY}', scale.reshape(())

    @torch.no_grad()
    def export_table_to_hf(self, hf_state_dict, prefix=''):
        """Reverse of ``fill_table_from_hf``: write the current table back as HF
        shards so a full-parameter checkpoint is self-contained. Works for both
        the host-resident (``NGRAM_EMBEDDING_CPU_OFFLOAD=1``) table and the TP-sharded
        ``VocabParallelEmbedding`` (the default, which receives gradients and so
        must not be silently dropped from the checkpoint). Bulk API: consumes
        :meth:`iter_export_table_to_hf` into ``hf_state_dict``.
        """
        for key, tensor in self.iter_export_table_to_hf(prefix):
            hf_state_dict[key] = tensor

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        # Mirrors transformers `_shift_right_ignore_eos`: segment-aware shift,
        # segments are reset after every eos token (request boundary logic).
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [rows, L] -> embeddings [rows, L, embedding_dim].

        Training variant of transformers ``forward(input_ids, past_key_values,
        layer_idx)``: the previous context is eos padding (fresh request),
        prepended here and dropped from the ids afterwards.
        """
        input_ids = input_ids.long()
        # mcore-specific: training has no conv-state cache; a fresh request
        # starts from eos context, so prepend it (the no-previous-state branch
        # of transformers `forward` returns exactly this). The latest HF
        # constructor also takes a cache-only `layer_idx`, which has no
        # training counterpart.
        previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]
        if self.cpu_offload:
            return self._host_lookup(ngram_ids)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def _init_host_table(self, config, padded_vocab_size, head_dim):
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        per_partition = (padded_vocab_size + tp_size - 1) // tp_size
        self.vocab_start = tp_rank * per_partition
        self.vocab_end = min(padded_vocab_size, self.vocab_start + per_partition)
        n_local = max(0, self.vocab_end - self.vocab_start)
        self.host_table = torch.empty(n_local, head_dim, dtype=config.params_dtype, pin_memory=True)
        self._tp_size = tp_size
        self._tp_group = parallel_state.get_tensor_model_parallel_group()

    def _host_lookup(self, ngram_ids):
        # ngram_ids: [rows, L, nH] global ids in [0, padded_vocab_size).
        # Triton fast path reads rows straight out of the pinned host table;
        # otherwise fall back to a torch gather. Either way, all-reduce across
        # TP (partitions are disjoint, so the sum is the full embedding).
        rows = gather_ple_rows(self.host_table, ngram_ids.reshape(-1), self.vocab_start, self.vocab_end)
        if rows is not None:
            rows = rows.view(*ngram_ids.shape, self.host_table.shape[-1])
            if self._tp_size > 1:
                torch.distributed.all_reduce(rows, group=self._tp_group)
            return rows.flatten(-2)
        ids_cpu = ngram_ids.detach().to('cpu')
        local_mask = (ids_cpu >= self.vocab_start) & (ids_cpu < self.vocab_end)
        local_ids = (ids_cpu - self.vocab_start).clamp(0, max(self.host_table.shape[0] - 1, 0))
        rows = self.host_table[local_ids]  # [rows, L, nH, head_dim]
        rows = rows * local_mask.unsqueeze(-1).to(rows.dtype)
        rows = rows.to(ngram_ids.device, non_blocking=True)
        if self._tp_size > 1:
            torch.distributed.all_reduce(rows, group=self._tp_group)
        return rows.flatten(-2)


class _PlePiece(NamedTuple):
    """Rows ``[start, end)`` of the local block, plus the tail slot that supplies its left context."""
    run_local_start: int
    start: int
    end: int
    global_start: int
    owner: int
    slot: int


class _PleHaloGather(torch.autograd.Function):
    """All-gather the per-rank tails along the outermost axis.

    A tail has at most one consumer, so the backward is a plain reduce-scatter to its owner.
    """

    @staticmethod
    def forward(ctx, tails: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        world = torch.distributed.get_world_size(group)
        gathered = tails.new_empty((world, ) + tuple(tails.shape))
        torch.distributed.all_gather_into_tensor(gathered, tails.contiguous(), group=group)
        return gathered

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        world = grad_output.shape[0]
        flat = grad_output.reshape(world, -1).contiguous()
        reduced = torch.empty_like(flat[:1])
        torch.distributed.reduce_scatter_tensor(reduced, flat, group=ctx.group)
        return reduced.view(grad_output.shape[1:]), None


class Qwen4ExpTextPLELayer(nn.Module):
    """Inject hashed n-gram features into every hyper-connection stream;
    mirrors transformers ``Qwen4ExpTextPLELayer`` (training variant without
    the inference cache and ``conv_mask``).

    PLE projects each token's concatenated n-gram embedding to a shared value
    and one key per residual stream. The normalized stream activations gate
    those values, then a dilated depthwise convolution adds local lexical
    context. The returned tensor has width ``hc_count * hidden_size``.

    Checkpoint names under the layer prefix ``ple.``:
        ple_embedding.{layer_multipliers,ngram_heads_offsets,ngram_heads_vocab_sizes}
        ple_embedding.ngram_embedding.shard_{i}.weight
        key_proj.weight / value_proj.weight
        norm_key.weight / norm_query.weight / norm_conv.weight
        conv1d.weight
    """

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        # Visit the embedding's sharded_state_dict instead of flattening its weights.
        tp_group = parallel_state.get_tensor_model_parallel_group()
        state_dict = {}
        for name, module in self.named_children():
            state_dict.update(
                sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group))
        return state_dict

    def __init__(self, config, ple_layer_index: int, pg_collection=None):
        super().__init__()
        self.config = config
        self.pg_collection = pg_collection
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        ple_embed_dim = int(config.ple_embed_dim)
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpTextNGramEmbedding(config, ple_layer_index)
        conv_kernel_size = int(config.ple_conv_kernel_size)
        conv_dilation = int(config.ngram_size)
        self.ngram_size = conv_dilation
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        # The conv's receptive field plus the n-gram warm-up its first rows need for their ids.
        self.context_rows = self.short_conv_state_len + self.ngram_size - 1
        # Replicated projections in params_dtype.
        self.key_proj = TELinear(
            input_size=ple_embed_dim,
            output_size=hc_hidden_size,
            parallel_mode='duplicated',
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
        )
        self.value_proj = TELinear(
            input_size=ple_embed_dim,
            output_size=self.hidden_size,
            parallel_mode='duplicated',
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
        )
        # mcore's config field layernorm_epsilon corresponds to HF's rms_norm_eps;
        # the grouped norm is the mcore subclass adding dtype/SP-flag construction.
        self.norm_key = Qwen4ExpTextGroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        self.norm_query = Qwen4ExpTextGroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        self.norm_conv = Qwen4ExpTextGroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=conv_kernel_size,
            groups=hc_hidden_size,
            dilation=conv_dilation,
            bias=False,
            dtype=config.params_dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        replicated_prefixes = ('key_proj', 'value_proj', 'norm_key', 'norm_query', 'norm_conv', 'conv1d')
        for name, param in self.named_parameters():
            if name.startswith(replicated_prefixes) or name.endswith('norm.weight'):
                # Sharded compute leaves a partial gradient per rank: SUM. Duplicated
                # compute (SP off) already has the whole one: AVG, or SUM inflates it by tp.
                setattr(param, 'sequence_parallel', self.sp_size > 1)
                setattr(param, 'average_gradients_across_tp_domain',
                        bool(config.sequence_parallel) and self.sp_size == 1)

    @property
    def _tp_group(self):
        return self.pg_collection.tp if self.pg_collection is not None else parallel_state.get_tensor_model_parallel_group()

    @property
    def _cp_group(self):
        return self.pg_collection.cp if self.pg_collection is not None else parallel_state.get_context_parallel_group()

    @property
    def tp_size(self) -> int:
        return self._tp_group.size()

    @property
    def cp_size(self) -> int:
        return self._cp_group.size()

    @property
    def sp_size(self) -> int:
        """Rows each rank holds over TP: TP only splits the sequence under sequence parallelism."""
        return self.tp_size if self.config.sequence_parallel else 1

    @property
    def tp_rank(self) -> int:
        return self._tp_group.rank() if self.sp_size > 1 else 0

    @property
    def cp_rank(self) -> int:
        return self._cp_group.rank() if self.cp_size > 1 else 0

    @property
    def tp_cp_stride(self) -> int:
        """The tp-cp group is laid out tp-fastest, so a rank's index is tp_rank + tp * cp_rank."""
        return self.sp_size

    def _short_conv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Training variant of transformers `_short_conv` (no conv state cache):
        # causal alignment is achieved by left-padding the conv input with
        # zeros (equivalent to trimming the padded conv output), which is a
        # no-op on fresh-request context.
        hidden_states = hidden_states.transpose(1, 2)
        sequence_length = hidden_states.shape[-1]
        conv_input = F.pad(hidden_states, (self.short_conv_state_len, 0))
        conv_input = conv_input[..., -(self.short_conv_state_len + sequence_length):]
        return F.silu(self.conv1d(conv_input)).transpose(1, 2)

    def compute(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Mirrors transformers ``Qwen4ExpTextPLELayer.forward`` (training
        variant); hidden_states/input_ids: [rows, L, nH]/[rows, L]."""
        return self._mix(hidden_states, self.ple_embedding(input_ids))  # mcore-specific: no past_key_values cache arg

    def _mix(self, hidden_states: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Gate + short conv + residual on precomputed embeddings.

        Split out of ``compute`` so the table lookup (a TP all-reduce) happens once per block
        instead of once per document: a differing collective count across ranks deadlocks.
        """
        if use_ple_fused_kernel() and embeddings.device.type != 'cpu':
            fused = self._compute_fused(hidden_states, embeddings)
            if fused is not None:
                return fused
        key_normed = self.norm_key(self.key_proj(embeddings)[0]).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)[0]
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        # (transformers also applies conv_mask here; training relies on the
        # loss mask instead.)
        output = gated_value + self._short_conv(gated_value_normed)
        return output

    def _compute_fused(self, hidden_states: torch.Tensor, embeddings: torch.Tensor):
        """Fused triton chain (gate + norm_conv + causal conv + SiLU + residual);
        fp32 accumulation, output dtype = params_dtype. Returns None when the
        kernel path is unavailable so the caller falls back to torch."""
        rows, seq_len = embeddings.shape[:2]
        key = self.key_proj(embeddings)[0].reshape(rows * seq_len, -1)
        value = self.value_proj(embeddings)[0].reshape(rows * seq_len, -1)
        query = hidden_states.reshape(rows * seq_len, -1)
        out = ple_gate_conv_triton(
            query,
            key,
            value,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.conv1d.weight,
            self.hc_count,
            self.config.layernorm_epsilon,
            int(self.config.ngram_size),
            seq_len,
        )
        if out is None:
            return None
        return out.view(rows, seq_len, -1)

    @staticmethod
    def _normalize_cu_seqlens(cu: Optional[torch.Tensor], total: int) -> Optional[torch.Tensor]:
        """Align a (possibly padded/offset) cu_seqlens against the gathered full length.

        Mirrors GatedDeltaNet._resolve_cu_seqlens for the SP-only case: accept a
        global cumulative layout (possibly missing its leading 0 or carrying a
        per-batch offset), and return a cu whose last entry equals ``total`` so it
        indexes the gathered full-sequence tensor directly.
        """
        if cu is None:
            return None
        cu = cu.reshape(-1)
        if cu.numel() > 0 and int(cu[0]) != 0:
            if int(cu[-1]) == total:
                cu = torch.cat([cu.new_zeros(1), cu])
            elif int(cu[-1]) - int(cu[0]) == total:
                cu = cu - cu[0]
        if not (cu.numel() > 0 and int(cu[-1]) == total):
            raise ValueError(f'PLE cannot align cu_seqlens (last={int(cu[-1]) if cu.numel() else None}, '
                             f'first={int(cu[0]) if cu.numel() else None}) with the gathered sequence '
                             f'length {total} under sequence parallelism.')
        return cu

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """hidden_states: [s, b, nH] (bsh) or thd [T, 1, nH]; input_ids: [b, s] or [1, T].

        mcore-specific wrapper (no counterpart in transformers): TP needs no
        special handling (replicated weights, full-width hidden between blocks).

        The increment is token-local up to a left window, so shards are not gathered: each rank
        computes its rows plus ``context_rows`` published by the preceding rank. ``_forward_impl``
        remains the full-sequence reference and serves short, unsharded inputs.
        """
        thd = packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', 'bshd') == 'thd'
        if self.sp_size == 1 and self.cp_size == 1 and self._impl_rows(
                packed_seq_params, hidden_states.shape[0], thd) <= ple_chunk_tokens(hidden_states.shape[-1]):
            return self._forward_impl(hidden_states, input_ids, packed_seq_params)
        return self._forward_windowed(hidden_states, input_ids, packed_seq_params, thd)

    @staticmethod
    def _impl_rows(packed_seq_params, rows: int, thd: bool) -> int:
        """Rows ``_forward_impl`` materialises: it pads every document to ``max_seqlen_q``."""
        if not thd:
            return rows
        max_seqlen_q = packed_seq_params.max_seqlen_q
        max_len = int(max_seqlen_q.item() if torch.is_tensor(max_seqlen_q) else max_seqlen_q)
        return get_num_samples(packed_seq_params) * max_len

    def _cp_chunks(self, doc_length: int) -> List[tuple]:
        """``(slot, cp-local offset, global offset, length)`` of this rank's two halves of a
        document: CP splits it into ``2*cp`` chunks and keeps ``cp_rank`` / ``2*cp-1-cp_rank``."""
        if self.cp_size == 1:
            return [(0, 0, 0, doc_length)]
        if doc_length % (2 * self.cp_size):
            raise ValueError(f'PLE cannot shard a document of {doc_length} tokens into '
                             f'2*cp={2 * self.cp_size} CP chunks.')
        piece = doc_length // (2 * self.cp_size)
        return [(0, 0, self.cp_rank * piece, piece),
                (1, piece, (2 * self.cp_size - 1 - self.cp_rank) * piece, piece)]

    def _cp_local_offsets(self, documents: List[tuple]) -> List[int]:
        """Offset of each document inside the CP-local tensor."""
        offsets, cursor = [], 0
        for doc_start, doc_end in documents:
            offsets.append(cursor)
            cursor += sum(chunk[3] for chunk in self._cp_chunks(doc_end - doc_start))
        return offsets

    def _local_runs(self, total: int, documents: List[tuple],
                    tp_rank: Optional[int] = None) -> List[tuple]:
        """``[(local_start, run_len, global_start, slot), ...]`` in local row order: the TP slice of
        the CP-local tensor intersected with the per-document chunks."""
        if total % (self.sp_size * self.cp_size):
            raise ValueError(f'PLE needs the gathered sequence length {total} to be divisible by '
                             f'sp*cp={self.sp_size * self.cp_size}.')
        tp_rank = self.tp_rank if tp_rank is None else tp_rank
        local_len = total // (self.sp_size * self.cp_size)
        k0 = tp_rank * local_len
        offsets = self._cp_local_offsets(documents)
        runs = []
        for doc_index, (doc_start, doc_end) in enumerate(documents):
            for slot, local_offset, global_offset, piece in self._cp_chunks(doc_end - doc_start):
                cp_start = offsets[doc_index] + local_offset
                lo, hi = max(k0, cp_start), min(k0 + local_len, cp_start + piece)
                if hi > lo:
                    runs.append((lo - k0, hi - lo, doc_start + global_offset + (lo - cp_start),
                                 2 * doc_index + slot))
        runs.sort(key=lambda run: run[0])
        return runs

    def _owner_of(self, token: int, total: int, documents: List[tuple]) -> tuple:
        """``(tp_rank, cp_rank, slot)`` owning global ``token``, i.e. the tail that ends after it."""
        offsets = self._cp_local_offsets(documents)
        local_len = total // (self.sp_size * self.cp_size)
        for doc_index, (doc_start, doc_end) in enumerate(documents):
            if token >= doc_end:
                continue
            offset = token - doc_start
            if self.cp_size == 1:
                cp_rank, half, cp_local = 0, 0, offsets[doc_index] + offset
            else:
                piece = (doc_end - doc_start) // (2 * self.cp_size)
                chunk = offset // piece
                half = 0 if chunk < self.cp_size else 1
                cp_rank = chunk if half == 0 else 2 * self.cp_size - 1 - chunk
                cp_local = offsets[doc_index] + half * piece + offset % piece
            return cp_local // local_len, cp_rank, 2 * doc_index + half
        raise ValueError(f'PLE: global token {token} lies outside the documents.')

    def _halo_shortfall(self, total: int, documents: List[tuple]) -> Optional[str]:
        """Why the tail exchange cannot be used for this microbatch, ``None`` when it can.
        Geometry only, so every rank picks the same branch (different collectives would deadlock)."""
        local_len = total // (self.sp_size * self.cp_size)
        shortest = min(run[1] for tp_rank in range(self.sp_size)
                       for run in self._local_runs(total, documents, tp_rank))
        if shortest < self.context_rows:
            return (f'the shortest run is {shortest} rows, below the {self.context_rows}-row window '
                    f'(a packed document needs about {2 * self.cp_size * self.context_rows} tokens for cp={self.cp_size})')
        slots = 2 * len(documents) * self.context_rows
        if slots > local_len:
            return f'the {2 * len(documents)} published tails ({slots} rows) exceed the {local_len}-row local block'
        return None

    def _cp_local_pieces(self, documents: List[tuple]) -> List[tuple]:
        """This CP rank's chunks as ``(cp_local_start, length, global_start)``."""
        offsets = self._cp_local_offsets(documents)
        pieces = []
        for doc_index, (doc_start, doc_end) in enumerate(documents):
            for _slot, local_offset, global_offset, piece_len in self._cp_chunks(doc_end - doc_start):
                pieces.append((offsets[doc_index] + local_offset, piece_len, doc_start + global_offset))
        return pieces

    @staticmethod
    def _cp_partition_mode(packed_seq_params) -> str:
        """CP split layout. It rides on the batch (mcore's ``PackedSeqParams`` has no such field,
        ms-swift attaches it); the geometry below is zigzag only, so contiguous must be rejected."""
        return getattr(packed_seq_params, 'cp_partition_mode', 'zigzag')

    @staticmethod
    def _document_start(token: int, documents: List[tuple]) -> int:
        for doc_start, doc_end in documents:
            if token < doc_end:
                return doc_start
        raise ValueError(f'PLE: global token {token} lies outside the documents.')

    def _exchange_halo(self, hidden_states: torch.Tensor, runs: List[tuple],
                       documents: List[tuple]) -> torch.Tensor:
        """Publish each local run's trailing ``context_rows`` rows -> ``[tp*cp, 2*docs, rows, ...]``."""
        tails = hidden_states.new_zeros((2 * len(documents), self.context_rows) + tuple(hidden_states.shape[1:]))
        for local_start, run_len, _global_start, slot in runs:
            take = min(self.context_rows, run_len)
            tails[slot, self.context_rows - take:] = hidden_states[local_start + run_len - take:local_start + run_len]
        return _PleHaloGather.apply(tails, parallel_state.get_tensor_and_context_parallel_group())

    def _gather_full(self, hidden_states: torch.Tensor, packed_seq_params) -> torch.Tensor:
        """Materialise the whole sequence: the fallback when a local block is shorter than one window."""
        if self.sp_size > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=True, group=self.pg_collection.tp)
        if self.cp_size > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=True, group=parallel_state.get_context_parallel_group())
            hidden_states = _undo_attention_load_balancing(hidden_states,
                                                           parallel_state.get_context_parallel_world_size(),
                                                           packed_seq_params)
        return hidden_states

    def _global_input_ids(self, input_ids: torch.Tensor, packed_seq_params, total: int) -> torch.Tensor:
        """The full, unsharded ids: windows are indexed by global position, so a CP-sharded copy is rebuilt."""
        if input_ids.shape[-1] == total:
            return input_ids
        if self.cp_size > 1:
            ids = reconstruct_tensor_cp(input_ids, packed_seq_params, dim=1,
                                        cp_partition_mode=self._cp_partition_mode(packed_seq_params))
            if ids.shape[-1] == total:
                return ids
        raise ValueError(f'PLE needs the full (unsharded) input_ids: got {input_ids.shape[-1]} tokens for a '
                         f'gathered sequence of {total}. gpt_model is expected to pass the full input_ids.')

    def _document_bounds(self, packed_seq_params, total: int) -> List[tuple]:
        """Per-document ``[start, end)`` global token ranges (both the n-gram shift
        and the short conv restart at a document boundary)."""
        cu = getattr(packed_seq_params, 'cu_seqlens_q', None) if packed_seq_params is not None else None
        if cu is None:
            return [(0, total)]
        bounds = self._normalize_cu_seqlens(cu, total).tolist()
        return [(int(start), int(end)) for start, end in zip(bounds[:-1], bounds[1:]) if int(end) > int(start)]

    def _forward_windowed(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, packed_seq_params,
                          thd: bool) -> torch.Tensor:
        rows = hidden_states.shape[0]
        total = rows * self.sp_size * self.cp_size
        psp = packed_seq_params if thd else None
        if thd:
            # The data pipeline may hand over a cu without its leading 0 or with a per-batch
            # offset; align it once against the gathered total for the gather and the ids.
            psp = copy.copy(packed_seq_params)
            psp.cu_seqlens_q = self._normalize_cu_seqlens(packed_seq_params.cu_seqlens_q, total)
        if self.cp_size > 1 and self._cp_partition_mode(psp) != 'zigzag':
            raise NotImplementedError(f'PLE windowed forward supports the zigzag CP partition only, got '
                                      f'cp_partition_mode={self._cp_partition_mode(psp)!r}.')
        ids = self._global_input_ids(input_ids, psp, total)
        documents = self._document_bounds(psp, total)
        runs = self._local_runs(total, documents)

        sharded = self.sp_size > 1 or self.cp_size > 1
        # SP off with TP > 1 duplicates the compute, so every copy would consume the same tail and
        # the reduce-scatter would count its gradient tp times.
        tp_duplicated = self.tp_size > 1 and self.sp_size == 1
        shortfall = self._halo_shortfall(total, documents) if sharded and not tp_duplicated else None
        if sharded and shortfall is None and not tp_duplicated:
            source, halo, source_is_global = hidden_states, self._exchange_halo(hidden_states, runs, documents), False
        elif sharded:
            reason = shortfall if shortfall is not None else (
                f'sequence parallelism is off while tp={self.config.tensor_model_parallel_size} duplicates the '
                'compute, so the tail exchange would count its gradient tp times')
            get_logger().warning_once(
                f'PLE windowed forward falls back to the full-sequence gather: {reason}. The PLE block keeps '
                'an O(sequence) peak in this step; pad or merge the short documents of the pack (or keep '
                'sequence_parallel on) to get the windowed path back.',
                hash_id='ple-halo-shortfall')
            source, halo, source_is_global = self._gather_full(hidden_states, psp), None, True
        else:
            source, halo, source_is_global = hidden_states, None, True

        plan = []
        for local_start, run_len, global_start, _slot in runs:
            owner = slot = 0
            if halo is not None and global_start > 0:
                owner_tp, owner_cp, slot = self._owner_of(global_start - 1, total, documents)
                owner = owner_tp + self.tp_cp_stride * owner_cp
            for doc_start, doc_end in documents:
                if doc_end <= global_start:
                    continue
                if doc_start >= global_start + run_len:
                    break
                start = local_start + max(0, doc_start - global_start)
                end = local_start + min(run_len, doc_end - global_start)
                if start >= end:
                    continue
                plan.append(
                    _PlePiece(
                        run_local_start=local_start, start=start, end=end, global_start=global_start, owner=owner,
                        slot=slot))

        # One lookup for the whole CP-local block: the vocabulary-parallel lookup all-reduces over
        # TP, so every TP rank must feed the same ids, and each piece is preceded by the
        # `ngram_size - 1` eos ids that keep documents from leaking into each other's hash.
        eos = self.ple_embedding.eos_token_id
        cp_pieces = self._cp_local_pieces(documents)
        blocks, window_starts = [], []
        for cp_start, piece_len, global_start in cp_pieces:
            context = min(self.context_rows, global_start - self._document_start(global_start, documents))
            window_starts.append(global_start - context)
            blocks.append(
                torch.cat(
                    [ids.new_full((ids.shape[0], self.ngram_size - 1), eos),
                     ids[:, global_start - context:global_start + piece_len]], dim=1))
        width = sum(block.shape[1] for block in blocks)
        capacity = total // self.cp_size + 2 * len(documents) * (self.context_rows + self.ngram_size - 1)
        if width > capacity:
            raise ValueError(f'PLE windowed block ({width} rows) exceeds its geometry bound ({capacity}).')
        feed = blocks + ([ids.new_full((ids.shape[0], capacity - width), eos)] if width < capacity else [])
        embeddings = self.ple_embedding(torch.cat(feed, dim=1))

        # A piece belongs to the CP chunk containing its first row; windows overlap by
        # `context_rows`, so matching on the window would hit two chunks.
        chunk_rows = ple_chunk_tokens(hidden_states.shape[-1])
        tp_base = self.tp_rank * rows
        located = []
        embedding_offset = 0
        # Rows still holding fp32 kernel state. `_PLEGateConv` saves `conv_pre` per call, so the
        # *total* has to stay in budget: a pack of short documents would otherwise never checkpoint.
        retained = 0
        for (cp_start, piece_len, _global_start), block, window_start in zip(cp_pieces, blocks, window_starts):
            window = embeddings[:, embedding_offset:embedding_offset + block.shape[1]].contiguous()
            embedding_offset += block.shape[1]
            for piece in plan:
                if not cp_start <= piece.start + tp_base < cp_start + piece_len:
                    continue
                tensor, retained = self._compute_piece(source, window, halo, piece, chunk_rows, source_is_global,
                                                       window_start, retained)
                located.append((piece.start, tensor))
        if sorted(start for start, _bytes in located) != [piece.start for piece in plan]:
            raise ValueError(f'PLE matched {len(located)} pieces to the lookup blocks, expected the {len(plan)} '
                             'planned pieces exactly once.')
        located.sort(key=lambda item: item[0])
        expected = 0
        for start, tensor in located:
            if start != expected:
                raise ValueError(f'PLE pieces do not tile the row block: expected row {expected}, got {start}.')
            expected += tensor.shape[0]
        if expected != rows:
            raise ValueError(f'PLE pieces cover {expected} of {rows} rows.')
        pieces = [tensor for _start, tensor in located]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    def _compute_piece(self, source: torch.Tensor, embeddings: torch.Tensor, halo, piece: _PlePiece,
                       chunk_rows: int, source_is_global: bool, window_global_start: int,
                       retained: int) -> tuple:
        """Chunk one piece, each chunk with the ``context_rows`` window the full-sequence path sees.

        ``retained`` is how many rows this rank already holds un-recomputed fp32 state for; a chunk
        is checkpointed once keeping it would exceed ``chunk_rows``. Returns ``(output, retained)``.
        """
        source_base = piece.global_start if source_is_global else piece.run_local_start
        outs = []
        position = piece.start
        while position < piece.end:
            chunk_end = min(piece.end, position + chunk_rows)
            checkpointed = retained + (chunk_end - position) > chunk_rows
            if not checkpointed:
                retained += chunk_end - position
            global_position = piece.global_start + (position - piece.run_local_start)
            # Never reach back past the document start: that is where the full-sequence path
            # zero-pads the conv and restarts the hash with eos.
            context = min(self.context_rows, global_position - window_global_start)
            available = global_position if source_is_global else position - piece.run_local_start
            owned = min(context, available)
            window_start = source_base + (position - piece.run_local_start - owned)
            window_end = source_base + (chunk_end - piece.run_local_start)
            state = source[window_start:window_end]
            if halo is not None:
                # Tails are right-aligned in their slot, so the missing rows are its last
                # `context - owned` entries (offsetting by `context` only holds when `owned == 0`).
                tail = halo[piece.owner, piece.slot, self.context_rows - (context - owned):]
                # Cat even when empty: autograd drops the backward of an unused node, so a rank
                # needing no tail would skip the reduce-scatter and deadlock the others.
                state = torch.cat([tail, state], dim=0)
            embedding_start = self.ngram_size - 1 + (global_position - context - window_global_start)
            window = embeddings[:, embedding_start:embedding_start + context + (chunk_end - position)]
            state = state.transpose(0, 1)
            if checkpointed:
                result = checkpoint(self._mix, state, window, use_reentrant=False)
            else:
                result = self._mix(state, window)
            outs.append(result[:, context:].transpose(0, 1))
            position = chunk_end
        return (outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)), retained

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """hidden_states: [s, b, nH] (bsh) or thd [T, 1, nH]; input_ids: [b, s] or [1, T]."""
        thd = packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', 'bshd') == 'thd'
        if thd:
            num_samples = get_num_samples(packed_seq_params)
            # PackedSeqParams.max_seqlen_q is declared `int` in mcore and swift
            # normalizes it to int, so `.item()` would raise AttributeError;
            # tolerate a 0-d tensor from other callers.
            max_seqlen_q = packed_seq_params.max_seqlen_q
            max_len = int(max_seqlen_q.item() if torch.is_tensor(max_seqlen_q) else max_seqlen_q)
            cu = packed_seq_params.cu_seqlens_q
            total = hidden_states.shape[0]
            hid = hidden_states.new_zeros((num_samples, max_len, hidden_states.shape[-1]))
            toks = input_ids.new_full((num_samples, max_len), self.ple_embedding.eos_token_id)
            for i in range(num_samples):
                start, end = int(cu[i]), int(cu[i + 1])
                hid[i, :end - start] = hidden_states[start:end, 0]
                toks[i, :end - start] = input_ids[0, start:end]
            res = self.compute(hid, toks)
            out = res.new_zeros((total, 1, res.shape[-1]))
            for i in range(num_samples):
                start, end = int(cu[i]), int(cu[i + 1])
                out[start:end, 0] = res[i, :end - start]
            return out
        else:
            # [s, b, nH] -> [b, s, nH]; input_ids [b, s]
            hid = hidden_states.transpose(0, 1)
            res = self.compute(hid, input_ids)
            return res.transpose(0, 1).contiguous()
