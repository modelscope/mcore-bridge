# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.tensor_parallel import VocabParallelEmbedding
from megatron.core.tensor_parallel.mappings import (gather_from_sequence_parallel_region,
                                                    scatter_to_sequence_parallel_region)
from torch import nn
from typing import List, Optional

from ...utils import get_env_args, get_logger
from ...utils.megatron_utils import reconstruct_tensor_cp, split_cp_inputs
from .hyper_connection_gated import Qwen4ExpTextGroupedRMSNorm
from .kernels import gather_ple_rows, ple_gate_conv_triton

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def use_ple_cpu_offload() -> bool:
    return get_env_args('PLE_CPU_OFFLOAD', bool, False)


def use_ple_fused_kernel() -> bool:
    # On by default; PLE_FUSED_KERNEL=0 forces the eager chain.
    return get_env_args('PLE_FUSED_KERNEL', bool, True)


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
        self.cpu_offload = use_ple_cpu_offload()
        if self.cpu_offload:
            # Warn rather than raise: whether this is safe depends on the train type,
            # which the model config does not carry (see use_ple_cpu_offload).
            get_logger().warning_once('PLE_CPU_OFFLOAD=1: the n-gram table is host-resident and receives no '
                                      'gradient, so it stays frozen.')
            self._init_host_table(config, padded_vocab_size, head_dim_per_ngram)
        else:
            # mcore-specific: TP-sharded table (a replicated nn.Embedding would be ~80GB).
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                head_dim_per_ngram,
                init_method=torch.nn.init.normal_,
                config=config,
            )

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
        for i in range(parts):
            key = f'ple.ple_embedding.ngram_embedding.shard_{i}.weight'
            if key not in hf_state_dict:
                continue
            cs, ce = i * shard_size, min((i + 1) * shard_size, total)
            s, e = max(cs, tp_start), min(ce, tp_end)
            if s >= e:
                continue
            weight = hf_state_dict[key].load()
            if self.cpu_offload:
                self.host_table[s - tp_start:e - tp_start] = weight[s - cs:e - cs].to(
                    dtype=dtype, device=self.host_table.device)
            else:
                emb.weight.data[s - tp_start:e - tp_start] = weight[s - cs:e - cs].to(dtype)

    @torch.no_grad()
    def export_table_to_hf(self, hf_state_dict, prefix=''):
        """Reverse of ``fill_table_from_hf``: write the offloaded table back as HF
        shards so a full-parameter checkpoint is self-contained.
        """
        if not self.cpu_offload:
            return
        total = self.padded_vocab_size
        parts = self.split_ngram_parts
        shard_size = (total + parts - 1) // parts
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        for i in range(parts):
            cs, ce = i * shard_size, min((i + 1) * shard_size, total)
            # Reduce on GPU: NCCL has no CPU backend, and the host table is pinned
            # CPU. Each rank scatters its owned rows into a full shard, sums across
            # TP (rows are disjoint, so sum == gather), then rank 0 keeps the CPU copy.
            device = torch.cuda.current_device()
            local = torch.zeros(ce - cs, self.host_table.shape[-1], dtype=self.host_table.dtype, device=device)
            s, e = max(cs, self.vocab_start), min(ce, self.vocab_end)
            if s < e:
                local[s - cs:e - cs] = self.host_table[s - self.vocab_start:e - self.vocab_start].to(device)
            if self._tp_size > 1:
                torch.distributed.all_reduce(local, group=tp_group)
            if tp_rank == 0:
                hf_state_dict[f'{prefix}ple.ple_embedding.ngram_embedding.shard_{i}.weight'] = local.cpu()

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
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
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
                # Replicated across TP; reduce grads across TP when SP is on.
                setattr(param, 'sequence_parallel', config.sequence_parallel)

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
        embeddings = self.ple_embedding(input_ids)  # mcore-specific: no past_key_values cache arg
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
        special handling (replicated weights, full-width hidden between
        blocks); under SP/CP the shard is gathered to the full sequence
        (undoing SP first, then the CP zigzag), ``compute`` runs on the full
        sequence, and the additive output is scattered back.
        """
        sp_on = (
            self.pg_collection is not None and self.config.sequence_parallel
            and self.config.tensor_model_parallel_size > 1)
        cp_on = self.config.context_parallel_size > 1
        if not (sp_on or cp_on):
            return self._forward_impl(hidden_states, input_ids, packed_seq_params)

        thd = packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', 'bshd') == 'thd'

        # ---- gather the shard into the full sequence (SP shard is innermost) ----
        if sp_on:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.pg_collection.tp)
        if cp_on:
            # For packed (thd) inputs the data pipeline (swift get_batch_on_this_cp_rank)
            # zigzag-splits hidden/input_ids per sample via cu_seqlens_q while keeping
            # packed_seq_params itself global, so the undo uses the same cu.
            psp_for_cp = packed_seq_params if thd else None
            hidden_states = reconstruct_tensor_cp(hidden_states, psp_for_cp, dim=0)
            # The data pipeline may hand us either a CP-sharded or a full copy of
            # input_ids; re-align only when the lengths disagree.
            if input_ids.shape[-1] != hidden_states.shape[0]:
                input_ids = reconstruct_tensor_cp(input_ids, psp_for_cp, dim=1)
        elif input_ids.shape[-1] != hidden_states.shape[0]:
            raise ValueError(f'PLE input_ids length {input_ids.shape[-1]} does not match the gathered '
                             f'sequence length {hidden_states.shape[0]} under sequence parallelism; '
                             'gpt_model is expected to pass the full, unsharded input_ids.')

        if thd:
            # SP keeps one global cu_seqlens copy per rank; normalize padded/offset
            # forms against the gathered total so cu indexes the full sequence.
            # copy.copy (not dataclasses.replace) preserves dynamically attached
            # fields such as `num_samples` that the data pipeline relies on.
            cu = self._normalize_cu_seqlens(getattr(packed_seq_params, 'cu_seqlens_q', None), hidden_states.shape[0])
            psp = copy.copy(packed_seq_params)
            psp.cu_seqlens_q = cu
            packed_seq_params = psp

        out = self._forward_impl(hidden_states, input_ids, packed_seq_params)

        # ---- scatter the additive output back to the caller's shard ----
        if cp_on:
            out = split_cp_inputs(out, getattr(packed_seq_params, 'cu_seqlens_q', None) if thd else None, dim=0)
        if sp_on:
            out = scatter_to_sequence_parallel_region(out, group=self.pg_collection.tp)
        return out

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """hidden_states: [s, b, nH] (bsh) or thd [T, 1, nH]; input_ids: [b, s] or [1, T]."""
        thd = packed_seq_params is not None and getattr(packed_seq_params, 'qkv_format', 'bshd') == 'thd'
        if thd:
            num_samples = packed_seq_params.num_samples
            max_len = int(packed_seq_params.max_seqlen_q)
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
