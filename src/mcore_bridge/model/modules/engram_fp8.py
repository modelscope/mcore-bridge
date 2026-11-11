# Copyright (c) ModelScope Contributors. All rights reserved.
"""FP8-resident Engram tables.

DeepSeek-V4.1 ships its n-gram tables as fp8 + e8m0 block scales and never trains them, but
Megatron's ``EPShardedEmbeddingTable`` materializes a ``bf16`` row shard (~2x the checkpoint
bytes) because its tables are trainable in general. These subclasses keep the raw fp8 bits and
decode only the rows a forward actually touches, reusing upstream's EP routing untouched: only
``EPShardedEmbeddingTable.forward`` is replaced, and every other table attribute is preserved so
``EPShardedMultiTableEmbedding`` keeps working.

Load with these tables is a byte copy (checkpoint fp8 -> store fp8), so it no longer upcasts
183 GiB of Engram weights to bf16.
"""
import torch
from torch import Tensor, nn

from ...utils import get_current_device, get_logger

try:
    from megatron.core.models.engram.distributed_embedding import (EPShardedEmbeddingTable,
                                                                   EPShardedMultiTableEmbedding,
                                                                   get_contiguous_row_range)
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.utils import get_pg_rank, get_pg_size
except ImportError:
    EPShardedEmbeddingTable = None
    EPShardedMultiTableEmbedding = None
    MegatronModule = None

logger = get_logger()


def fp8_quant_declared(config) -> bool:
    """Whether the checkpoint declares DeepSeek-V4.1's official quantized layout.

    ``scale_fmt: ue8m0`` or a 32x32 block scale is what both consumers key off: the fp8-resident
    Engram store and the export requantizer. The storage follows the checkpoint declaration rather
    than a user switch.
    """
    hf_config = getattr(config, 'hf_config', None)
    quantization = getattr(hf_config, 'quantization_config', None)
    if quantization is None:
        return False
    lookup = quantization.get if isinstance(quantization, dict) else lambda key: getattr(quantization, key, None)
    scale_fmt = lookup('scale_fmt')
    if scale_fmt is not None and str(scale_fmt).lower() in ('ue8m0', 'e8m0'):
        return True
    block = lookup('weight_block_size')
    return block is not None and list(block) == [32, 32]


def decode_fp8_rows(rows: Tensor, scales: Tensor, block_size: int, dtype) -> Tensor:
    """Dequantize ``[n, D]`` fp8 rows with per-row e8m0 block scales ``[n, ceil(D/block)]``."""
    weight = rows.view(torch.float8_e4m3fn).float()
    scale = scales.view(torch.float8_e8m0fnu).float()
    width = weight.shape[-1]
    pad = -width % block_size
    if pad:
        weight = torch.nn.functional.pad(weight, (0, pad))
    values = (weight.unflatten(-1, (-1, block_size)) * scale.unsqueeze(-1)).flatten(-2)
    return values[..., :width].to(dtype)


def is_fp8_table(module) -> bool:
    return isinstance(module, Fp8EngramTable)


def reinterpret_u8(tensor: Tensor) -> Tensor:
    """Reinterpret fp8/e8m0 payloads as raw bytes without converting their values."""
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


if EPShardedEmbeddingTable is not None:

    class Fp8EngramTable(EPShardedEmbeddingTable):
        """Row shard stored as fp8 bits + e8m0 block scales."""

        def __init__(self,
                     config,
                     global_num_embeddings,
                     embedding_dim,
                     init_method=None,
                     ep_group=None,
                     tp_group=None,
                     expt_dp_group=None,
                     block_size=32):
            MegatronModule.__init__(self, config=config)
            self.global_num_embeddings = global_num_embeddings
            self.embedding_dim = embedding_dim
            self.tp_group = tp_group
            self.expt_dp_group = expt_dp_group
            self.deterministic_mode = getattr(config, 'deterministic_mode', False)
            self.ep_rank = get_pg_rank(ep_group)
            self.ep_size = get_pg_size(ep_group)
            self.row_start, self.row_end = get_contiguous_row_range(global_num_embeddings, self.ep_rank, self.ep_size)
            self.block_size = block_size
            device = torch.device('cpu') if config.use_cpu_initialization else get_current_device()
            local_rows = self.local_num_embeddings
            # Zero-filled, not empty: an unfilled row must read as 0 rather than as garbage bits.
            # Non-persistent buffers so `.to(device)` moves them while `.to(dtype)` leaves the bytes alone.
            self.register_buffer(
                'fp8_rows',
                torch.zeros((local_rows, embedding_dim), dtype=torch.uint8, device=device),
                persistent=False)
            self.register_buffer(
                'fp8_scales',
                torch.zeros((local_rows, -(-embedding_dim // block_size)), dtype=torch.uint8, device=device),
                persistent=False)
            self._filled_rows = 0
            # Zero-row bf16 placeholder: upstream allocates temporaries and probes the checkpoint
            # dtype through `weight`, so it must exist with the right dtype/device but hold no data.
            self.weight = nn.Parameter(
                torch.empty(0, embedding_dim, dtype=config.params_dtype, device=device), requires_grad=False)
            self.weight.allreduce = False
            self.weight.is_engram_embedding = True
            self.weight.is_embedding_or_output_parameter = True
            if config.sequence_parallel:
                self.weight.sequence_parallel = True

        @property
        def local_parameter_count(self) -> int:
            return self.fp8_rows.numel()

        @property
        def local_storage_bytes(self) -> int:
            return self.fp8_rows.numel() + self.fp8_scales.numel()

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            super()._save_to_state_dict(destination, prefix, keep_vars)
            # `weight` stays a zero-row placeholder; the payload travels under the checkpoint names so
            # plain state_dict() and DCP both carry the table.
            destination[prefix + 'weight'] = self.fp8_rows.view(torch.float8_e4m3fn)
            destination[prefix + 'weight_scale_inv'] = self.fp8_scales.view(torch.float8_e8m0fnu)

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                  error_msgs):
            rows = state_dict.pop(prefix + 'weight', None)
            scales = state_dict.pop(prefix + 'weight_scale_inv', None)
            if rows is not None:
                # Keep the inherited shape check quiet; the payload is written into the buffers below.
                state_dict[prefix + 'weight'] = self.weight.detach()
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                          error_msgs)
            if rows is None and scales is None:
                return
            expected = (self.local_num_embeddings, self.embedding_dim)
            if rows is not None and rows.dtype != torch.float8_e4m3fn:
                # Old (pre-fp8) Engram DCP checkpoints only carry a bf16 `weight` row shard.
                error_msgs.append(f'{prefix}weight is {rows.dtype}, not fp8: this checkpoint stores the Engram table '
                                  'in the pre-fp8 layout; re-convert it from the HF checkpoint')
            elif rows is None or scales is None:
                error_msgs.append(f'{prefix}fp8 Engram table needs both weight and weight_scale_inv')
            elif scales.dtype != torch.float8_e8m0fnu:
                error_msgs.append(f'{prefix}weight_scale_inv is {scales.dtype}, not e8m0: re-convert the '
                                  'Engram table from the HF checkpoint')
            elif tuple(rows.shape) != expected:
                error_msgs.append(f'{prefix}weight shape {tuple(rows.shape)} != local shard {expected}')
            else:
                self.fp8_rows.copy_(rows.view(torch.uint8))
                self.fp8_scales.copy_(scales.view(torch.uint8))
                self._filled_rows = self.local_num_embeddings

        def forward(self, local_row_ids: Tensor) -> Tensor:
            if self._filled_rows < self.local_num_embeddings:
                # A DCP restore writes the planes directly, bypassing fill_rows.
                logger.warning_once(f'Engram table ({self.global_num_embeddings} rows) has only '
                                    f'{self._filled_rows}/{self.local_num_embeddings} rows loaded on this rank; '
                                    'the remaining rows decode to zero.')
            flat = local_row_ids.reshape(-1).to(torch.int64)
            rows = self.fp8_rows.index_select(0, flat)
            scales = self.fp8_scales.index_select(0, flat)
            values = decode_fp8_rows(rows, scales, self.block_size, self.weight.dtype)
            return values.view(*local_row_ids.shape, self.embedding_dim)

        @torch.no_grad()
        def fill_rows(self, local_start: int, local_end: int, rows: Tensor, scales: Tensor) -> None:
            """Copy checkpoint rows (fp8 bits + e8m0 bits) into the local shard."""
            count = local_end - local_start
            self.fp8_rows[local_start:local_end].copy_(reinterpret_u8(rows).reshape(count, self.embedding_dim))
            self.fp8_scales[local_start:local_end].copy_(
                reinterpret_u8(scales).reshape(count, self.fp8_scales.shape[1]))
            self._filled_rows += count

        def sharded_state_dict(self, prefix: str = '', sharded_offsets: tuple = (), metadata=None):
            """Expose both planes so DCP saves the store verbatim (no bf16 materialization)."""
            from megatron.core.dist_checkpointing.mapping import ShardedTensor
            del metadata
            prepend = len(sharded_offsets)
            out = {}
            for name, tensor, width in (('weight', self.fp8_rows.view(torch.float8_e4m3fn), self.embedding_dim),
                                        ('weight_scale_inv', self.fp8_scales.view(torch.float8_e8m0fnu),
                                         self.fp8_scales.shape[1])):
                key = f'{prefix}{name}'
                global_shape = [1] * prepend + [self.global_num_embeddings, width]
                global_offset = [0] * prepend + [self.row_start, 0]
                out[key] = ShardedTensor(
                    key=key,
                    data=tensor,
                    dtype=tensor.dtype,
                    local_shape=tuple(tensor.shape),
                    global_shape=tuple(global_shape),
                    global_offset=tuple(global_offset),
                    axis_fragmentations=None,
                    replica_id=(0, get_pg_rank(self.tp_group), get_pg_rank(self.expt_dp_group)),
                    prepend_axis_num=prepend)
            return out

    class Fp8EngramEmbedding(EPShardedMultiTableEmbedding):
        """``EPShardedMultiTableEmbedding`` whose tables are the fp8-resident subclass."""

        def __init__(self,
                     config,
                     table_sizes,
                     embedding_dim,
                     init_method=None,
                     ep_group=None,
                     tp_group=None,
                     expt_dp_group=None,
                     block_size=32):
            MegatronModule.__init__(self, config=config)
            self.table_sizes = tuple(table_sizes)
            self.embedding_dim = embedding_dim
            self.ep_group = ep_group
            self.ep_size = get_pg_size(ep_group)
            self.tables = nn.ModuleList([
                Fp8EngramTable(
                    config=config,
                    global_num_embeddings=size,
                    embedding_dim=embedding_dim,
                    init_method=init_method,
                    ep_group=ep_group,
                    tp_group=tp_group,
                    expt_dp_group=expt_dp_group,
                    block_size=block_size) for size in self.table_sizes
            ])
            parts = [divmod(size, self.ep_size) for size in self.table_sizes]
            device = self.tables[0].fp8_rows.device
            self.register_buffer(
                'table_row_base',
                torch.tensor([base for base, _ in parts], dtype=torch.int64, device=device),
                persistent=False)
            self.register_buffer(
                'table_row_remainder',
                torch.tensor([rem for _, rem in parts], dtype=torch.int64, device=device),
                persistent=False)

        @property
        def local_parameter_count(self) -> int:
            return sum(table.local_parameter_count for table in self.tables)

else:
    Fp8EngramTable = None
    Fp8EngramEmbedding = None
