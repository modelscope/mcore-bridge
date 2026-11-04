# Copyright (c) ModelScope Contributors. All rights reserved.
# code borrowed from modelscope/ms-swift
import inspect
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing
from megatron.core.transformer.multi_token_prediction import roll_tensor as mcore_roll_tensor
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import set_seed
from typing import Optional

from .logger import get_logger

# Megatron dev (0.20+) refactored `roll_tensor` to take a LIST of tensors and return a list of rolled
# tensors; releases take a single tensor and return `(rolled, rolled.sum())`. mcore-bridge's callers
# (gpt_model / mtp_layer) use the single-tensor form, so detect the installed signature once.
_ROLL_TAKES_LIST = 'tensors' in inspect.signature(mcore_roll_tensor).parameters

logger = get_logger()


def split_cp_inputs(inputs: torch.Tensor,
                    cu_seqlens: Optional[torch.Tensor],
                    dim: int,
                    cp_partition_mode: str = 'zigzag'):
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    if cp_partition_mode == 'contiguous':
        # Rank r owns the contiguous block [r * local_rows, (r + 1) * local_rows) of the
        # flattened packed sequence. This must match the DSv4 THD CP forward, which assumes
        # each rank holds a single contiguous slice (global_start = cp_rank * local_rows).
        total_rows = inputs.shape[dim]
        assert total_rows % cp_size == 0, (
            f'Contiguous CP slicing requires dim size={total_rows} to be divisible by cp_size={cp_size}.')
        local_rows = total_rows // cp_size
        slices = [slice(None)] * inputs.ndim
        slices[dim] = slice(cp_rank * local_rows, (cp_rank + 1) * local_rows)
        return inputs[tuple(slices)].contiguous()
    new_inputs = []
    for i in range(1 if cu_seqlens is None else (cu_seqlens.shape[0] - 1)):
        if cu_seqlens is None:
            val = inputs
        else:
            slices = [slice(None)] * inputs.ndim
            slices[dim] = slice(cu_seqlens[i], cu_seqlens[i + 1])
            val = inputs[tuple(slices)]
        view_shape = (*inputs.shape[:dim], 2 * cp_size, val.shape[dim] // (2 * cp_size), *inputs.shape[dim + 1:])
        val = val.view(view_shape)
        # Keep the index on the same device as the activation. `.cuda()` sends an NPU
        # (or CPU) tensor's index to CUDA and index_select then fails before the CP slice.
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], dtype=torch.long, device=val.device)
        val = val.index_select(dim, index)
        view_shape = (*inputs.shape[:dim], -1, *inputs.shape[dim + 1:])
        new_inputs.append(val.view(view_shape))
    return torch.cat(new_inputs, dim=dim)


def get_num_samples(packed_seq_params) -> int:
    """Number of real (unpadded) samples packed into a THD micro-batch.

    ``cu_seqlens_q.numel() - 1`` is *not* this number: the data pipeline pads the packed
    sequence (see swift ``get_padding_to``) and the padding tokens carry ``position_ids == 0``,
    so the padded tail opens extra segments (one per token when cp_size == 1, one per
    ``2 * cp_size`` tokens otherwise). Using the segment count would inflate the
    ``[num_samples, max_seqlen_q, h]`` buffers built by the GDN/PLE THD paths by up to
    ``padding_to - 1`` extra rows.

    ``num_samples``/``seq_lens`` are attached to ``PackedSeqParams`` by the data pipeline;
    they are not mcore dataclass fields. Fall back to the segment count only when neither is
    available, i.e. for callers that build ``PackedSeqParams`` directly (the convert-precision
    check and the README example), which run tiny batches.
    """
    num_samples = getattr(packed_seq_params, 'num_samples', None)
    if num_samples is not None:
        return int(num_samples)
    seq_lens = getattr(packed_seq_params, 'seq_lens', None)
    if seq_lens is not None:
        return int(seq_lens.shape[0])
    return int(packed_seq_params.cu_seqlens_q.numel()) - 1


def reconstruct_tensor_cp(tensor, packed_seq_params, dim: int, cp_partition_mode: str = 'zigzag') -> torch.Tensor:
    """In CP mode, all-gather and undo the chunking produced by
    ``split_cp_inputs``, restoring the full sequence in original token order
    along ``dim``.

    Args:
        tensor: CP-sharded local tensor whose sequence dim is at ``dim``.
        packed_seq_params: ``PackedSeqParams`` for THD inputs, or ``None`` for
            regular ``[B, S, ...]`` inputs.
        dim: Sequence dimension index of ``tensor`` (default: 1).
        cp_partition_mode: CP partition layout, either ``zigzag`` or ``contiguous``.
            It must match the mode used by ``split_cp_inputs``.

    Returns:
        torch.Tensor: Full-sequence tensor with the same shape as ``tensor``
        except the size at ``dim`` is multiplied by ``cp_size``.

    Gradient: only the local shard keeps its autograd graph; the slices gathered
    from other ranks are constants. That is correct when the tensor needs no
    grad, or when every rank computes the same full-sequence loss from the
    result (each rank then owns the gradient of its own shard). It drops
    gradient when the consumer mixes tokens across shards and splits its output
    back to the local shard; use a differentiable gather there
    (``gather_from_sequence_parallel_region(..., tensor_parallel_output_grad=True,
    group=cp_group)``), as PLE does.
    """

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size <= 1:
        return tensor

    cp_rank = mpu.get_context_parallel_rank()
    cp_group = mpu.get_context_parallel_group()

    # All-gather across CP ranks (preserve local autograd graph for `tensor`).
    output_list = [torch.empty_like(tensor) for _ in range(cp_size)]
    torch.distributed.all_gather(output_list, tensor.contiguous(), group=cp_group)
    output_list[cp_rank] = tensor
    gathered = torch.cat(output_list, dim=dim)

    if cp_partition_mode == 'contiguous':
        # Rank r owns block r, so concatenating the shards in rank order already
        # restores the original token order.
        return gathered

    # `_undo_attention_load_balancing` assumes sequence dim is 0; transpose if needed.
    if dim != 0:
        gathered = gathered.transpose(0, dim).contiguous()
    out = _undo_attention_load_balancing(gathered, cp_size, packed_seq_params)
    if dim != 0:
        out = out.transpose(0, dim).contiguous()
    return out


def get_local_layer_specs(config, layer_specs, vp_stage=None):
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if getattr(config, 'pipeline_model_parallel_layout', None) is not None:
        from megatron.core.transformer.enums import LayerType
        local_layer_specs = [
            layer_specs[layer_id] for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage)
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset:offset + num_layers_to_build]
    return local_layer_specs


def set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (1009 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (11 * mpu.get_data_parallel_rank())
        set_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker,
                                                            use_cudagraphable_rng)
    else:
        raise ValueError(f'Seed ({seed_}) should be a positive integer.')


def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
    """Roll one tensor, returning ``(rolled, rolled.sum())`` as the releases do."""
    if _ROLL_TAKES_LIST:
        rolled = mcore_roll_tensor([tensor],
                                   shifts=shifts,
                                   dims=dims,
                                   cp_group=cp_group,
                                   packed_seq_params=packed_seq_params)[0]
        return rolled, rolled.sum()
    return mcore_roll_tensor(tensor, shifts=shifts, dims=dims, cp_group=cp_group, packed_seq_params=packed_seq_params)
