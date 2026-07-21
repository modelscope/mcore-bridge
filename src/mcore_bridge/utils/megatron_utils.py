# Copyright (c) ModelScope Contributors. All rights reserved.
# code borrowed from modelscope/ms-swift
import megatron.core
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import FullyShardedDataParallel as megatron_FSDP
from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.multi_token_prediction import roll_tensor as mcore_roll_tensor
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from packaging import version
from transformers import set_seed
from typing import Optional

from .logger import get_logger

mcore_016 = version.parse(megatron.core.__version__) >= version.parse('0.16.0rc0')

logger = get_logger()


def unwrap_model(models, module_instances=None):
    """Unwrap_model to return the final model instance"""
    try:
        from megatron.core.utils import unwrap_model
        return unwrap_model(models, module_instances)
    except ImportError:
        pass
    if module_instances is None:
        from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP
        module_instances = (DDP, torch_FSDP, Float16Module, megatron_FSDP)

    return_list = True
    if not isinstance(models, list):
        models = [models]
        return_list = False
    unwrapped_model = []
    for model in models:
        while isinstance(model, module_instances):
            model = model.module
        unwrapped_model.append(model)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


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
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device='cpu',
                             pin_memory=True).cuda(non_blocking=True)
        val = val.index_select(dim, index)
        view_shape = (*inputs.shape[:dim], -1, *inputs.shape[dim + 1:])
        new_inputs.append(val.view(view_shape))
    return torch.cat(new_inputs, dim=dim)


def reconstruct_tensor_cp(tensor, packed_seq_params, dim: int) -> torch.Tensor:
    """In CP mode, all-gather and undo the load-balanced (zigzag) chunking
    produced by ``split_cp_inputs``, restoring the full sequence in original
    token order along ``dim``.

    Args:
        tensor: CP-sharded local tensor whose sequence dim is at ``dim``.
        packed_seq_params: ``PackedSeqParams`` for THD inputs, or ``None`` for
            regular ``[B, S, ...]`` inputs.
        dim: Sequence dimension index of ``tensor`` (default: 1).

    Returns:
        torch.Tensor: Full-sequence tensor with the same shape as ``tensor``
        except the size at ``dim`` is multiplied by ``cp_size``.
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


# code borrowed from NVIDIA/Megatron-LM
def _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group=None):
    """Roll tensor with packed sequence support.
    This function handles rolling for packed sequences by respecting sequence boundaries
    """

    # Notice: This is a naive implementation to test the correctness,
    # a better solution will only sync the boundary tokens once.
    assert (dims == -1 or dims == tensor.dim() - 1), 'Packed sequence roll only supports the last dimension.'
    assert shifts == -1, 'Packed sequence roll only supports a single-token left shift.'
    cu_seqlens = packed_seq_params.cu_seqlens_q
    assert cu_seqlens is not None, 'Packed sequence parameters must provide cu_seqlens_q.'

    rolled_tensor = tensor.clone()

    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size == 1:
        # CP disabled: roll each packed sequence independently within its boundaries
        for i in range(len(cu_seqlens) - 1):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i + 1]
            seq_slice = tensor[..., start_idx:end_idx]
            rolled_seq = torch.roll(seq_slice, shifts=shifts, dims=dims)
            # Zero out the last position(s) that would cross sequence boundaries
            rolled_seq[..., shifts:] = 0
            rolled_tensor[..., start_idx:end_idx] = rolled_seq
        return rolled_tensor, rolled_tensor.sum()

    # CP enabled: each rank owns two chunks per sequence (front and mirrored tail).
    local_rank = torch.distributed.get_rank(group=cp_group)
    global_ranks = torch.distributed.get_process_group_ranks(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % cp_size]
    prev_rank = global_ranks[(local_rank - 1) % cp_size]

    # Iterate over each sequence individually
    for i in range(len(cu_seqlens) - 1):
        start_idx = cu_seqlens[i]
        end_idx = cu_seqlens[i + 1]

        # the idx has been multiplied by cp_size, need to divide it by cp_size to get the local idx
        local_start_idx = start_idx // cp_size
        local_end_idx = end_idx // cp_size

        # Skip empty sequences - this can happen when a sequence is very short and
        # after dividing by cp_size, the local slice has zero length
        local_seq_len = local_end_idx - local_start_idx
        if local_seq_len == 0:
            continue

        tensor_slice = rolled_tensor[..., local_start_idx:local_end_idx].clone()

        # The following code is very similar as the code in roll_tensor function
        local_chunks = tensor_slice.chunk(2, dim=dims)
        rolled_chunks = [torch.roll(chunk, shifts=shifts, dims=dims) for chunk in local_chunks]

        tensor_send_list = []
        tensor_recv_list = []
        for chunk in rolled_chunks:
            # Skip empty chunks that can occur when the sequence slice is very small
            if chunk.size(dims) == 0:
                tensor_send_list.append(torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device))
                tensor_recv_list.append(torch.empty(chunk.shape[:-1], dtype=chunk.dtype, device=chunk.device))
                continue
            boundary = chunk.select(dims, shifts).contiguous().clone()
            tensor_send_list.append(boundary)
            tensor_recv_list.append(torch.empty_like(boundary))

        ops = []
        if local_rank != 0:
            ops.append(torch.distributed.isend(tensor=tensor_send_list[0], dst=prev_rank))
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[1], src=prev_rank))
        else:
            tensor_recv_list[1].zero_()

        if local_rank != cp_size - 1:
            ops.append(torch.distributed.irecv(tensor=tensor_recv_list[0], src=next_rank))
            ops.append(torch.distributed.isend(tensor=tensor_send_list[1], dst=next_rank))
        else:
            tensor_recv_list[0].copy_(tensor_send_list[1])

        for op in ops:
            op.wait()

        index = [slice(None)] * rolled_chunks[0].dim()
        index[dims] = shifts
        for chunk, recv in zip(rolled_chunks, tensor_recv_list):
            # Skip empty chunks
            if chunk.size(dims) == 0:
                continue
            chunk[tuple(index)] = recv

        seq_result = torch.cat(rolled_chunks, dim=dims)

        # update the rolled tensor
        rolled_tensor[..., local_start_idx:local_end_idx] = seq_result

    return rolled_tensor, rolled_tensor.sum()


def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None):
    if mcore_016 or packed_seq_params is None:
        kwargs = {'packed_seq_params': packed_seq_params} if mcore_016 else {}
        return mcore_roll_tensor(tensor, shifts=shifts, dims=dims, cp_group=cp_group, **kwargs)
    # mcore 0.15 & packed_seq_params
    return _roll_tensor_packed_seq(tensor, shifts, dims, packed_seq_params, cp_group)
