# Copyright (c) ModelScope Contributors. All rights reserved.
# code borrowed from modelscope/ms-swift
import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.transformer.module import Float16Module
from typing import Optional

from .logger import get_logger

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
        module_instances = (DDP, torch_FSDP, Float16Module)

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


def split_cp_inputs(inputs: torch.Tensor, cu_seqlens: Optional[torch.Tensor], dim: int):
    if dim < 0:
        dim = (dim + inputs.ndim) % inputs.ndim
    new_inputs = []
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
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
