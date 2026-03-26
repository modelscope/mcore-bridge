# Copyright (c) ModelScope Contributors. All rights reserved.
# code borrowed from modelscope/ms-swift
import gc
import hashlib
import os
import time
import torch
import torch.distributed as dist
import uuid
from contextlib import contextmanager
from datasets.utils.filelock import FileLock
from modelscope.hub.utils.utils import get_cache_dir
from transformers.utils import is_torch_cuda_available, is_torch_mps_available, is_torch_npu_available
from typing import Any, Mapping, Optional, Union

from .env import get_dist_setting, is_dist, is_local_master, is_master
from .logger import get_logger

logger = get_logger()


def _find_local_mac() -> str:
    mac = uuid.getnode()
    mac_address = ':'.join(('%012x' % mac)[i:i + 2] for i in range(0, 12, 2))
    return mac_address


def synchronize(device: Union[torch.device, str, int, None] = None):
    if is_torch_npu_available():
        torch.npu.synchronize(device)
    elif is_torch_cuda_available():
        torch.cuda.synchronize(device)
    else:
        torch.cuda.synchronize(device)


def time_synchronize() -> float:
    synchronize()
    return time.perf_counter()  # second


@contextmanager
def safe_ddp_context(hash_id: Optional[str], use_barrier: bool = True):
    if use_barrier and dist.is_initialized():
        if is_dist():
            if not is_master():
                dist.barrier()
            if not is_local_master():
                # Compatible with multi-machine scenarios,
                # where each machine uses different storage hardware.
                dist.barrier()
        yield
        if is_dist():
            if is_master():
                dist.barrier()
            if is_local_master():
                dist.barrier()
    elif hash_id is not None:
        lock_dir = os.path.join(get_cache_dir(), 'lockers')
        os.makedirs(lock_dir, exist_ok=True)
        file_path = hashlib.sha256(hash_id.encode('utf-8')).hexdigest() + '.lock'
        file_path = os.path.join(lock_dir, file_path)
        with FileLock(file_path):
            yield
    else:
        yield


def get_device(local_rank: Optional[Union[str, int]] = None) -> str:
    if local_rank is None:
        local_rank = max(0, get_dist_setting()[1])
    local_rank = str(local_rank)
    if is_torch_npu_available():
        device = 'npu:{}'.format(local_rank)
    elif is_torch_mps_available():
        device = 'mps:{}'.format(local_rank)
    elif is_torch_cuda_available():
        device = 'cuda:{}'.format(local_rank)
    else:
        device = 'cpu'

    return device


def get_current_device():
    if is_torch_npu_available():
        current_device = torch.npu.current_device()
    elif is_torch_cuda_available():
        current_device = torch.cuda.current_device()
    elif is_torch_mps_available():
        current_device = 'mps'
    else:
        current_device = 'cpu'
    return current_device


def get_torch_device():
    if is_torch_cuda_available():
        return torch.cuda
    elif is_torch_npu_available():
        return torch.npu
    elif is_torch_mps_available():
        return torch.mps
    else:
        return torch.cpu


def set_device(local_rank: Optional[Union[str, int]] = None):
    if local_rank is None:
        local_rank = max(0, get_dist_setting()[1])
    if is_torch_npu_available():
        torch.npu.set_device(local_rank)
    elif is_torch_cuda_available():
        torch.cuda.set_device(local_rank)


def get_device_count() -> int:
    if is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def empty_cache():
    if is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()


def gc_collect() -> None:
    gc.collect()
    empty_cache()


def to_float_dtype(data: Any, dtype: torch.dtype) -> Any:
    """Change the float inputs to a dtype"""
    if isinstance(data, Mapping):
        return type(data)({k: to_float_dtype(v, dtype) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_float_dtype(v, dtype) for v in data)
    elif isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        return data.to(dtype=dtype)
    else:
        return data


def to_device(data: Any, device: Union[str, torch.device, int], non_blocking: bool = False) -> Any:
    """Move inputs to a device"""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device, non_blocking) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device, non_blocking) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    else:
        return data
