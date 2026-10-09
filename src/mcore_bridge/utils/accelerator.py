# Copyright (c) ModelScope Contributors. All rights reserved.
"""Process accelerator, without assuming CUDA."""
import torch


def accelerator_device() -> torch.device:
    """Device used for collectives and newly built modules.

    ``torch.device('cuda')`` and ``torch.cuda.current_device()`` fail, or target
    the wrong chip, when the process is on Ascend NPU or CPU. CUDA keeps the
    current CUDA device, which is what ``device='cuda'`` already meant.
    """
    npu = getattr(torch, 'npu', None)
    if npu is not None and npu.is_available():
        return torch.device('npu', npu.current_device())
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
