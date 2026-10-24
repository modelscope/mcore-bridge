# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import torch
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/utils/accelerator.py'
    spec = importlib.util.spec_from_file_location('accelerator_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_when_no_accelerator_is_available():
    device = _load().accelerator_device()
    if torch.cuda.is_available() or (getattr(torch, 'npu', None) is not None and torch.npu.is_available()):
        assert device.type in {'cuda', 'npu'}
    else:
        assert device.type in {'cpu', 'mps'}


def test_cuda_current_device_is_not_called_on_cpu(monkeypatch):
    if torch.cuda.is_available():
        return
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: (_ for _ in ()).throw(AssertionError('cuda')))
    device = _load().accelerator_device()
    assert device.type != 'cuda'
