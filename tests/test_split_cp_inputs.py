# Copyright (c) ModelScope Contributors. All rights reserved.
"""Zigzag CP slicing must keep its index on the activation device.

Loads megatron_utils without importing the package (that pulls peft and a
real Megatron install). The parallel-state calls are stubbed.
"""
import importlib.util
import sys
import torch
import types
from pathlib import Path


def _load_megatron_utils():
    root = Path(__file__).resolve().parents[1] / 'src' / 'mcore_bridge'
    pkg = types.ModuleType('mcore_bridge')
    pkg.__path__ = [str(root)]
    pkg.__package__ = 'mcore_bridge'
    utils = types.ModuleType('mcore_bridge.utils')
    utils.__path__ = [str(root / 'utils')]
    utils.__package__ = 'mcore_bridge.utils'
    sys.modules['mcore_bridge'] = pkg
    sys.modules['mcore_bridge.utils'] = utils

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load('mcore_bridge.utils.logger', root / 'utils' / 'logger.py')

    megatron = types.ModuleType('megatron')
    core = types.ModuleType('megatron.core')
    core.__version__ = '0.16.1'
    megatron.core = core
    mpu = types.SimpleNamespace(get_context_parallel_world_size=lambda: 2, get_context_parallel_rank=lambda: 0)
    core.mpu = mpu
    core.tensor_parallel = types.SimpleNamespace()
    distributed = types.ModuleType('megatron.core.distributed')
    distributed.DistributedDataParallel = type('DDP', (), {})
    distributed.FullyShardedDataParallel = type('FSDP', (), {})
    ssm = types.ModuleType('megatron.core.ssm.mamba_context_parallel')
    ssm._undo_attention_load_balancing = lambda tensor, *args, **kwargs: tensor
    module_mod = types.ModuleType('megatron.core.transformer.module')
    module_mod.Float16Module = type('Float16Module', (), {})
    mtp = types.ModuleType('megatron.core.transformer.multi_token_prediction')

    def roll_tensor(tensor, shifts, dims):
        return tensor, tensor.sum()

    mtp.roll_tensor = roll_tensor
    block = types.ModuleType('megatron.core.transformer.transformer_block')
    block.get_num_layers_to_build = lambda *args, **kwargs: 0
    layer = types.ModuleType('megatron.core.transformer.transformer_layer')
    layer.get_transformer_layer_offset = lambda *args, **kwargs: 0
    for name, module in {
            'megatron': megatron,
            'megatron.core': core,
            'megatron.core.distributed': distributed,
            'megatron.core.ssm': types.ModuleType('megatron.core.ssm'),
            'megatron.core.ssm.mamba_context_parallel': ssm,
            'megatron.core.transformer': types.ModuleType('megatron.core.transformer'),
            'megatron.core.transformer.module': module_mod,
            'megatron.core.transformer.multi_token_prediction': mtp,
            'megatron.core.transformer.transformer_block': block,
            'megatron.core.transformer.transformer_layer': layer,
    }.items():
        sys.modules[name] = module

    loaded = load('mcore_bridge.utils.megatron_utils', root / 'utils' / 'megatron_utils.py')
    loaded.mpu = mpu
    return loaded


def test_zigzag_split_keeps_index_on_cpu():
    saved = sys.modules.copy()
    try:
        utils = _load_megatron_utils()
        # 8 tokens, cp=2, rank 0 owns chunks 0 and 3 (zigzag).
        values = torch.arange(8, dtype=torch.float32)
        local = utils.split_cp_inputs(values, None, 0, 'zigzag')
    finally:
        sys.modules.clear()
        sys.modules.update(saved)
    assert local.device.type == 'cpu'
    assert local.tolist() == [0, 1, 6, 7]
