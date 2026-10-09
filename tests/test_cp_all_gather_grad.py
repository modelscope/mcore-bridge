# Copyright (c) ModelScope Contributors. All rights reserved.
"""CP reconstruct must reduce-scatter gradients, not drop other ranks."""
import importlib.util
import sys
import torch
import types
from pathlib import Path


def _load():
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
    group = types.SimpleNamespace(size=lambda: 2)
    core.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 2,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: group,
    )
    core.tensor_parallel = types.SimpleNamespace()
    distributed = types.ModuleType('megatron.core.distributed')
    distributed.DistributedDataParallel = type('DDP', (), {})
    distributed.FullyShardedDataParallel = type('FSDP', (), {})
    ssm = types.ModuleType('megatron.core.ssm.mamba_context_parallel')
    ssm._undo_attention_load_balancing = lambda tensor, *args, **kwargs: tensor
    module_mod = types.ModuleType('megatron.core.transformer.module')
    module_mod.Float16Module = type('Float16Module', (), {})
    mtp = types.ModuleType('megatron.core.transformer.multi_token_prediction')
    mtp.roll_tensor = lambda tensor, shifts, dims: (tensor, tensor.sum())
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
    return loaded, group


def test_requires_grad_reduce_scatters_the_full_sequence():
    saved = sys.modules.copy()
    try:
        utils, group = _load()
        local = torch.tensor([[1.0], [2.0]], requires_grad=True)
        seen = {}

        def all_gather(parts, tensor, group=None):
            parts[0].copy_(tensor.detach())
            parts[1].copy_(tensor.detach() + 10)

        def reduce_scatter(output, chunks, group=None):
            seen['chunks'] = [chunk.detach().clone() for chunk in chunks]
            output.copy_(sum(chunks))

        torch.distributed.all_gather = all_gather
        torch.distributed.reduce_scatter = reduce_scatter
        gathered = utils.reconstruct_tensor_cp(local, None, 0, 'contiguous')
        assert torch.equal(gathered.detach(), torch.tensor([[1.0], [2.0], [11.0], [12.0]]))
        gathered.sum().backward()
    finally:
        sys.modules.clear()
        sys.modules.update(saved)
    # Each rank's compute contributes 1 to every gathered row. reduce-scatter
    # sums those contributions onto the local shard.
    assert torch.equal(local.grad, torch.tensor([[2.0], [2.0]]))
    assert len(seen['chunks']) == 2
