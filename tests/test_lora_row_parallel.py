# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compare sharded TE LoRA forward/backward with an unsharded reference."""
import importlib.util
import pytest
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, init_file, tp_size, device_type):
    # MegatronAdaptor parses sys.argv when building TransformerConfig.
    sys.argv = [sys.argv[0]]
    if device_type == 'npu':
        import megatron_adaptor  # noqa: F401
        import torch_npu  # noqa: F401
    device_module = getattr(torch, device_type)
    device_module.set_device(rank)
    backend = 'hccl' if device_type == 'npu' else 'nccl'
    dist.init_process_group(backend, init_method=f'file://{init_file}', rank=rank, world_size=tp_size)
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TERowParallelLinear
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    from mcore_bridge.tuners.lora import LoraParallelLinear

    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            tensor_model_parallel_size=tp_size,
            params_dtype=torch.float32,
            gradient_accumulation_fusion=False,
        )
        base = TERowParallelLinear(
            input_size=32,
            output_size=16,
            config=config,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        )
        layer = LoraParallelLinear(base, 'default', r=4, lora_alpha=8)
        lora_a, lora_b = layer.lora_A['default'], layer.lora_B['default']
        assert tuple(lora_a.weight.shape) == (4, 32 // tp_size)
        assert tuple(lora_b.weight.shape) == (16, 4)
        generator = torch.Generator().manual_seed(7)

        def integers(*shape):
            return torch.randint(-1, 2, shape, generator=generator).to(f'{device_type}:{rank}', torch.float32)

        weight = integers(16, 32)
        ref_a = integers(4, 32).requires_grad_(True)
        ref_b = integers(16, 4).requires_grad_(True)
        ref_x = integers(4, 1, 32).requires_grad_(True)
        probe = integers(4, 1, 16)
        shard = slice(rank * 32 // tp_size, (rank + 1) * 32 // tp_size)
        with torch.no_grad():
            base.weight.copy_(weight[:, shard])
            lora_a.weight.copy_(ref_a[:, shard])
            lora_b.weight.copy_(ref_b)
        x = ref_x.detach()[..., shard].clone().requires_grad_(True)
        out, _ = layer(x)
        (out * probe).sum().backward()

        ref_out = ref_x @ weight.T + layer.scaling['default'] * (ref_x @ ref_a.T) @ ref_b.T
        (ref_out * probe).sum().backward()
        torch.testing.assert_close(out, ref_out.detach(), rtol=0, atol=0)
        torch.testing.assert_close(x.grad, ref_x.grad[..., shard], rtol=0, atol=0)
        torch.testing.assert_close(lora_a.weight.grad, ref_a.grad[:, shard], rtol=0, atol=0)
        torch.testing.assert_close(lora_b.weight.grad, ref_b.grad, rtol=0, atol=0)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize('device_type', ['cuda', 'npu'])
@pytest.mark.parametrize('tp_size', [1, 2])
def test_row_parallel_lora_matches_reference(tmp_path, tp_size, device_type):
    if device_type == 'npu':
        pytest.importorskip('torch_npu')
        if importlib.util.find_spec('megatron_adaptor') is None:
            pytest.skip('requires MegatronAdaptor for TENPU')
    device_module = getattr(torch, device_type)
    if not device_module.is_available() or device_module.device_count() < tp_size:
        pytest.skip(f'requires {tp_size} {device_type} devices')
    init_file = str(tmp_path / 'lora-row-init')
    mp.spawn(_worker, args=(init_file, tp_size, device_type), nprocs=tp_size, join=True)
