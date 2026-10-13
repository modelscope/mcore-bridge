# Copyright (c) ModelScope Contributors. All rights reserved.
"""LoRA on TP-replicated (``parallel_mode='duplicated'``) TELinear bases.

Megatron's ``TELinear`` hands TE ``parallel_mode=None`` for duplicated layers, so the
``'duplicated'`` tag is gone after construction; the layout survives only as
``weight.tensor_model_parallel=False``. Before the fix, LoRA on such a base built a
``TEColumnParallelLinear`` ``lora_B`` and copied the erased ``parallel_mode=None`` onto it:
forward stayed local, but ``lora_B`` was flagged tensor parallel and never got the
``sequence_parallel`` flag, so under sequence parallel its grad (from this rank's sequence
shard only) was not summed over TP and the replicated copies drifted apart.

Two real NCCL TP ranks are launched from pytest; CPU or single-GPU jobs report a skip.
"""
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

requires_two_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='requires two CUDA devices')
requires_nccl = pytest.mark.skipif(not dist.is_nccl_available(), reason='requires NCCL')

TP = 2
SEQ, BATCH, HIDDEN, OUT, RANK, ALPHA = 8, 2, 32, 24, 4, 8


def _config(sequence_parallel):
    from megatron.core.transformer import TransformerConfig
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=TP,
        tensor_model_parallel_size=TP,
        sequence_parallel=sequence_parallel,
        params_dtype=torch.float32,
    )


def _duplicated_linear(config, cls=None):
    from megatron.core.extensions.transformer_engine import TELinear
    return (cls or TELinear)(
        HIDDEN,
        OUT,
        parallel_mode='duplicated',
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=True,
        skip_weight_param_allocation=False,
    )


def _wrap(base):
    from mcore_bridge.tuners.lora import LoraParallelLinear
    return LoraParallelLinear(
        base, 'default', r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, init_lora_weights=True, lora_bias=False)


def _check_detection(config):
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear

    from mcore_bridge.model.gpt_model import OutputLayerLinear
    from mcore_bridge.tuners.lora import _is_replicated_base

    duplicated = _duplicated_linear(config)
    # Megatron erases the tag; the replicated layout is only recorded on the weight.
    assert duplicated.parallel_mode is None
    assert _is_replicated_base(duplicated)
    assert _is_replicated_base(_duplicated_linear(config, OutputLayerLinear))

    kwargs = dict(config=config, init_method=config.init_method, bias=False, skip_bias_add=True, is_expert=False)
    column = TEColumnParallelLinear(HIDDEN, OUT, gather_output=False, **kwargs)
    # moe_shared_expert_overlap clears parallel_mode on still-sharded linears.
    column.parallel_mode = None
    row = TERowParallelLinear(HIDDEN, OUT, input_is_parallel=True, **kwargs)
    assert not _is_replicated_base(column)
    assert not _is_replicated_base(row)
    assert isinstance(_wrap(column).lora_B['default'], TEColumnParallelLinear)


def _worker(rank, init_file, sequence_parallel, check_detection):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=f'file://{init_file}', rank=rank, world_size=TP)
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=TP)
        model_parallel_cuda_manual_seed(1234)
        config = _config(sequence_parallel)
        lora = _wrap(_duplicated_linear(config))
        lora_a, lora_b = lora.lora_A['default'], lora.lora_B['default']
        # TE runs fp32 GEMMs in TF32; small integers keep every product and sum exact.
        gen = torch.Generator(device='cuda').manual_seed(0)

        def ints(*shape):
            return torch.randint(-2, 3, shape, device='cuda', generator=gen).float()

        weight = ints(OUT, HIDDEN)
        weight_a = ints(RANK, HIDDEN)
        weight_b = ints(OUT, RANK)
        x = ints(SEQ, BATCH, HIDDEN)
        probe = ints(SEQ, BATCH, OUT)
        with torch.no_grad():
            lora.base_layer.weight.copy_(weight)
            lora_a.weight.copy_(weight_a)
            lora_b.weight.copy_(weight_b)
        scaling = lora.scaling['default']

        # Under sequence parallel the replicated base sees this rank's sequence shard.
        if sequence_parallel:
            local = slice(rank * SEQ // TP, (rank + 1) * SEQ // TP)
            x_local, probe_local = x[local], probe[local]
        else:
            x_local, probe_local = x, probe
        x_local = x_local.clone().requires_grad_(True)
        out, _ = lora(x_local)
        (out * probe_local).sum().backward()

        # Megatron's finalize_model_grads sums sequence_parallel-flagged grads over TP.
        if sequence_parallel:
            for p in lora.parameters():
                if p.requires_grad and getattr(p, 'sequence_parallel', False):
                    dist.all_reduce(p.grad, group=parallel_state.get_tensor_model_parallel_group())

        ref_a = weight_a.clone().requires_grad_(True)
        ref_b = weight_b.clone().requires_grad_(True)
        ref_x = x.clone().requires_grad_(True)
        ref_out = ref_x @ weight.T + scaling * (ref_x @ ref_a.T) @ ref_b.T
        (ref_out * probe).sum().backward()
        ref_x_local = ref_x.grad[local] if sequence_parallel else ref_x.grad
        ref_out_local = ref_out[local] if sequence_parallel else ref_out

        tol = dict(rtol=0, atol=0)
        torch.testing.assert_close(out, ref_out_local.detach(), **tol)
        torch.testing.assert_close(x_local.grad, ref_x_local, **tol)
        torch.testing.assert_close(lora_a.weight.grad, ref_a.grad, **tol)
        torch.testing.assert_close(lora_b.weight.grad, ref_b.grad, **tol)

        # Replicated factors: full-size B, both factors outside TP sharding.
        assert tuple(lora_b.weight.shape) == (OUT, RANK)
        for factor in (lora_a, lora_b):
            assert getattr(factor.weight, 'tensor_model_parallel', None) is False
            if sequence_parallel:
                assert factor.weight.sequence_parallel
        if check_detection:
            _check_detection(config)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@requires_two_gpus
@requires_nccl
@pytest.mark.parametrize('sequence_parallel', [False, True], ids=['tp2', 'tp2_sp'])
def test_lora_on_replicated_base_matches_reference(tmp_path, sequence_parallel):
    """Forward and LoRA grads on a duplicated TELinear must match a single-device reference."""
    init_file = str(tmp_path / 'lora-replicated-init')
    mp.spawn(_worker, args=(init_file, sequence_parallel, sequence_parallel), nprocs=TP, join=True)
