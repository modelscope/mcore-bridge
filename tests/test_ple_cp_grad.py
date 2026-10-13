# Copyright (c) ModelScope Contributors. All rights reserved.
"""GPU regression test: PLE under context parallelism must match cp=1 in forward and backward.

PLE gathers the CP shards, runs a causal conv over the full sequence and splits the output back
to the local shard, so the gradient of a token near a shard boundary is partly produced on another
CP rank. The test drives the real ``Qwen4ExpTextPLELayer.forward`` on real NCCL ranks and compares
the local input gradient against a cp=1 run of the same layer. A negative control switches the CP
gather back to a local-slice backward (the ``reconstruct_tensor_cp`` semantics) and must fail.
"""
import itertools
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import TransformerConfig

from mcore_bridge.model.modules import ple as ple_module
from mcore_bridge.utils.megatron_utils import split_cp_inputs

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
requires_nccl = pytest.mark.skipif(not dist.is_nccl_available(), reason='requires NCCL')


def _config(cp_size):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        context_parallel_size=cp_size,
        params_dtype=torch.float32,
        perform_initialization=False,  # every parameter is overwritten in _layer
    )
    # PLE fields normally filled in by the qwen4_exp config parser.
    config.hc_count = 2
    config.ple_embed_dim = 16
    config.ple_conv_kernel_size = 4
    config.ngram_size = 3  # also the conv dilation: the receptive field spans 9 previous tokens
    config.heads_per_ngram = 2
    config.eos_token_id = 0
    config.split_ngram_parts = 1
    config.ple_seed = 1234
    config.padded_vocab_size = 64
    config.ngram_vocab_size_base = 31
    config.make_ngram_vocab_size_divisible_by = 4
    return config


def _layer(cp_size, device):
    layer = ple_module.Qwen4ExpTextPLELayer(_config(cp_size), ple_layer_index=0).to(device)
    generator = torch.Generator(device='cpu').manual_seed(2026)
    with torch.no_grad():
        for name, param in layer.named_parameters():
            # conv1d is zero-initialized; a zero conv would hide the cross-shard gradient.
            if 'norm' in name:
                value = 0.5 + torch.rand(param.shape, generator=generator)
            else:
                value = 0.5 * torch.randn(param.shape, generator=generator)
            param.copy_(value.to(param.dtype))
    return layer


def _packed_params(lengths, device):
    boundaries = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device)
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
    )


def _inputs(layout, cp_size, device):
    generator = torch.Generator(device='cpu').manual_seed(7)
    width = 16  # hc_count * hidden_size
    if layout == 'thd':
        packed = _packed_params([4 * cp_size, 12 * cp_size], device)
        total, batch = 16 * cp_size, 1
    else:
        packed, total, batch = None, 16 * cp_size, 2
    hidden = torch.randn(total, batch, width, generator=generator).to(device)
    probe = torch.randn(total, batch, width, generator=generator).to(device)
    input_ids = torch.randint(0, 64, (batch, total), generator=generator).to(device)
    return hidden, probe, input_ids, packed


def _run_cp(layer, hidden, probe, input_ids, packed):
    cu = None if packed is None else packed.cu_seqlens_q
    local_hidden = split_cp_inputs(hidden, cu, dim=0).detach().clone().requires_grad_(True)
    local_ids = split_cp_inputs(input_ids, cu, dim=1)
    layer.zero_grad(set_to_none=True)
    local_out = layer(local_hidden, local_ids, packed)
    (local_out * split_cp_inputs(probe, cu, dim=0)).sum().backward()
    param_grads = {name: param.grad.detach().clone() for name, param in layer.named_parameters()}
    for grad in param_grads.values():
        dist.all_reduce(grad, group=parallel_state.get_context_parallel_group())
    return local_out.detach(), local_hidden.grad, param_grads


def _local_slice_backward_gather(input_, tensor_parallel_output_grad=True, group=None, **kwargs):
    return _megatron_gather(input_, tensor_parallel_output_grad=False, group=group, **kwargs)


_megatron_gather = ple_module.gather_from_sequence_parallel_region


def _cp_grad_worker(rank, world_size, init_file):
    # TE runs fp32 GEMMs in TF32 by default (~3e-4 relative on the projection grads); pin true fp32
    # before the first cuBLAS handle so the cp=1 vs cp=N comparison can use tight tolerances.
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    try:
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        dist.init_process_group(backend='nccl', init_method=f'file://{init_file}', rank=rank, world_size=world_size)
        parallel_state.initialize_model_parallel(context_parallel_size=world_size)
        for layout, fused in itertools.product(('sbhd', 'thd'), ('1', '0')):
            os.environ['PLE_FUSED_KERNEL'] = fused
            case = f'layout={layout} fused={fused} cp={world_size}'
            layer = _layer(world_size, device)
            hidden, probe, input_ids, packed = _inputs(layout, world_size, device)
            cu = None if packed is None else packed.cu_seqlens_q

            # cp=1 reference: the same layer on the full sequence.
            full_hidden = hidden.detach().clone().requires_grad_(True)
            layer.zero_grad(set_to_none=True)
            full_out = layer._forward_impl(full_hidden, input_ids, packed)
            (full_out * probe).sum().backward()
            ref_grads = {name: param.grad.detach().clone() for name, param in layer.named_parameters()}
            ref_out = split_cp_inputs(full_out.detach(), cu, dim=0)
            ref_hidden_grad = split_cp_inputs(full_hidden.grad, cu, dim=0)

            out, hidden_grad, param_grads = _run_cp(layer, hidden, probe, input_ids, packed)
            torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5, msg=lambda m: f'{case} output: {m}')
            torch.testing.assert_close(
                hidden_grad, ref_hidden_grad, atol=1e-5, rtol=1e-5, msg=lambda m: f'{case} hidden grad: {m}')
            assert param_grads.keys() == ref_grads.keys()
            for name in param_grads:
                torch.testing.assert_close(
                    param_grads[name], ref_grads[name], atol=1e-4, rtol=1e-4, msg=lambda m: f'{case} {name}: {m}')

            # Negative control: a local-slice gather backward must lose the cross-shard gradient.
            ple_module.gather_from_sequence_parallel_region = _local_slice_backward_gather
            try:
                _, bad_hidden_grad, _ = _run_cp(layer, hidden, probe, input_ids, packed)
            finally:
                ple_module.gather_from_sequence_parallel_region = _megatron_gather
            error = (bad_hidden_grad - ref_hidden_grad).abs().max()
            dist.all_reduce(error, op=dist.ReduceOp.MAX)
            assert error > 1e-2, f'{case}: negative control did not fail (max error {error.item():.3e})'
            if rank == 0:
                print(f'{case}: ok, negative-control max hidden-grad error {error.item():.3e}')
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_cuda
@requires_nccl
@pytest.mark.parametrize('world_size', [2, 4])
def test_ple_cp_backward_matches_cp1(tmp_path, world_size):
    """Real NCCL CP ranks: PLE output, input gradient and summed parameter gradients match cp=1."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f'requires {world_size} CUDA devices')
    init_file = str(tmp_path / f'ple-cp{world_size}-init')
    mp.spawn(_cp_grad_worker, args=(world_size, init_file), nprocs=world_size, join=True)
