# Copyright (c) ModelScope Contributors. All rights reserved.
"""GPU regression tests for DeepSeek-V4.1 packed and context-parallel backward.

The fixture is deliberately checkpoint-free: it drives the real V4.1 Engram hashing,
packed-boundary, contiguous-CP gather/slice and differentiable lookup/projection path with a
tiny parameter set.  The CP case launches two local NCCL workers from pytest, so ordinary
CPU or single-GPU jobs collect it safely and report a skip.
"""
import itertools
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams

from mcore_bridge.model.modules import engram as engram_adapter

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
requires_two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two CUDA devices')
requires_nccl = pytest.mark.skipif(not dist.is_nccl_available(), reason='requires NCCL')
requires_native_engram = pytest.mark.skipif(
    not engram_adapter.has_native_engram(), reason='requires Megatron-Core Engram support')


def _packed_params(lengths, device):
    boundaries = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device)
    return PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=boundaries,
        cu_seqlens_kv=boundaries,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        cp_partition_mode='contiguous',
    )


def _tiny_engram(context_parallel_size, device):
    """Construct the real adapter around tiny deterministic PyTorch projections."""
    module = engram_adapter.DeepseekV41Engram.__new__(engram_adapter.DeepseekV41Engram)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        context_parallel_size=context_parallel_size,
        sequence_parallel=False,
        cp_partition_mode='contiguous',
    )
    module.engram_config = SimpleNamespace(
        excluded_token_ids=(),
        max_ngram_order=3,
        num_hash_heads=1,
        num_tables=2,
        hash_boundary_token_id=0,
        boundary_token_id=0,
        variant_spec=SimpleNamespace(resets_windows_at_boundary_token=False),
    )
    module.num_streams = 2
    module.hidden_size = 4
    module.tp_group = None
    module.tokenizer_remap = None
    module.register_buffer('hash_multipliers', torch.tensor([17, 31, 43], dtype=torch.int64, device=device))
    module.register_buffer('table_sizes', torch.tensor([17, 19], dtype=torch.int64, device=device))
    module.embedding = torch.nn.Embedding(19, 2, device=device)
    module.value_projection = torch.nn.Linear(4, 4, bias=False, device=device)
    module.key_projection = torch.nn.Linear(4, 8, bias=False, device=device)
    module.key_norm = torch.nn.Identity()
    module.query_norm = torch.nn.Identity()
    return module


def _fixture_tensors(device):
    input_ids = torch.tensor([[5, 6, 7, 11, 12, 13, 14, 15]], dtype=torch.long, device=device)
    hidden = torch.linspace(-0.75, 0.75, steps=64, device=device).view(8, 1, 8)
    probe = torch.linspace(0.5, -0.25, steps=64, device=device).view_as(hidden)
    return input_ids, hidden, probe


def _parameter_grads(module):
    return {name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()}


def _assert_finite_nonzero(grads):
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), f'{name} gradient is not finite'
        assert grad.abs().sum() > 0, f'{name} gradient is all zero'


@requires_cuda
@requires_native_engram
def test_deepseek_v41_packed_backward_matches_separate_documents():
    """A packed document boundary must isolate both activations and parameter/input gradients."""
    if dist.is_initialized():
        pytest.skip('single-rank packing test')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.manual_seed(2026)
    packed_model = _tiny_engram(1, device)
    separate_model = _tiny_engram(1, device)
    separate_model.load_state_dict(packed_model.state_dict())
    input_ids, hidden, probe = _fixture_tensors(device)

    packed_hidden = hidden.detach().clone().requires_grad_(True)
    packed_model._bridge_packed_seq_params = _packed_params([3, 5], device)
    packed_output = packed_model(packed_hidden, input_ids)
    (packed_output * probe).sum().backward()
    packed_grads = _parameter_grads(packed_model)

    separate_hidden = hidden.detach().clone().requires_grad_(True)
    separate_outputs = []
    offset = 0
    for length in (3, 5):
        separate_outputs.append(
            separate_model(
                separate_hidden[offset:offset + length],
                input_ids[:, offset:offset + length],
            ))
        offset += length
    separate_output = torch.cat(separate_outputs, dim=0)
    (separate_output * probe).sum().backward()
    separate_grads = _parameter_grads(separate_model)

    torch.testing.assert_close(packed_output, separate_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(packed_hidden.grad, separate_hidden.grad, atol=1e-6, rtol=1e-6)
    assert packed_grads.keys() == separate_grads.keys()
    for name in packed_grads:
        torch.testing.assert_close(packed_grads[name], separate_grads[name], atol=1e-6, rtol=1e-6)
    _assert_finite_nonzero(packed_grads)


def _cp_backward_worker(rank, world_size, init_file):
    try:
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        dist.init_process_group(
            backend='nccl',
            init_method=f'file://{init_file}',
            rank=rank,
            world_size=world_size,
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=world_size,
        )
        torch.manual_seed(2026)
        full_model = _tiny_engram(1, device)
        cp_model = _tiny_engram(world_size, device)
        cp_model.load_state_dict(full_model.state_dict())
        input_ids, hidden, probe = _fixture_tensors(device)
        packed = _packed_params([3, 5], device)

        full_hidden = hidden.detach().clone().requires_grad_(True)
        full_model._bridge_packed_seq_params = packed
        full_output = full_model(full_hidden, input_ids)
        (full_output * probe).sum().backward()
        full_grads = _parameter_grads(full_model)

        local_length = hidden.shape[0] // world_size
        start = rank * local_length
        stop = start + local_length
        local_hidden = hidden[start:stop].detach().clone().requires_grad_(True)
        cp_model._bridge_packed_seq_params = packed
        local_output = cp_model(local_hidden, input_ids[:, start:stop])
        (local_output * probe[start:stop]).sum().backward()
        cp_grads = _parameter_grads(cp_model)
        for grad in cp_grads.values():
            dist.all_reduce(grad)

        output_shards = [torch.empty_like(local_output) for _ in range(world_size)]
        hidden_grad_shards = [torch.empty_like(local_hidden.grad) for _ in range(world_size)]
        dist.all_gather(output_shards, local_output.detach())
        dist.all_gather(hidden_grad_shards, local_hidden.grad)

        torch.testing.assert_close(torch.cat(output_shards), full_output, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.cat(hidden_grad_shards), full_hidden.grad, atol=1e-6, rtol=1e-6)
        assert cp_grads.keys() == full_grads.keys()
        for name in cp_grads:
            torch.testing.assert_close(cp_grads[name], full_grads[name], atol=1e-5, rtol=1e-5)
        _assert_finite_nonzero(cp_grads)
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_cuda
@requires_two_gpus
@requires_nccl
@requires_native_engram
def test_deepseek_v41_packed_cp2_backward_matches_cp1(tmp_path):
    """Two real NCCL CP ranks must reconstruct the packed forward and summed backward."""
    init_file = str(tmp_path / 'deepseek-v41-cp2-init')
    mp.spawn(_cp_backward_worker, args=(2, init_file), nprocs=2, join=True)
