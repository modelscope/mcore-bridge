# Copyright (c) ModelScope Contributors. All rights reserved.
"""GPU regression test: PLE parameter grads under sequence parallel must not be scaled by TP.

PLE's SP forward gathers the full sequence, runs ``compute`` and scatters the additive
output back to the local shard, so every TP rank already produces the complete
full-sequence gradient for the replicated key_proj/value_proj/norm_*/conv1d parameters.
The finalizer must therefore sync them with ``average_gradients_across_tp_domain`` (AVG);
the old ``sequence_parallel`` flag (SUM) multiplied them by tp_size (issue #233).

The test drives the real ``Qwen4ExpTextPLELayer.forward`` on real NCCL TP ranks, pushes
the grads through the real ``finalize_model_grads``, and compares against the same
layer's full-sequence ``_forward_impl`` (the TP=1, SP=off reference). A negative control
restores the old flag and must reproduce the tp_size amplification.

The second test covers LoRA adapters wrapped around the same projections (a pure-LoRA
setup with frozen base weights): they inherit the base's AVG sync through
``tuners/lora.py``, so their grads must stay at 1x as well.
"""
import itertools
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import types
from megatron.core.packed_seq_params import PackedSeqParams

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
requires_nccl = pytest.mark.skipif(not dist.is_nccl_available(), reason='requires NCCL')

PARAMS = ('key_proj.weight', 'value_proj.weight', 'norm_key.weight', 'norm_query.weight', 'norm_conv.weight',
          'conv1d.weight')
LORA_PARAMS = ('key_proj.lora_A.default.weight', 'key_proj.lora_B.default.weight', 'value_proj.lora_A.default.weight',
               'value_proj.lora_B.default.weight')


def _config(tp_size):
    from megatron.core.transformer import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=tp_size,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=tp_size > 1,
        params_dtype=torch.float32,
        perform_initialization=False,  # every parameter is overwritten in _layer
    )
    # PLE fields normally filled in by the qwen4_exp config parser.
    config.hc_count = 2
    config.ple_embed_dim = 16
    config.ple_conv_kernel_size = 4
    config.ngram_size = 3  # also the conv dilation
    config.heads_per_ngram = 2
    config.eos_token_id = 0
    config.split_ngram_parts = 1
    config.ple_seed = 1234
    config.padded_vocab_size = 64
    config.ngram_vocab_size_base = 31  # 4 heads -> primes 31/37/41/43 -> 152 rows (divisible by 4)
    config.make_ngram_vocab_size_divisible_by = 4
    return config


def _layer(config, tp_rank, device):
    from megatron.core.process_groups_config import ProcessGroupCollection

    from mcore_bridge.model.modules import ple as ple_module
    layer = ple_module.Qwen4ExpTextPLELayer(
        config, ple_layer_index=0, pg_collection=ProcessGroupCollection.use_mpu_process_groups()).to(device)
    generator = torch.Generator(device='cpu').manual_seed(2026)
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if name == 'ple_embedding.ngram_embedding.weight':
                continue  # TP-sharded table, filled below
            # conv1d is zero-initialized; a zero conv would hide the conv grad.
            if 'norm' in name:
                value = 0.5 + torch.rand(param.shape, generator=generator)
            else:
                value = 0.5 * torch.randn(param.shape, generator=generator)
            param.copy_(value.to(param.dtype))
        # The table is TP-sharded: build the same GLOBAL table for every tp_size and give
        # each rank its own row range, so activations match the reference.
        emb = layer.ple_embedding.ngram_embedding
        total = layer.ple_embedding.padded_vocab_size
        table = 0.5 * torch.randn(total, emb.weight.shape[-1], generator=torch.Generator(device='cpu').manual_seed(99))
        per = emb.num_embeddings_per_partition
        emb.weight.copy_(table[tp_rank * per:(tp_rank + 1) * per].to(emb.weight.dtype))
        emb.weight.requires_grad_(False)  # the n-gram table is frozen (issue #233 setup)
    return layer


def _lora_layer(config, tp_rank, device):
    """Pure-LoRA setup: adapters on the gathered-input projections, base weights frozen."""
    from mcore_bridge.tuners.lora import LoraParallelLinear

    layer = _layer(config, tp_rank, device)
    for name in ('key_proj', 'value_proj'):
        setattr(
            layer, name,
            LoraParallelLinear(
                getattr(layer, name),
                'default',
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                init_lora_weights=True,
                lora_bias=False))
    generator = torch.Generator(device='cpu').manual_seed(1234)
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if 'lora_' in name:
                param.copy_((0.5 * torch.randn(param.shape, generator=generator)).to(param.dtype))
    for name, param in layer.named_parameters():
        if 'lora_' not in name:
            param.requires_grad_(False)
    return layer


def _inputs(tp_size, layout, device):
    generator = torch.Generator(device='cpu').manual_seed(7)
    width = 16  # hc_count * hidden_size
    if layout == 'thd':
        lengths = [4 * tp_size, 12 * tp_size]
        boundaries = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=device)
        packed = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=boundaries,
            cu_seqlens_kv=boundaries,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths))
        total, batch = sum(lengths), 1
    else:
        packed, total, batch = None, 16 * tp_size, 2
    hidden = torch.randn(total, batch, width, generator=generator).to(device)
    probe = torch.randn(total, batch, width, generator=generator).to(device)
    input_ids = torch.randint(0, 64, (batch, total), generator=generator).to(device)
    return hidden, probe, input_ids, packed


def _grads(layer):
    return {
        name: param.grad.detach().clone()
        for name, param in layer.named_parameters() if param.requires_grad and param.grad is not None
    }


class _FinalizeChunk(torch.nn.Module):
    """Standalone chunk with the DDP surface ``finalize_model_grads`` touches."""

    def __init__(self, module, config):
        super().__init__()
        self.module = module
        self.config = config
        self.ddp_config = types.SimpleNamespace(use_megatron_fsdp=False)

    def finish_grad_sync(self, force_all_reduce=False):
        pass

    def scale_gradients(self, scaling):
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.mul_(scaling)


def _worker(rank, tp_size, init_file, layout, fused, lora=False):
    # TE runs fp32 GEMMs in TF32 by default (~3e-4 relative); pin true fp32 before the
    # first cuBLAS handle so the reference comparison can use tight tolerances.
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    os.environ['PLE_FUSED_KERNEL'] = fused
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', init_method=f'file://{init_file}', rank=rank, world_size=tp_size)
    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        _run_case(rank, tp_size, device, layout, fused, lora)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _run_case(rank, tp_size, device, layout, fused, lora=False):
    from megatron.core.distributed.finalize_model_grads import (_allreduce_non_tensor_model_parallel_grads,
                                                                finalize_model_grads)

    case = f'layout={layout} fused={fused} tp={tp_size}' + (' lora' if lora else '')
    config = _config(tp_size)
    layer = _lora_layer(config, rank, device) if lora else _layer(config, rank, device)
    checked = LORA_PARAMS if lora else PARAMS
    hidden, probe, input_ids, packed = _inputs(tp_size, layout, device)
    local = slice(rank * hidden.shape[0] // tp_size, (rank + 1) * hidden.shape[0] // tp_size)

    # The constructor (and lora.py for the adapters) must sync replicas with AVG, never
    # with the SP SUM.
    for name, param in layer.named_parameters():
        if name in checked:
            assert getattr(param, 'sequence_parallel', False) is False, f'{case} {name}: sequence_parallel'
            assert getattr(param, 'average_gradients_across_tp_domain', False), \
                f'{case} {name}: average_gradients_across_tp_domain'

    # --- reference: full-sequence compute on the same layer (== TP=1, SP=off math) ---
    full_hidden = hidden.detach().clone().requires_grad_(True)
    layer.zero_grad(set_to_none=True)
    full_out = layer._forward_impl(full_hidden, input_ids, packed)
    (full_out * probe).sum().backward()
    ref = _grads(layer)
    ref_hidden = full_hidden.grad.detach().clone()

    def sp_backward():
        local_hidden = hidden[local].detach().clone().requires_grad_(True)
        layer.zero_grad(set_to_none=True)
        out = layer(local_hidden, input_ids, packed)
        (out * probe[local]).sum().backward()
        return out.detach(), local_hidden.grad.detach().clone()

    def backward_then_finalize():
        out, hidden_grad = sp_backward()
        chunk = _FinalizeChunk(layer, config)
        try:
            finalize_model_grads([chunk], num_tokens=None)
        except Exception:  # older chunk API: redo the backward, then the exact helper
            out, hidden_grad = sp_backward()
            _allreduce_non_tensor_model_parallel_grads([chunk], config)
        return out, hidden_grad, _grads(layer)

    # --- the SP forward/backward already produces the complete gradient on every rank ---
    out, hidden_grad = sp_backward()
    torch.testing.assert_close(out, full_out[local], rtol=1e-5, atol=1e-5, msg=lambda m: f'{case} forward: {m}')
    torch.testing.assert_close(
        hidden_grad, ref_hidden[local], rtol=1e-5, atol=1e-5, msg=lambda m: f'{case} input grad: {m}')
    pre = _grads(layer)
    for name in checked:
        torch.testing.assert_close(
            pre[name], ref[name], rtol=1e-4, atol=1e-5, msg=lambda m: f'{case} pre-finalize {name}: {m}')

    # --- the regression assertion: the real finalizer keeps the grads at 1x ---
    _, _, post = backward_then_finalize()
    for name in checked:
        torch.testing.assert_close(
            post[name], ref[name], rtol=1e-4, atol=1e-5, msg=lambda m: f'{case} post-finalize {name}: {m}')

    # --- negative control: the old flag must reproduce the tp_size amplification ---
    for name, param in layer.named_parameters():
        if name in checked:
            param.sequence_parallel = True
            param.average_gradients_across_tp_domain = False
    _, _, buggy = backward_then_finalize()
    for name in checked:
        torch.testing.assert_close(
            buggy[name],
            ref[name] * tp_size,
            rtol=2e-3,
            atol=1e-4,
            msg=lambda m: f'{case} negative control {name}: {m}')
    if rank == 0:
        print(f'{case}: ok', flush=True)


@requires_cuda
@requires_nccl
@pytest.mark.parametrize('world_size', [2, 4])
@pytest.mark.parametrize('layout', ['sbhd', 'thd'])
@pytest.mark.parametrize('fused', ['0', '1'])
def test_ple_sp_tp_grads_not_scaled(tmp_path, world_size, layout, fused):
    """SP+TP: PLE param grads after finalize must equal the TP=1 full-sequence reference."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f'requires {world_size} CUDA devices')
    init_file = str(tmp_path / f'ple-sp-tp{world_size}-{layout}-{fused}-init')
    mp.spawn(_worker, args=(world_size, init_file, layout, fused), nprocs=world_size, join=True)


@requires_cuda
@requires_nccl
@pytest.mark.parametrize('world_size', [2, 4])
@pytest.mark.parametrize('layout', ['sbhd', 'thd'])
def test_ple_lora_factors_not_scaled(tmp_path, world_size, layout):
    """SP+TP: LoRA factor grads on PLE's gathered-input projections must stay at 1x."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f'requires {world_size} CUDA devices')
    init_file = str(tmp_path / f'ple-lora-sp-tp{world_size}-{layout}-init')
    mp.spawn(_worker, args=(world_size, init_file, layout, '0', True), nprocs=world_size, join=True)
