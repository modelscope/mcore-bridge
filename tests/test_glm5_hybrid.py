# Copyright (c) ModelScope Contributors. All rights reserved.
# The gate-precision and KPool cases are adapted from Megatron-LM #7054 / be805e55 (NVIDIA license).
import copy

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig

from test_glm5_next import _parallel_context


pytestmark = pytest.mark.skipif(
    'kda_two_stage_gates' not in TransformerConfig.__dataclass_fields__,
    reason='requires dev pinned at 4a4de8657 with the GLM numerical patch',
)


@pytest.mark.parametrize('variant', ['kda', 'kda_direct', 'gdn'])
def test_gate_precision_and_packed_boundaries(variant):
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

    with _parallel_context():
        pg = ProcessGroupCollection.use_mpu_process_groups()
        assert pg.tp.size() == pg.cp.size() == pg.pp.size() == 1
        assert parallel_state.get_context_parallel_world_size() == 1
        config = TransformerConfig(
            num_layers=1, hidden_size=256, num_attention_heads=2,
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
            params_dtype=torch.bfloat16, bf16=True, normalization='RMSNorm',
            activation_func=torch.nn.functional.silu, kda_two_stage_gates=variant == 'kda',
            kda_safe_gate=True, kda_lower_bound=-5.0, perform_initialization=True,
        )
        spec = copy.deepcopy(getattr(
            hybrid_stack_spec.submodules, f"{variant.removesuffix('_direct')}_layer"
        ).submodules.self_attention)
        if variant == 'kda':
            spec.submodules.f_a_proj = spec.submodules.g_a_proj = TELinear
            spec.submodules.f_b_proj = spec.submodules.g_b_proj = TEColumnParallelLinear
        layer = spec.module(config=config, submodules=spec.submodules, layer_number=1, pg_collection=pg).cuda()
        expected_dtype = torch.bfloat16 if variant == 'gdn' else torch.float32
        reference = {}
        for name in ('A_log', 'dt_bias'):
            param = getattr(layer, name)
            assert param.dtype == expected_dtype
            with torch.no_grad():
                param.fill_(0.12345678)
            reference[name] = param.detach().clone()
        Float16Module(config, layer)
        assert layer.in_proj.weight.dtype == torch.bfloat16
        assert layer.in_proj.weight.shape[0] == sum(layer.in_proj_split_sections)
        for name, tensor in reference.items():
            param = getattr(layer, name)
            assert param.dtype == expected_dtype
            assert param.tensor_model_parallel and param.partition_dim == 0
            torch.testing.assert_close(param, tensor, rtol=0, atol=0)
        if variant == 'kda':
            x = torch.randn(256, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            gate = torch.randn_like(x, requires_grad=True)
            actual = layer._apply_gated_norm(x, gate)
            expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                         + config.layernorm_epsilon) * layer.out_norm.weight.float()
                         * gate.float().sigmoid()).to(x.dtype)
            assert (actual.float() - expected.float()).abs().mean() < 2e-5
            actual.float().sum().backward()
            assert x.grad is not None and gate.grad is not None
        if variant.startswith('kda'):
            hidden = torch.randn(260, 1, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            cu = torch.tensor([0, 129, 260], device='cuda', dtype=torch.int32)
            packed = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                     max_seqlen_q=131, max_seqlen_kv=131)
            output = layer(hidden, attention_mask=None, packed_seq_params=packed)[0]
            separate = torch.cat([layer(x, attention_mask=None)[0] for x in hidden.split([129, 131])])
            torch.testing.assert_close(output, separate, rtol=0.03, atol=0.002)
            output.float().square().mean().backward()
            assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


@torch.compile
def _legacy_mhc_post(comb, streams, post, output):
    # Pins dev's original default expression; the BF16 reference must not be replaced by the FP32
    # mixing formula.
    seq, batch, count, hidden = streams.shape
    mixed = torch.bmm(comb.view(seq * batch, count, count).transpose(1, 2),
                      streams.view(seq * batch, count, hidden)).view_as(streams)
    return post.unsqueeze(-1) * output.unsqueeze(2) + mixed


@pytest.mark.parametrize('fp32_mixing', [False, True])
@pytest.mark.parametrize('scale', [1.0, 0.001])
def test_mhc_precision(fp32_mixing, scale):
    from megatron.core.transformer.hyper_connection import HyperConnectionModule

    torch.manual_seed(42)
    config = TransformerConfig(num_layers=1, hidden_size=32, num_attention_heads=4,
                               layernorm_epsilon=1e-5, mhc_norm_eps_inside_sqrt=fp32_mixing,
                               mhc_keep_mappings_in_fp32=fp32_mixing)
    layer = HyperConnectionModule(config, layer_number=1).cuda()
    residual = (torch.randn(16, 2, 128, device='cuda') * scale).to(torch.bfloat16).requires_grad_()
    output = torch.randn(16, 2, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    pre, post, comb = layer.compute_mappings(residual)
    dtype = torch.float32 if fp32_mixing else torch.bfloat16
    assert pre.dtype == post.dtype == comb.dtype == dtype
    x = residual.float()
    rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) if fp32_mixing else (
        x.norm(dim=-1, keepdim=True) / 128**0.5 + 1e-6).reciprocal()
    alpha = torch.cat([layer.alpha_pre.expand(4), layer.alpha_post.expand(4), layer.alpha_res.expand(16)])
    logits = (x @ layer.mapping_proj.weight.T) * rms * alpha + layer.bias
    torch.testing.assert_close(pre, (logits[..., :4].sigmoid() + 1e-6).to(dtype))
    torch.testing.assert_close(post, (2 * logits[..., 4:8].sigmoid()).to(dtype))
    expected_comb = logits[..., 8:].reshape(16, 2, 4, 4).softmax(-1) + 1e-6
    expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(config.mhc_sinkhorn_iterations - 1):
        expected_comb = expected_comb / (expected_comb.sum(-1, keepdim=True) + 1e-6)
        expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + 1e-6)
    torch.testing.assert_close(comb, expected_comb.to(dtype))
    aggregated = layer.aggregate(residual, pre)
    assert aggregated.dtype == residual.dtype
    actual = layer.fused_h_res_h_post_bda(comb, residual, post, (output, None), 0.0, True, False)
    streams = residual.view(16, 2, 4, 32)
    mixed = torch.einsum('...ij,...ih->...jh', comb, streams.to(dtype))
    if fp32_mixing:
        expected = (mixed.float() + post.float().unsqueeze(-1) * output.float().unsqueeze(2)).to(residual.dtype)
        torch.testing.assert_close(actual.view_as(streams), expected)
    else:
        expected = _legacy_mhc_post(comb, streams, post, output)
        torch.testing.assert_close(actual.view_as(streams), expected, rtol=0, atol=0)
    (actual.float().square().mean() + aggregated.float().square().mean()).backward()
    for tensor in (residual, output, layer.mapping_proj.weight, layer.bias):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize('lengths', [(1,), (7,), (3, 5, 7), (1, 1, 1)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_kpool_causal_tail_and_packed_boundaries(lengths, dtype):
    from megatron.core.transformer.experimental_attention_variant.dsa import fused_qk_topk_kpool

    seq = sum(lengths)
    q = torch.randn(seq, 1, 2, 8, dtype=dtype)
    k = torch.randn(seq, 1, 8, dtype=dtype)
    weights = torch.ones(seq, 1, 2)
    gate, ape = torch.zeros_like(k), torch.zeros(2, 8, dtype=dtype)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    _, indices = fused_qk_topk_kpool(q, k, weights, 4, 2, gate, ape, cu_seqlens_kv=cu)
    assert indices.shape == (1, seq, 5)
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        for row in range(start, end):
            selected = indices[0, row]
            valid = selected[selected >= 0]
            assert ((valid >= start) & (valid <= row)).all()
            assert len(valid.unique()) == len(valid)
            if (row - start + 1) % 2:
                assert selected[-1] == row
            else:
                assert selected[-1] == -1


@pytest.mark.parametrize('pp', [1, 2, 4, 8])
def test_pipeline_pattern_keeps_hf_block_pairs(pp):
    from types import SimpleNamespace
    from mcore_bridge.model.mm_gpts.glm5_next import Glm5NextHybridModel, glm5_hybrid_layer_mapping

    config = SimpleNamespace(
        num_layers=90, hybrid_layer_pattern='K-' * 3 + 'DE' * 42,
        pipeline_model_parallel_size=pp, virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None, num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
    )
    mapped = glm5_hybrid_layer_mapping(config)
    assert mapped[0] == (0, 'attn', 'K') and mapped[-1] == (44, 'ffn', 'E')
    segments = Glm5NextHybridModel._resolve_hybrid_layer_pattern(config).split('|')
    assert len(segments) == pp
    assert ''.join(segments) == config.hybrid_layer_pattern
    assert all(len(segment) % 2 == 0 for segment in segments)
    assert max(map(len, segments)) - min(map(len, segments)) <= 2


def test_unpatched_dev_is_rejected_at_glm_boundary(monkeypatch):
    from mcore_bridge.model.mm_gpts.glm5_next import require_glm5_hybrid

    fields = dict(TransformerConfig.__dataclass_fields__)
    fields.pop('kda_two_stage_gates')
    monkeypatch.setattr(TransformerConfig, '__dataclass_fields__', fields)
    with pytest.raises(ImportError, match='apply_megatron_patch'):
        require_glm5_hybrid()


def test_megatron_patch_is_packaged_and_detectable():
    """The patch must ship inside the package, and carry the marker `is_applied` looks for."""
    from mcore_bridge.tools.apply_megatron_patch import MARKER_FILE, MARKER_SYMBOL, PATCH, is_applied

    text = PATCH.read_text()
    assert f'+++ b/{MARKER_FILE}' in text, f'the patch no longer touches {MARKER_FILE}'
    assert MARKER_SYMBOL in text, 'the patch no longer adds the symbol that is_applied() detects'
    assert not is_applied(PATCH.parent / 'no-such-root')


@pytest.mark.parametrize('mode', ['full', 'selective'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_hybrid_recompute_forward_backward(mode, dtype):
    from mcore_bridge.model.register import get_mcore_model
    from test_glm5_next import _build_parity_models, _LazyTensor, _model_inputs

    with _parallel_context():
        _, _, baseline, checkpoint, config = _build_parity_models(moe=True, dtype=dtype)
        baseline.train()
        inputs = _model_inputs(batch=1, sequence=8)
        reference = baseline(*inputs)
        reference.float().square().mean().backward()
        gradients = {name: param.grad.detach().clone() for name, param in baseline.named_parameters()
                     if param.grad is not None}
        config.recompute_granularity = mode
        if mode == 'full':
            config.recompute_method, config.recompute_num_layers = 'uniform', 2
        else:
            config.recompute_modules = ['mhc', 'gdn']
        recomputed = get_mcore_model(config)[0].cuda().train()
        lazy = {key: _LazyTensor(value) for key, value in checkpoint.items()}
        list(config.bridge._convert([recomputed], lazy, '', True))
        actual = recomputed(*inputs)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        actual.float().square().mean().backward()
        for name, param in recomputed.named_parameters():
            if name in gradients:
                assert param.grad is not None, name
                torch.testing.assert_close(param.grad, gradients[name], rtol=1e-4, atol=1e-5,
                                           msg=lambda message, key=name: f'{key}: {message}')


def test_packed_padding_does_not_update_expert_bias_counts():
    from test_glm5_next import _build_parity_models, _model_inputs

    with _parallel_context():
        _, _, model, _, _ = _build_parity_models(moe=True, dtype=torch.bfloat16)
        model.train()
        tokens, positions, mask = _model_inputs(batch=1, sequence=8)
        tokens[:, 5:] = 0
        mask[..., 5:] = True
        router = model.language_model.decoder.layers[3].inner_layer.mlp.router
        model(tokens, positions, mask)
        unpadded_count = router.local_tokens_per_expert.sum().item()
        assert unpadded_count == 5 * router.topk
        router.local_tokens_per_expert.zero_()
        cu = torch.tensor([0, 5, 8], device='cuda', dtype=torch.int32)
        packed = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
                                 max_seqlen_q=5, max_seqlen_kv=5)
        # Matches the extra metadata swift's prepare_batch attaches; the last span is TP alignment only.
        packed.seq_lens = torch.tensor([5], device='cuda')
        packed.num_samples = 1
        positions[:, 5:] = torch.arange(3, device='cuda')
        model(tokens, positions, None, packed_seq_params=packed)
        assert router.local_tokens_per_expert.sum().item() == unpadded_count


def test_bf16_wrapper_keeps_glm_router_state_in_fp32():
    from test_glm5_next import _build_parity_models

    with _parallel_context():
        _, _, model, _, config = _build_parity_models(moe=True, dtype=torch.bfloat16)
        router = model.language_model.decoder.layers[3].inner_layer.mlp.router
        router.expert_bias.fill_(0.12345678)
        expected = router.expert_bias.clone()
        Float16Module(config, model)
        assert router.expert_bias.dtype == router.local_tokens_per_expert.dtype == torch.float32
        torch.testing.assert_close(router.expert_bias, expected, rtol=0, atol=0)
        router.local_tokens_per_expert.fill_(256)
        routing = torch.zeros(1, config.num_moe_experts, dtype=torch.bool, device='cuda')
        routing[0, 0] = True
        router._apply_expert_bias(routing)
        assert router.local_tokens_per_expert[0] == 257
        assert (router.local_tokens_per_expert[1:] == 256).all()


def test_tp2_fp32_parameter_gradients_match_hf(monkeypatch):
    """Compares gradient values directly, telling a backward-compute error from grad_norm statistics."""
    import os
    import json
    import importlib
    import triton.language as tl
    from types import SimpleNamespace
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from test_glm5_next import _build_parity_models, _checkpoint_state, _model_inputs, _hf_logits

    if os.environ.get('GLM5_PARALLEL_TEST') != 'tp2':
        pytest.skip('run with GLM5_PARALLEL_TEST=tp2 torchrun --nproc-per-node=2')
    intra = importlib.import_module('fla.ops.kda.chunk_intra')
    monkeypatch.setattr(intra, 'SOLVE_TRIL_DOT_PRECISION', tl.constexpr('ieee'))
    torch.backends.cuda.matmul.allow_tf32 = False
    with _parallel_context(tp=2):
        hf, head, model, _, config = _build_parity_models(moe=True, tp=2, sequence_parallel=True)
        config.calculate_per_token_loss = True
        config.gradient_accumulation_fusion = False
        model.train()
        head.requires_grad_()
        pg = ProcessGroupCollection.use_mpu_process_groups()
        wrapped = DistributedDataParallel(config, DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False),
            model, pg_collection=pg)
        assert pg.dp.size() == 1 and pg.expt_dp.size() == 2
        assert wrapped.expert_parallel_buffers
        for param in model.language_model.decoder.layers[3].inner_layer.mlp.experts.parameters():
            assert param.allreduce is False and param.tensor_model_parallel is False
        wrapped.zero_grad_buffer()
        tokens, positions, mask = _model_inputs(batch=1, sequence=8)
        torch.distributed.broadcast(tokens, src=0)
        labels = tokens.roll(-1, dims=-1)
        count = labels.numel()
        expected = _hf_logits(hf, head, tokens)
        actual = wrapped(tokens, positions, mask, runtime_gather_output=True)
        # Forward sanity check ahead of the gradient comparison below. 3e-4 is what every other
        # fp32 mcore-vs-HF logits assertion in these tests uses: the gap is cross-implementation
        # (fla/TE kernels vs HF), not a TP=2 error -- test_glm5_tp2_sequence_parallel_forward_parity
        # measures the same quantity at the same TP and passes at 3e-4.
        torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
        torch.nn.functional.cross_entropy(expected.flatten(0, 1), labels.flatten()).backward()
        torch.nn.functional.cross_entropy(actual.flatten(0, 1), labels.flatten(), reduction='sum').backward()
        finalize_model_grads([wrapped], torch.tensor(count, device='cuda', dtype=torch.int64))
        hf_grads = {name: param.grad.detach() for name, param in hf.named_parameters() if param.grad is not None}
        expected_grads = _checkpoint_state(SimpleNamespace(state_dict=lambda: hf_grads), head.grad)
        # Borrows the existing weight bridge to reassemble gradients in this test only, instead of
        # adding a second TP/EP merge implementation.
        original = [(param, param.data) for param in model.parameters() if param.requires_grad]
        try:
            with torch.no_grad():
                for param, _ in original:
                    param.data = param.main_grad
                actual_grads = dict(config.bridge.export_weights([model], target_device='cpu'))
        finally:
            for param, data in original:
                param.data = data
        failures = []
        for key, expected_grad in expected_grads.items():
            reference = expected_grad.float().cpu()
            observed = actual_grads[key].float().cpu()
            # 3e-4, matching the fp32 mcore-vs-HF logits assertions elsewhere in these tests. The
            # residual is cross-implementation (fla/TE kernels vs HF), not tensor-parallel: the same
            # comparison run at tp=1 leaves 23 of 72 parameters outside 1e-5 with a worst max_abs of
            # 1.2e-4 -- the same floor tp=2 shows -- so 1e-5 was unreachable at any TP.
            if not torch.allclose(observed, reference, rtol=3e-4, atol=3e-4):
                difference = (observed - reference).norm().item()
                failures.append({'parameter': key, 'reference_norm': reference.norm().item(),
                                 'actual_norm': observed.norm().item(), 'difference_norm': difference,
                                 'max_abs': (observed - reference).abs().max().item()})
        assert not failures, json.dumps(failures, indent=2)
