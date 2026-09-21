# Copyright (c) ModelScope Contributors. All rights reserved.
"""MTP (multi-token prediction) tests for Qwen3.8-Flash-Next (mcore type `qwen4_exp`).

The MTP head is the `residual_linear_shared` fusion: separate fc_embedding/fc_hidden (e_proj/h_proj),
a joint multi-stream pre_fc_norm_hidden, and a hyper_connection_mixer contraction -- built on the
gated-HC backbone but NOT Megatron's enable_hyper_connections mHC. Fixtures are tiny synthetic
configs (last backbone layer is full-attention so the MTP inner block, which is always
full-attention, matches). The QSA indexer's hard top-k selection is non-differentiable in the
sbhd/bool-mask path, so its params legitimately carry no gradient there -- exactly as the backbone's
full-attention indexer does -- and are excluded from the grad-flow assertion.
"""
import os
import pytest
import torch
import uuid
from contextlib import contextmanager
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from mcore_bridge.config import ModelConfig
from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.register import get_mcore_model


def _tiny_qwen4exp_config():
    cfg = pytest.importorskip('transformers.models.qwen4_exp.configuration_qwen4_exp')
    text = {
        'model_type': 'qwen4_exp_text',
        'vocab_size': 512,
        'hidden_size': 128,
        'head_dim': 64,
        'num_attention_heads': 4,
        'num_key_value_heads': 4,
        'num_hidden_layers': 4,
        # Last layer full-attention so layer_specs[-1] (the MTP inner block) is full-attention + QSA.
        'layer_types': ['linear_attention', 'linear_attention', 'linear_attention', 'qwen_sparse_attention'],
        'full_attention_interval': 4,
        'num_experts': 4,
        'num_experts_per_tok': 2,
        'moe_intermediate_size': 64,
        'shared_expert_intermediate_size': 64,
        'hc_count': 4,
        'hc_lowrank': 32,
        'output_gate_type': 'sigmoid',
        'partial_rotary_factor': 0.25,
        'rope_parameters': {
            'mrope_interleaved': True,
            'mrope_section': [11, 11, 10],
            'partial_rotary_factor': 0.25,
            'rope_theta': 10000000,
            'rope_type': 'default'
        },
        'linear_num_key_heads': 16,
        'linear_num_value_heads': 48,
        'linear_key_head_dim': 128,
        'linear_value_head_dim': 128,
        'linear_conv_kernel_dim': 4,
        'indexer_n_heads': 4,
        'indexer_kv_heads': 1,
        'indexer_head_dim': 128,
        'indexer_budget': 2048,
        'indexer_compress_ratio': 4,
        'ple_layer_ids': [2],
        'ple_embed_dim': 32,
        'ple_conv_kernel_size': 4,
        'ngram_size': 3,
        'heads_per_ngram': 8,
        'ngram_vocab_size_base': 256,
        'make_ngram_vocab_size_divisible_by': 128,
        'split_ngram_parts': 2,
        'rms_norm_eps': 1e-6,
        'eos_token_id': 248044,
        'seed': 1234,
        'mtp_num_hidden_layers': 1,
        'mtp_use_dedicated_embeddings': False,
        'mtp': {
            'hybrid': True,
            'layer_types': ['full_attention'],
            'mtp_use_hidden_state_from_layer': None,
            'num_hidden_layers': 1,
            'rope_theta': 10000000
        },
    }
    vision = {
        'model_type': 'qwen4_exp',
        'depth': 2,
        'hidden_size': 128,
        'intermediate_size': 256,
        'num_heads': 4,
        'out_hidden_size': 128,
        'in_channels': 3,
        'patch_size': 16,
        'spatial_merge_size': 2,
        'temporal_patch_size': 2,
        'num_position_embeddings': 2304,
    }
    return cfg.Qwen4ExpConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        language_model_only=True,
    )


def _mcore_config(hf_config, mtp=1, dtype=torch.bfloat16, tp=1, pp=1, ep=1, recompute=False, shared=False):
    values = hf_to_mcore_config(hf_config)
    values['mcore_model_type'] = 'qwen4_exp'
    values['hf_config'] = hf_config
    values.update(
        params_dtype=dtype,
        pipeline_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        perform_initialization=True,
        use_cpu_initialization=False,
        language_model_only=True,
        moe_grouped_gemm=True,
        overlap_p2p_comm=False,
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        recompute_granularity='full' if recompute else None,
        recompute_method='uniform' if recompute else None,
        recompute_num_layers=1 if recompute else None,
        mtp_shared_weights=shared,
    )
    if mtp:
        values.update(mtp_num_layers=mtp, mtp_loss_scaling_factor=0.1)
    return ModelConfig(**values)


@pytest.fixture(scope='session', autouse=True)
def _distributed_session():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@contextmanager
def _parallel_context(tp=1, pp=1, ep=1):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != max(tp * pp, ep):
        pytest.skip(f'requires world size {max(tp * pp, ep)}')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    # Another test module in the same pytest process may already have initialized (and not torn
    # down) the default group and/or the model-parallel state, so repair both instead of asserting.
    created_pg = False
    if not torch.distributed.is_initialized():
        if world_size == 1:
            torch.distributed.init_process_group(
                'nccl', init_method=f'file:///tmp/q4e-mtp-{uuid.uuid4().hex}', rank=0, world_size=1)
        else:
            torch.distributed.init_process_group('nccl')
        created_pg = True
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(123)
    try:
        yield
    finally:
        if world_size > 1:
            torch.distributed.barrier()
        parallel_state.destroy_model_parallel()
        from mcore_bridge.bridge import gpt_bridge
        gpt_bridge.EP_PP_GROUP = gpt_bridge.EP_PP_RANK = gpt_bridge.EP_PP_SIZE = None
        if created_pg:
            torch.distributed.destroy_process_group()


class _Lazy:

    def __init__(self, t):
        self.t = t

    def load(self):
        return self.t


def _build(mtp=1, seed=5, dtype=torch.bfloat16, tp=1, pp=1, ep=1, recompute=False, shared=False):
    torch.manual_seed(seed)
    hf_config = _tiny_qwen4exp_config()
    config = _mcore_config(hf_config, mtp=mtp, dtype=dtype, tp=tp, pp=pp, ep=ep, recompute=recompute, shared=shared)
    model = get_mcore_model(config)[0].cuda()
    return model, config


def _lm(model):
    return model.language_model if hasattr(model, 'language_model') else model


def _export(config, model):
    out = {}
    for k, v in config.bridge._convert([model], {}, '', False, 'Exporting test: '):
        out[k] = v.load() if hasattr(v, 'load') else v
    return out


def test_qwen4exp_mtp_builds_residual_linear_shared_head():
    """The MTP head uses e_proj/h_proj + joint hnorm + mixer, not a fused eh_proj/final_layernorm."""
    with _parallel_context():
        model, config = _build(mtp=1)
        lm = _lm(model)
        assert lm.mtp_process and len(lm.mtp.layers) == 1
        ml = lm.mtp.layers[0]
        assert type(ml).__name__ == 'Qwen4ExpMultiTokenPredictionLayer'
        assert ml.mhc_enabled is True  # selects the e_proj/h_proj branch
        assert ml.eh_proj is None and not hasattr(ml, 'final_layernorm') and not hasattr(ml, 'hc_head_fn')
        assert type(ml.e_proj).__name__ == 'TEColumnParallelLinear'
        assert type(ml.h_proj).__name__ == 'TEColumnParallelLinear'
        # joint multi-stream norm: hc_count * hidden_size
        assert ml.hnorm.weight.shape[0] == config.hc_count * config.hidden_size
        assert type(ml.hyper_connection_mixer).__name__ == 'Qwen4ExpTextGatedResidual'
        inner = ml.transformer_layer
        assert type(inner).__name__ == 'Qwen4ExpMTPInnerLayer'
        assert getattr(inner.self_attention, 'indexer', None) is not None  # full-attention + QSA
        assert inner.ple is None  # no PLE in the MTP block


def test_qwen4exp_mtp_bridge_roundtrip_matches_checkpoint_layout():
    """mcore -> hf export matches the real Qwen3.8-Flash-Next mtp.* layout and re-imports bit-exactly."""
    with _parallel_context():
        model_a, config = _build(mtp=1, seed=5)
        exported = _export(config, model_a)
        for key in ('mtp.fc_embedding.weight', 'mtp.fc_hidden.weight', 'mtp.pre_fc_norm_embedding.weight',
                    'mtp.pre_fc_norm_hidden.weight', 'mtp.hyper_connection_mixer.hc_norm.weight',
                    'mtp.hyper_connection_mixer.input_mix_weight_down.weight',
                    'mtp.hyper_connection_mixer.input_mix_weight_up.weight', 'mtp.layers.0.self_attn.q_proj.weight',
                    'mtp.layers.0.self_attn.indexer.index_qk_proj.weight', 'mtp.layers.0.mlp.gate.weight',
                    'mtp.layers.0.attn_hyper_connection.hc_norm.weight',
                    'mtp.layers.0.mlp_hyper_connection.hc_norm.weight'):
            assert key in exported, f'MTP export missing {key}'
        assert not any(k.startswith('mtp.') and 'eh_proj' in k for k in exported)

        model_b, _ = _build(mtp=1, seed=999)
        list(
            config.bridge._convert([model_b], {
                k: _Lazy(v)
                for k, v in exported.items()
            }, '', True, 'Reloading test: '))
        sd_a = _lm(model_a).mtp.state_dict()
        sd_b = _lm(model_b).mtp.state_dict()
        assert sd_a.keys() == sd_b.keys()
        for key in sd_a:
            torch.testing.assert_close(
                sd_b[key].cpu(), sd_a[key].cpu(), atol=0, rtol=0, msg=lambda m, k=key: f'{k}: {m}')


def test_qwen4exp_mtp_forward_backward_flows_grads_to_head():
    """The MTP loss path runs; the head and inner block get gradients (except the non-differentiable indexer)."""
    with _parallel_context():
        model, _ = _build(mtp=1)
        lm = _lm(model).train()
        b, s = 2, 16
        g = torch.Generator(device='cuda').manual_seed(23)
        input_ids = torch.randint(1, 500, (b, s), device='cuda', generator=g)
        position_ids = torch.arange(s, device='cuda').unsqueeze(0).expand(b, -1)
        attention_mask = torch.triu(torch.ones(b, 1, s, s, device='cuda', dtype=torch.bool), diagonal=1)
        labels = torch.randint(1, 500, (b, s), device='cuda', generator=g)
        loss_mask = torch.ones(b, s, device='cuda', dtype=torch.bool)
        out = lm(input_ids, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
        loss = out if out.dim() == 0 else out.float().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        # The QSA indexer's hard top-k selection is non-differentiable in the bool-mask path (the
        # backbone's full-attention indexer carries no gradient there either), so exclude it.
        nograd = [n for n, p in lm.mtp.named_parameters() if p.requires_grad and p.grad is None and 'indexer' not in n]
        assert not nograd, f'MTP params without grad: {nograd}'
        head = dict(lm.mtp.named_parameters())
        for probe in ('layers.0.e_proj.weight', 'layers.0.h_proj.weight', 'layers.0.enorm.weight',
                      'layers.0.hnorm.weight', 'layers.0.hyper_connection_mixer.input_mix_weight_down.weight'):
            assert head[probe].grad is not None and torch.isfinite(head[probe].grad.float()).all()


def test_mtp_inner_layer_kwargs_are_model_opt_in():
    from mcore_bridge.model.modules.mtp_layer import MultiTokenPredictionLayer
    layer = object.__new__(MultiTokenPredictionLayer)
    assert layer._get_inner_layer_kwargs(None, None) == {}


def test_qwen4exp_shared_mtp_recompute_restores_each_depth_ids(monkeypatch):
    """Checkpoint recompute must replay each shared MTP depth with that depth's rolled IDs."""
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
    monkeypatch.setattr(MTPLossLoggingHelper, 'tracker', {})
    with _parallel_context():
        model, config = _build(mtp=2, recompute=True, shared=True)
        lm = _lm(model).train()
        assert config.mtp_num_layers == 1 and config.mtp_unroll_steps == 2
        assert len(lm.mtp.layers) == 1

        calls = []
        layer = lm.mtp.layers[0]
        original = layer._proj_and_transformer_layer

        def record_ids(*args, **kwargs):
            calls.append((kwargs['input_ids'].detach().clone(), kwargs['position_ids'].detach().clone()))
            return original(*args, **kwargs)

        layer._proj_and_transformer_layer = record_ids
        b, s = 1, 16
        generator = torch.Generator(device='cuda').manual_seed(29)
        input_ids = torch.randint(1, 500, (b, s), device='cuda', generator=generator)
        position_ids = torch.arange(s, device='cuda').unsqueeze(0)
        attention_mask = torch.triu(torch.ones(b, 1, s, s, device='cuda', dtype=torch.bool), diagonal=1)
        labels = torch.randint(1, 500, (b, s), device='cuda', generator=generator)
        loss_mask = torch.ones(b, s, device='cuda', dtype=torch.bool)

        output = lm(input_ids, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
        loss = output if output.dim() == 0 else output.float().mean()
        loss.backward()

        assert len(calls) == 4, f'expected two forwards and two recomputes, got {len(calls)}'
        assert not torch.equal(calls[0][0], calls[1][0])
        for field in range(2):
            torch.testing.assert_close(calls[2][field], calls[1][field], atol=0, rtol=0)
            torch.testing.assert_close(calls[3][field], calls[0][field], atol=0, rtol=0)
        assert not hasattr(layer, '_mtp_input_ids')
        assert not hasattr(layer, '_mtp_position_ids')


def test_qwen4exp_mtp_pp2_forward_backward():
    """PP transports both n*H gated-HC states and the H-wide embedding needed by MTP."""
    if os.environ.get('QWEN4EXP_MTP_PARALLEL_TEST') != 'pp2':
        pytest.skip('run with QWEN4EXP_MTP_PARALLEL_TEST=pp2 torchrun --nproc-per-node=2')
    with _parallel_context(pp=2):
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        model, _ = _build(mtp=1, pp=2)
        b, s = 2, 16
        generator = torch.Generator(device='cuda').manual_seed(23)
        input_ids = torch.randint(1, 500, (b, s), device='cuda', generator=generator)
        position_ids = torch.arange(s, device='cuda').unsqueeze(0).expand(b, -1)
        attention_mask = torch.triu(torch.ones(b, 1, s, s, device='cuda', dtype=torch.bool), diagonal=1)
        labels = torch.randint(1, 500, (b, s), device='cuda', generator=generator)
        loss_mask = torch.ones(b, s, device='cuda', dtype=torch.bool)

        def forward_step(data_iterator, stage_model):
            ids, positions, mask, target, target_mask = next(data_iterator)
            output = stage_model(ids, positions, mask, labels=target, loss_mask=target_mask)

            def loss_func(tensor):
                loss = tensor if tensor.dim() == 0 else tensor.float().mean()
                return loss, {'loss': loss.detach()}

            return output, loss_func

        losses = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter([(input_ids, position_ids, attention_mask, labels, loss_mask)]),
            model=model,
            num_microbatches=1,
            seq_length=s,
            micro_batch_size=b,
            forward_only=False,
        )
        if parallel_state.is_pipeline_last_stage():
            assert len(losses) == 1 and torch.isfinite(losses[0]['loss']).all()
            lm = _lm(model)
            missing = [
                name for name, param in lm.mtp.named_parameters()
                if param.requires_grad and param.grad is None and 'indexer' not in name
            ]
            assert not missing, f'MTP params without grad: {missing}'
        else:
            assert losses == []
