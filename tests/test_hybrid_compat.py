# Copyright (c) ModelScope Contributors. All rights reserved.
"""The shared dev numerical patch must not change other models' assembly or forward/backward."""
import pytest
import torch
from test_glm5_next import _parallel_context
from transformers import AutoConfig

from mcore_bridge.config import ModelConfig
from mcore_bridge.config.parser import hf_to_mcore_config
from mcore_bridge.model.register import get_mcore_model

_TE_ATTN_ENV_VARS = ('NVTE_FLASH_ATTN', 'NVTE_FUSED_ATTN', 'NVTE_UNFUSED_ATTN')


@pytest.fixture(autouse=True)
def _isolate_te_attention_backend(monkeypatch):
    """Keep Megatron's process-wide attention backend selection local to each test."""
    for variable in _TE_ATTN_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)


def _config(name):
    if name == 'qwen3_5':
        from transformers import Qwen3_5Config
        hf = Qwen3_5Config(
            text_config={
                'vocab_size': 128,
                'hidden_size': 256,
                'intermediate_size': 512,
                'num_hidden_layers': 2,
                'num_attention_heads': 4,
                'num_key_value_heads': 2,
                'head_dim': 64,
                'linear_num_key_heads': 4,
                'linear_num_value_heads': 4,
                'linear_key_head_dim': 64,
                'linear_value_head_dim': 64,
                'linear_conv_kernel_dim': 4,
                'layer_types': ['linear_attention', 'full_attention'],
                'tie_word_embeddings': False,
            })
        values = hf_to_mcore_config(hf)
        values['language_model_only'] = True
    elif name == 'deepseek_v4':
        hf = AutoConfig.for_model(
            'deepseek_v4',
            vocab_size=128,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=32,
            qk_rope_head_dim=16,
            q_lora_rank=64,
            o_lora_rank=32,
            o_groups=2,
            moe_intermediate_size=128,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hash_layers=0,
            hc_mult=2,
            hc_sinkhorn_iters=4,
            sliding_window=32,
            compress_ratios=[0, 0],
            index_head_dim=32,
            index_n_heads=4,
            index_topk=8,
            max_position_embeddings=256,
            norm_topk_prob=True,
            scoring_func='sqrtsoftplus',
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
            tie_word_embeddings=False,
        )
        values = hf_to_mcore_config(hf)
        values['mtp_num_layers'] = None
    else:
        from mcore_bridge.config.parser import squared_relu
        values = dict(
            hf_model_type='nemotron_h',
            llm_model_type='nemotron_h',
            mcore_model_type='nemotron_h',
            num_layers=4,
            hidden_size=256,
            ffn_hidden_size=512,
            num_attention_heads=4,
            num_query_groups=2,
            padded_vocab_size=128,
            max_position_embeddings=256,
            hybrid_layer_pattern='M*-E',
            is_hybrid_model=True,
            position_embedding_type='none',
            mamba_num_heads=8,
            mamba_head_dim=64,
            mamba_num_groups=1,
            mamba_state_dim=16,
            num_moe_experts=4,
            moe_ffn_hidden_size=128,
            moe_shared_expert_intermediate_size=128,
            swiglu=False,
            gated_linear_unit=False,
            activation_func=squared_relu,
            add_qkv_bias=False,
            moe_router_load_balancing_type='none',
        )
    values.update(
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        perform_initialization=True,
        use_cpu_initialization=False,
        overlap_p2p_comm=False)
    return ModelConfig(**values)


@pytest.mark.parametrize('name', ['qwen3_5', 'deepseek_v4', 'nemotron_h'])
def test_other_model_forward_backward(name):
    from megatron.core.transformer.module import Float16Module
    from megatron.core.utils import get_attr_wrapped_model

    if name == 'deepseek_v4':
        from mcore_bridge.model.gpts.deepseek_v4 import McoreDSv4HybridSelfAttention
        if McoreDSv4HybridSelfAttention is object:
            pytest.skip('the installed release has no DSv4; that existing dev dependency is not a new requirement')
    elif name == 'nemotron_h':
        from megatron.core.ssm.mamba_mixer import HAVE_MAMBA_SSM
        if not HAVE_MAMBA_SSM:
            pytest.skip('requires the optional mamba-ssm dependency')
    with _parallel_context():
        config = _config(name)
        model = get_mcore_model(config)[0].cuda().train()
        wrapped = Float16Module(config, model)
        assert get_attr_wrapped_model(wrapped, 'get_input_tensor', return_model_obj=True) is model
        assert get_attr_wrapped_model(wrapped, 'vp_stage') == model.vp_stage
        tokens = torch.randint(1, 128, (1, 64), device='cuda')
        positions = torch.arange(64, device='cuda')[None]
        mask = torch.triu(torch.ones(1, 1, 64, 64, dtype=torch.bool, device='cuda'), diagonal=1)
        logits = wrapped(tokens, positions, mask)
        assert logits.shape == (1, 64, 128)
        assert torch.isfinite(logits).all()
        logits.float().square().mean().backward()
        grads = [param.grad for param in model.parameters() if param.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
