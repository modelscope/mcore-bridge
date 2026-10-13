# Copyright (c) ModelScope Contributors. All rights reserved.
"""Expert LoRA adapters must carry Megatron's expert topology metadata.

Megatron's parameter/gradient buffer classifies a parameter as an expert
parameter through ``not getattr(param, 'allreduce', True)``. The NPU grouped
LoRA adapters register one weight (and optional bias) per local expert; leaving
that attribute unset made them look like ordinary dense parameters, so their
gradients could be reduced over the dense TP/DP groups instead of the expert
groups. Dense adapters must keep the default so they are still reduced normally.
"""
import torch


def _module(is_expert, num_gemms=2, bias=True):
    from megatron.core.transformer import TransformerConfig

    from mcore_bridge.tuners.npu_lora import NpuGroupedLoraLinear
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
    )
    return NpuGroupedLoraLinear(num_gemms, 32, 64, config=config, bias=bias, is_expert=is_expert)


def test_expert_adapters_marked_as_expert_params():
    module = _module(is_expert=True)
    params = list(module.named_parameters())
    assert len(params) == 4, [name for name, _ in params]
    for name, param in params:
        assert not getattr(param, 'allreduce', True), f'{name} is not marked as an expert parameter'


def test_dense_adapters_keep_default_reduction():
    module = _module(is_expert=False)
    for name, param in module.named_parameters():
        assert getattr(param, 'allreduce', True), f'{name} must not be marked as an expert parameter'
