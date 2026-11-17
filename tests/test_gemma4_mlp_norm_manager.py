# Copyright (c) ModelScope Contributors. All rights reserved.
"""Gemma4TransformerLayer._forward_mlp must not keep the graph of its last forward alive (CPU only)."""
import gc
import torch
import weakref
from types import SimpleNamespace

from mcore_bridge.model.mm_gpts.gemma4 import Gemma4TransformerLayer


class _Interface:
    """Same contract as megatron's FineGrainedActivationOffloadingInterface with offloading disabled: the constructor
    keeps the tensor it is given."""

    def __init__(self, offload, tensor, name):
        self.offload, self.tensor, self.name = offload, tensor, name

    def __enter__(self):
        return self.tensor

    def __exit__(self, *args):
        pass

    def group_offload(self, tensor, forced_released_tensors=None, delay_offload=False):
        return tensor


def _bda(training, fused):

    def fn(x_with_bias, residual, prob):
        return x_with_bias[0] + residual

    return fn


def _make_layer():
    linear = torch.nn.Linear(4, 4)
    layer = SimpleNamespace(
        training=True,
        enable_moe_block=False,
        hidden_dropout=0.0,
        config=SimpleNamespace(bias_dropout_fusion=False),
        mlp_norm_manager=None,
        mlp=lambda x, padding_mask=None: (linear(x), None),
        post_feedforward_layernorm=lambda x: x,
        mlp_bda=_bda,
    )

    def pre_mlp_layernorm(hidden_states):
        # what mcore's TransformerLayer._forward_pre_mlp_layernorm does with the manager
        layer.mlp_norm_manager = _Interface(False, hidden_states, 'mlp_norm')
        with layer.mlp_norm_manager as hidden_states:
            return hidden_states * 2

    layer._forward_pre_mlp_layernorm = pre_mlp_layernorm
    return layer


def test_mlp_norm_manager_is_released():
    layer = _make_layer()
    x = torch.randn(3, 1, 4, requires_grad=True)
    out = Gemma4TransformerLayer._forward_mlp(layer, x * 1.0)
    assert layer.mlp_norm_manager is None
    out.sum().backward()
    assert layer.mlp_norm_manager is None


def test_layer_input_is_freed_after_backward():
    """The layer input (a tensor with an autograd history) must not stay referenced by the layer once the graph is
    consumed: a leftover reference on `mlp_norm_manager` kept every layer's last forward graph alive."""
    layer = _make_layer()
    x = torch.randn(3, 1, 4, requires_grad=True)
    hidden = x * 1.0  # non-leaf, like a real layer input
    ref = weakref.ref(hidden)
    out = Gemma4TransformerLayer._forward_mlp(layer, hidden)
    del hidden
    out.sum().backward()
    del out
    gc.collect()
    assert ref() is None
