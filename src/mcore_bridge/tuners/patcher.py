# Copyright (c) ModelScope Contributors. All rights reserved.
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.router import TopKRouter
from peft import LoraModel
from peft.tuners.lora import Linear as LoraLinear
from peft.tuners.lora import model
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn
from typing import Optional

from mcore_bridge.utils import patch_deepcopy

from .lora import LoraParallelLinear


def dispatch_megatron(
    target: nn.Module,
    adapter_name: str,
    lora_config=None,
    **kwargs,
) -> Optional[nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    linear_cls = (TELayerNormColumnParallelLinear, TELinear, TEGroupedLinear, TopKRouter)
    if isinstance(target_base_layer, linear_cls):
        new_module = LoraParallelLinear(base_layer=target, adapter_name=adapter_name, **kwargs)

    return new_module


model.dispatch_megatron = dispatch_megatron


def _patch_lora_model():
    if hasattr(LoraModel, '_mcore_patched'):
        return

    __origin_init__ = LoraModel.__init__

    def __new_init__(self, *args, **kwargs):
        with patch_deepcopy():
            __origin_init__(self, *args, **kwargs)
        if not isinstance(self.model, MegatronModule):
            return
        for m in self.model.modules():
            if isinstance(m, LoraLinear):
                assert not isinstance(m, LoraParallelLinear)
                for p in m.parameters():
                    if p.requires_grad:
                        p.average_gradients_across_tp_domain = True

    LoraModel.__init__ = __new_init__
    LoraModel._mcore_patched = True


_patch_lora_model()
