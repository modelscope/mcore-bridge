# Copyright (c) ModelScope Contributors. All rights reserved.

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model


class HyV3Bridge(GPTBridge):
    hf_gate_key = 'router.gate.weight'
    hf_expert_bias_key = 'expert_bias'
    hf_shared_expert_key = 'shared_mlp'


register_model(ModelMeta(
    ModelType.hy_v3,
    ['hy_v3'],
    bridge_cls=HyV3Bridge,
))
