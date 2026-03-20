# Copyright (c) ModelScope Contributors. All rights reserved.
from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model

register_model(
    ModelMeta(
        ModelType.gpt,
        [
            'qwen2', 'llama', 'qwen3', 'qwen2_moe', 'qwen3_moe', 'internlm3', 'mimo', 'deepseek', 'deepseek_v2',
            'deepseek_v3', 'deepseek_v32', 'kimi_k2', 'dots1', 'ernie4_5', 'ernie4_5_moe', 'glm4_moe', 'glm4_moe_lite',
            'glm_moe_dsa'
        ],
    ))


class GptOssBridge(GPTBridge):
    hf_gate_key = 'router.weight'


register_model(ModelMeta(
    ModelType.gpt_oss,
    ['gpt_oss'],
    bridge_cls=GptOssBridge,
))
