# Copyright (c) ModelScope Contributors. All rights reserved.
from ..constant import ModelType
from ..register import ModelMeta, register_model
from . import glm4, minimax_m2, olmoe, qwen3_emb, qwen3_next

register_model(
    ModelMeta(
        ModelType.gpt,
        [
            'qwen2', 'llama', 'qwen3', 'qwen2_moe', 'qwen3_moe', 'internlm3', 'mimo', 'deepseek', 'deepseek_v2',
            'deepseek_v3', 'deepseek_v32', 'kimi_k2', 'dots1', 'ernie4_5', 'ernie4_5_moe', 'glm4_moe', 'glm4_moe_lite',
            'glm_moe_dsa', 'gpt_oss'
        ],
    ))
