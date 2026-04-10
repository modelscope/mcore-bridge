# Copyright (c) ModelScope Contributors. All rights reserved.
class LLMModelType:
    gpt = 'gpt'
    gpt_oss = 'gpt_oss'

    qwen3_next = 'qwen3_next'
    olmoe = 'olmoe'
    glm4 = 'glm4'
    minimax_m2 = 'minimax_m2'

    qwen3_emb = 'qwen3_emb'


class MLLMModelType:
    qwen2_vl = 'qwen2_vl'
    qwen2_5_vl = 'qwen2_5_vl'
    qwen3_vl = 'qwen3_vl'
    qwen2_5_omni = 'qwen2_5_omni'
    qwen3_omni = 'qwen3_omni'
    qwen3_5 = 'qwen3_5'
    ovis2_5 = 'ovis2_5'

    internvl_chat = 'internvl_chat'
    internvl = 'internvl'
    glm4v = 'glm4v'
    glm4v_moe = 'glm4v_moe'
    kimi_vl = 'kimi_vl'
    llama4 = 'llama4'
    gemma4 = 'gemma4'


class ModelType(LLMModelType, MLLMModelType):
    pass
