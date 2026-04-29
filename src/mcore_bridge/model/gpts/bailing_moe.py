# Copyright (c) ModelScope Contributors. All rights reserved.
from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model


class BailingMoeBridge(GPTBridge):
    hf_embed_key = 'model.word_embeddings.weight'
    hf_q_norm_key = 'query_layernorm.weight'
    hf_k_norm_key = 'key_layernorm.weight'
    hf_expert_bias_key = 'gate.expert_bias'


register_model(ModelMeta(
    ModelType.bailing_moe,
    ['bailing_moe'],
    bridge_cls=BailingMoeBridge,
))
