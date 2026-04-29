# Copyright (c) ModelScope Contributors. All rights reserved.
from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model


class BailingMoeBridge(GPTBridge):
    hf_embed_key = 'model.word_embeddings.weight'
    hf_attn_prefix = 'attention'
    hf_q_norm_key = 'query_layernorm.weight'
    hf_k_norm_key = 'key_layernorm.weight'
    hf_expert_bias_key = 'gate.expert_bias'
    hf_o_proj_key = 'dense'

    def _set_qkv(self, mg_attn, hf_state_dict, to_mcore: bool):
        self._set_state_dict(mg_attn, 'linear_qkv.weight', hf_state_dict, 'query_key_value.weight', to_mcore)
        return hf_state_dict


register_model(ModelMeta(
    ModelType.bailing_moe,
    ['bailing_moe'],
    bridge_cls=BailingMoeBridge,
))
