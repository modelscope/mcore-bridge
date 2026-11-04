# Copyright (c) ModelScope Contributors. All rights reserved.
from mcore_bridge.utils import get_env_args

from ..constant import ModelType
from ..gpts.qwen3_next_gdn import Qwen3NextGDNBridgeMixin, Qwen3NextLoader
from ..register import ModelMeta, register_model
from .qwen3_5 import Qwen3_5Vit


class Qwen3_5Bridge(Qwen3NextGDNBridgeMixin):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.language_model.norm.weight'


use_mcore_gdn = get_env_args('USE_MCORE_GDN', bool, True)

if use_mcore_gdn:
    register_model(
        ModelMeta(
            ModelType.qwen3_5,
            ['qwen3_5', 'qwen3_5_moe'],
            bridge_cls=Qwen3_5Bridge,
            visual_cls=Qwen3_5Vit,
            loader=Qwen3NextLoader,
            mtp_prefix='mtp',
            mtp_depth_field='mtp_num_hidden_layers',
        ))
