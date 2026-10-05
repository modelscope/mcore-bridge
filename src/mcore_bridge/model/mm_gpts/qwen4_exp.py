# Copyright (c) ModelScope Contributors. All rights reserved.
from mcore_bridge.utils import get_env_args

from ..constant import ModelType
from ..gpts.qwen4_exp import Qwen4ExpBridge, Qwen4ExpLoader, Qwen4ExpTransformerBlock
from ..register import ModelMeta, register_model
from .qwen3_5 import Qwen3_5Vit


class Qwen4ExpMMBridge(Qwen4ExpBridge):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_mixer_prefix = 'model.language_model.'


class Qwen4ExpMMLoader(Qwen4ExpLoader):
    transformer_block = Qwen4ExpTransformerBlock


use_mcore_gdn = get_env_args('USE_MCORE_GDN', bool, True)

if use_mcore_gdn:
    register_model(
        ModelMeta(
            ModelType.qwen4_exp,
            ['qwen4_exp'],
            bridge_cls=Qwen4ExpMMBridge,
            visual_cls=Qwen3_5Vit,
            loader=Qwen4ExpMMLoader,
        ))
