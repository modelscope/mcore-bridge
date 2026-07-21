# Copyright (c) ModelScope Contributors. All rights reserved.
import megatron.core
from packaging import version
from typing import Optional

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model

try:
    from megatron.core.transformer.experimental_attention_variant.dsa import (DSAIndexerLossAutoScaler,
                                                                              DSAIndexerLossLoggingHelper)
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention as McoreDSAttention
    from megatron.core.transformer.experimental_attention_variant.dsa import FusedDSAIndexerLoss, unfused_dsa_fn
except ImportError:
    McoreDSAttention = object

mcore_019 = version.parse(megatron.core.__version__) >= version.parse('0.19.0rc0')


class DSAttention(McoreDSAttention):

    def _get_index_share_carrier(self, packed_seq_params, attention_mask):
        """Return the object that carries DSA top-k sharing state for this forward."""
        if packed_seq_params is not None and packed_seq_params.qkv_format is not None:
            return packed_seq_params
        return attention_mask if attention_mask is not None else self.config


class GlmMoeDsaLoader(ModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        transformer_layer_spec = super().get_transformer_layer_spec(vp_stage)
        if self.config.dsa_indexer_topk_freq > 1 and getattr(DSAttention, '_HOLDER_ATTR', None) is None:
            raise ImportError(
                'Please install the megatron-core main branch to support the "shared" indexer layer of `glm_moe_dsa`: '
                '`pip install git+https://github.com/NVIDIA/Megatron-LM.git`')
        if self.config.dsa_indexer_topk_freq > 1:
            for layer_spec in transformer_layer_spec.layer_specs:
                core_attn = layer_spec.submodules.self_attention.submodules.core_attention
                if hasattr(core_attn, 'module') and issubclass(core_attn.module, McoreDSAttention):
                    core_attn.module = DSAttention

        return transformer_layer_spec


register_model(ModelMeta(
    ModelType.glm_moe_dsa,
    ['glm_moe_dsa'],
    loader=GlmMoeDsaLoader,
))
