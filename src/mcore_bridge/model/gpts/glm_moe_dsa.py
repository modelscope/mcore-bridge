# Copyright (c) ModelScope Contributors. All rights reserved.
from megatron.core.transformer.experimental_attention_variant.dsa import (DSAIndexerLossAutoScaler,
                                                                          DSAIndexerLossLoggingHelper)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention as McoreDSAttention
from megatron.core.transformer.experimental_attention_variant.dsa import FusedDSAIndexerLoss, unfused_dsa_fn

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model


class DSAttention(McoreDSAttention):

    def _get_index_share_carrier(self, packed_seq_params, attention_mask):
        """Return the object that carries DSA top-k sharing state for this forward."""
        if packed_seq_params is not None and packed_seq_params.qkv_format is not None:
            return packed_seq_params
        return attention_mask if attention_mask is not None else self.config


class GlmMoeDsaLoader(ModelLoader):

    def _build_decoder_layer_specs(self):
        layer_specs = super()._build_decoder_layer_specs()
        if self.config.dsa_indexer_topk_freq > 1:
            if getattr(DSAttention, '_HOLDER_ATTR', None) is None:
                raise ImportError('glm_moe_dsa with dsa_indexer_topk_freq > 1 requires megatron-core >= 0.19: '
                                  'the shared-indexer carrier (`DSAttention._HOLDER_ATTR`) entered the release in 0.19 '
                                  'and is absent from 0.18.')
            for layer_spec in layer_specs:
                core_attn = layer_spec.submodules.self_attention.submodules.core_attention
                if hasattr(core_attn, 'module') and issubclass(core_attn.module, McoreDSAttention):
                    core_attn.module = DSAttention
        return layer_specs


register_model(ModelMeta(
    ModelType.glm_moe_dsa,
    ['glm_moe_dsa'],
    loader=GlmMoeDsaLoader,
    mtp_prefix='model.layers',
))
