# Copyright (c) ModelScope Contributors. All rights reserved.
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import (gather_from_tensor_model_parallel_region,
                                                    scatter_to_tensor_model_parallel_region)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from typing import Optional

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.config import ModelConfig

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model


class OLMoESelfAttention(SelfAttention):

    def __init__(
        self,
        config: ModelConfig,
        submodules: SelfAttentionSubmodules,
        *args,
        **kwargs,
    ):
        q_layernorm = submodules.q_layernorm
        k_layernorm = submodules.k_layernorm
        submodules.q_layernorm = IdentityOp
        submodules.k_layernorm = IdentityOp
        try:
            super().__init__(config, submodules, *args, **kwargs)
        finally:
            submodules.q_layernorm = q_layernorm
            submodules.k_layernorm = k_layernorm
        # OLMoE applies q/k RMSNorm over the full Q/K channels (not per-head).
        self.q_norm = build_module(
            submodules.q_layernorm,
            hidden_size=self.hidden_size_per_attention_head * config.num_attention_heads,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.k_norm = build_module(
            submodules.k_layernorm,
            hidden_size=self.hidden_size_per_attention_head * config.num_query_groups,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        # These norms operate on the gathered (TP-replicated) tensor, so they must
        # NOT participate in the cross-TP SP/QK-layernorm grad AllReduce,
        # otherwise the gradients will be amplified by tp_size.
        for norm in (self.q_norm, self.k_norm):
            for p in norm.parameters():
                p.sequence_parallel = False
                p.average_gradients_across_tp_domain = True

    def get_query_key_value_tensors(self, *_args, **kwargs):
        enable_tp = mpu.get_tensor_model_parallel_world_size() > 1
        query, key, value = super().get_query_key_value_tensors(*_args, **kwargs)
        query = query.reshape(*query.shape[:-2], -1)
        key = key.reshape(*key.shape[:-2], -1)
        if enable_tp:
            query = gather_from_tensor_model_parallel_region(query)
            key = gather_from_tensor_model_parallel_region(key)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if enable_tp:
            query = scatter_to_tensor_model_parallel_region(query)
            key = scatter_to_tensor_model_parallel_region(key)
        query = query.view(*query.shape[:2], -1, self.hidden_size_per_attention_head)
        key = key.view(*key.shape[:2], -1, self.hidden_size_per_attention_head)
        return query, key, value


class OLMoEBridge(GPTBridge):

    def _set_qk_layernorm(self, mg_attn, hf_state_dict, to_mcore, **kwargs):
        self._set_state_dict(mg_attn, 'q_norm.weight', hf_state_dict, 'q_norm.weight', to_mcore)
        self._set_state_dict(mg_attn, 'k_norm.weight', hf_state_dict, 'k_norm.weight', to_mcore)


class OLMoELoader(ModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        transformer_layer_spec = super().get_transformer_layer_spec(vp_stage)
        for layer_spec in transformer_layer_spec.layer_specs:
            layer_spec.submodules.self_attention.module = OLMoESelfAttention
        return transformer_layer_spec


register_model(ModelMeta(
    ModelType.olmoe,
    ['olmoe'],
    bridge_cls=OLMoEBridge,
    loader=OLMoELoader,
))
