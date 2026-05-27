# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from megatron.core.transformer.attention import SelfAttention
from torch import Tensor
from typing import Optional

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model


class BailingMoeSelfAttention(SelfAttention):

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        *args,
        **kwargs,
    ):
        """Override to handle BailingMoE's non-interleaved QKV weight layout.

        BailingMoE stores weights as [Q_all | K_all | V_all] (split by head count),
        not Megatron's interleaved [q1 q2 k1 v1 | q3 q4 k2 v2 | ...].
        """
        # [sq, b, h] --> [sq, b, (num_heads + 2 * num_kv_heads) * head_dim]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, (num_heads + 2 * num_kv_heads) * head_dim]
        # --> [sq, b, num_heads + 2 * num_kv_heads, head_dim]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_attention_heads_per_partition + 2 * self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        # Split by head count: [sq, b, num_heads, hn], [sq, b, num_kv_heads, hn], [sq, b, num_kv_heads, hn]
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.num_attention_heads_per_partition, self.num_query_groups_per_partition,
                self.num_query_groups_per_partition
            ],
            dim=2,
        )

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value


class BailingMoeLoader(ModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        transformer_layer_spec = super().get_transformer_layer_spec(vp_stage)
        for layer_spec in transformer_layer_spec.layer_specs:
            layer_spec.submodules.self_attention.module = BailingMoeSelfAttention
        return transformer_layer_spec


class BailingMoeBridge(GPTBridge):
    hf_embed_key = 'model.word_embeddings.weight'
    hf_attn_prefix = 'attention'
    hf_q_norm_key = 'query_layernorm.weight'
    hf_k_norm_key = 'key_layernorm.weight'
    hf_expert_bias_key = 'gate.expert_bias'
    hf_o_proj_key = 'dense'

    def _set_qkv(self, mg_attn, hf_state_dict, to_mcore: bool, **kwargs):
        self._set_state_dict(mg_attn, 'linear_qkv.weight', hf_state_dict, 'query_key_value.weight', to_mcore)
        assert not self.config.add_bias_linear
        return hf_state_dict


register_model(
    ModelMeta(
        ModelType.bailing_moe,
        ['bailing_moe'],
        bridge_cls=BailingMoeBridge,
        loader=BailingMoeLoader,
    ))
