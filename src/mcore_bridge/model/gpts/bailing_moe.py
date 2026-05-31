# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.tuners import LoraParallelLinear

from ..constant import ModelType
from ..register import ModelMeta, register_model


class BailingMoeBridge(GPTBridge):
    hf_embed_key = 'model.word_embeddings.weight'
    hf_attn_prefix = 'attention'
    hf_q_norm_key = 'query_layernorm.weight'
    hf_k_norm_key = 'key_layernorm.weight'
    hf_expert_bias_key = 'gate.expert_bias'
    hf_o_proj_key = 'dense'
    hf_mtp_final_layernorm_key = 'final_layernorm.weight'

    def _set_qkv(self, mg_attn, hf_state_dict, to_mcore: bool, **kwargs):
        config = self.config
        num_heads = config.num_attention_heads
        num_query_groups = config.num_query_groups if config.num_query_groups is not None else num_heads
        assert num_heads % num_query_groups == 0, (
            f'num_attention_heads ({num_heads}) must be divisible by num_query_groups ({num_query_groups})')
        q_per_group = num_heads // num_query_groups
        head_dim = config.kv_channels
        hidden_size = config.hidden_size
        hidden_size_block = hidden_size // self.fp8_block_size

        def hf_to_mg(w, per_head_rows, last_dim):
            # HF [Q_all (N*r) | K_all (G*r) | V_all (G*r)] -> MG grouped interleaved (G,(qpg+2)*r)
            total_q = num_heads * per_head_rows
            total_kv = num_query_groups * per_head_rows
            q = w[:total_q].reshape(num_query_groups, q_per_group, per_head_rows, last_dim)
            k = w[total_q:total_q + total_kv].reshape(num_query_groups, 1, per_head_rows, last_dim)
            v = w[total_q + total_kv:].reshape(num_query_groups, 1, per_head_rows, last_dim)
            return torch.cat([q, k, v], dim=1).reshape(-1, last_dim)

        def mg_to_hf(w, per_head_rows, last_dim):
            # MG grouped interleaved -> HF [Q_all | K_all | V_all]
            w = w.reshape(num_query_groups, q_per_group + 2, per_head_rows, last_dim)
            q = w[:, :q_per_group, :, :].reshape(-1, last_dim)
            k = w[:, q_per_group:q_per_group + 1, :, :].reshape(-1, last_dim)
            v = w[:, q_per_group + 1:, :, :].reshape(-1, last_dim)
            return torch.cat([q, k, v], dim=0)

        if to_mcore:
            if isinstance(mg_attn.linear_qkv, LoraParallelLinear):
                # LoRA on fused QKV: lora_A is shared (input side), lora_B needs same row layout transform.
                lora_A = hf_state_dict['query_key_value.lora_A.weight'].load()
                lora_B = hf_state_dict['query_key_value.lora_B.weight'].load()
                lora_B = hf_to_mg(lora_B, head_dim, lora_B.shape[-1])
                self._set_weight(mg_attn.linear_qkv.lora_A[self._adapter_name].weight, lora_A,
                                 'linear_qkv.lora_A.weight')
                self._set_weight(mg_attn.linear_qkv.lora_B[self._adapter_name].weight, lora_B,
                                 'linear_qkv.lora_B.weight')
            elif not self._peft_format:
                qkv = hf_state_dict['query_key_value.weight'].load()
                qkv = hf_to_mg(qkv, head_dim, hidden_size)
                qkv_scale_inv = None
                if 'query_key_value.weight_scale_inv' in hf_state_dict:
                    assert head_dim % self.fp8_block_size == 0, (
                        f'head_dim ({head_dim}) must be divisible by fp8_block_size ({self.fp8_block_size})')
                    head_dim_block = head_dim // self.fp8_block_size
                    qkv_scale_inv = hf_state_dict['query_key_value.weight_scale_inv'].load()
                    qkv_scale_inv = hf_to_mg(qkv_scale_inv, head_dim_block, hidden_size_block)
                self._set_weight(mg_attn.linear_qkv.weight, qkv, 'linear_qkv.weight', hf_scale_inv=qkv_scale_inv)
        else:
            is_lora = False if mg_attn is None else (isinstance(mg_attn.linear_qkv, LoraParallelLinear)
                                                     and self._peft_format)
            is_lora = torch.tensor([is_lora], dtype=torch.bool, device='cuda')
            if self.pp_size > 1:
                dist.all_reduce(is_lora, group=self.pp_group)
            if is_lora:
                lora_A, _ = self._get_weight(
                    None if mg_attn is None else mg_attn.linear_qkv.lora_A[self._adapter_name].weight.data,
                    f'linear_qkv.lora_A.{self._adapter_name}.weight')
                lora_B, _ = self._get_weight(
                    None if mg_attn is None else mg_attn.linear_qkv.lora_B[self._adapter_name].weight.data,
                    f'linear_qkv.lora_B.{self._adapter_name}.weight')
                if lora_A is not None:
                    self._peft_target_modules.update({'query_key_value'})
                    hf_state_dict['query_key_value.lora_A.weight'] = lora_A.clone()
                    hf_state_dict['query_key_value.lora_B.weight'] = mg_to_hf(lora_B, head_dim, lora_B.shape[-1])
            elif not self._peft_format:
                mg_w, scale_inv = self._get_weight(None if mg_attn is None else mg_attn.linear_qkv.weight.data,
                                                   'linear_qkv.weight')
                if mg_w is not None:
                    hf_state_dict['query_key_value.weight'] = mg_to_hf(mg_w, head_dim, hidden_size)
                if scale_inv is not None:
                    assert head_dim % self.fp8_block_size == 0, (
                        f'head_dim ({head_dim}) must be divisible by fp8_block_size ({self.fp8_block_size})')
                    head_dim_block = head_dim // self.fp8_block_size
                    hf_state_dict['query_key_value.weight_scale_inv'] = mg_to_hf(scale_inv, head_dim_block,
                                                                                 hidden_size_block)
                del mg_w
        assert not self.config.add_bias_linear
        return hf_state_dict


register_model(ModelMeta(
    ModelType.bailing_moe,
    ['bailing_moe'],
    bridge_cls=BailingMoeBridge,
))
