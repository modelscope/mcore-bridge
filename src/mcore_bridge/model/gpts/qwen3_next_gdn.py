# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
import torch.distributed as dist
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.transformer.attention import SelfAttention
from transformers.utils import is_torch_npu_available
from typing import Optional

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.tuners import LoraParallelLinear
from mcore_bridge.utils import get_env_args

from ..constant import ModelType
from ..modules import GatedDeltaNet, GatedSelfAttention
from ..register import ModelLoader, ModelMeta, register_model
from .qwen3_next import Qwen3NextBridge


class Qwen3NextGDNBridgeMixin(GPTBridge):
    hf_mtp_prefix = 'mtp.layers'

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        is_linear_attention = self.config.linear_attention_freq[layer_idx]
        if is_linear_attention:
            hf_state_dict.update(
                self._set_linear_attn_state(mg_attn, hf_state_dict, 'linear_attn.', layer_idx, to_mcore))

            if self.config.linear_decoupled_in_proj:
                self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key,
                                     to_mcore)
            else:
                self._set_state_dict(mg_layer, 'self_attention.in_proj.layer_norm_weight', hf_state_dict,
                                     self.hf_input_layernorm_key, to_mcore)
        else:
            hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
            self._set_state_dict(mg_layer, 'self_attention.linear_qkv.layer_norm_weight', hf_state_dict,
                                 self.hf_input_layernorm_key, to_mcore)
        return hf_state_dict

    def _convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict):
        return Qwen3NextBridge._convert_mtp_extra(self, mtp_layer, hf_state_dict, to_mcore, origin_hf_state_dict)


class Qwen3NextGDNBridge(Qwen3NextGDNBridgeMixin):

    def _set_linear_in_proj(self, mg_attn, hf_state_dict, to_mcore: bool):
        config = self.config
        num_key_heads = config.linear_num_key_heads
        key_dim = config.linear_key_head_dim
        value_dim = config.linear_value_head_dim * config.linear_num_value_heads // num_key_heads
        if to_mcore:
            if isinstance(mg_attn.in_proj, LoraParallelLinear):
                lora_A = hf_state_dict['in_proj_qkvz.lora_A.weight'].load()
                assert (lora_A == hf_state_dict['in_proj_ba.lora_A.weight'].load()).all(), \
                       'Need to ensure QKVZBA\'s lora_A are consistent'
                qkvz_lora_B = hf_state_dict['in_proj_qkvz.lora_B.weight'].load()
                ba_lora_B = hf_state_dict['in_proj_ba.lora_B.weight'].load()
                lora_B = torch.cat([
                    *(x.reshape(num_key_heads, -1, qkvz_lora_B.shape[-1]) for x in [qkvz_lora_B, ba_lora_B]),
                ],
                                   dim=1).reshape(-1, qkvz_lora_B.shape[-1])
                self._set_weight(mg_attn.in_proj.lora_A[self._adapter_name].weight, lora_A, 'in_proj.lora_A.weight')
                self._set_weight(mg_attn.in_proj.lora_B[self._adapter_name].weight, lora_B, 'in_proj.lora_B.weight')
            elif not self._peft_format:
                qkvz = hf_state_dict['in_proj_qkvz.weight'].load()
                ba = hf_state_dict['in_proj_ba.weight'].load()
                in_proj_weight = torch.cat([
                    *(x.reshape(num_key_heads, -1, config.hidden_size) for x in [qkvz, ba]),
                ],
                                           dim=1).reshape((-1, config.hidden_size))
                self._set_weight(mg_attn.in_proj.weight, in_proj_weight, 'in_proj.weight')
        else:
            qkvz_dim = key_dim * 2 + value_dim * 2
            is_lora = False if mg_attn is None else isinstance(mg_attn.in_proj,
                                                               LoraParallelLinear) and self._peft_format
            is_lora = torch.tensor([is_lora], dtype=torch.bool, device='cuda')
            if self.pp_size > 1:
                dist.all_reduce(is_lora, group=self.pp_group)
            if is_lora:
                lora_A, _ = self._get_weight(
                    None if mg_attn is None else mg_attn.in_proj.lora_A[self._adapter_name].weight.data,
                    f'in_proj.lora_A.{self._adapter_name}.weight')
                lora_B, _ = self._get_weight(
                    None if mg_attn is None else mg_attn.in_proj.lora_B[self._adapter_name].weight.data,
                    f'in_proj.lora_B.{self._adapter_name}.weight')
                if lora_A is not None:
                    lora_B = lora_B.reshape(num_key_heads, -1, lora_B.shape[-1])
                    self._peft_target_modules.update({'in_proj_qkvz', 'in_proj_ba'})
                    for key in ['in_proj_qkvz', 'in_proj_ba']:
                        hf_state_dict[f'{key}.lora_A.weight'] = lora_A.clone()
                    hf_state_dict['in_proj_qkvz.lora_B.weight'] = lora_B[:, :qkvz_dim].reshape(
                        -1, lora_B.shape[-1]).clone()
                    hf_state_dict['in_proj_ba.lora_B.weight'] = lora_B[:, qkvz_dim:].reshape(-1,
                                                                                             lora_B.shape[-1]).clone()
            elif not self._peft_format:
                in_proj_weight, _ = self._get_weight(None if mg_attn is None else mg_attn.in_proj.weight.data,
                                                     'in_proj.weight')
                if in_proj_weight is not None:
                    in_proj_weight = in_proj_weight.reshape(num_key_heads, -1, config.hidden_size)
                    hf_state_dict['in_proj_qkvz.weight'] = in_proj_weight[:, :qkvz_dim].reshape(
                        -1, config.hidden_size).clone()
                    hf_state_dict['in_proj_ba.weight'] = in_proj_weight[:,
                                                                        qkvz_dim:].reshape(-1,
                                                                                           config.hidden_size).clone()
        return hf_state_dict

    def _set_linear_decoupled_in_proj(self, mg_attn, hf_state_dict, to_mcore: bool):
        self._set_state_dict(mg_attn, 'in_proj_qkvz.weight', hf_state_dict, 'in_proj_qkvz.weight', to_mcore)
        self._set_state_dict(mg_attn, 'in_proj_ba.weight', hf_state_dict, 'in_proj_ba.weight', to_mcore)
        return hf_state_dict


class Qwen3NextLoader(ModelLoader):
    gated_delta_net = GatedDeltaNet

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import \
            get_transformer_block_with_experimental_attention_variant_spec
        layer_specs = get_transformer_block_with_experimental_attention_variant_spec(self.config, vp_stage)
        for layer_spec in layer_specs.layer_specs:
            attn_module = layer_spec.submodules.self_attention.module
            if issubclass(attn_module, SelfAttention):
                layer_spec.submodules.self_attention.module = GatedSelfAttention
            else:
                layer_spec.submodules.self_attention.module = self.gated_delta_net
                if self.config.linear_decoupled_in_proj:
                    layer_spec.submodules.input_layernorm = TENorm
                    layer_spec.submodules.self_attention.submodules.in_proj = TEColumnParallelLinear
        return layer_specs

    def build_model(
        self,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
    ):
        model = super().build_model(pre_process, post_process, vp_stage)
        lm_model = model.language_model if hasattr(model, 'language_model') else model
        for layer in lm_model.decoder.layers:
            if hasattr(layer.self_attention, 'out_norm'):
                out_norm = layer.self_attention.out_norm
                out_norm.zero_centered_gamma = False
                if not is_torch_npu_available():
                    assert hasattr(out_norm, 'zero_centered_gamma')
                if hasattr(out_norm, 'config'):
                    out_norm.config = copy.copy(out_norm.config)
                    out_norm.config.layernorm_zero_centered_gamma = False
        return model


use_mcore_gdn = get_env_args('USE_MCORE_GDN', bool, True)

if use_mcore_gdn:
    register_model(
        ModelMeta(
            ModelType.qwen3_next,
            ['qwen3_next'],
            bridge_cls=Qwen3NextGDNBridge,
            loader=Qwen3NextLoader,
        ))
