# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
from contextlib import contextmanager
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import _yarn_get_concentration_factor_from_config
from megatron.core.tensor_parallel.mappings import (gather_from_tensor_model_parallel_region,
                                                    scatter_to_tensor_model_parallel_region)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_range_pop, nvtx_range_push
from torch import Tensor, nn
from typing import Optional, Tuple

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model
from .bailing_moe import BailingMoeBridge

try:
    from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
except ImportError:
    fused_recurrent_simple_gla = None


class BailingHybridBridge(BailingMoeBridge):
    additional_dim0_keys = {'g_proj'}

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        layer_type = self.config.hf_config.attention_layer_type[layer_idx]
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        if layer_type == 'attention':
            hf_state_dict.update(
                self._set_mla_attn_state(mg_attn, hf_state_dict, f'{self.hf_attn_prefix}.', layer_idx, to_mcore))

        elif layer_type == 'linear_attention':
            hf_state_dict.update(
                self._set_attn_state(mg_attn, hf_state_dict, f'{self.hf_attn_prefix}.', layer_idx, to_mcore))
            for key in ['g_proj', 'g_norm']:
                self._set_state_dict(mg_layer, f'self_attention.{key}.weight', hf_state_dict, f'attention.{key}.weight',
                                     to_mcore)
        self._set_state_dict(mg_layer, 'input_layernorm.weight', hf_state_dict, self.hf_input_layernorm_key, to_mcore)
        return hf_state_dict


class BailingMoeV2_5GroupRMSNorm(nn.Module):

    def __init__(self, config, hidden_size, group_norm_size, eps=1e-6):
        super().__init__()
        self.config = config
        assert hidden_size % group_norm_size == 0, 'hidden_size must be divisible by group_norm_size'
        self.hidden_size = hidden_size
        self.group_norm_size = group_norm_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()
        group_input_shape = input_shape[:-1] + (self.group_norm_size, input_shape[-1] // self.group_norm_size)
        hidden_states = hidden_states.view(group_input_shape)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype).view(input_shape)


class LinearAttention(SelfAttention):

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        if fused_recurrent_simple_gla is None:
            raise ImportError('flash-linear-attention is required but not installed. '
                              'Please install it via: '
                              "`pip install -U 'flash-linear-attention' --no-build-isolation`")
        super().__init__(config, *args, **kwargs)
        self.g_proj = TEColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.query_projection_size,
            bias=False,
            skip_bias_add=False,
            init_method=config.init_method,
            skip_weight_param_allocation=False,
            gather_output=False,
            is_expert=False,
            config=config,
        )
        self.g_norm = BailingMoeV2_5GroupRMSNorm(
            config,
            self.query_projection_size,
            group_norm_size=config.hf_config.group_norm_size,
            eps=config.layernorm_epsilon)
        self.g_norm.weight.average_gradients_across_tp_domain = True  # No need to set `sequence_parallel`.
        # https://github.com/sgl-project/sglang/blob/8e0ed75f2d5417015329095dc9a1626df2895acf/python/sglang/srt/layers/attention/linear/lightning_backend.py#L144C12-L149  # noqa
        slope = -self.build_slope_tensor(config.num_attention_heads) * (1 - (self.layer_number - 1) /
                                                                        (config.num_layers - 1) + 1e-5)
        # Slice slope to current TP rank: each rank only owns `num_attention_heads_per_partition` heads.
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        heads_per_partition = self.num_attention_heads_per_partition
        slope = slope[tp_rank * heads_per_partition:(tp_rank + 1) * heads_per_partition].contiguous()
        self.register_buffer('slope', slope, persistent=False)

    @staticmethod
    def build_slope_tensor(n_attention_heads: int):
        """
        Build a tensor of slopes for Lightning Attention-2 as described in the paper:
        "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models"
        (https://arxiv.org/abs/2401.04658)
        This function computes the slope values that control the decay rate of attention scores
        based on the number of attention heads. The slopes are designed to have specific
        mathematical properties that work optimally when the number of heads is a power of 2.
        For non-power-of-2 head counts, a workaround is implemented to maintain similar properties.
        Args:
            n_attention_heads (int): Number of attention heads in the model
        Returns:
            torch.Tensor: A tensor of shape [n_attention_heads] containing the computed slopes
        Note:
            Code copied from: https://github.com/OpenNLPLab/lightning-attention/blob/d15c38529bbd5c2c82b44ddda3cac885825aa873/lightning_attn/utils/utils.py#L6  # noqa
        """

        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n)  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(
                    math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.
                return (get_slopes_power_of_2(closest_power_of_2)
                        + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float)
        return slopes

    @contextmanager
    def _patch_attention_scaling(self):
        multi_latent_attention = self.config.multi_latent_attention
        self.config.multi_latent_attention = False
        try:
            yield
        finally:
            self.config.multi_latent_attention = multi_latent_attention

    def _apply_rotary(self, query, key, rotary_pos_emb, cu_seqlens=None):
        if cu_seqlens is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
        nvtx_range_push(suffix='rotary_pos_emb')
        q_pos_emb, k_pos_emb = rotary_pos_emb

        if q_pos_emb is not None:
            # TODO VIJAY: simplify
            query = apply_rotary_pos_emb(
                query,
                q_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens,
                mscale=_yarn_get_concentration_factor_from_config(self.config),
                cp_group=self.pg_collection.cp,
            )
        if k_pos_emb is not None:
            key = apply_rotary_pos_emb(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens,
                mscale=_yarn_get_concentration_factor_from_config(self.config),
                cp_group=self.pg_collection.cp,
            )
        nvtx_range_pop(suffix='rotary_pos_emb')
        if cu_seqlens is not None:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
        return query, key

    def _forward_core_attention(self, query, key, value, attention_mask, cu_seqlens=None):
        nvtx_range_push(suffix='core_attention')
        query = query.transpose(0, 1)
        core_attn_out, _ = fused_recurrent_simple_gla(
            q=query,
            k=key.transpose(0, 1),
            v=value.transpose(0, 1),
            g=self.slope[None, None, :].expand(*query.shape[:2], self.num_attention_heads_per_partition),
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        nvtx_range_pop(suffix='core_attention')
        core_attn_out = core_attn_out.view(*core_attn_out.shape[:2], -1)
        return core_attn_out.transpose(0, 1)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        rotary_pos_emb = kwargs.get('rotary_pos_emb')
        packed_seq_params = kwargs.get('packed_seq_params')
        query, key, value = self.get_query_key_value_tensors(hidden_states)
        if isinstance(rotary_pos_emb, torch.Tensor):
            rotary_pos_emb = (rotary_pos_emb, ) * 2
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
        else:
            cu_seqlens_q = None
        with self._patch_attention_scaling():
            query, key = self._apply_rotary(query, key, rotary_pos_emb, cu_seqlens_q)
        core_attn_out = self._forward_core_attention(query, key, value, attention_mask, cu_seqlens_q)
        enable_tp = self.config.tensor_model_parallel_size > 1
        if enable_tp:
            core_attn_out = gather_from_tensor_model_parallel_region(core_attn_out)
        core_attn_out = self.g_norm(core_attn_out)
        if enable_tp:
            core_attn_out = scatter_to_tensor_model_parallel_region(core_attn_out)
        g_proj = self.g_proj(hidden_states)[0]
        core_attn_out = core_attn_out * torch.sigmoid_(g_proj)
        nvtx_range_push(suffix='linear_proj')
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix='linear_proj')
        return output, bias


class BailingHybridLoader(ModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        hf_config = self.config.hf_config
        num_layers = hf_config.num_hidden_layers
        group_size = hf_config.layer_group_size
        tail_start = num_layers // group_size * group_size
        hf_config.attention_layer_type = [
            'attention' if (layer_idx + 1) % group_size == 0 or layer_idx >= tail_start else 'linear_attention'
            for layer_idx in range(num_layers)
        ]
        layer_specs = super().get_transformer_layer_spec(vp_stage=vp_stage)
        multi_latent_attention = self.config.multi_latent_attention
        self.config.multi_latent_attention = False
        linear_layer_specs = super().get_transformer_layer_spec(vp_stage=vp_stage)
        self.config.multi_latent_attention = multi_latent_attention
        for i, layer_spec in enumerate(layer_specs.layer_specs):
            if hf_config.attention_layer_type[i] == 'linear_attention':
                linear_spec = linear_layer_specs.layer_specs[i].submodules.self_attention
                linear_spec.module = LinearAttention
                linear_spec.submodules.linear_qkv = TEColumnParallelLinear
                layer_spec.submodules.self_attention = linear_spec
        return layer_specs


register_model(
    ModelMeta(
        ModelType.bailing_hybrid,
        ['bailing_hybrid'],
        bridge_cls=BailingHybridBridge,
        loader=BailingHybridLoader,
    ))
