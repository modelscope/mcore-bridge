# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import torch
from contextlib import contextmanager
from megatron.core.extensions.transformer_engine import (SplitAlongDim, TEColumnParallelLinear, TENorm,
                                                         TERowParallelLinear)
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import _yarn_get_concentration_factor_from_config
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_tensor_model_parallel_rank
from megatron.core.tensor_parallel import VocabParallelEmbedding, all_gather_last_dim_from_tensor_parallel_region
from megatron.core.tensor_parallel.mappings import (gather_from_tensor_model_parallel_region,
                                                    scatter_to_tensor_model_parallel_region)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import build_module
from megatron.core.utils import make_viewless_tensor, nvtx_range_pop, nvtx_range_push
from swift.utils import get_logger
from torch import Tensor, nn
from transformers import AutoModel, PretrainedConfig
from transformers.utils.versions import require_version
from typing import Optional, Tuple

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.config import ModelConfig

from ..constant import ModelType
from ..gpt_model import GPTModel
from ..mm_gpt_model import MultimodalGPTModel
from ..modules import TransformerBlock, TransformerLayer
from ..register import ModelLoader, ModelMeta, register_model
from ..rope import get_rope_inv_freq
from .utils import HuggingFaceVit

logger = get_logger()


class Gemma4RMSNormNoScale(torch.nn.Module):
    """RMSNorm without learnable scale, mirroring HF `Gemma4RMSNorm(with_scale=False)`."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(orig_dtype)


class Gemma4Vit(HuggingFaceVit):
    module_mapping = {
        'model.vision_tower': 'vision_tower',
        'model.embed_vision': 'embed_vision',
        'model.audio_tower': 'audio_tower',
        'model.embed_audio': 'embed_audio',
    }
    _vision_tower = ['vision_tower', 'audio_tower']
    _aligner = ['embed_vision', 'embed_audio']

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model, Gemma4MultimodalEmbedder
        self.vision_tower = AutoModel.from_config(hf_config.vision_config)
        dtype = self.vision_tower.dtype
        self.audio_tower = AutoModel.from_config(hf_config.audio_config) if hf_config.audio_config is not None else None
        self.embed_vision = Gemma4MultimodalEmbedder(hf_config.vision_config, hf_config.text_config).to(dtype)
        self.embed_audio = (
            Gemma4MultimodalEmbedder(hf_config.audio_config, hf_config.text_config).to(dtype)
            if hf_config.audio_config is not None else None)
        self.prepare_language_model(hf_config)
        self.model_cls = Gemma4Model

    def prepare_language_model(self, hf_config: PretrainedConfig):
        self.register_buffer(
            'embed_scale', torch.tensor(hf_config.hidden_size**0.5).to(hf_config.torch_dtype), persistent=False)

    def get_inputs_embeds_language_model(self, inputs_embeds, **kwargs):
        inputs_embeds = inputs_embeds * self.embed_scale.to(inputs_embeds.dtype)
        return {'inputs_embeds': inputs_embeds, 'llm_input_ids': kwargs.get('input_ids')}

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        hf_config = self.hf_config
        res = self.get_inputs_embeds_language_model(inputs_embeds, **kwargs)

        pixel_values = kwargs.get('pixel_values')
        pixel_values_videos = kwargs.get('pixel_values_videos')
        input_features = kwargs.get('input_features')
        input_features_mask = kwargs.get('input_features_mask')
        image_position_ids = kwargs.get('image_position_ids')
        video_position_ids = kwargs.get('video_position_ids')

        input_ids = kwargs.get('input_ids')
        image_mask = input_ids == hf_config.image_token_id
        video_mask = input_ids == hf_config.video_token_id
        audio_mask = input_ids == hf_config.audio_token_id
        multimodal_mask = image_mask | video_mask | audio_mask
        llm_input_ids = input_ids.clone()
        llm_input_ids[multimodal_mask] = hf_config.text_config.pad_token_id

        inputs_embeds = res['inputs_embeds']

        if pixel_values is not None:
            with self.patch_hf_config():
                image_features = self.model_cls.get_image_features(
                    self, pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask_e = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_e, image_features)

        if pixel_values_videos is not None:
            with self.patch_hf_config():
                video_features = self.model_cls.get_video_features(
                    self, pixel_values_videos, video_position_ids, return_dict=True).pooler_output
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask_e = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask_e, video_features)

        if (input_features is not None and input_features_mask is not None and self.audio_tower is not None):
            with self.patch_hf_config():
                audio_output = self.model_cls.get_audio_features(
                    self, input_features, input_features_mask, return_dict=True)
            audio_features = audio_output.pooler_output
            audio_features = audio_features[audio_output.attention_mask]
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask_e = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask_e, audio_features)
        res['inputs_embeds'] = inputs_embeds
        res['llm_input_ids'] = llm_input_ids
        return res


class Gemma4SelfAttention(SelfAttention):

    def __init__(
        self,
        config: ModelConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        *args,
        **kwargs,
    ):
        text_config = config.hf_config.text_config
        layer_idx = layer_number - 1

        # Layer type / sliding attention
        self.layer_type = text_config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == 'sliding_attention'

        # Head dim: global layers may use a different head dim than sliding ones
        self.head_dim = (
            text_config.global_head_dim
            if not self.is_sliding and text_config.global_head_dim else text_config.head_dim)

        self.use_alternative_attention = (text_config.attention_k_eq_v and not self.is_sliding)
        self.num_key_value_heads = (
            text_config.num_global_key_value_heads
            if self.use_alternative_attention else text_config.num_key_value_heads)
        # Shared KV across the trailing layers
        self.num_kv_shared_layers = getattr(text_config, 'num_kv_shared_layers', 0)
        if self.num_kv_shared_layers:
            first_kv_shared_layer_idx = config.num_layers - self.num_kv_shared_layers
            self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
            prev_layers = text_config.layer_types[:first_kv_shared_layer_idx]
            self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == len(
                prev_layers) - 1 - prev_layers[::-1].index(text_config.layer_types[layer_idx])
        else:
            self.is_kv_shared_layer = False
            self.store_full_length_kv = False
        orig_kv_channels = config.kv_channels
        orig_num_query_groups = config.num_query_groups
        orig_k_layernorm = submodules.k_layernorm
        config.kv_channels = self.head_dim
        config.num_query_groups = self.num_key_value_heads
        if text_config.use_bidirectional_attention == 'vision':
            kwargs['attn_mask_type'] = AttnMaskType.arbitrary
        if self.is_kv_shared_layer:
            submodules.k_layernorm = IdentityOp
        try:
            super().__init__(config, submodules, layer_number, *args, **kwargs)
        finally:
            config.kv_channels = orig_kv_channels
            config.num_query_groups = orig_num_query_groups
            submodules.k_layernorm = orig_k_layernorm

        if self.is_kv_shared_layer or self.use_alternative_attention:
            linear_qkv_dim = self.query_projection_size
            if not self.is_kv_shared_layer:
                linear_qkv_dim += self.kv_projection_size
            self.linear_qkv = submodules.linear_qkv(
                self.config.hidden_size,
                linear_qkv_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkv',
                tp_group=self.pg_collection.tp,
            )

        if not self.is_kv_shared_layer:
            self.v_norm = Gemma4RMSNormNoScale(self.head_dim, eps=self.config.layernorm_epsilon)

    def _forward_core_attention(
        self,
        query,
        key,
        value,
        attention_mask,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        nvtx_range_push(suffix='core_attention')
        attn_mask_type = self.attn_mask_type
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix='core_attention')
        return core_attn_out

    @contextmanager
    def _patch_attention_scaling(self):
        attention_scaling = self.config.attention_scaling
        if not self.is_sliding:
            self.config.attention_scaling = self.config.full_attention_scaling
        try:
            yield
        finally:
            self.config.attention_scaling = attention_scaling

    def _apply_rotary(self, query, key, rotary_pos_emb, packed_seq_params):
        nvtx_range_push(suffix='rotary_pos_emb')
        q_pos_emb, k_pos_emb = rotary_pos_emb

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
            if packed_seq_params.cu_seqlens_kv_padded is not None:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
            else:
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        else:
            cu_seqlens_q = cu_seqlens_kv = None

        if q_pos_emb is not None:
            # TODO VIJAY: simplify
            query = apply_rotary_pos_emb(
                query,
                q_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_q,
                mscale=_yarn_get_concentration_factor_from_config(self.config),
                cp_group=self.pg_collection.cp,
            )
        if not self.is_kv_shared_layer and k_pos_emb is not None:
            key = apply_rotary_pos_emb(
                key,
                k_pos_emb,
                config=self.config,
                cu_seqlens=cu_seqlens_kv,
                mscale=_yarn_get_concentration_factor_from_config(self.config),
                cp_group=self.pg_collection.cp,
            )
        nvtx_range_pop(suffix='rotary_pos_emb')
        return query, key

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        shared_kv_states = kwargs['shared_kv_states']
        rotary_pos_emb = kwargs.get('rotary_pos_emb')
        packed_seq_params = kwargs.get('packed_seq_params')
        attention_bias = kwargs.get('attention_bias')
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        if getattr(self, 'world_size', None) is not None and self.num_key_value_heads < self.world_size:
            mixed_qkv = all_gather_last_dim_from_tensor_parallel_region(mixed_qkv)
            idx = get_tensor_model_parallel_rank() // (self.world_size // self.num_key_value_heads)
            size = mixed_qkv.size()[-1] // self.num_key_value_heads
            mixed_qkv = mixed_qkv[:, :, idx * size:(idx + 1) * size]

        thd_format = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        if self.is_kv_shared_layer:
            query = mixed_qkv
            key, value = shared_kv_states[self.layer_type]
        else:
            kv_heads_per_group = 1 if self.use_alternative_attention else 2
            num_query_heads_per_group = (self.num_attention_heads_per_partition // self.num_query_groups_per_partition)
            new_tensor_shape = mixed_qkv.size()[:-1] + (
                self.num_query_groups_per_partition,
                (num_query_heads_per_group + kv_heads_per_group) * self.hidden_size_per_attention_head,
            )
            mixed_qkv = mixed_qkv.view(*new_tensor_shape)
            split_arg_list = [num_query_heads_per_group * self.hidden_size_per_attention_head
                              ] + [self.hidden_size_per_attention_head] * kv_heads_per_group
            if SplitAlongDim is not None:
                qkv = SplitAlongDim(mixed_qkv, len(split_arg_list), split_arg_list)
            else:
                qkv = torch.split(mixed_qkv, split_arg_list, dim=3)
            if self.use_alternative_attention:
                query, key = qkv
                value = key
            else:
                query, key, value = qkv
            key = self.k_layernorm(key)
            value = self.v_norm(value)
            if thd_format:
                key = key.squeeze(1)
                value = value.squeeze(1)
        # Query [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        if getattr(self, 'world_size', None) is not None and self.num_key_value_heads < self.world_size:
            idx = get_tensor_model_parallel_rank() % (self.world_size // self.num_key_value_heads)
            size = query.shape[2] // (self.world_size // self.num_key_value_heads)
            query = query[:, :, idx * size:(idx + 1) * size, :]
        query = self.q_layernorm(query)
        if isinstance(rotary_pos_emb, torch.Tensor):
            rotary_pos_emb = (rotary_pos_emb, ) * 2
        if thd_format:
            query = query.squeeze(1)
        with self._patch_attention_scaling():
            query, key = self._apply_rotary(query, key, rotary_pos_emb, packed_seq_params)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = key, value
        core_attn_out = self._forward_core_attention(query, key, value, attention_mask, attention_bias,
                                                     packed_seq_params)

        nvtx_range_push(suffix='linear_proj')
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix='linear_proj')
        return output, bias


class Gemma4MLP(MLP):

    def __init__(
        self,
        config: ModelConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        *args,
        **kwargs,
    ):
        self.layer_number = layer_number
        text_config = config.hf_config.text_config
        first_kv_shared_layer_idx = config.num_layers - text_config.num_kv_shared_layers
        is_kv_shared_layer = layer_number > first_kv_shared_layer_idx > 0
        self.use_double_wide_mlp = text_config.use_double_wide_mlp and is_kv_shared_layer
        ffn_hidden_size = config.ffn_hidden_size
        config.ffn_hidden_size = config.ffn_hidden_size * (2 if self.use_double_wide_mlp else 1)
        try:
            super().__init__(config, submodules, *args, **kwargs)
        finally:
            config.ffn_hidden_size = ffn_hidden_size

    @classmethod
    def as_mlp_submodule(cls, *args, layer_number: int, **kwargs) -> MLP:
        pg_collection = kwargs.pop('pg_collection')
        kwargs.pop('is_mtp_layer', None)
        assert hasattr(pg_collection, 'tp'), 'TP process group is required for MLP in TransformerLayer'
        kwargs['tp_group'] = pg_collection.tp
        return cls(*args, layer_number=layer_number, **kwargs)


class Gemma4MoELayer(MoELayer):

    def __init__(self, config, *args, **kwargs):
        require_version('megatron-core>=0.16.0.dev', 'Gemma4MoELayer requires megatron-core>=0.16.0')
        super().__init__(config, *args, **kwargs)
        self.pre_feedforward_layernorm_2 = build_module(
            TENorm, hidden_size=config.hidden_size, config=config, eps=config.layernorm_epsilon)
        self.norm = Gemma4RMSNormNoScale(config.hidden_size, eps=self.config.layernorm_epsilon)
        self.scalar_root_size = config.hidden_size**-0.5
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_moe_experts))
        self.scale.sequence_parallel = config.sequence_parallel
        self.per_expert_scale.sequence_parallel = config.sequence_parallel

    def route(self, hidden_states: torch.Tensor, *args, **kwargs):
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        probs, routing_map = super().route(hidden_states)
        probs = probs * self.per_expert_scale
        return probs, routing_map

    def preprocess(self, hidden_states: torch.Tensor, *args, **kwargs):
        hidden_states = self.pre_feedforward_layernorm_2(hidden_states)
        return super().preprocess(hidden_states, *args, **kwargs)


class Gemma4Bridge(MultimodalGPTBridge):
    additional_dim0_keys = {'embed_tokens_per_layer', 'per_layer_input_gate', 'per_layer_model_projection'}
    additional_dim1_keys = {'per_layer_projection'}

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.text_config = config.hf_config.text_config
        self.hidden_size_per_layer_input = self.text_config.hidden_size_per_layer_input

    def _set_qk_layernorm(self, mg_attn, hf_state_dict, to_mcore, **kwargs):
        layer_idx = kwargs['layer_idx']
        is_kv_shared_layer = self._get_is_kv_shared_layer(layer_idx)
        self._set_state_dict(mg_attn, 'q_layernorm.weight', hf_state_dict, self.hf_q_norm_key, to_mcore)
        if not is_kv_shared_layer:
            self._set_state_dict(mg_attn, 'k_layernorm.weight', hf_state_dict, self.hf_k_norm_key, to_mcore)

    def _get_is_kv_shared_layer(self, layer_idx):
        text_config = self.text_config
        num_kv_shared_layers = getattr(text_config, 'num_kv_shared_layers', 0)
        first_kv_shared_layer_idx = self.config.num_layers - num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        return is_kv_shared_layer

    def _set_qkv(self, mg_attn, hf_state_dict, to_mcore: bool, **kwargs):
        text_config = self.text_config
        layer_idx = kwargs['layer_idx']
        is_sliding = text_config.layer_types[layer_idx] == 'sliding_attention'
        head_dim = (
            text_config.global_head_dim if not is_sliding and text_config.global_head_dim else text_config.head_dim)
        is_kv_shared_layer = self._get_is_kv_shared_layer(layer_idx)
        if is_kv_shared_layer:
            self._set_state_dict(mg_attn, 'linear_qkv.weight', hf_state_dict, 'q_proj.weight', to_mcore)
            return hf_state_dict
        else:
            kwargs['kv_channels'] = head_dim
            use_alternative_attention = text_config.attention_k_eq_v and not is_sliding
            kwargs['attention_k_eq_v'] = use_alternative_attention
            kwargs['num_query_groups'] = (
                text_config.num_global_key_value_heads
                if use_alternative_attention else text_config.num_key_value_heads)
            return super()._set_qkv(mg_attn, hf_state_dict, to_mcore, **kwargs)

    def _set_router(self, mg_mlp, hf_state_dict, to_mcore, **kwargs):
        self._set_state_dict(mg_mlp, 'router.weight', hf_state_dict, 'router.proj.weight', to_mcore)
        for key in ['per_expert_scale', 'scale']:
            self._set_state_dict(mg_mlp, key, hf_state_dict, f'router.{key}', to_mcore)

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool, is_mtp: bool = False):
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        hf_state_dict.update(self._set_mlp_state(mg_mlp, hf_state_dict, f'{self.hf_mlp_prefix}.', layer_idx, to_mcore))
        self._set_state_dict(mg_layer, 'mlp.linear_fc1.layer_norm_weight', hf_state_dict,
                             'pre_feedforward_layernorm.weight', to_mcore)
        if self.text_config.enable_moe_block:
            mg_experts = None if mg_layer is None else mg_layer.experts_mlp
            hf_state_dict.update(self._set_moe_state(mg_experts, hf_state_dict, '', layer_idx, to_mcore, is_mtp=is_mtp))
        return hf_state_dict

    def _set_layer_state(self, mg_layer, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        hf_prefix = f'{hf_prefix}{layer_idx}.'
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        hf_state_dict.update(self._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore))
        hf_state_dict.update(self._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore))
        for key in ['post_attention_layernorm', 'post_feedforward_layernorm']:
            self._set_state_dict(mg_layer, f'{key}.weight', hf_state_dict, f'{key}.weight', to_mcore)
        if self.hidden_size_per_layer_input:
            for key in ['per_layer_input_gate', 'per_layer_projection', 'post_per_layer_input_norm']:
                self._set_state_dict(mg_layer, f'{key}.weight', hf_state_dict, f'{key}.weight', to_mcore)
        self._set_state_dict(mg_layer, 'layer_scalar', hf_state_dict, 'layer_scalar', to_mcore)
        if self.text_config.enable_moe_block:
            for key in ['post_feedforward_layernorm_1', 'post_feedforward_layernorm_2']:
                self._set_state_dict(mg_layer, f'{key}.weight', hf_state_dict, f'{key}.weight', to_mcore)
            self._set_state_dict(mg_layer, 'experts_mlp.pre_feedforward_layernorm_2.weight', hf_state_dict,
                                 'pre_feedforward_layernorm_2.weight', to_mcore)
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict

    def _set_word_embeddings(self, mg_model, hf_state_dict, to_mcore):
        lm_model = getattr(mg_model, 'language_model') if self.is_multimodal else mg_model
        self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key, to_mcore)
        if self.hidden_size_per_layer_input:
            for key in ['embed_tokens_per_layer', 'per_layer_model_projection', 'per_layer_projection_norm']:
                self._set_state_dict(lm_model, f'{key}.weight', hf_state_dict, f'model.language_model.{key}.weight',
                                     to_mcore)


class Gemma4TextGPTModel(GPTModel):
    extra_forward_keys = ['mm_token_type_ids']

    def __init__(self, config, *args, **kwargs):
        # If set to "vision", pass attention_mask manually.
        text_config = config.hf_config.text_config
        if text_config.use_bidirectional_attention == 'vision':
            if config.attention_backend.name != 'unfused':
                logger.warning(
                    f'attention_backend {config.attention_backend.name} does not support use_bidirectional_attention '
                    'for vision. Setting `use_bidirectional_attention` to None. Note: This may cause computational '
                    'errors in multimodal scenarios. Please always pass pure text data.')
                text_config.use_bidirectional_attention = None
            else:
                config.window_size = None
                config.window_attn_skip_freq = None
        super().__init__(config, *args, **kwargs)
        self.num_query_groups_per_partition = self.decoder.layers[0].self_attention.num_query_groups_per_partition
        self.text_config = text_config
        self.num_kv_shared_layers = getattr(text_config, 'num_kv_shared_layers', 0)
        self.unique_layer_types = set(text_config.layer_types)
        self.hidden_size_per_layer_input = text_config.hidden_size_per_layer_input
        self.final_logit_softcapping = text_config.final_logit_softcapping
        if self.hidden_size_per_layer_input and self.pre_process:
            total_dim = self.config.num_layers * self.hidden_size_per_layer_input
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                num_embeddings=self.vocab_size,
                embedding_dim=total_dim,
                init_method=self.config.init_method,
                config=self.config,
                tp_group=self.pg_collection.tp,
            )
            self.embed_tokens_per_layer_scale = self.hidden_size_per_layer_input**0.5
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = build_module(
                TEColumnParallelLinear,
                self.config.hidden_size,
                total_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='per_layer_model_projection',
                tp_group=self.pg_collection.tp,
            )
            self.per_layer_model_projection_scale = self.config.hidden_size**-0.5
            self.per_layer_projection_norm = build_module(
                TENorm,
                hidden_size=self.hidden_size_per_layer_input,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

    def _get_rotary_pos_emb(self, decoder_input, position_ids, packed_seq_params, inference_context=None):
        rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder, decoder_input,
                                                            self.config, packed_seq_params)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        full_rotary_pos_emb = self.full_rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        rotary_pos_emb = {'sliding_attention': rotary_pos_emb, 'full_attention': full_rotary_pos_emb}
        return rotary_pos_emb, None, None

    def _set_inv_freq(self):
        rope_scaling = self.config.rope_scaling
        self.config.rope_scaling = rope_scaling['sliding_attention']
        new_inv_freq, attention_scaling = get_rope_inv_freq(self.config)
        self.config.attention_scaling = attention_scaling
        self.rotary_pos_emb.inv_freq = new_inv_freq.to(self.rotary_pos_emb.inv_freq.device)
        # full
        self.full_rotary_pos_emb = copy.copy(self.rotary_pos_emb)
        self.config.rope_scaling = rope_scaling['full_attention']
        kwargs = {'layer_type': 'full_attention'}
        if self.config.rope_scaling['rope_type'] == 'proportional':
            kwargs['head_dim_key'] = 'global_head_dim'
        new_inv_freq, attention_scaling = get_rope_inv_freq(
            self.config, text_config=self.config.hf_config.text_config, **kwargs)
        self.full_rotary_pos_emb.inv_freq = new_inv_freq
        self.config.full_attention_scaling = attention_scaling

        self.config.rope_scaling = rope_scaling

    def forward(self, *args, **kwargs):
        extra_block_kwargs = kwargs.pop('extra_block_kwargs', None) or {}
        llm_input_ids = extra_block_kwargs.pop('llm_input_ids', None)
        decoder_input = kwargs.get('decoder_input')
        mm_token_type_ids = extra_block_kwargs.pop('mm_token_type_ids', None)
        shared_kv_states = {}
        if self.hidden_size_per_layer_input:
            assert self.num_kv_shared_layers > 0, 'not support'
            if decoder_input is None:
                per_layer_inputs, shared_kv_states = self.unpack_pp_input()
            else:
                inputs_embeds = decoder_input
                per_layer_inputs = self.embed_tokens_per_layer(llm_input_ids) * self.embed_tokens_per_layer_scale
                per_layer_inputs = per_layer_inputs.reshape(*per_layer_inputs.shape[:-1], self.config.num_layers,
                                                            -1).transpose(0, 1)
                per_layer_projection = self.per_layer_model_projection(
                    inputs_embeds)[0] * self.per_layer_model_projection_scale
                per_layer_projection = gather_from_tensor_model_parallel_region(per_layer_projection)
                per_layer_projection = per_layer_projection.reshape(*per_layer_projection.shape[:-1],
                                                                    self.config.num_layers, -1)
                per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
                per_layer_inputs = (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
                per_layer_inputs = scatter_to_tensor_model_parallel_region(per_layer_inputs)
            extra_block_kwargs['per_layer_inputs'] = per_layer_inputs
        else:
            assert self.num_kv_shared_layers == 0, 'not support'
        extra_block_kwargs['shared_kv_states'] = shared_kv_states
        kwargs['extra_block_kwargs'] = extra_block_kwargs
        attention_mask = kwargs.get('attention_mask')
        attention_mask = {'sliding_attention': attention_mask, 'full_attention': attention_mask}
        if self.text_config.use_bidirectional_attention == 'vision':
            self._update_attention_mask(attention_mask, mm_token_type_ids)
        kwargs['attention_mask'] = attention_mask
        hidden_states = super().forward(*args, **kwargs)
        if self.hidden_size_per_layer_input and not self.post_process:
            hidden_states = self._pack_pp_output(hidden_states, per_layer_inputs, shared_kv_states)
        return hidden_states

    def _update_attention_mask(self, attention_mask, mm_token_type_ids):
        sliding_attention = attention_mask['sliding_attention']
        full_attention = attention_mask['full_attention']
        # sliding
        window_size = self.text_config.sliding_window - 1
        seq_len = sliding_attention.shape[-1]
        window_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=sliding_attention.device)
        window_mask = ~torch.triu(window_mask, diagonal=-window_size)
        sliding_attention = sliding_attention | window_mask
        if mm_token_type_ids is not None:
            is_vision = mm_token_type_ids > 0
            is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
            is_prev_vision[:, 0] = False
            vision_group_ids = torch.cumsum((is_vision & ~is_prev_vision).int(), dim=1) - 1
            vision_group_ids = torch.where(is_vision, vision_group_ids, torch.full_like(vision_group_ids, -1))
            q_group = vision_group_ids.unsqueeze(1).unsqueeze(-1)
            k_group = vision_group_ids.unsqueeze(1).unsqueeze(-2)
            same_vision_group = (q_group == k_group) & (q_group >= 0) & (k_group >= 0)
            sliding_attention = sliding_attention & ~same_vision_group
            full_attention = full_attention & ~same_vision_group
        attention_mask['sliding_attention'] = sliding_attention
        attention_mask['full_attention'] = full_attention

    def _pack_pp_output(self, hidden_states, per_layer_inputs, shared_kv_states):
        per_layer_inputs = per_layer_inputs.view(*hidden_states.shape[:2], -1)
        hidden_states = torch.concat([hidden_states, per_layer_inputs], dim=-1)
        flag = per_layer_inputs.new_zeros(*hidden_states.shape[:2], 2)
        if 'sliding_attention' in shared_kv_states:
            flag[0, 0, 0] = 1
            sliding_states = torch.concat(shared_kv_states['sliding_attention'], -1)
            sliding_states = sliding_states.view(*hidden_states.shape[:2], -1)
            hidden_states = torch.concat([hidden_states, sliding_states], dim=-1)
        if 'full_attention' in shared_kv_states:
            flag[0, 0, 1] = 1
            full_states = torch.concat(shared_kv_states['full_attention'], -1)
            full_states = full_states.view(*hidden_states.shape[:2], -1)
            hidden_states = torch.concat([hidden_states, full_states], dim=-1)
        hidden_states = torch.concat([hidden_states, flag], dim=-1)
        return hidden_states

    def unpack_pp_input(self):
        tp_size = self.config.tensor_model_parallel_size
        shared_kv_states = {}
        input_tensor = self.get_input_tensor()
        sequence_len = input_tensor.shape[0] * tp_size if self.config.sequence_parallel else input_tensor.shape[0]
        input_tensor, flag = input_tensor.split([input_tensor.shape[-1] - 2, 2], dim=-1)
        flag = flag.detach()
        per_layer_inputs_shape = [
            sequence_len, input_tensor.shape[1], self.config.num_layers, self.hidden_size_per_layer_input // tp_size
        ]
        full_head_dim = self.text_config.global_head_dim
        full_states_shape = per_layer_inputs_shape[:2] + [self.num_query_groups_per_partition, full_head_dim * 2]
        sliding_states_shape = full_states_shape[:3] + [self.config.kv_channels * 2]
        per_layer_inputs_dim = math.prod(per_layer_inputs_shape) // math.prod(input_tensor.shape[:2])
        full_states_dim = math.prod(full_states_shape) // math.prod(input_tensor.shape[:2])
        sliding_states_dim = math.prod(sliding_states_shape) // math.prod(input_tensor.shape[:2])
        if flag[0, 0, 1].item() != 0:
            input_tensor, full_states = input_tensor.split([input_tensor.shape[-1] - full_states_dim, full_states_dim],
                                                           dim=-1)
            full_states = full_states.reshape(*full_states_shape)
            shared_kv_states['full_attention'] = full_states.chunk(2, -1)
        if flag[0, 0, 0].item() != 0:
            input_tensor, sliding_states = input_tensor.split(
                [input_tensor.shape[-1] - sliding_states_dim, sliding_states_dim], dim=-1)
            sliding_states = sliding_states.reshape(*sliding_states_shape)
            shared_kv_states['sliding_attention'] = sliding_states.chunk(2, -1)
        input_tensor, per_layer_inputs = input_tensor.split(
            [input_tensor.shape[-1] - per_layer_inputs_dim, per_layer_inputs_dim], dim=-1)
        self.set_input_tensor(input_tensor)
        per_layer_inputs = per_layer_inputs.reshape(*per_layer_inputs_shape)
        return per_layer_inputs, shared_kv_states

    def _forward_output_layer(self, hidden_states, *args, **kwargs):
        logits, _ = self.output_layer(hidden_states, *args, **kwargs)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits


class Gemma4TransformerLayer(TransformerLayer):

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        text_config = config.hf_config.text_config
        self.layer_type = text_config.layer_types[self.layer_number - 1]
        self.enable_moe_block = text_config.enable_moe_block
        if self.enable_moe_block:
            self.experts_mlp = self._build_mlp(submodules.experts_mlp)
        hidden_size = self.config.hidden_size
        eps = self.config.layernorm_epsilon

        self.post_attention_layernorm = build_module(TENorm, hidden_size=hidden_size, config=self.config, eps=eps)
        self.post_feedforward_layernorm = build_module(TENorm, hidden_size=hidden_size, config=self.config, eps=eps)

        self.register_buffer('layer_scalar', torch.ones(1))

        self.hidden_size_per_layer_input = text_config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            from transformers.activations import ACT2FN
            self.act_fn = ACT2FN[text_config.hidden_activation]
            self.per_layer_input_gate = build_module(
                TEColumnParallelLinear,
                hidden_size,
                self.hidden_size_per_layer_input,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='per_layer_input_gate',
                tp_group=self.pg_collection.tp,
            )
            self.per_layer_projection = build_module(
                TERowParallelLinear,
                self.hidden_size_per_layer_input,
                hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='per_layer_projection',
                tp_group=self.pg_collection.tp,
            )
            self.post_per_layer_input_norm = build_module(TENorm, hidden_size=hidden_size, config=self.config, eps=eps)

        if self.enable_moe_block:
            self.post_feedforward_layernorm_1 = build_module(
                TENorm, hidden_size=hidden_size, config=self.config, eps=eps)
            self.post_feedforward_layernorm_2 = build_module(
                TENorm, hidden_size=hidden_size, config=self.config, eps=eps)

    def _forward_attention(self, hidden_states: Tensor, **kwargs):
        kwargs['rotary_pos_emb'] = kwargs['rotary_pos_emb'][self.layer_type]
        kwargs['attention_mask'] = kwargs['attention_mask'][self.layer_type]
        context = kwargs.pop('context', None)
        residual = hidden_states
        input_layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        nvtx_range_push(suffix='self_attention')
        attention_output, bias = self.self_attention(input_layernorm_output, **kwargs)
        nvtx_range_pop(suffix='self_attention')
        attention_output = self.post_attention_layernorm(attention_output)

        hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
            (attention_output, bias), residual, self.hidden_dropout)
        return hidden_states, context

    def _forward_mlp(self, hidden_states, inference_context=None, padding_mask=None):
        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self._forward_pre_mlp_layernorm(hidden_states)
        mlp_output, bias = self.mlp(pre_mlp_layernorm_output, padding_mask=padding_mask)
        if self.enable_moe_block:
            mlp_output_1 = self.post_feedforward_layernorm_1(mlp_output)
            mlp_output_2, bias = self.experts_mlp(residual, padding_mask=padding_mask)
            mlp_output_2 = self.post_feedforward_layernorm_2(mlp_output_2)

            # Combine mlp and moe outputs
            mlp_output = mlp_output_1 + mlp_output_2

        mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)((mlp_output, bias), residual,
                                                                                     self.hidden_dropout)
        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        return output

    def forward(self, hidden_states, *args, **kwargs):
        per_layer_input = kwargs.pop('per_layer_input', None)
        hidden_states, context = super().forward(hidden_states, *args, **kwargs)
        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states, _ = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states, _ = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states, context


class Gemma4GPTModel(MultimodalGPTModel):
    language_model_cls = Gemma4TextGPTModel


class Gemma4TransformerBlock(TransformerBlock):

    def _layer_forward(self, layer, hidden_states, **kwargs):
        per_layer_inputs = kwargs.pop('per_layer_inputs', None)
        if per_layer_inputs is not None:
            kwargs['per_layer_input'] = per_layer_inputs[:, :, layer.layer_number - 1]
        return super()._layer_forward(layer, hidden_states, **kwargs)


class Gemma4Loader(ModelLoader):
    model_cls = Gemma4GPTModel
    transformer_block = Gemma4TransformerBlock

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        num_moe_experts = self.config.num_moe_experts
        self.config.num_moe_experts = None
        layer_specs = get_gpt_decoder_block_spec(
            self.config, use_transformer_engine=True, normalization=self.config.normalization, vp_stage=vp_stage)
        for layer_spec in layer_specs.layer_specs:
            layer_spec.submodules.self_attention.module = Gemma4SelfAttention
            self._set_mlp_spec(layer_spec.submodules, Gemma4MLP)
        if num_moe_experts is not None:
            self.config.num_moe_experts = num_moe_experts
            moe_layer_specs = get_gpt_decoder_block_spec(
                self.config, use_transformer_engine=True, normalization=self.config.normalization, vp_stage=vp_stage)
            for layer_spec, moe_layer_spec in zip(layer_specs.layer_specs, moe_layer_specs.layer_specs):
                layer_spec.submodules.experts_mlp = moe_layer_spec.submodules.mlp
                self._set_mlp_spec(layer_spec.submodules, Gemma4MoELayer, mlp_key='experts_mlp')
        return layer_specs

    def _set_transformer_layer(self, transformer_layer_spec):
        for layer_spec in transformer_layer_spec.layer_specs:
            layer_spec.module = Gemma4TransformerLayer

    def _replace_router(self, transformer_layer_spec, mlp_key='experts_mlp'):
        super()._replace_router(transformer_layer_spec, mlp_key=mlp_key)


register_model(
    ModelMeta(
        ModelType.gemma4,
        ['gemma4'],
        bridge_cls=Gemma4Bridge,
        visual_cls=Gemma4Vit,
        loader=Gemma4Loader,
    ))


class Gemma4UnifiedVit(Gemma4Vit):
    module_mapping = {
        'model.embed_vision': 'embed_vision',
        'model.embed_audio': 'embed_audio',
    }
    _vision_tower = []

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.gemma4_unified.modeling_gemma4_unified import (Gemma4UnifiedModel,
                                                                                Gemma4UnifiedMultimodalEmbedder,
                                                                                Gemma4UnifiedVisionEmbedder)
        dtype = hf_config.torch_dtype
        self.embed_vision = (
            Gemma4UnifiedVisionEmbedder(hf_config.vision_config, hf_config.text_config).to(dtype)
            if hf_config.vision_config is not None else None)

        self.embed_audio = (
            Gemma4UnifiedMultimodalEmbedder(hf_config.audio_config, hf_config.text_config).to(dtype)
            if hf_config.audio_config is not None else None)
        self.prepare_language_model(hf_config)
        self.model_cls = Gemma4UnifiedModel


class Gemma4UnifiedBridge(Gemma4Bridge):

    def _convert_hf_state_dict(self, hf_state_dict, to_mcore):
        res = super()._convert_hf_state_dict(hf_state_dict, to_mcore)
        new_state_dict = {}
        if to_mcore:
            for k, v in res.items():
                if k.startswith('model.embed_vision.embedding_projection.'):
                    new_state_dict['model.embed_vision.multimodal_embedder.embedding_projection.'
                                   + k[len('model.embed_vision.embedding_projection.'):]] = v
                elif k.startswith('model.vision_embedder.'):
                    new_state_dict['model.embed_vision.' + k[len('model.vision_embedder.'):]] = v
                else:
                    new_state_dict[k] = v
        else:
            for k, v in res.items():
                if k.startswith('model.embed_vision.multimodal_embedder.'):
                    new_state_dict['model.embed_vision.' + k[len('model.embed_vision.multimodal_embedder.'):]] = v
                elif k.startswith('model.embed_vision.'):
                    new_state_dict['model.vision_embedder.' + k[len('model.embed_vision.'):]] = v
                else:
                    new_state_dict[k] = v
        res = new_state_dict
        return res


register_model(
    ModelMeta(
        ModelType.gemma4_unified,
        ['gemma4_unified'],
        bridge_cls=Gemma4UnifiedBridge,
        visual_cls=Gemma4UnifiedVit,
        loader=Gemma4Loader,
    ))
