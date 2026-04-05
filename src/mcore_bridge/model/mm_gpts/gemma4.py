# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
from torch import nn
from transformers import PretrainedConfig

from megatron.core.tensor_parallel import VocabParallelEmbedding
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_layer import TransformerLayer

from mcore_bridge.bridge import MultimodalGPTBridge
from mcore_bridge.model.gpt_model import GPTModel
from mcore_bridge.model.mm_gpt_model import MultimodalGPTModel
from mcore_bridge.utils import get_logger

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model
from .utils import HuggingFaceVit

logger = get_logger()


class Gemma4RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * self.weight.float()).to(input_dtype)


class Gemma4SelfAttention(SelfAttention):

    def __init__(self, config, submodules, *args, **kwargs):
        layer_number = kwargs.get('layer_number', 1)
        layer_idx = layer_number - 1
        layer_types = getattr(config, 'layer_types', None) or []
        layer_type = layer_types[layer_idx] if layer_idx < len(layer_types) else 'sliding_attention'
        is_sliding = layer_type == 'sliding_attention'

        local_config = copy.copy(config)
        local_config.kv_channels = config.kv_channels if is_sliding else (config.global_kv_channels or config.kv_channels)
        local_config.num_query_groups = (config.num_query_groups if is_sliding or config.num_global_query_groups is None
                                         else config.num_global_query_groups)
        super().__init__(local_config, submodules, *args, **kwargs)
        self.layer_type = layer_type
        self.post_self_attn_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states, *args, **kwargs):
        output, bias = super().forward(hidden_states, *args, **kwargs)
        if bias is not None:
            output = output + bias
            bias = None
        output = self.post_self_attn_layernorm(output)
        return output, bias


class Gemma4MLP(MLP):

    def __init__(self, config, submodules, *, layer_number: int, tp_group=None):
        local_config = copy.copy(config)
        first_kv_shared_layer_idx = config.num_layers - getattr(config, 'num_kv_shared_layers', 0)
        is_kv_shared_layer = layer_number - 1 >= first_kv_shared_layer_idx > 0
        if getattr(config, 'use_double_wide_mlp', False) and is_kv_shared_layer:
            local_config.ffn_hidden_size = config.ffn_hidden_size * 2
        super().__init__(local_config, submodules, tp_group=tp_group)
        self.post_mlp_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states, *args, **kwargs):
        output, bias = super().forward(hidden_states, *args, **kwargs)
        if bias is not None:
            output = output + bias
            bias = None
        output = self.post_mlp_layernorm(output)
        return output, bias


class Gemma4TransformerLayer(TransformerLayer):

    def __init__(self, config, submodules, *args, **kwargs):
        super().__init__(config, submodules, *args, **kwargs)
        self.mlp = Gemma4MLP(
            config,
            submodules.mlp.submodules,
            layer_number=self.layer_number,
            tp_group=self.pg_collection.tp,
        )
        self.hidden_size_per_layer_input = getattr(config, 'hidden_size_per_layer_input', 0) or 0
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(config.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, *args, per_layer_input=None, **kwargs):
        hidden_states, context = super().forward(*args, **kwargs)
        if per_layer_input is not None and self.hidden_size_per_layer_input:
            if per_layer_input.dim() == hidden_states.dim() + 1:
                per_layer_input = per_layer_input[..., self.layer_number - 1, :]
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = torch.nn.functional.gelu(hidden_states, approximate='tanh')
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states
        return hidden_states, context


class Gemma4GPTModel(GPTModel):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.hidden_size_per_layer_input = getattr(config, 'hidden_size_per_layer_input', 0) or 0
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                config.vocab_size_per_layer_input,
                config.num_layers * self.hidden_size_per_layer_input,
                init_method=config.init_method,
                reduce_scatter_embeddings=False,
                config=config,
                tp_group=self.pg_collection.tp,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_layers * self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(self.hidden_size_per_layer_input, eps=config.layernorm_epsilon)
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection_scale = config.hidden_size**-0.5

    def get_per_layer_inputs(self, input_ids: torch.Tensor):
        per_layer_inputs = self.embed_tokens_per_layer(input_ids)
        return per_layer_inputs.reshape(*input_ids.shape, self.config.num_layers, self.hidden_size_per_layer_input)

    def project_per_layer_inputs(self, inputs_embeds: torch.Tensor, per_layer_inputs: torch.Tensor | None = None):
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs: dict = None,
        runtime_gather_output=None,
        *,
        inference_params=None,
        loss_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        extra_block_kwargs = dict(extra_block_kwargs or {})
        if self.hidden_size_per_layer_input and decoder_input is not None:
            per_layer_inputs = self.get_per_layer_inputs(input_ids)
            extra_block_kwargs['per_layer_input'] = self.project_per_layer_inputs(decoder_input, per_layer_inputs)
        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
            **kwargs,
        )


class Gemma4MultimodalGPTModel(MultimodalGPTModel):

    def __init__(self, config, transformer_layer_spec, pre_process=True, post_process=True, *_args, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.language_model = Gemma4GPTModel(config, transformer_layer_spec, pre_process, post_process, *_args, **kwargs)
        self.vp_stage = self.language_model.vp_stage
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights
        self.model_meta = config.model_meta
        self.visual = None
        if pre_process and self.model_meta.visual_cls is not None:
            self.visual = self.model_meta.visual_cls(config)


class Gemma4Bridge(MultimodalGPTBridge):

    @staticmethod
    def _is_sliding_layer(config, layer_idx: int) -> bool:
        layer_types = getattr(config, 'layer_types', None) or []
        return layer_idx >= len(layer_types) or layer_types[layer_idx] == 'sliding_attention'

    def _patch_layer_attention_config(self, layer_idx: int):
        class _Ctx:

            def __init__(self, bridge):
                self.bridge = bridge
                self.orig_kv_channels = bridge.config.kv_channels
                self.orig_num_query_groups = bridge.config.num_query_groups

            def __enter__(self):
                if not Gemma4Bridge._is_sliding_layer(self.bridge.config, layer_idx):
                    self.bridge.config.kv_channels = self.bridge.config.global_kv_channels or self.bridge.config.kv_channels
                    if self.bridge.config.num_global_query_groups is not None:
                        self.bridge.config.num_query_groups = self.bridge.config.num_global_query_groups

            def __exit__(self, exc_type, exc, tb):
                self.bridge.config.kv_channels = self.orig_kv_channels
                self.bridge.config.num_query_groups = self.orig_num_query_groups

        return _Ctx(self)

    def _get_tp_split_dim(self, mg_key):
        if mg_key == 'embed_tokens_per_layer.weight':
            return 0
        return super()._get_tp_split_dim(mg_key)

    def _set_attn_state(self, mg_attn, hf_state_dict, hf_prefix: str, layer_idx: int, to_mcore: bool):
        with self._patch_layer_attention_config(layer_idx):
            return super()._set_attn_state(mg_attn, hf_state_dict, hf_prefix, layer_idx, to_mcore)

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        hf_state_dict = super()._set_layer_attn(mg_layer, hf_state_dict, layer_idx, to_mcore)
        self._set_state_dict(mg_layer, 'self_attention.post_self_attn_layernorm.weight', hf_state_dict,
                             'post_attention_layernorm.weight', to_mcore)
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        hf_state_dict = super()._set_layer_mlp(mg_layer, hf_state_dict, layer_idx, to_mcore)
        self._set_state_dict(mg_layer, 'mlp.post_mlp_layernorm.weight', hf_state_dict,
                             'post_feedforward_layernorm.weight', to_mcore)
        if getattr(self.config, 'hidden_size_per_layer_input', 0):
            self._set_state_dict(mg_layer, 'per_layer_input_gate.weight', hf_state_dict, 'per_layer_input_gate.weight',
                                 to_mcore)
            self._set_state_dict(mg_layer, 'per_layer_projection.weight', hf_state_dict, 'per_layer_projection.weight',
                                 to_mcore)
            self._set_state_dict(mg_layer, 'post_per_layer_input_norm.weight', hf_state_dict,
                                 'post_per_layer_input_norm.weight', to_mcore)
        return hf_state_dict

    def _convert_pre_process(self, mg_model, hf_state_dict, hf_prefix: str, to_mcore):
        if to_mcore:
            hf_state_dict = self._remove_prefix(hf_state_dict, hf_prefix)
        else:
            hf_state_dict = {}
        lm_model = getattr(mg_model, 'language_model') if self.is_multimodal else mg_model
        self._set_state_dict(lm_model, 'embedding.word_embeddings.weight', hf_state_dict, self.hf_embed_key, to_mcore)
        if getattr(self.config, 'hidden_size_per_layer_input', 0):
            self._set_state_dict(lm_model, 'embed_tokens_per_layer.weight', hf_state_dict,
                                 'model.language_model.embed_tokens_per_layer.weight', to_mcore)
            self._set_state_dict(lm_model, 'per_layer_model_projection.weight', hf_state_dict,
                                 'model.language_model.per_layer_model_projection.weight', to_mcore)
            self._set_state_dict(lm_model, 'per_layer_projection_norm.weight', hf_state_dict,
                                 'model.language_model.per_layer_projection_norm.weight', to_mcore)
        if self.is_multimodal:
            for prefix, mg_prefix in self.module_mapping.items():
                mg_module = getattr(mg_model.visual, mg_prefix)
                hf_state_dict.update(self._set_module(mg_module, hf_state_dict, f'{hf_prefix}{prefix}.', to_mcore))
        if to_mcore:
            hf_state_dict = {}
        else:
            hf_state_dict = self._add_prefix(hf_state_dict, hf_prefix)
        return hf_state_dict


class Gemma4Loader(ModelLoader):
    model_cls = Gemma4MultimodalGPTModel

    def get_transformer_layer_spec(self, vp_stage=None):
        self.config.qk_layernorm = True
        layer_spec = self._get_transformer_layer_spec()
        layer_spec.module = Gemma4TransformerLayer
        layer_spec.submodules.self_attention.module = Gemma4SelfAttention
        return layer_spec


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
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4AudioModel,
            Gemma4Model,
            Gemma4MultimodalEmbedder,
            Gemma4VisionModel,
        )

        self.vision_tower = Gemma4VisionModel._from_config(hf_config.vision_config)
        self.embed_vision = Gemma4MultimodalEmbedder(hf_config.vision_config, hf_config.text_config).to(
            self.vision_tower.dtype)
        self.audio_tower = None
        self.embed_audio = None
        if hf_config.audio_config is not None:
            self.audio_tower = Gemma4AudioModel._from_config(hf_config.audio_config)
            self.embed_audio = Gemma4MultimodalEmbedder(hf_config.audio_config, hf_config.text_config).to(
                self.audio_tower.dtype)
        self.model_cls = Gemma4Model

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        pixel_values_videos = kwargs.get('pixel_values_videos')
        input_features = kwargs.get('input_features')
        input_features_mask = kwargs.get('input_features_mask')

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids=input_ids, inputs_embeds=inputs_embeds)

        if pixel_values is None and pixel_values_videos is None and input_features is None:
            dummy = self._get_dummy_dependency(inputs_embeds)
            return inputs_embeds + dummy * 0.

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_position_ids=kwargs.get('image_position_ids'),
                return_dict=True,
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos=pixel_values_videos,
                video_position_ids=kwargs.get('video_position_ids'),
                return_dict=True,
            ).pooler_output
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

        if input_features is not None and input_features_mask is not None:
            audio_output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
            audio_features = audio_output.pooler_output
            audio_mask_from_encoder = audio_output.attention_mask
            audio_features = audio_features[audio_mask_from_encoder]
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        return inputs_embeds

    def _get_dummy_dependency(self, inputs_embeds):
        deps = []
        for module_name in ('vision_tower', 'embed_vision', 'audio_tower', 'embed_audio'):
            module = getattr(self, module_name, None)
            if module is None:
                continue
            try:
                deps.append(next(module.parameters()).mean())
            except StopIteration:
                continue
        if not deps:
            return inputs_embeds.new_zeros(())
        return sum(dep.to(inputs_embeds.device, inputs_embeds.dtype) for dep in deps)

    def get_placeholder_mask(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_placeholder_mask(self, *args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_image_features(self, *args, **kwargs)

    def get_video_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_video_features(self, *args, **kwargs)

    def get_audio_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_audio_features(self, *args, **kwargs)


register_model(
    ModelMeta(
        ModelType.gemma4,
        ['gemma4'],
        bridge_cls=Gemma4Bridge,
        visual_cls=Gemma4Vit,
        loader=Gemma4Loader,
    ))
