# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from copy import deepcopy
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import PretrainedConfig
from typing import Optional

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelLoader, ModelMeta, register_model
from .utils import HuggingFaceVit


class Llama4Vit(HuggingFaceVit):
    module_mapping = {'multi_modal_projector': 'multi_modal_projector', 'vision_model': 'vision_model'}
    _vision_tower = ['vision_model']
    _aligner = ['multi_modal_projector']

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.llama4.modeling_llama4 import (Llama4ForConditionalGeneration,
                                                                Llama4MultiModalProjector, Llama4VisionModel)
        self.vision_model = Llama4VisionModel._from_config(hf_config.vision_config)
        self.multi_modal_projector = Llama4MultiModalProjector(hf_config).to(self.vision_model.dtype)
        self.model_cls = Llama4ForConditionalGeneration

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        pixel_values = kwargs.get('pixel_values')
        input_ids = kwargs.get('input_ids')
        vision_feature_select_strategy = self.hf_config.vision_config.vision_feature_select_strategy
        origin_pixel_values = pixel_values
        if pixel_values is None:
            pixel_values = torch.zeros((1, 3, 336, 336), dtype=self.vision_model.dtype, device=inputs_embeds.device)
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )
        if hasattr(image_features, 'last_hidden_state'):
            image_features = image_features.last_hidden_state
        vision_flat = image_features.view(-1, image_features.size(-1))
        projected_vision_flat = self.multi_modal_projector(vision_flat).to(inputs_embeds.device, inputs_embeds.dtype)
        if origin_pixel_values is None:
            inputs_embeds = inputs_embeds + projected_vision_flat.mean() * 0.
        else:
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=projected_vision_flat)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, projected_vision_flat)
        return inputs_embeds

    def get_placeholder_mask(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_placeholder_mask(self, *args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_image_features(self, *args, **kwargs)


class Llama4Bridge(GPTBridge):
    hf_layers_prefix = 'language_model.model.layers'
    hf_embed_key = 'language_model.model.embed_tokens.weight'
    hf_final_layernorm_key = 'language_model.model.norm.weight'
    hf_lm_head_key = 'language_model.lm_head.weight'
    hf_score_key = 'language_model.score.weight'

    hf_mlp_prefix = 'feed_forward'
    hf_gate_key = 'router.weight'


class Llama4Loader(ModelLoader):

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        layer_specs = super().get_transformer_layer_spec(vp_stage)
        for i, layer_spec in enumerate(layer_specs.layer_specs):
            global_i = i + get_transformer_layer_offset(self.config, vp_stage)
            no_rope = self.config.no_rope_freq[global_i]
            layer_spec = deepcopy(layer_spec)
            if no_rope:
                layer_spec.submodules.self_attention.submodules.q_layernorm = IdentityOp
                layer_spec.submodules.self_attention.submodules.k_layernorm = IdentityOp
                layer_specs.layer_specs[i] = layer_spec
        return layer_specs


register_model(
    ModelMeta(
        ModelType.llama4,
        ['llama4'],
        bridge_cls=Llama4Bridge,
        visual_cls=Llama4Vit,
        loader=Llama4Loader,
    ))
