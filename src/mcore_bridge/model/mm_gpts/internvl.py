# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mcore_bridge.bridge import GPTBridge, MultimodalGPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


class InternvlBridge(GPTBridge):
    hf_layers_prefix = 'language_model.model.layers'
    hf_embed_key = 'language_model.model.embed_tokens.weight'
    hf_final_layernorm_key = 'language_model.model.norm.weight'
    hf_lm_head_key = 'language_model.lm_head.weight'
    hf_score_key = 'language_model.score.weight'

    def get_hf_meta_model(self):
        model_cls = []
        class_names = ['Qwen2ForCausalLM', 'Qwen3ForCausalLM', 'Qwen3MoeForCausalLM', 'GptOssForCausalLM']
        module = importlib.import_module('transformers')
        for cls_name in class_names:
            try:
                model_cls.append(getattr(module, cls_name))
            except (ImportError, AttributeError):
                pass
        contexts = self._get_meta_model_context(model_cls)
        hf_config = self.config.hf_config
        model_cls = get_class_from_dynamic_module('modeling_internvl_chat.InternVLChatModel', hf_config.name_or_path)
        with contexts:
            model = model_cls(hf_config)
        model._auto_class = 'AutoModelForCausalLM'
        return model


class InternvlVit(HuggingFaceVit):
    module_mapping = {'vision_model': 'vision_model', 'mlp1': 'mlp1'}
    _vision_tower = ['vision_model']
    _aligner = ['mlp1']

    def prepare_attn_impl(self):
        vit_attn_impl = self.config.vit_attn_impl or 'flash_attention_2'
        if self.config.attention_backend.name == 'flash' and 'flash' in vit_attn_impl:
            use_flash_attn = True
        else:
            use_flash_attn = False
        self.hf_config.vision_config.use_flash_attn = use_flash_attn

    def prepare_model(self, hf_config: PretrainedConfig):
        llm_model_type = self.config.llm_model_type
        if llm_model_type not in ['qwen2', 'qwen3', 'qwen3_moe', 'gpt_oss']:
            raise ValueError(f'{llm_model_type} is not supported for internvl_chat model')
        InternVisionModel = get_class_from_dynamic_module('modeling_internvl_chat.InternVisionModel',
                                                          hf_config.name_or_path)
        self.model_cls = get_class_from_dynamic_module('modeling_internvl_chat.InternVLChatModel',
                                                       hf_config.name_or_path)

        self.vision_model = InternVisionModel._from_config(hf_config.vision_config)
        vit_hidden_size = hf_config.vision_config.hidden_size
        llm_hidden_size = hf_config.llm_config.hidden_size
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / hf_config.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / hf_config.downsample_ratio)**2, llm_hidden_size), nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)).to(self.vision_model.dtype)
        self.select_layer = hf_config.select_layer
        self.downsample_ratio = hf_config.downsample_ratio
        self.ps_version = hf_config.ps_version
        self.tokenizer = AutoTokenizer.from_pretrained(hf_config.name_or_path, trust_remote_code=True)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        if pixel_values is None:
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), dtype=self.vision_model.dtype, device=inputs_embeds.device)
            vit_embeds = self.extract_feature(dummy_pixel_values)
            inputs_embeds = inputs_embeds + vit_embeds.mean() * 0.
        else:
            vit_embeds = self.extract_feature(pixel_values.to(self.vision_model.dtype))
            selected = (input_ids == self.tokenizer.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1]).to(dtype=inputs_embeds.dtype)
        return inputs_embeds

    def extract_feature(self, pixel_values):
        with self.patch_hf_config():
            return self.model_cls.extract_feature(self, pixel_values)

    def pixel_shuffle(self, x, scale_factor=0.5):
        with self.patch_hf_config():
            return self.model_cls.pixel_shuffle(self, x, scale_factor=scale_factor)


register_model(
    ModelMeta(
        ModelType.internvl_chat,
        ['internvl_chat'],
        bridge_cls=InternvlBridge,
        visual_cls=InternvlVit,
    ))


class InternvlHfBridge(MultimodalGPTBridge):
    hf_state_dict_mapping = {
        'language_model.lm_head': 'lm_head',
        'language_model.model': 'model.language_model',
        'vision_tower': 'model.vision_tower',
        'multi_modal_projector': 'model.multi_modal_projector',
    }


class InternvlHfVit(HuggingFaceVit):
    module_mapping = {'model.vision_tower': 'vision_tower', 'model.multi_modal_projector': 'multi_modal_projector'}
    _vision_tower = ['vision_tower']
    _aligner = ['multi_modal_projector']

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.internvl.modeling_internvl import InternVLModel, InternVLMultiModalProjector
        self.vision_tower = AutoModel.from_config(hf_config.vision_config)
        self.multi_modal_projector = InternVLMultiModalProjector(hf_config).to(self.vision_tower.dtype)
        self.model_cls = InternVLModel
        self.dtype = self.vision_tower.dtype

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        device = self.vision_tower.device
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device)
            image_features = self.get_image_features(
                pixel_values,
                vision_feature_layer=self.hf_config.vision_feature_layer,
                vision_feature_select_strategy=self.hf_config.vision_feature_select_strategy,
            )
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            special_image_mask = input_ids == self.hf_config.image_token_id
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        else:
            dummy_pixel_values = torch.zeros((1, 3, 32, 32), device=device, dtype=self.vision_tower.dtype)
            image_features = self.get_image_features(
                dummy_pixel_values,
                vision_feature_layer=self.hf_config.vision_feature_layer,
                vision_feature_select_strategy=self.hf_config.vision_feature_select_strategy,
            )
            if hasattr(image_features, 'pooler_output'):
                image_features = image_features.pooler_output
            inputs_embeds = inputs_embeds + image_features.mean() * 0.
        return inputs_embeds

    def get_image_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_image_features(self, *args, **kwargs)

    def pixel_shuffle(self, x, scale_factor=0.5):
        with self.patch_hf_config():
            return self.model_cls.pixel_shuffle(self, x, scale_factor=scale_factor)


register_model(ModelMeta(
    ModelType.internvl,
    ['internvl'],
    bridge_cls=InternvlHfBridge,
    visual_cls=InternvlHfVit,
))
