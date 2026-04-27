# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from transformers import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mcore_bridge.bridge import MultimodalGPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


class KimiVLBridge(MultimodalGPTBridge):
    hf_layers_prefix = 'language_model.model.layers'
    hf_embed_key = 'language_model.model.embed_tokens.weight'
    hf_final_layernorm_key = 'language_model.model.norm.weight'
    hf_lm_head_key = 'language_model.lm_head.weight'
    hf_score_key = 'language_model.score.weight'


class KimiVLVit(HuggingFaceVit):
    module_mapping = {'vision_tower': 'vision_tower', 'multi_modal_projector': 'multi_modal_projector'}
    _vision_tower = ['vision_tower']
    _aligner = ['multi_modal_projector']

    def prepare_model(self, hf_config: PretrainedConfig):
        MoonVitPretrainedModel = get_class_from_dynamic_module('modeling_kimi_vl.MoonVitPretrainedModel',
                                                               hf_config.name_or_path)
        KimiVLMultiModalProjector = get_class_from_dynamic_module('modeling_kimi_vl.KimiVLMultiModalProjector',
                                                                  hf_config.name_or_path)
        self.vision_tower = MoonVitPretrainedModel._from_config(hf_config.vision_config)
        self.multi_modal_projector = KimiVLMultiModalProjector(hf_config).to(self.vision_tower.dtype)
        self.model_cls = get_class_from_dynamic_module('modeling_kimi_vl.KimiVLForConditionalGeneration',
                                                       hf_config.name_or_path)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        vision_config = self.hf_config.vision_config
        if pixel_values is not None and pixel_values.size(0) > 0:
            pixel_values = pixel_values.to(self.vision_tower.dtype)
            image_features: torch.Tensor = self._extract_image_features(pixel_values, kwargs['image_grid_hws'])
            inputs_embeds = inputs_embeds.to(image_features[0].dtype).clone()
            inputs_embeds = self._merge_with_image_features(inputs_embeds, input_ids, image_features)
        else:
            pixel_values = torch.zeros((16, 3, vision_config.patch_size, vision_config.patch_size),
                                       dtype=self.vision_tower.dtype,
                                       device=input_ids.device)
            image_grid_hws = input_ids.new_tensor([[4, 4]])
            image_features: torch.Tensor = self._extract_image_features(pixel_values, image_grid_hws)
            inputs_embeds = inputs_embeds + image_features.mean() * 0.
        return inputs_embeds

    def _extract_image_features(self, pixel_values, image_grid_hws):
        with self.patch_hf_config():
            return self.model_cls._extract_image_features(self, pixel_values, image_grid_hws)

    def _merge_with_image_features(self, inputs_embeds, input_ids, image_features):
        with self.patch_hf_config():
            return self.model_cls._merge_with_image_features(self, inputs_embeds, input_ids, image_features)


register_model(ModelMeta(
    ModelType.kimi_vl,
    ['kimi_vl'],
    bridge_cls=KimiVLBridge,
    visual_cls=KimiVLVit,
))


class KimiK25Vit(HuggingFaceVit):
    module_mapping = {'vision_tower': 'vision_tower', 'mm_projector': 'mm_projector'}
    _vision_tower = ['vision_tower']
    _aligner = ['mm_projector']
    support_multimodal = False

    def prepare_model(self, hf_config: PretrainedConfig):
        output = []
        for key in ['MoonViT3dPretrainedModel', 'PatchMergerMLP', 'VisionTowerConfig', 'ProjectorConfig']:
            output.append(get_class_from_dynamic_module(f'modeling_kimi_k25.{key}', hf_config.name_or_path))
        MoonViT3dPretrainedModel, PatchMergerMLP, VisionTowerConfig, ProjectorConfig = output
        assert hf_config.vision_config.mm_projector_type == 'patchmerger'
        vit_config = VisionTowerConfig(hf_config.vision_config)
        proj_config = ProjectorConfig(hf_config.vision_config)
        self.vision_tower = MoonViT3dPretrainedModel._from_config(vit_config)
        self.mm_projector = PatchMergerMLP(proj_config).to(self.vision_tower.dtype)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        pixel_values = kwargs.pop('pixel_values', None)
        if pixel_values is not None:
            raise NotImplementedError('Kimi-K25 currently only supports plain text training.')
        return inputs_embeds


register_model(ModelMeta(
    ModelType.kimi_k25,
    ['kimi_k25'],
    bridge_cls=KimiVLBridge,
    visual_cls=KimiK25Vit,
))
