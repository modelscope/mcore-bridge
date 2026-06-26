# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from transformers import PretrainedConfig

from ..constant import ModelType
from ..gpts.qwen3_next_gdn import Qwen3NextGDNBridgeMixin, Qwen3NextLoader
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


class MiniCPMV46Vit(HuggingFaceVit):
    module_mapping = {
        'model.vision_tower': 'vision_tower',
        'model.merger': 'merger',
    }
    _vision_tower = ['vision_tower']
    _aligner = ['merger']

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
            MiniCPMV4_6VisionModel,
            MiniCPMV4_6Merger,
            MiniCPMV4_6Model
        )
        self.vision_tower = MiniCPMV4_6VisionModel._from_config(hf_config.vision_config)
        self.merger = MiniCPMV4_6Merger(hf_config).to(dtype=self.vision_tower.dtype)
        self.model_cls = MiniCPMV4_6Model

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs.get('input_ids')
        pixel_values = kwargs.get('pixel_values')
        pixel_values_videos = kwargs.get('pixel_values_videos')
        target_sizes = kwargs.get('target_sizes')
        target_sizes_videos = kwargs.get('target_sizes_videos')
        hf_config = self.hf_config

        if pixel_values is None and pixel_values_videos is None:
            patch_size = hf_config.vision_config.patch_size
            dummy_pv = torch.zeros(
                1, 3, 4 * patch_size, 4 * patch_size,
                device=inputs_embeds.device, dtype=self.vision_tower.dtype)
            dummy_ts = torch.tensor(
                [[4, 4]], device=inputs_embeds.device, dtype=torch.int32)
            with self.patch_hf_config():
                vision_output = self.model_cls.get_image_features(self, dummy_pv, dummy_ts)
            image_embeds = torch.cat(vision_output.pooler_output, dim=0)
            inputs_embeds = inputs_embeds + image_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                num_beams = pixel_values.shape[0]
                with self.patch_hf_config():
                    vision_output = self.model_cls.get_image_features(
                        self, pixel_values[:1].to(dtype=self.vision_tower.dtype), target_sizes)
                    image_features = (
                        torch.cat(vision_output.pooler_output, dim=0)
                        .to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        .repeat(num_beams, 1))
                    mask = self.model_cls.get_placeholder_mask(
                        self, input_ids, inputs_embeds, image_features, hf_config.image_token_id)
                inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)

            if pixel_values_videos is not None:
                num_beams = pixel_values_videos.shape[0]
                with self.patch_hf_config():
                    vision_output = self.model_cls.get_video_features(
                        self, pixel_values_videos[:1].to(dtype=self.vision_tower.dtype), target_sizes_videos)
                    video_features = (
                        torch.cat(vision_output.pooler_output, dim=0)
                        .to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        .repeat(num_beams, 1))
                    mask = self.model_cls.get_placeholder_mask(
                        self, input_ids, inputs_embeds, video_features, hf_config.video_token_id)
                inputs_embeds = inputs_embeds.masked_scatter(mask, video_features)

        return inputs_embeds


class MiniCPMV46Bridge(Qwen3NextGDNBridgeMixin):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.language_model.norm.weight'


register_model(
    ModelMeta(
        ModelType.minicpmv4_6,
        ['minicpmv4_6'],
        bridge_cls=MiniCPMV46Bridge,
        visual_cls=MiniCPMV46Vit,
        loader=Qwen3NextLoader,
    ))
