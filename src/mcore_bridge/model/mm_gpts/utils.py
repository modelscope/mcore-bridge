# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from abc import ABC, abstractmethod
from contextlib import contextmanager
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from transformers import PretrainedConfig, dynamic_module_utils

from mcore_bridge.config import ModelConfig
from mcore_bridge.utils import safe_ddp_context


@contextmanager
def patch_get_dynamic_module():
    origin_get_cached_module_file = dynamic_module_utils.get_cached_module_file

    def new_get_cached_module_file(pretrained_model_name_or_path, *args, **kwargs):
        with safe_ddp_context(hash_id=str(pretrained_model_name_or_path), use_barrier=False):
            return origin_get_cached_module_file(pretrained_model_name_or_path, *args, **kwargs)

    dynamic_module_utils.get_cached_module_file = new_get_cached_module_file
    try:
        yield
    finally:
        dynamic_module_utils.get_cached_module_file = origin_get_cached_module_file


class HuggingFaceVit(_HuggingFaceModule, ABC):
    module_mapping = {}  # hf -> mcore
    support_multimodal = True

    @contextmanager
    def patch_hf_config(self):
        config = self.config
        self.config = self.hf_config
        try:
            yield
        finally:
            self.config = config

    def __init__(self, config: ModelConfig, ignore_init_model_cls=None):
        super().__init__(config)
        hf_config = config.hf_config
        hf_config.torch_dtype = config.params_dtype
        self.hf_config = hf_config
        self.prepare_attn_impl()
        with patch_get_dynamic_module():
            self.prepare_model(hf_config)
        self.to(device='cuda')

    @abstractmethod
    def prepare_model(self, hf_config: PretrainedConfig):
        pass

    def prepare_attn_impl(self):
        vit_attn_impl = self.config.vit_attn_impl or 'flash_attention_2'
        if self.config.attention_backend.name == 'flash':
            self.hf_config._attn_implementation = vit_attn_impl

    @abstractmethod
    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        pass

    @staticmethod
    def _get_vision_config(hf_config):
        for k in ['vision_config', 'vit_config']:
            if hasattr(hf_config, k):
                return getattr(hf_config, k)

    @staticmethod
    def _hf_get_inputs_embeds(inputs_embeds, inputs, visual, hf_config):
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        dtype = visual.dtype
        vision_config = HuggingFaceVit._get_vision_config(hf_config)
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            hidden_size = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
            pixel_values = torch.zeros(16 * 16, hidden_size, dtype=dtype, device=input_ids.device)
            image_grid_thw = input_ids.new_tensor([[1, 16, 16]])
            image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
            if hasattr(image_embeds, 'pooler_output'):
                image_embeds = image_embeds.pooler_output
            inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
        else:
            if pixel_values is None:
                pixel_values_mixed = pixel_values_videos
                grid_thw = video_grid_thw
            elif pixel_values_videos is None:
                pixel_values_mixed = pixel_values
                grid_thw = image_grid_thw
            else:
                pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
                grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
            pixel_values_mixed = pixel_values_mixed.type(dtype)
            mixed_embeds = visual(pixel_values_mixed, grid_thw=grid_thw)
            if hasattr(mixed_embeds, 'pooler_output'):
                mixed_embeds = mixed_embeds.pooler_output
            if pixel_values is None:
                image_embeds = None
                video_embeds = mixed_embeds
            elif pixel_values_videos is None:
                image_embeds = mixed_embeds
                video_embeds = None
            else:
                merge_length = vision_config.spatial_merge_size**2
                image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
                image_embeds = mixed_embeds[:image_tokens]
                video_embeds = mixed_embeds[image_tokens:]

            if image_embeds is not None:
                image_mask = (input_ids == hf_config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask = image_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_mask = (input_ids == hf_config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                video_mask = video_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds
