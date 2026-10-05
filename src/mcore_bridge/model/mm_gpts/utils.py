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
    test_mm_type = 'image'

    @contextmanager
    def patch_hf_config(self):
        config = self.config
        self.config = self.hf_config
        try:
            yield
        finally:
            self.config = config

    @staticmethod
    def set_torch_dtype(hf_config, torch_dtype):
        for key, value in hf_config.__dict__.items():
            if isinstance(value, PretrainedConfig):
                HuggingFaceVit.set_torch_dtype(value, torch_dtype)
            elif key in {'torch_dtype', 'params_dtype', 'dtype'}:
                setattr(hf_config, key, torch_dtype)

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        hf_config = config.hf_config
        self.set_torch_dtype(hf_config, config.params_dtype)
        self.hf_config = hf_config
        self.prepare_attn_impl()
        with patch_get_dynamic_module():
            if config.language_model_only:
                self.prepare_language_model(hf_config)
            else:
                self.prepare_model(hf_config)

        self.to(device='cuda')

    @abstractmethod
    def prepare_model(self, hf_config: PretrainedConfig):
        pass

    def prepare_language_model(self, hf_config: PretrainedConfig):
        pass

    def prepare_attn_impl(self):
        vit_attn_impl = self.config.vit_attn_impl or 'flash_attention_2'
        if self.config.attention_backend.name == 'flash':
            self.hf_config._attn_implementation = vit_attn_impl

    @abstractmethod
    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        pass

    def get_inputs_embeds_language_model(self, inputs_embeds, **kwargs):
        return inputs_embeds

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
        token_types = inputs.get('mm_token_type_ids')
        if token_types is not None:
            if token_types.shape != input_ids.shape:
                raise ValueError('mm_token_type_ids must match input_ids for vision embedding.')
            token_types = token_types.to(device=input_ids.device)
            torch._assert_async(((token_types == 0) | (token_types == 1) | (token_types == 2)).all(),
                                'Unsupported vision modality type.')
            for modality, name in ((1, 'image_token_id'), (2, 'video_token_id')):
                token_id = getattr(hf_config, name, None)
                if token_id is None:
                    torch._assert_async((token_types != modality).all(), 'Unsupported vision modality.')
                else:
                    torch._assert_async(((token_types != modality) | (input_ids == token_id)).all(),
                                        'Vision modality type disagrees with placeholder token id.')
            if pixel_values is None:
                torch._assert_async((token_types != 1).all(), 'Image placeholders require pixel_values.')
            if pixel_values_videos is None:
                torch._assert_async((token_types != 2).all(), 'Video placeholders require pixel_values_videos.')
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
                # Generated special tokens retain their text embedding. Only
                # input placeholders carry a nonzero modality type.
                image_positions = input_ids == hf_config.image_token_id if token_types is None else token_types == 1
                torch._assert_async(image_positions.sum() == image_embeds.shape[0],
                                    'Image placeholder and embedding counts differ.')
                image_mask = image_positions.unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask = image_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if video_embeds is not None:
                video_positions = input_ids == hf_config.video_token_id if token_types is None else token_types == 2
                torch._assert_async(video_positions.sum() == video_embeds.shape[0],
                                    'Video placeholder and embedding counts differ.')
                video_mask = video_positions.unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                video_mask = video_mask.to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds
