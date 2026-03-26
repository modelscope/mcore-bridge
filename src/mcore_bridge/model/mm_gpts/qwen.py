# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mcore_bridge.bridge import GPTBridge, MultimodalGPTBridge
from mcore_bridge.utils import get_env_args

from ..constant import ModelType
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


class Qwen2_5VL_Vit(HuggingFaceVit):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']
    version = 'v2_5'

    def prepare_model(self, hf_config: PretrainedConfig):
        if self.version == 'v2_5':
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
                Qwen2_5_VisionTransformerPretrainedModel as VisionModel
        elif self.version == 'v2':
            from transformers.models.qwen2_vl.modeling_qwen2_vl import \
                Qwen2VisionTransformerPretrainedModel as VisionModel
        self.visual = VisionModel._from_config(hf_config.vision_config)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return self._hf_get_inputs_embeds(inputs_embeds, kwargs, self.visual, self.processor, self.hf_config)


class Qwen2_5VLBridge(MultimodalGPTBridge):
    # Compatible with older versions of transformers
    hf_state_dict_mapping = {
        'model.layers': 'model.language_model.layers',
        'model.embed_tokens': 'model.language_model.embed_tokens',
        'model.norm': 'model.language_model.norm',
        'visual': 'model.visual',
    }


register_model(ModelMeta(
    ModelType.qwen2_5_vl,
    ['qwen2_5_vl'],
    bridge_cls=Qwen2_5VLBridge,
    visual_cls=Qwen2_5VL_Vit,
))


class Qwen2VL_Vit(Qwen2_5VL_Vit):
    version = 'v2'


register_model(ModelMeta(
    ModelType.qwen2_vl,
    ['qwen2_vl'],
    bridge_cls=Qwen2_5VLBridge,
    visual_cls=Qwen2VL_Vit,
))


class Qwen2_5OmniBridge(GPTBridge):
    hf_layers_prefix = 'thinker.model.layers'
    hf_embed_key = 'thinker.model.embed_tokens.weight'
    hf_final_layernorm_key = 'thinker.model.norm.weight'
    hf_lm_head_key = 'thinker.lm_head.weight'
    hf_score_key = 'thinker.score.weight'


class Qwen2_5Omni_Vit(HuggingFaceVit):
    module_mapping = {'thinker.audio_tower': 'audio_tower', 'thinker.visual': 'visual'}
    _vision_tower = ['audio_tower', 'visual']
    _aligner = ['audio_tower.proj', 'visual.merger']
    _generator = ['talker', 'token2wav']

    def prepare_model(self, hf_config: PretrainedConfig):
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (Qwen2_5OmniAudioEncoder,
                                                                            Qwen2_5OmniThinkerForConditionalGeneration,
                                                                            Qwen2_5OmniVisionEncoder)
        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(hf_config.thinker_config.audio_config)
        self.visual = Qwen2_5OmniVisionEncoder._from_config(hf_config.thinker_config.vision_config)
        self.model_cls = Qwen2_5OmniThinkerForConditionalGeneration

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        thinker_config = self.hf_config.thinker_config
        inputs_embeds = self._hf_get_inputs_embeds(inputs_embeds, kwargs, self.visual, self.processor, thinker_config)
        input_ids = kwargs['input_ids']
        input_features = kwargs.get('input_features')
        feature_attention_mask = kwargs.get('feature_attention_mask')

        if input_features is None:
            input_features = input_ids.new_zeros([1, 128, 128], dtype=self.audio_tower.dtype)
            feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
            audio_res = self.get_audio_features(input_features, feature_attention_mask)
            if hasattr(audio_res, 'last_hidden_state'):
                audio_embeds = audio_res.last_hidden_state
            else:
                audio_embeds = audio_res
            inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_res = self.get_audio_features(input_features, feature_attention_mask)
            if hasattr(audio_res, 'last_hidden_state'):
                audio_embeds = audio_res.last_hidden_state
            else:
                audio_embeds = audio_res
            audio_mask = (input_ids == thinker_config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        return inputs_embeds

    def get_audio_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_audio_features(self, *args, **kwargs)


register_model(
    ModelMeta(
        ModelType.qwen2_5_omni,
        ['qwen2_5_omni'],
        bridge_cls=Qwen2_5OmniBridge,
        visual_cls=Qwen2_5Omni_Vit,
    ))


class Ovis2_5Bridge(GPTBridge):
    hf_layers_prefix = 'llm.model.layers'
    hf_embed_key = 'llm.model.embed_tokens.weight'
    hf_final_layernorm_key = 'llm.model.norm.weight'
    hf_lm_head_key = 'llm.lm_head.weight'
    hf_score_key = 'llm.score.weight'


class Ovis2_5Vit(HuggingFaceVit):
    module_mapping = {'visual_tokenizer': 'visual_tokenizer', 'vte': 'vte'}
    _vision_tower = ['visual_tokenizer.vit', 'vte']
    _aligner = ['visual_tokenizer.head']

    def prepare_model(self, hf_config):
        self.min_pixels = get_env_args('min_pixels', int, 448 * 448)
        self.max_pixels = get_env_args('max_pixels', int, 1344 * 1792)
        VisualEmbedding = get_class_from_dynamic_module('modeling_ovis2_5.VisualEmbedding', hf_config.name_or_path)
        INDICATOR_IDS = get_class_from_dynamic_module('modeling_ovis2_5.INDICATOR_IDS', hf_config.name_or_path)
        VisualTokenizer = get_class_from_dynamic_module('modeling_ovis2_5.VisualTokenizer', hf_config.name_or_path)
        vit = AutoModel.from_config(hf_config.vit_config)
        self.visual_tokenizer = VisualTokenizer(
            vit=vit, visual_vocab_size=hf_config.visual_vocab_size, image_processor_name_or_path=hf_config.name_or_path)
        self.visual_tokenizer.head.to(dtype=vit.dtype)
        self.vte = VisualEmbedding(
            hf_config.visual_vocab_size, hf_config.hidden_size, device=vit.device, dtype=vit.dtype)
        self.indicator_token_indices = torch.arange(
            hf_config.visual_vocab_size - len(INDICATOR_IDS), hf_config.visual_vocab_size, dtype=torch.long)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values', None)
        grid_thws = kwargs.get('grid_thws')
        INDICATOR_IDS = [-301, -302, -303, -304]
        VISUAL_ATOM_ID = -300
        device = inputs_embeds.device
        visual_indicator_embeds = self.vte(self.indicator_token_indices.to(device=device)).to(
            dtype=inputs_embeds.dtype, device=device)
        inputs_embeds = inputs_embeds.clone()
        for i, indicator_id in enumerate(INDICATOR_IDS):
            inputs_embeds[input_ids == indicator_id] = visual_indicator_embeds[i]
        if pixel_values is None:
            pixel_values, grid_thws = self.visual_tokenizer.preprocess(
                Image.new('RGB', (32, 32), (0, 0, 0)), min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            pixel_values = pixel_values.to(device=inputs_embeds.device)
            grid_thws = grid_thws.to(device=inputs_embeds.device)
            visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = self.vte(visual_tokens).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds += visual_embeds.mean() * 0.
        else:
            visual_tokens = self.visual_tokenizer(pixel_values, grid_thws)
            visual_embeds = self.vte(visual_tokens).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds[input_ids == VISUAL_ATOM_ID] = visual_embeds
        return inputs_embeds


register_model(ModelMeta(
    ModelType.ovis2_5,
    ['ovis2_5'],
    bridge_cls=Ovis2_5Bridge,
    visual_cls=Ovis2_5Vit,
))
