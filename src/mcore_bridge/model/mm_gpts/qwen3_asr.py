# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from transformers import PretrainedConfig

from mcore_bridge.bridge import MultimodalGPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model
from .utils import HuggingFaceVit


class Qwen3ASRBridge(MultimodalGPTBridge):
    hf_layers_prefix = 'thinker.model.layers'
    hf_embed_key = 'thinker.model.embed_tokens.weight'
    hf_final_layernorm_key = 'thinker.model.norm.weight'
    hf_lm_head_key = 'thinker.lm_head.weight'
    hf_score_key = 'thinker.score.weight'


class Qwen3ASRVit(HuggingFaceVit):
    module_mapping = {'thinker.audio_tower': 'audio_tower'}
    _vision_tower = ['audio_tower']
    _aligner = ['audio_tower.proj1', 'audio_tower.proj2']
    test_mm_type = 'audio'
    media_input_keys = ('input_features',)

    def prepare_model(self, hf_config: PretrainedConfig):
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (Qwen3ASRAudioEncoder,
                                                                           Qwen3ASRThinkerForConditionalGeneration)
        self.audio_tower = Qwen3ASRAudioEncoder._from_config(hf_config.thinker_config.audio_config)
        self.model_cls = Qwen3ASRThinkerForConditionalGeneration

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        hf_config = self.hf_config.thinker_config
        input_features = kwargs.get('input_features')
        feature_attention_mask = kwargs.get('feature_attention_mask')

        if input_features is None:
            input_features = input_ids.new_zeros([1, 128, 128], dtype=self.audio_tower.dtype)
            feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
            audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
            inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        else:
            audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
            audio_mask = (input_ids == hf_config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        return inputs_embeds

    def get_audio_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_audio_features(self, *args, **kwargs)


register_model(ModelMeta(
    ModelType.qwen3_asr,
    ['qwen3_asr'],
    bridge_cls=Qwen3ASRBridge,
    visual_cls=Qwen3ASRVit,
))
