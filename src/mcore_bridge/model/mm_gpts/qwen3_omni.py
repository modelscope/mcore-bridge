# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from mcore_bridge.bridge import GPTBridge

from ..constant import ModelType
from ..register import ModelMeta, register_model
from .qwen3_vl import Qwen3VL_Vit, Qwen3VLLoader
from .utils import HuggingFaceVit


class Qwen3OmniBridge(GPTBridge):
    hf_layers_prefix = 'thinker.model.layers'
    hf_embed_key = 'thinker.model.embed_tokens.weight'
    hf_final_layernorm_key = 'thinker.model.norm.weight'
    hf_lm_head_key = 'thinker.lm_head.weight'
    hf_score_key = 'thinker.score.weight'


class Qwen3Omni_Vit(HuggingFaceVit):
    module_mapping = {'thinker.audio_tower': 'audio_tower', 'thinker.visual': 'visual'}
    _vision_tower = ['audio_tower', 'visual']
    _aligner = ['audio_tower.proj1', 'audio_tower.proj2', 'visual.merger', 'visual.merger_list']
    _generator = ['talker', 'code2wav']

    def prepare_model(self, hf_config):
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeVisionEncoder)
        self.model_cls = Qwen3OmniMoeThinkerForConditionalGeneration
        self.audio_tower = Qwen3OmniMoeAudioEncoder._from_config(hf_config.thinker_config.audio_config)
        self.visual = Qwen3OmniMoeVisionEncoder._from_config(hf_config.thinker_config.vision_config)

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        input_ids = kwargs['input_ids']
        visual = self.visual
        hf_config = self.hf_config.thinker_config
        res = Qwen3VL_Vit._get_inputs_embeds(self, inputs_embeds, kwargs, visual, hf_config)
        inputs_embeds = res['inputs_embeds']
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
            audio_mask = (input_ids == hf_config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        res['inputs_embeds'] = inputs_embeds
        return res

    def get_audio_features(self, *args, **kwargs):
        with self.patch_hf_config():
            return self.model_cls.get_audio_features(self, *args, **kwargs)


register_model(
    ModelMeta(
        ModelType.qwen3_omni,
        ['qwen3_omni_moe'],
        bridge_cls=Qwen3OmniBridge,
        visual_cls=Qwen3Omni_Vit,
        loader=Qwen3VLLoader,
    ))
